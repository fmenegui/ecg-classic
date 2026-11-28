"""Start, follow and terminate a RunPod GPU pod (used by .github/workflows/gpu-train.yml).

    python hybrid/gpu/runpod_ctl.py start            # prints pod_id=... (GitHub step output)
    python hybrid/gpu/runpod_ctl.py follow --out DIR  # streams the pod log, fetches the result
    python hybrid/gpu/runpod_ctl.py stop             # terminates $POD_ID and this run's other pods
    python hybrid/gpu/runpod_ctl.py run --out DIR    # start + follow, replacing pods whose CUDA fails

Environment: RUNPOD_API_KEY; start: GPU, MAX_HOURS, JOB, REF; follow/stop: POD_ID.
Standard library only.
"""
from __future__ import annotations

import argparse
import io
import json
import os
import re
import sys
import tarfile
import time
import urllib.error
import urllib.request
from pathlib import Path

API = "https://api.runpod.io/graphql"
IMAGE = "runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04"
# cheap cards only, best cost per training step first (the crops come from the CPU generator, so a
# 4090 mostly waits for data: 16/11, A5000 0.27 USD/h vs 4090 0.89). A card outside this list is used
# only when asked for by name (workflow input "gpu").
FALLBACK_GPUS = ["NVIDIA RTX A5000", "NVIDIA GeForce RTX 3090", "NVIDIA RTX A4500", "NVIDIA RTX 4000 Ada Generation",
                 "NVIDIA RTX A4000", "NVIDIA L4", "NVIDIA RTX 2000 Ada Generation", "NVIDIA GeForce RTX 3080"]
WAIT_MIN = float(os.environ.get("WAIT_MIN", "45"))   # no cheap card free: retry every 3 min for this long
MAX_USD_H = float(os.environ.get("MAX_USD_H", "0.5"))   # never start a pod above this price (unless GPU asks)
# one name per workflow run: "stop" must only terminate this run's pods, never a parallel training
NAME = "ecgdig-train-" + os.environ.get("GITHUB_RUN_ID", "local")


def gql(query, variables=None):
    key = os.environ.get("RUNPOD_API_KEY", "").strip().strip("\"'")
    if not key:
        sys.exit("RUNPOD_API_KEY is not set (repository secret missing?)")
    body = json.dumps({"query": query, "variables": variables or {}}).encode()
    hdr = {"Content-Type": "application/json", "User-Agent": "ecgdig-gpu-train"}
    try:   # bearer header first; older keys only work as the api_key query parameter
        req = urllib.request.Request(API, data=body, headers=dict(hdr, Authorization=f"Bearer {key}"))
        with urllib.request.urlopen(req, timeout=60) as r:
            out = json.loads(r.read())
    except urllib.error.HTTPError as e:
        if e.code not in (401, 403):
            raise
        req = urllib.request.Request(f"{API}?api_key={key}", data=body, headers=hdr)
        with urllib.request.urlopen(req, timeout=60) as r:
            out = json.loads(r.read())
    if out.get("errors"):
        raise RuntimeError("; ".join(e.get("message", str(e)) for e in out["errors"]))
    return out["data"]


def check():
    """Fails fast with a clear message when the key is not accepted."""
    try:
        me = gql("query { myself { id clientBalance } }")["myself"]
    except urllib.error.HTTPError as e:
        k = os.environ.get("RUNPOD_API_KEY", "").strip()
        print(f"key length {len(k)}, starts with {k[:4]!r}...", file=sys.stderr)
        try:   # does the REST API (v1) accept it? tells a wrong key from a key without GraphQL rights
            rq = urllib.request.Request("https://rest.runpod.io/v1/pods", headers={"Authorization": f"Bearer {k}",
                                                                                  "User-Agent": "ecgdig-gpu-train"})
            with urllib.request.urlopen(rq, timeout=60) as r:
                print(f"REST API accepts the key (HTTP {r.status})", file=sys.stderr)
        except urllib.error.HTTPError as e2:
            print(f"REST API: HTTP {e2.code} {e2.read()[:200]!r}", file=sys.stderr)
        except Exception as e2:  # noqa: BLE001
            print(f"REST API not reachable: {e2}", file=sys.stderr)
        try:
            print(f"GraphQL said: {e.read()[:300]!r}", file=sys.stderr)
        except Exception:  # noqa: BLE001
            pass
        sys.exit(f"RunPod rejected the API key (HTTP {e.code}): check that the secret RUNPOD_API_KEY holds a "
                 "valid key with read/write (GraphQL) permission")
    bal = float(me.get("clientBalance") or 0.0)
    print(f"RunPod account ok, balance {bal:.2f} USD", file=sys.stderr)
    # a pod is stopped by RunPod when the balance reaches 0 (two runs were lost that way, at
    # 4 h of 5 h): do not start one the balance cannot pay for, at the price cap
    need = float(os.environ.get("MAX_HOURS", "1")) * MAX_USD_H
    if bal < need:
        sys.exit(f"RunPod balance {bal:.2f} USD is below the {need:.2f} USD this run may cost "
                 f"({os.environ.get('MAX_HOURS', '1')} h at up to {MAX_USD_H} USD/h): add funds")


def start(exclude=(), cloud="ALL"):
    check()
    ref = os.environ["REF"]
    init = _init_files()
    extra = []
    if os.environ.get("UPLOAD_TOKEN"):
        extra.append({"key": "UPLOAD_TOKEN", "value": os.environ["UPLOAD_TOKEN"]})
    if init:   # the job starts once the checkpoints to continue from are on the pod
        extra.append({"key": "WAIT_IN", "value": "1"})
    env = extra + [{"key": "REF", "value": ref}, {"key": "JOB", "value": os.environ["JOB"]},
           {"key": "MAX_HOURS", "value": os.environ.get("MAX_HOURS", "1")}]
    for k in ("KAGGLE_API_TOKEN", "KAGGLE_USERNAME"):   # only when the run asks for it (input kaggle)
        if os.environ.get(k):
            env.append({"key": k, "value": os.environ[k]})
    if os.environ.get("RENDERER_TOKEN"):   # read-only token for the private ecg-torch-renderer repo, if set
        env.append({"key": "RENDERER_TOKEN", "value": os.environ["RENDERER_TOKEN"]})
    cmd = ("bash -c 'curl -fsSL https://raw.githubusercontent.com/fmenegui/ecg-classic/"
           "${REF}/hybrid/gpu/pod.sh -o /pod.sh && bash /pod.sh'")
    asked = os.environ.get("GPU") or ""
    gpus = ([asked] if asked else []) + [g for g in FALLBACK_GPUS if g != asked]
    gpus = [g for g in gpus if g not in exclude] + [g for g in gpus if g in exclude]
    q = """mutation($input: PodFindAndDeployOnDemandInput) {
             podFindAndDeployOnDemand(input: $input) { id machineId costPerHr gpuCount } }"""
    last = None
    vid, vdc = os.environ.get("VOLUME_ID", ""), os.environ.get("DATA_CENTER", "")
    if vid and vdc:
        # first the data center of the volume: the bank and the real strips are then cached
        run = os.environ.get("GITHUB_RUN_ID", "local")
        venv = env + [{"key": "OUT_DIR", "value": f"/workspace/runs/{run}/out"},
                      {"key": "IN_DIR", "value": f"/workspace/runs/{run}/in"},
                      {"key": "REPO_DIR", "value": "/root/repo"}, {"key": "CACHE", "value": "/workspace/cache"}]
        for attempt in range(int(os.environ.get("VOLUME_TRIES", "3"))):
            got = _try_cards(gpus, asked, q, "SECURE", venv, cmd, (vid, vdc))
            if isinstance(got, tuple):
                print(f"on the data volume {vid} ({vdc})", file=sys.stderr)
                return got
            print(f"no cheap card in {vdc} with the volume ({got}); retry {attempt + 1}", file=sys.stderr)
            time.sleep(60)
        if os.environ.get("VOLUME_ONLY") == "true":
            # the job needs what is on the volume (built strips): a pod elsewhere would only stop
            sys.exit(f"no cheap card in {vdc} with the volume after {os.environ.get('VOLUME_TRIES', '3')} tries")
        print("training without the data volume (nothing cached)", file=sys.stderr)
    env = env + [{"key": "CACHE", "value": "/workspace/cache"}]
    t_end = time.time() + 60 * WAIT_MIN
    while True:
        got = _try_cards(gpus, asked, q, cloud, env, cmd)
        if isinstance(got, tuple):
            return got
        last = got
        if time.time() > t_end:
            break
        print(f"no cheap card free, retrying in 3 min (until {WAIT_MIN:.0f} min)", file=sys.stderr, flush=True)
        time.sleep(180)
    sys.exit(f"no GPU available: {last}")


def _try_cards(gpus, asked, q, cloud, env, cmd, vol=None):
    last = None
    # preferred GPU first, with as many vCPU as possible (the crops are generated on the CPU)
    # a cheap card with 8 vCPU beats an expensive one with more: every card at 16 vCPU first, then at 8
    dcs = [d for d in os.environ.get("GPU_DCS", "").split(",") if d] if not vol else []
    for vcpu, g, dc in [(16, g, dc) for g in gpus for dc in (dcs or [None])] + [(8, g, dc) for g in gpus for dc in (dcs or [None])]:
        inp = dict(cloudType=cloud, gpuCount=1, volumeInGb=0, containerDiskInGb=int(os.environ.get("DISK_GB") or 60), minVcpuCount=vcpu,
                   minMemoryInGb=int(os.environ.get("MIN_RAM_GB", "46")), gpuTypeId=g, name=NAME, imageName=IMAGE, dockerArgs=cmd,
                   ports="8000/http", volumeMountPath="/workspace", env=env)
        if vol:   # the data volume (caches of the bank and of the real strips): its data center only
            inp.update(cloudType="SECURE", networkVolumeId=vol[0], dataCenterId=vol[1])
        elif dc:  # GPU_DCS: only the data centers asked for (one that does not answer through the proxy left out)
            inp.update(dataCenterId=dc)
        try:
            d = gql(q, {"input": inp})["podFindAndDeployOnDemand"]
        except Exception as e:  # noqa: BLE001
            last = e
            print(f"{g} ({vcpu} vCPU{', ' + dc if dc else ''}): {e}", file=sys.stderr)
            continue
        if d and d.get("id") and g != asked and float(d.get("costPerHr") or 0) > MAX_USD_H:
            print(f"{g}: {d.get('costPerHr')} USD/h is above MAX_USD_H={MAX_USD_H}, terminating it", file=sys.stderr)
            gql("mutation($input: PodTerminateInput!) { podTerminate(input: $input) }", {"input": {"podId": d["id"]}})
            continue
        if d and d.get("id"):
            print(f"started {d['id']} on {g} (>= {vcpu} vCPU, {cloud} cloud), {d.get('costPerHr')} USD/h", file=sys.stderr)
            print(f"pod_id={d['id']}")
            # an annotation: the pod id is visible on the run while it runs (to look at its log)
            print(f"::notice title=pod::{d['id']} {g}", flush=True)
            print(f"gpu={g}")
            return d["id"], g
    return last


def _get(url, timeout=30, total=None):
    """GET with a hard limit on the whole transfer (`total`, default 4 x timeout), not only on each
    socket operation: a response trickling through the proxy once hung the follower for 5 h, so
    the run saw neither the log nor the checkpoints of a pod that kept training."""
    total = total or 4 * timeout
    t_end = time.time() + total
    req = urllib.request.Request(url, headers={"User-Agent": "ecgdig-gpu-train"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        buf = []
        while True:
            if time.time() > t_end:
                raise TimeoutError(f"{url.rsplit('/', 1)[-1]}: no complete answer in {total:.0f} s")
            b = r.read(1 << 20)
            if not b:
                return b"".join(buf)
            buf.append(b)


LOST_MIN = 20
BOOT_MIN = float(os.environ.get("BOOT_MIN", "35"))   # image pull + setup took up to ~25 min on slow hosts
STALL_MIN = 25     # pod.sh logs the GPU use every 2 min: a log that stops growing means a dead job
CKPT_MIN = 30
CKPT_FILES = ["page_best.onnx", "page_best.pt", "unet_best.pt", "trace_seg_best.onnx", "dewarp_best.onnx",
              "dewarp_best.pt", "val.json"]


def _copy_checkpoints(base, out):
    """Copies the best checkpoints the training has written so far: a pod that dies late in a
    run (it happened at 4 h 10 min of 5 h) no longer takes the whole run with it."""
    got = []
    for f in CKPT_FILES:
        try:
            data = _get(base + "/" + f, timeout=600)
        except Exception:  # noqa: BLE001  (not written yet, or not this kind of training)
            continue
        tmp = out / (f + ".part")
        tmp.write_bytes(data)
        tmp.replace(out / f)
        got.append(f"{f} {len(data) // 1_000_000} MB")
    if got:
        print(f"[checkpoints copied: {', '.join(got)}]", flush=True)


def follow(out: Path):
    pod = os.environ["POD_ID"]
    base = f"https://{pod}-8000.proxy.runpod.net"
    limit = time.time() + float(os.environ.get("MAX_HOURS", "1")) * 3600 + 40 * 60   # + setup margin
    if os.environ.get("FOLLOW_UNTIL"):
        # the GitHub job has a hard time limit: stop following before it, so that the checkpoints
        # are copied and uploaded (a job cancelled by the limit uploads nothing)
        limit = min(limit, float(os.environ["FOLLOW_UNTIL"]))
    seen = 0
    status = ""
    out.mkdir(parents=True, exist_ok=True)
    last_ok, last_ckpt, last_grow = time.time(), time.time(), time.time()
    t_start = time.time()
    while time.time() < limit:
        try:
            status = _get(base + "/status.txt").decode().strip()
            log = _get(base + "/log.txt", timeout=60).decode(errors="replace")
            if len(log) > seen:
                print(log[seen:], end="", flush=True)
                seen = len(log)
                last_grow = time.time()
            last_ok = time.time()
        except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as e:   # pod still booting
            print(f"[waiting: {getattr(e, 'code', '')} {type(e).__name__}]", flush=True)
        if status in ("done", "nogpu"):
            break
        if not seen and time.time() - t_start > BOOT_MIN * 60:
            # a pod that never answered (EUR-IS-1 pods with the volume, Nov 19-20) will not:
            # stop paying for it
            print(f"the pod never answered in {BOOT_MIN} min: giving up on it", flush=True)
            status = "lost"
            break
        if seen and time.time() - last_ok > LOST_MIN * 60:
            # the pod answered before and has not for LOST_MIN minutes (host reset or lost):
            # stop waiting and keep the last checkpoints already copied
            print(f"pod unreachable for {LOST_MIN} min: giving up on it", flush=True)
            status = "lost"
            break
        # any status but done: on Nov 27 two pods on ecgdig-data3 kept answering through the proxy
        # with status.txt empty and a log that had stopped (~23:20 UTC); `status == "running"`
        # never matched, and both ran 3 more hours to their time limit (~3 USD for nothing)
        if seen and status not in ("done", "nogpu") and time.time() - last_grow > STALL_MIN * 60:
            print(f"the pod log has not grown for {STALL_MIN} min (status {status!r}): the job is stuck",
                  flush=True)
            status = "stalled"
            break
        if seen and time.time() - last_ok < 120 and time.time() - last_ckpt > CKPT_MIN * 60:
            _copy_checkpoints(base, out)
            last_ckpt = time.time()
        time.sleep(60)
    if status not in ("done", "nogpu", "lost"):
        try:   # time limit or stalled job: the latest checkpoints before giving up
            _copy_checkpoints(base, out)
        except Exception as e:  # noqa: BLE001
            print(f"last checkpoint copy failed: {e}", flush=True)
    if status == "lost":
        sys.exit("pod lost; the artifact holds the last checkpoints copied")
    if status == "stalled":
        sys.exit("job stalled on the pod; the artifact holds the last checkpoints copied")
    if status == "nogpu":
        return status
    try:
        data = _get(base + "/result.tar.gz", timeout=600)
        tarfile.open(fileobj=io.BytesIO(data), mode="r:gz").extractall(out)
        print(f"result: {sorted(p.name for p in out.iterdir())}", flush=True)
    except Exception as e:  # noqa: BLE001
        print(f"no result archive ({e}); keeping the log only", flush=True)
        try:
            (out / "log.txt").write_bytes(_get(base + "/log.txt", timeout=60))
        except Exception:  # noqa: BLE001
            pass
    if status != "done":
        sys.exit(f"pod did not finish in time (status: {status!r}); the artifact holds the last checkpoints")
    # the pod finishes either way; the job's own exit code decides whether the run succeeded
    # (a failed training must not start its benchmark)
    try:
        full = (out / "log.txt").read_text(errors="replace") if (out / "log.txt").exists() else \
            _get(base + "/log.txt", timeout=60).decode(errors="replace")
    except Exception:  # noqa: BLE001
        full = ""
    codes = re.findall(r"^exit code (\d+)\s*$", full, flags=re.M)
    if codes and codes[-1] != "0":
        sys.exit(f"the job failed on the pod (exit code {codes[-1]})")
    return status


def _init_files():
    d = Path(os.environ.get("INIT_DIR", "init"))
    return sorted(p for p in d.glob("*") if p.is_file()) if d.exists() else []


def _send_init(pod):
    """Checkpoints of an earlier run (workflow input init_run) to the pod's /workspace/in, then
    READY: a training longer than one GitHub job continues from them."""
    files = _init_files()
    if not files:
        return
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import runpod_v2
    os.environ["POD_ID"] = pod
    ready = Path("init_READY"); ready.write_text("ok")
    runpod_v2.put([str(f) for f in files] + [str(ready.rename(ready.with_name("READY")))])


def run(out: Path, tries=3):
    """start + follow; a pod whose CUDA does not work is terminated and replaced (up to `tries`)."""
    bad = []
    for _ in range(tries):
        # after a broken pod: secure cloud (datacenter hosts), where CUDA has always worked
        pod, g = start(cloud="SECURE" if bad else "ALL")
        os.environ["POD_ID"] = pod
        Path("pod_ids.txt").open("a").write(pod + "\n")
        _send_init(pod)
        if follow(out) != "nogpu":
            return
        print(f"pod {pod} ({g}): CUDA not usable, terminating it and trying another pod", flush=True)
        stop()
        bad.append(g)
    sys.exit("no pod with a working GPU")


def stop_all():
    """Terminates every training pod of this account (names starting with ecgdig-train) and reports
    what is left running."""
    mine = gql("query { myself { clientBalance pods { id name costPerHr desiredStatus } } }")["myself"]
    only = [x.strip() for x in os.environ.get("POD_IDS", "").split(",") if x.strip()]
    for p in mine["pods"] or []:
        if (p["id"] in only) if only else (p.get("name") or "").startswith("ecgdig-train"):
            gql("mutation($input: PodTerminateInput!) { podTerminate(input: $input) }", {"input": {"podId": p["id"]}})
            print(f"terminated {p['id']} {p['name']} ({p.get('costPerHr')} USD/h)")
        else:
            print(f"left running: {p['id']} {p.get('name')} ({p.get('costPerHr')} USD/h)")
    print(f"balance {mine.get('clientBalance')} USD")


def stop():
    pods = []
    if os.environ.get("POD_ID"):
        pods.append(os.environ["POD_ID"])
    try:
        mine = gql("query { myself { pods { id name } } }")["myself"]["pods"] or []
        pods += [p["id"] for p in mine if p.get("name") == NAME and p["id"] not in pods]
    except Exception as e:  # noqa: BLE001
        print(f"could not list pods: {e}", file=sys.stderr)
    for p in pods:
        try:
            gql("mutation($input: PodTerminateInput!) { podTerminate(input: $input) }", {"input": {"podId": p}})
            print(f"terminated {p}")
        except Exception as e:  # noqa: BLE001
            print(f"terminate {p}: {e}", file=sys.stderr)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["start", "follow", "stop", "run", "stop-all"])
    ap.add_argument("--out", type=Path, default=Path("result"))
    a = ap.parse_args()
    {"start": start, "follow": lambda: follow(a.out), "stop": stop, "run": lambda: run(a.out),
     "stop-all": stop_all}[a.cmd]()
