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
    env = [{"key": "REF", "value": ref}, {"key": "JOB", "value": os.environ["JOB"]},
           {"key": "MAX_HOURS", "value": os.environ.get("MAX_HOURS", "1")}]
    for k in ("KAGGLE_API_TOKEN", "KAGGLE_USERNAME"):   # only when the run asks for it (input kaggle)
        if os.environ.get(k):
            env.append({"key": k, "value": os.environ[k]})
    if os.environ.get("RENDERER_TOKEN"):   # read-only token for the private ecg-torch-renderer repo, if set
        env.append({"key": "RENDERER_TOKEN", "value": os.environ["RENDERER_TOKEN"]})
    cmd = ("bash -c 'curl -fsSL https://raw.githubusercontent.com/fmenegui/ecg-digitalization-classic/"
           "${REF}/hybrid/gpu/pod.sh -o /pod.sh && bash /pod.sh'")
    asked = os.environ.get("GPU") or ""
    gpus = ([asked] if asked else []) + [g for g in FALLBACK_GPUS if g != asked]
    gpus = [g for g in gpus if g not in exclude] + [g for g in gpus if g in exclude]
    q = """mutation($input: PodFindAndDeployOnDemandInput) {
             podFindAndDeployOnDemand(input: $input) { id machineId costPerHr gpuCount } }"""
    last = None
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


def _try_cards(gpus, asked, q, cloud, env, cmd):
    last = None
    # preferred GPU first, with as many vCPU as possible (the crops are generated on the CPU)
    # a cheap card with 8 vCPU beats an expensive one with more: every card at 16 vCPU first, then at 8
    for vcpu, g in [(16, g) for g in gpus] + [(8, g) for g in gpus]:
        inp = dict(cloudType=cloud, gpuCount=1, volumeInGb=0, containerDiskInGb=60, minVcpuCount=vcpu,
                   minMemoryInGb=30, gpuTypeId=g, name=NAME, imageName=IMAGE, dockerArgs=cmd,
                   ports="8000/http", volumeMountPath="/workspace", env=env)
        try:
            d = gql(q, {"input": inp})["podFindAndDeployOnDemand"]
        except Exception as e:  # noqa: BLE001
            last = e
            print(f"{g} ({vcpu} vCPU): {e}", file=sys.stderr)
            continue
        if d and d.get("id") and g != asked and float(d.get("costPerHr") or 0) > MAX_USD_H:
            print(f"{g}: {d.get('costPerHr')} USD/h is above MAX_USD_H={MAX_USD_H}, terminating it", file=sys.stderr)
            gql("mutation($input: PodTerminateInput!) { podTerminate(input: $input) }", {"input": {"podId": d["id"]}})
            continue
        if d and d.get("id"):
            print(f"started {d['id']} on {g} (>= {vcpu} vCPU, {cloud} cloud), {d.get('costPerHr')} USD/h", file=sys.stderr)
            print(f"pod_id={d['id']}")
            print(f"gpu={g}")
            return d["id"], g
    return last


def _get(url, timeout=30):
    req = urllib.request.Request(url, headers={"User-Agent": "ecgdig-gpu-train"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


LOST_MIN = 20
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
    seen = 0
    status = ""
    out.mkdir(parents=True, exist_ok=True)
    last_ok, last_ckpt = time.time(), time.time()
    while time.time() < limit:
        try:
            status = _get(base + "/status.txt").decode().strip()
            log = _get(base + "/log.txt", timeout=60).decode(errors="replace")
            if len(log) > seen:
                print(log[seen:], end="", flush=True)
                seen = len(log)
            last_ok = time.time()
        except (urllib.error.URLError, TimeoutError, ConnectionError) as e:   # pod still booting
            print(f"[waiting: {getattr(e, 'code', '')} {type(e).__name__}]", flush=True)
        if status in ("done", "nogpu"):
            break
        if seen and time.time() - last_ok > LOST_MIN * 60:
            # the pod answered before and has not for LOST_MIN minutes (host reset or lost):
            # stop waiting and keep the last checkpoints already copied
            print(f"pod unreachable for {LOST_MIN} min: giving up on it", flush=True)
            status = "lost"
            break
        if seen and time.time() - last_ok < 120 and time.time() - last_ckpt > CKPT_MIN * 60:
            _copy_checkpoints(base, out)
            last_ckpt = time.time()
        time.sleep(60)
    if status == "lost":
        sys.exit("pod lost; the artifact holds the last checkpoints copied")
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
        sys.exit(f"pod did not finish in time (status: {status!r})")
    return status


def run(out: Path, tries=3):
    """start + follow; a pod whose CUDA does not work is terminated and replaced (up to `tries`)."""
    bad = []
    for _ in range(tries):
        # after a broken pod: secure cloud (datacenter hosts), where CUDA has always worked
        pod, g = start(cloud="SECURE" if bad else "ALL")
        os.environ["POD_ID"] = pod
        Path("pod_ids.txt").open("a").write(pod + "\n")
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
