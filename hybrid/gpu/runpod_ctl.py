"""Start, follow and terminate a RunPod GPU pod (used by .github/workflows/gpu-train.yml).

    python hybrid/gpu/runpod_ctl.py start            # prints pod_id=... (GitHub step output)
    python hybrid/gpu/runpod_ctl.py follow --out DIR  # streams the pod log, fetches the result
    python hybrid/gpu/runpod_ctl.py stop             # terminates $POD_ID (and stray ecgdig-* pods)
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
# best cost per training step first: the crops come from the CPU generator, so a mid-range 24 GB card
# is not slower in practice than a 4090 and costs a fraction (16/11: A5000 0.27 USD/h vs 4090 0.89)
FALLBACK_GPUS = ["NVIDIA RTX A5000", "NVIDIA GeForce RTX 3090", "NVIDIA RTX A6000", "NVIDIA A40",
                 "NVIDIA GeForce RTX 4090", "NVIDIA L40S"]
NAME = "ecgdig-train"


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
    print(f"RunPod account ok, balance {me.get('clientBalance')} USD", file=sys.stderr)


def start(exclude=(), cloud="ALL"):
    check()
    ref = os.environ["REF"]
    env = [{"key": "REF", "value": ref}, {"key": "JOB", "value": os.environ["JOB"]},
           {"key": "MAX_HOURS", "value": os.environ.get("MAX_HOURS", "1")}]
    cmd = ("bash -c 'curl -fsSL https://raw.githubusercontent.com/fmenegui/ecg-digitalization-classic/"
           "${REF}/hybrid/gpu/pod.sh -o /pod.sh && bash /pod.sh'")
    gpus = [os.environ.get("GPU") or FALLBACK_GPUS[0]] + [g for g in FALLBACK_GPUS if g != os.environ.get("GPU")]
    gpus = [g for g in gpus if g not in exclude] + [g for g in gpus if g in exclude]
    q = """mutation($input: PodFindAndDeployOnDemandInput) {
             podFindAndDeployOnDemand(input: $input) { id machineId costPerHr gpuCount } }"""
    last = None
    # preferred GPU first, with as many vCPU as possible (the crops are generated on the CPU)
    for vcpu, g in [(v, g) for g in gpus for v in (24, 16)] + [(8, g) for g in gpus]:
        inp = dict(cloudType=cloud, gpuCount=1, volumeInGb=0, containerDiskInGb=60, minVcpuCount=vcpu,
                   minMemoryInGb=30, gpuTypeId=g, name=NAME, imageName=IMAGE, dockerArgs=cmd,
                   ports="8000/http", volumeMountPath="/workspace", env=env)
        try:
            d = gql(q, {"input": inp})["podFindAndDeployOnDemand"]
        except Exception as e:  # noqa: BLE001
            last = e
            print(f"{g} ({vcpu} vCPU): {e}", file=sys.stderr)
            continue
        if d and d.get("id"):
            print(f"started {d['id']} on {g} (>= {vcpu} vCPU, {cloud} cloud), {d.get('costPerHr')} USD/h", file=sys.stderr)
            print(f"pod_id={d['id']}")
            print(f"gpu={g}")
            return d["id"], g
    sys.exit(f"no GPU available: {last}")


def _get(url, timeout=30):
    req = urllib.request.Request(url, headers={"User-Agent": "ecgdig-gpu-train"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def follow(out: Path):
    pod = os.environ["POD_ID"]
    base = f"https://{pod}-8000.proxy.runpod.net"
    limit = time.time() + float(os.environ.get("MAX_HOURS", "1")) * 3600 + 40 * 60   # + setup margin
    seen = 0
    status = ""
    out.mkdir(parents=True, exist_ok=True)
    while time.time() < limit:
        try:
            status = _get(base + "/status.txt").decode().strip()
            log = _get(base + "/log.txt", timeout=60).decode(errors="replace")
            if len(log) > seen:
                print(log[seen:], end="", flush=True)
                seen = len(log)
        except (urllib.error.URLError, TimeoutError, ConnectionError) as e:   # pod still booting
            print(f"[waiting: {getattr(e, 'code', '')} {type(e).__name__}]", flush=True)
        if status in ("done", "nogpu"):
            break
        time.sleep(60)
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
    ap.add_argument("cmd", choices=["start", "follow", "stop", "run"])
    ap.add_argument("--out", type=Path, default=Path("result"))
    a = ap.parse_args()
    {"start": start, "follow": lambda: follow(a.out), "stop": stop, "run": lambda: run(a.out)}[a.cmd]()
