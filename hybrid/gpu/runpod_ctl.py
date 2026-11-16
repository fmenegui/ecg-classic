"""Start, follow and terminate a RunPod GPU pod (used by .github/workflows/gpu-train.yml).

    python hybrid/gpu/runpod_ctl.py start            # prints pod_id=... (GitHub step output)
    python hybrid/gpu/runpod_ctl.py follow --out DIR  # streams the pod log, fetches the result
    python hybrid/gpu/runpod_ctl.py stop             # terminates $POD_ID (and stray ecgdig-* pods)

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
FALLBACK_GPUS = ["NVIDIA GeForce RTX 4090", "NVIDIA RTX A6000", "NVIDIA A40", "NVIDIA RTX A5000",
                 "NVIDIA L40S", "NVIDIA GeForce RTX 3090"]
NAME = "ecgdig-train"


def gql(query, variables=None):
    key = os.environ.get("RUNPOD_API_KEY", "")
    if not key:
        sys.exit("RUNPOD_API_KEY is not set (repository secret missing?)")
    body = json.dumps({"query": query, "variables": variables or {}}).encode()
    req = urllib.request.Request(API, data=body, headers={"Content-Type": "application/json",
                                                          "Authorization": f"Bearer {key}",
                                                          "User-Agent": "ecgdig-gpu-train"})
    with urllib.request.urlopen(req, timeout=60) as r:
        out = json.loads(r.read())
    if out.get("errors"):
        raise RuntimeError("; ".join(e.get("message", str(e)) for e in out["errors"]))
    return out["data"]


def start():
    ref = os.environ["REF"]
    env = [{"key": "REF", "value": ref}, {"key": "JOB", "value": os.environ["JOB"]},
           {"key": "MAX_HOURS", "value": os.environ.get("MAX_HOURS", "1")}]
    cmd = ("bash -c 'curl -fsSL https://raw.githubusercontent.com/fmenegui/ecg-digitalization-classic/"
           "${REF}/hybrid/gpu/pod.sh -o /pod.sh && bash /pod.sh'")
    gpus = [os.environ.get("GPU") or FALLBACK_GPUS[0]] + [g for g in FALLBACK_GPUS if g != os.environ.get("GPU")]
    q = """mutation($input: PodFindAndDeployOnDemandInput) {
             podFindAndDeployOnDemand(input: $input) { id machineId costPerHr gpuCount } }"""
    last = None
    for g in gpus:
        inp = dict(cloudType="ALL", gpuCount=1, volumeInGb=0, containerDiskInGb=60, minVcpuCount=8,
                   minMemoryInGb=30, gpuTypeId=g, name=NAME, imageName=IMAGE, dockerArgs=cmd,
                   ports="8000/http", volumeMountPath="/workspace", env=env)
        try:
            d = gql(q, {"input": inp})["podFindAndDeployOnDemand"]
        except Exception as e:  # noqa: BLE001
            last = e
            print(f"{g}: {e}", file=sys.stderr)
            continue
        if d and d.get("id"):
            print(f"started {d['id']} on {g}, {d.get('costPerHr')} USD/h", file=sys.stderr)
            print(f"pod_id={d['id']}")
            print(f"gpu={g}")
            return
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
        if status == "done":
            break
        time.sleep(60)
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
    ap.add_argument("cmd", choices=["start", "follow", "stop"])
    ap.add_argument("--out", type=Path, default=Path("result"))
    a = ap.parse_args()
    {"start": start, "follow": lambda: follow(a.out), "stop": stop}[a.cmd]()
