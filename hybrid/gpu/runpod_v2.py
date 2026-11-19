"""Evaluation pods on RunPod (REST API v2): a network volume with the datasets, CPU pods that
mount it, model upload, follow and terminate. Used by .github/workflows/pod-eval.yml.

    python hybrid/gpu/runpod_v2.py volume                 # prints volume_id=..., data_center=...
    python hybrid/gpu/runpod_v2.py cpu-start               # prints pod_id=... (VOLUME_ID, DATA_CENTER, JOB, ...)
    python hybrid/gpu/runpod_v2.py put FILE [FILE ...]     # sends files to the pod's /workspace/in
    python hybrid/gpu/runpod_v2.py stop                    # terminates POD_ID

The volume (VOLUME_NAME, default ecgdig-data, VOLUME_GB) is created once in the first data center
of DATA_CENTERS that accepts it and is reused; the datasets on it are downloaded once (PM-ECG-ID
from Zenodo, the Kaggle training records) and never leave it: the pod's HTTP port serves only its
per-run output folder (logs, results.json, report.md). Never prints a pod's environment.
Standard library only.
"""
from __future__ import annotations

import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

API = "https://api.runpod.io/v2"
IMAGE = "runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04"
DATA_CENTERS = os.environ.get("DATA_CENTERS", "EUR-IS-1,EU-RO-1,EU-SE-1,CA-MTL-3,EU-CZ-1,EUR-NO-1").split(",")
NAME = "ecgdig-eval-" + os.environ.get("GITHUB_RUN_ID", "local") + "-" + os.environ.get("SHARD", "0")
MAX_USD_H = float(os.environ.get("MAX_USD_H", "1.2"))


def call(method, path, body=None, timeout=60):
    key = os.environ.get("RUNPOD_API_KEY", "").strip().strip("\"'")
    if not key:
        sys.exit("RUNPOD_API_KEY is not set")
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(API + path, data=data, method=method,
                                 headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json",
                                          "User-Agent": "ecgdig-eval"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            txt = r.read().decode()
            return r.status, (json.loads(txt) if txt else {})
    except urllib.error.HTTPError as e:
        txt = e.read().decode()[:400]
        return e.code, {"error": txt}


def out(k, v):
    print(f"{k}={v}")
    if os.environ.get("GITHUB_OUTPUT"):
        with open(os.environ["GITHUB_OUTPUT"], "a") as f:
            f.write(f"{k}={v}\n")


def volume():
    name = os.environ.get("VOLUME_NAME", "ecgdig-data")
    s, d = call("GET", "/network-volumes")
    for v in d.get("networkVolumes", []):
        if v.get("name") == name:
            print(f"volume {v['id']} {v.get('size')} GB in {v.get('dataCenter') or v.get('dataCenterId')}", file=sys.stderr)
            out("volume_id", v["id"]); out("data_center", v.get("dataCenter") or v.get("dataCenterId"))
            return
    size = int(os.environ.get("VOLUME_GB", "60"))
    for dc in DATA_CENTERS:
        s, d = call("POST", "/network-volumes", {"name": name, "size": size, "dataCenter": dc})
        if s in (200, 201) and d.get("id"):
            print(f"created volume {d['id']} ({size} GB) in {dc}", file=sys.stderr)
            out("volume_id", d["id"]); out("data_center", dc)
            return
        print(f"{dc}: {s} {d.get('error', '')[:200]}", file=sys.stderr)
    sys.exit("no data center accepted the volume")


def balance_guard(vcpu):
    """RunPod stops every pod when the balance reaches 0 (an evaluation pod never came up and a
    training was cut that way): refuse a pod the balance cannot pay for, with a reserve for the
    pods already running (BALANCE_RESERVE USD)."""
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import runpod_ctl
    bal = float(runpod_ctl.gql("query { myself { clientBalance } }")["myself"].get("clientBalance") or 0)
    need = float(os.environ.get("MAX_HOURS", "3")) * min(MAX_USD_H, 0.04 * vcpu) + \
        float(os.environ.get("BALANCE_RESERVE", "2"))
    print(f"RunPod balance {bal:.2f} USD, this pod may need {need:.2f} USD with the reserve", file=sys.stderr)
    if bal < need:
        sys.exit(f"RunPod balance {bal:.2f} USD is below {need:.2f} USD: add funds")


def cpu_start():
    vol, dc = os.environ["VOLUME_ID"], os.environ["DATA_CENTER"]
    balance_guard(int(os.environ.get("VCPU", "32")))
    run_dir = f"/workspace/runs/{os.environ.get('GITHUB_RUN_ID', 'local')}-{os.environ.get('SHARD', '0')}"
    env = {"REF": os.environ["REF"], "JOB": os.environ["JOB"], "MAX_HOURS": os.environ.get("MAX_HOURS", "3"),
           "CPU_ONLY": "1", "OUT_DIR": run_dir + "/out", "REPO_DIR": "/root/repo",
           "UPLOAD_TOKEN": os.environ["UPLOAD_TOKEN"], "IN_DIR": run_dir + "/in"}
    for k in ("KAGGLE_API_TOKEN", "KAGGLE_USERNAME"):   # only when the run asks for Kaggle data
        if os.environ.get(k):
            env[k] = os.environ[k]
    args = ("bash -c 'curl -fsSL https://raw.githubusercontent.com/fmenegui/ecg-digitalization-classic/"
            "${REF}/hybrid/gpu/pod.sh -o /pod.sh && bash /pod.sh'")
    vcpu = int(os.environ.get("VCPU", "32"))
    t_end = time.time() + 60 * float(os.environ.get("WAIT_MIN", "30"))
    last = None
    while True:
        for flavor in ("cpu5c", "cpu3c", "cpu5g", "cpu3g"):
            body = {"name": NAME, "image": IMAGE, "cpu": {"id": flavor, "vcpuCount": vcpu},
                    # container disk cap: 10 GB per vCPU on cpu3*, 15 on cpu5* (2 vCPU: 20 GB)
                    "disk": min(40, 10 * vcpu),
                    "cloud": "SECURE", "dataCenterIds": [dc], "ports": ["8000/http"], "env": env, "args": args,
                    "mounts": {"network": [{"volumeId": vol, "path": "/workspace"}]}}
            s, d = call("POST", "/pods", body)
            if s in (200, 201) and d.get("id"):
                cost = float(d.get("cost") or 0)
                if cost > MAX_USD_H:
                    call("DELETE", f"/pods/{d['id']}")
                    print(f"{flavor} x{vcpu}: {cost} USD/h above {MAX_USD_H}, terminated", file=sys.stderr)
                    continue
                print(f"started {d['id']} {flavor} x{vcpu} vCPU in {dc}, {cost} USD/h", file=sys.stderr)
                out("pod_id", d["id"])
                return
            last = f"{flavor}: {s} {str(d.get('error', ''))[:200]}"
            print(last, file=sys.stderr)
        if time.time() > t_end:
            sys.exit(f"no CPU pod in {dc}: {last}")
        time.sleep(120)


def put(files):
    base = f"https://{os.environ['POD_ID']}-8000.proxy.runpod.net"
    tok = os.environ["UPLOAD_TOKEN"]
    for f in files:
        p = Path(f)
        tries = int(os.environ.get("PUT_TRIES", "120"))
        for i in range(tries):   # the pod may still be booting (image pull: up to ~1 h)
            try:
                req = urllib.request.Request(f"{base}/in/{p.name}", data=p.read_bytes(), method="PUT",
                                             headers={"X-Upload-Token": tok, "User-Agent": "ecgdig-eval"})
                with urllib.request.urlopen(req, timeout=600) as r:
                    print(f"sent {p.name} ({p.stat().st_size // 1_000_000} MB): {r.status}", flush=True)
                    break
            except Exception as e:  # noqa: BLE001
                if i == tries - 1:
                    sys.exit(f"could not send {p.name}: {e}")
                time.sleep(30)


def stop():
    pid = os.environ.get("POD_ID")
    if pid:
        s, _ = call("DELETE", f"/pods/{pid}")
        print(f"terminate {pid}: {s}")
    s, d = call("GET", "/pods")   # this run's other pods, by name (the environment is never printed)
    for p in d.get("pods", []):
        if p.get("name") == NAME and p.get("id") != pid:
            s2, _ = call("DELETE", f"/pods/{p['id']}")
            print(f"terminate {p['id']}: {s2}")


if __name__ == "__main__":
    cmd = sys.argv[1]
    {"volume": volume, "cpu-start": cpu_start, "stop": stop}.get(cmd, lambda: put(sys.argv[2:]))()
