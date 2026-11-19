"""Prints what the RunPod REST API v2 offers for the evaluation infrastructure (read-only; no
secret is printed): data centers and availability, CPU and GPU catalog, network volumes, pods."""
import json
import os
import urllib.error
import urllib.request

KEY = os.environ.get("RUNPOD_API_KEY", "").strip().strip("\"'")


def get(path):
    req = urllib.request.Request("https://api.runpod.io/v2" + path,
                                 headers={"Authorization": f"Bearer {KEY}", "User-Agent": "ecgdig-probe"})
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            return r.status, r.read().decode()
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode()[:500]


# never /pods: a pod listing carries the pods' environment (the Kaggle token of a training pod)
for p in ["/catalog/datacenters", "/catalog/cpus", "/network-volumes"]:
    s, body = get(p)
    try:
        body = json.dumps(json.loads(body), separators=(",", ":"))
    except Exception:  # noqa: BLE001
        pass
    for k in ("KAGGLE", "TOKEN", "KEY", "SECRET"):
        if k in body.upper():
            body = "(withheld: the answer mentions a credential)"
    print(f"== GET {p} -> {s}\n{body[:5000]}\n")
