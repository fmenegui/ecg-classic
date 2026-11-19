"""Prints what the RunPod account and API offer for the evaluation infrastructure (no secret is
printed): data centers with their storage and GPU stock, CPU flavors, network volumes, and the
GraphQL inputs for network volumes and CPU pods."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from runpod_ctl import gql  # noqa: E402


def show(title, f):
    print(f"== {title}")
    try:
        print(json.dumps(f(), indent=1)[:6000])
    except Exception as e:  # noqa: BLE001
        print("error:", e)


def fields(t):
    q = """query($n: String!) { __type(name: $n) { name kind inputFields { name type { name kind ofType { name kind } } }
             fields { name args { name type { name kind ofType { name kind } } } } } }"""
    return gql(q, {"n": t})["__type"]


show("myself", lambda: gql("query { myself { clientBalance networkVolumes { id name size dataCenterId } } }"))
show("dataCenters", lambda: gql("query { dataCenters { id name location storageSupport listed } }"))
show("cpuFlavors", lambda: gql("query { cpuFlavors { id groupId displayName minVcpu maxVcpu ramMultiplier } }"))
for t in ["Mutation"]:
    show(t + " (volume / cpu)", lambda: [f["name"] for f in fields(t)["fields"] if any(k in f["name"].lower() for k in ("volume", "cpu", "deploy"))])
for t in ["CreateNetworkVolumeInput", "deployCpuPodInput", "DeployCpuPodInput", "PodFindAndDeployOnDemandInput"]:
    show(t, lambda: fields(t))
show("gpuTypes stock per data center (cheap cards)", lambda: gql("""query { gpuTypes { id displayName
     lowestPrice(input: {gpuCount: 1}) { minimumBidPrice uninterruptablePrice stockStatus } } }"""))
