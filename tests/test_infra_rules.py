"""Infrastructure rules checked on every push (docs/runpod-troubleshooting.md says why each exists)."""
import re
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
WF = ROOT / ".github" / "workflows"


def _src(p):
    return (ROOT / p).read_text()


def _wfs():
    return {f.name: (f.read_text(), yaml.safe_load(f.read_text())) for f in WF.glob("*.yml")}


def test_cheap_gpus_only():
    s = _src("hybrid/gpu/runpod_ctl.py")
    gpus = re.search(r"FALLBACK_GPUS = \[(.*?)\]", s, re.S).group(1)
    assert not re.search(r"4090|L40|A100|H100|H200|A6000", gpus), "expensive card in the default list"
    cap = float(re.search(r'MAX_USD_H", "([0-9.]+)"', s).group(1))
    assert cap <= 0.5


def test_balance_guards_present():
    assert "is below the" in _src("hybrid/gpu/runpod_ctl.py") or "add funds" in _src("hybrid/gpu/runpod_ctl.py")
    s = _src("hybrid/gpu/runpod_v2.py")
    assert "def balance_guard" in s and "balance_guard(" in s.split("def cpu_start")[1]


def test_follower_limits_present():
    s = _src("hybrid/gpu/runpod_ctl.py")
    for k in ("BOOT_MIN", "LOST_MIN", "STALL_MIN", "FOLLOW_UNTIL", "exit code"):
        assert k in s, k


def test_cpu_pods_use_the_pytorch_image():
    # the Kaggle archive answers 404 to pods on python:3.11-bookworm (Nov 20)
    s = _src("hybrid/gpu/runpod_v2.py")
    assert re.search(r'IMAGE = os.environ.get\("CPU_IMAGE", "runpod/pytorch', s)


def test_pods_always_terminated_and_followers_bounded():
    for name, (text, wf) in _wfs().items():
        starts = "cpu-start" in text or "runpod_ctl.py run" in text
        if not starts:
            continue
        for job_name, job in wf["jobs"].items():
            steps = job.get("steps", [])
            body = yaml.safe_dump(steps)
            if "cpu-start" not in body and "runpod_ctl.py run" not in body:
                continue
            stops = [s for s in steps if "stop" in str(s.get("run", "")) and ("runpod_v2.py stop" in str(s.get("run"))
                                                                          or "runpod_ctl.py stop" in str(s.get("run")))]
            assert stops and all(str(s.get("if", "")).startswith("always()") for s in stops), f"{name}:{job_name} pod not always terminated"
            assert int(job.get("timeout-minutes", 360)) <= 355, f"{name}:{job_name} timeout"
            assert "FOLLOW_UNTIL" in body, f"{name}:{job_name} follower not bounded"


def test_artifacts_never_carry_data():
    bad = re.compile(r"physionet/data|/sig\b|sig/|realstrips|visual_data|kaggle/train|\.png|/data/")
    for name, (text, wf) in _wfs().items():
        for job in wf["jobs"].values():
            for st in job.get("steps", []):
                if "upload-artifact" in str(st.get("uses", "")):
                    path = str((st.get("with") or {}).get("path", ""))
                    for line in path.split():
                        assert not bad.search(line) or "gallery" in line, f"{name}: artifact path {line}"


def test_no_pod_listing_printed():
    # a pod listing holds every pod's environment (Kaggle token): never printed
    for p in ("hybrid/gpu/runpod_ctl.py", "hybrid/gpu/runpod_v2.py", "hybrid/gpu/runpod_probe.py"):
        s = _src(p)
        assert not re.search(r"print\([^)\n]*(\[\"pods\"\]|get\(\"pods\"\)|json\.dumps\()", s), p
