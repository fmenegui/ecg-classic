"""The 15 local evaluation records are excluded from training everywhere with the same list."""
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FILES = ["hybrid/realstrips_build.py", "benchmarks/compare.py", "benchmarks/physionet/merge_shards.py"]


def _ids(f):
    s = (ROOT / f).read_text()
    m = re.search(r"LOCAL15 = [^=]*?\)\)?", s, re.S)
    return set(re.findall(r"\d{8,10}", m.group(0)))


def test_local15_identical():
    sets = [_ids(f) for f in FILES]
    assert all(len(s) == 15 for s in sets), [len(s) for s in sets]
    assert all(s == sets[0] for s in sets)
