"""eval_pm stops at ECGDIG_DEADLINE and returns what it has (no image left running)."""
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "benchmarks" / "pmcardio"))


def test_deadline_returns_partial(tmp_path, monkeypatch):
    import pickle
    import eval_pm
    root = tmp_path / "img"; root.mkdir()
    cache = tmp_path / "cache"; cache.mkdir()
    (root / "a.png").write_bytes(b"x"); (root / "b.png").write_bytes(b"x")
    (cache / "a.png.pkl").write_bytes(pickle.dumps(dict(ok=True, sec=1.0)))   # read in an earlier run
    monkeypatch.setenv("ECGDIG_DEADLINE", str(time.time() - 1))
    t0 = time.time()
    res = eval_pm._run_all([(str(root / "a.png"), {}), (str(root / "b.png"), {})], 2, cache, root)
    assert list(res) == ["a.png"] and time.time() - t0 < 20
