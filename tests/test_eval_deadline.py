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


def test_memory_limit_fails_one_image(monkeypatch):
    """With ECGDIG_MEM_LIMIT_MB a worker that allocates too much raises MemoryError and survives."""
    import subprocess, sys, textwrap
    code = textwrap.dedent('''
        import os, sys
        sys.path.insert(0, "benchmarks/pmcardio")
        os.environ["ECGDIG_MEM_LIMIT_MB"] = "300"
        import eval_pm
        from concurrent.futures import ProcessPoolExecutor
        def big():
            import numpy as np
            try:
                np.ones(800 << 20, np.uint8); return "allocated"
            except MemoryError:
                return "MemoryError"
        with ProcessPoolExecutor(1, initializer=eval_pm._limit_memory) as ex:
            print(ex.submit(big).result())
    ''')
    r = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True, timeout=120)
    assert "MemoryError" in r.stdout, r.stdout + r.stderr
