"""patch_4th: the 4th place writer keeps going (NaN) when an image has no prediction."""
import subprocess
import sys
from pathlib import Path

SRC = '''import numpy as np
def write(all_preds, rows):
    out = []
    for base_id, lead, n_rows in rows:
        values = all_preds.get((base_id, lead))
        if values is None:
            raise RuntimeError(
                "Missing prediction for submission row. "
                f"base_id={base_id}")
        out.append(values)
    return out
'''


def test_patch(tmp_path):
    f = tmp_path / "run_ecg_digitization_inference.py"
    f.write_text(SRC)
    subprocess.run([sys.executable, str(Path(__file__).resolve().parents[1] / "benchmarks/topsol/patch_4th.py"), str(f)],
                   check=True, capture_output=True)
    ns = {}
    exec(f.read_text(), ns)
    out = ns["write"]({("a", "I"): [1.0]}, [("a", "I", 1), ("b", "I", 3)])
    assert len(out) == 2 and len(out[1]) == 3
