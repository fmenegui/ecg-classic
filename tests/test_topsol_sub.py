"""to_npz_sub: a Kaggle submission.csv (keys with underscores) back on the 10 s, 500 Hz timeline;
nb2sh: notebook shell cells to a script."""
import csv
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
LEADS = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]


def test_submission_to_npz(tmp_path):
    key, fs = "photos_iphone__img_3_page_0", 500
    with open(tmp_path / "test.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["id", "lead", "fs", "number_of_rows"])
        for L in LEADS:
            w.writerow([key, L, fs, 5000 if L == "II" else 1250])
    with open(tmp_path / "sub.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["id", "value"])
        for i, L in enumerate(LEADS):
            for k in range(5000 if L == "II" else 1250):
                w.writerow([f"{key}_{k}_{L}", i + 1])
    subprocess.run([sys.executable, str(ROOT / "benchmarks/topsol/to_npz_sub.py"), str(tmp_path / "sub.csv"),
                    str(tmp_path / "test.csv"), str(tmp_path / "o")], check=True, capture_output=True)
    S = np.load(tmp_path / "o" / f"{key}.npz")["signals"]
    assert np.allclose(S[1], 2)
    for i, L in enumerate(LEADS):
        if L == "II":
            continue
        c = i // 3
        assert np.allclose(S[i, c * 1250:(c + 1) * 1250], i + 1), L
        assert np.isnan(np.delete(S[i], np.s_[c * 1250:(c + 1) * 1250])).all(), L


def test_nb2sh(tmp_path):
    nb = {"cells": [{"cell_type": "code", "source": ["%%capture\n", "!pip install x\n"]},
                    {"cell_type": "code", "source": ["%env A=/kaggle/input/d\n", "%env B=\"1,2;3\"\n",
                                                     "!python t.py --o /kaggle/working/s.csv\n"]}]}
    (tmp_path / "n.ipynb").write_text(json.dumps(nb))
    subprocess.run([sys.executable, str(ROOT / "benchmarks/topsol/nb2sh.py"), str(tmp_path / "n.ipynb"),
                    str(tmp_path / "r.sh")], check=True)
    s = (tmp_path / "r.sh").read_text()
    assert "export A=$TS_INPUT/d" in s and "export B='1,2;3'" in s and "python t.py --o $TS_WORK/s.csv" in s
    assert "%%capture" not in s and "pip install x" in s


def test_empty_values_are_nan(tmp_path):
    import csv as _csv
    sub = tmp_path / "sub.csv"; test = tmp_path / "test.csv"
    with open(test, "w", newline="") as f:
        w = _csv.writer(f); w.writerow(["id", "lead", "fs", "number_of_rows"]); w.writerow(["E2_000", "I", 500, 2])
    with open(sub, "w", newline="") as f:
        w = _csv.writer(f); w.writerow(["id", "value"]); w.writerow(["E2_000_0_I", ""]); w.writerow(["E2_000_1_I", "nan"])
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmarks" / "topsol"))
    import to_npz_sub as T
    T.main(str(sub), str(test), str(tmp_path / "out"))
