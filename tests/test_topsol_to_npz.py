"""to_npz puts each lead of a hengck23-style series in its own quarter of the 10 s timeline."""
import csv
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
LEADS = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
ROWS = [["I", "aVR", "V1", "V4"], ["II", "aVL", "V2", "V5"], ["III", "aVF", "V3", "V6"]]


def test_every_lead_in_place(tmp_path):
    for fs in (250, 500, 1000):
        q = int(round(2.5 * fs))
        ser = np.zeros((4, 4 * q))
        for r, names in enumerate(ROWS):
            for c, n in enumerate(names):
                ser[r, c * q:(c + 1) * q] = LEADS.index(n) + 1          # constant = lead number
        ser[3] = 100.0                                                   # rhythm strip
        d = tmp_path / str(fs); d.mkdir()
        np.save(d / "x.series.npy", ser)
        with open(d / "test.csv", "w", newline="") as f:
            w = csv.writer(f); w.writerow(["id", "lead", "fs", "number_of_rows"])
            for n in LEADS:
                w.writerow(["x", n, fs, 4 * q if n == "II" else q])
        subprocess.run([sys.executable, str(ROOT / "benchmarks/topsol/to_npz.py"), str(d), str(d / "test.csv"), str(d / "o")],
                       check=True, capture_output=True)
        S = np.load(d / "o" / "x.npz")["signals"]
        for r, names in enumerate(ROWS):
            for c, n in enumerate(names):
                if n == "II":
                    continue
                seg = S[LEADS.index(n), c * 1250:(c + 1) * 1250]
                assert np.allclose(seg, LEADS.index(n) + 1), (fs, n)
                assert np.isnan(np.delete(S[LEADS.index(n)], np.s_[c * 1250:(c + 1) * 1250])).all(), (fs, n)
        assert np.allclose(S[1], 100.0)
