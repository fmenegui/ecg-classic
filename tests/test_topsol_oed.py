"""Open-ECG-Digitizer adapter: CSV (uV, 5000 x 12, NaN outside each lead) -> our npz (mV);
prepare.py --resize keeps the aspect and the orientation."""
import csv
import subprocess
import sys

import cv2
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LEADS = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]


def test_csv_to_npz(tmp_path):
    out = tmp_path / "out" / "test"; out.mkdir(parents=True)
    data = np.full((5000, 12), np.nan)
    for i in range(12):
        c = i % 4
        data[c * 1250:(c + 1) * 1250, i] = (i + 1) * 1000.0            # lead number in mV, written in uV
    data[:, 1] = 2000.0
    np.savetxt(out / "k1_timeseries_canonical.csv", data, delimiter=",", header=",".join(LEADS), comments="")
    (out / "digitization_metadata.csv").write_text("file_path,matching_cost,is_flipped,lead_layout\nk1,0.1,False,standard_3x4_with_r1\n")
    with open(tmp_path / "test.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["id", "lead", "fs", "number_of_rows"])
        for k in ("k1", "k2"):
            for L in LEADS:
                w.writerow([k, L, 500, 5000 if L == "II" else 1250])
    r = subprocess.run([sys.executable, str(ROOT / "benchmarks/topsol/to_npz_oed.py"), str(tmp_path / "out"),
                        str(tmp_path / "test.csv"), str(tmp_path / "ext")], check=True, capture_output=True, text=True)
    assert "1 of 2" in r.stdout
    z = np.load(tmp_path / "ext" / "k1.npz")
    S = z["signals"]
    assert S.shape == (12, 5000) and list(z["leads"]) == LEADS and str(z["layout"]) == "standard_3x4_with_r1"
    for i in range(12):
        if i == 1:
            assert np.allclose(S[1], 2.0)
            continue
        c = i % 4
        assert np.allclose(S[i, c * 1250:(c + 1) * 1250], i + 1)
        assert np.isnan(np.delete(S[i], np.s_[c * 1250:(c + 1) * 1250])).all()


def test_prepare_resize(tmp_path):
    root = tmp_path / "pm" / "scen"; root.mkdir(parents=True)
    cv2.imwrite(str(root / "land.png"), np.zeros((1000, 3000, 3), np.uint8))   # wide
    cv2.imwrite(str(root / "port.png"), np.zeros((3000, 1500, 3), np.uint8))   # portrait photo
    (tmp_path / "sel.csv").write_text("path\nscen/land.png\nscen/port.png\n")
    subprocess.run([sys.executable, str(ROOT / "benchmarks/topsol/prepare.py"), "--out", str(tmp_path / "d"),
                    "--pm-root", str(tmp_path / "pm"), "--pm-list", str(tmp_path / "sel.csv"), "--resize", "2200x1700"],
                   check=True, capture_output=True)
    a = cv2.imread(str(tmp_path / "d" / "test" / "scen__land.png"))
    b = cv2.imread(str(tmp_path / "d" / "test" / "scen__port.png"))
    assert a.shape[:2] == (1700, 2200) and b.shape[:2] == (2200, 1700)
    dark = (a[..., 0] < 128)                                                   # content keeps 3:1, padding white
    ys, xs = np.where(dark)
    assert xs.max() - xs.min() + 1 == 2200 and abs((ys.max() - ys.min() + 1) - 733) <= 1
