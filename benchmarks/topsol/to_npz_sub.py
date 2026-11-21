"""A Kaggle submission.csv (id = <image>_<sample>_<lead>, value in mV) -> one npz per image for our
evaluators (ECGDIG_EXTERNAL).

    python benchmarks/topsol/to_npz_sub.py SUBMISSION.csv TEST_CSV OUT_DIR

Every lead is asked for on the Kaggle sheet's timeline (test.csv: number_of_rows at fs): lead II
for 10 s, the others for 2.5 s in their column (I II III 0-2.5 s, aVR aVL aVF 2.5-5, V1-V3 5-7.5,
V4-V6 7.5-10). They are put on a 10 s, 500 Hz timeline. Image keys may contain underscores.
"""
from __future__ import annotations

import csv
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

LEADS = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
COL = {L: i // 3 for i, L in enumerate(LEADS)}


def _resample(y, n):
    return np.interp(np.linspace(0, 1, n), np.linspace(0, 1, len(y)), y)


def main(sub, test_csv, out):
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    want = {(r["id"], r["lead"]) for r in csv.DictReader(open(test_csv))}
    keys = sorted({k for k, _ in want})
    vals = defaultdict(dict)
    for r in csv.DictReader(open(sub)):
        key, idx, lead = r["id"].rsplit("_", 2)
        vals[(key, lead)][int(idx)] = float(r["value"])
    done = 0
    for key in keys:
        if not any((key, L) in vals for L in LEADS):
            continue
        S = np.full((12, 5000), np.nan, np.float32)
        for L in LEADS:
            d = vals.get((key, L))
            if not d:
                continue
            y = np.array([d[i] for i in sorted(d)], np.float64)
            if not np.isfinite(y).any() or len(y) < 2:
                continue
            if L == "II":
                S[1] = _resample(y, 5000)
            else:
                c = COL[L]
                S[LEADS.index(L), c * 1250:(c + 1) * 1250] = _resample(y, 1250)
        np.savez(out / f"{key}.npz", leads=np.array(LEADS), signals=S, layout=np.array("3x4+1"))
        done += 1
    print(f"{done} of {len(keys)} images converted", flush=True)


if __name__ == "__main__":
    main(*sys.argv[1:4])
