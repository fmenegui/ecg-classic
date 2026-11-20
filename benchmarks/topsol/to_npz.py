"""A top solution's digitised series -> one npz per image, read by our evaluators
(ECGDIG_EXTERNAL=<dir> in benchmarks/pmcardio/eval_pm.py and benchmarks/physionet/eval_snr.py).

    python benchmarks/topsol/to_npz.py SERIES_DIR TEST_CSV OUT_DIR [--layout 3x4+1]

hengck23-style output: <id>.series.npy, 4 rows; rows 0-2 hold the four 2.5 s leads of the sheet's
three rows one after the other (I aVR V1 V4 / II aVL V2 V5 / III aVF V3 V6), row 3 the lead II
rhythm strip. They are put on a 10 s, 500 Hz timeline (lead in its column's quarter; II whole).
An image the solution failed on gets no file: the evaluator records a failure.
"""
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np

LEADS = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
ROWS = [["I", "aVR", "V1", "V4"], ["II", "aVL", "V2", "V5"], ["III", "aVF", "V3", "V6"]]


def _resample(y, n):
    x = np.linspace(0, 1, len(y))
    return np.interp(np.linspace(0, 1, n), x, y)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("series_dir", type=Path)
    ap.add_argument("test_csv", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--layout", default="3x4+1")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    n_rows = defaultdict(dict)
    for r in csv.DictReader(open(a.test_csv)):
        n_rows[r["id"]][r["lead"]] = int(r["number_of_rows"])
    done = 0
    for key, nr in n_rows.items():
        f = a.series_dir / f"{key}.series.npy"
        if not f.exists():
            continue
        S = np.load(f).astype(np.float64)
        out = np.full((12, 5000), np.nan, np.float32)
        for r, names in enumerate(ROWS):
            lens = [nr[names[0]] - sum(nr[n] for n in names[1:])] + [nr[n] for n in names[1:]]
            parts = np.split(S[r][: sum(lens)], np.cumsum(lens)[:-1])
            for c, (name, y) in enumerate(zip(names, parts)):
                if len(y) < 2:
                    continue
                out[LEADS.index(name), c * 1250:(c + 1) * 1250] = _resample(y, 1250)
        out[LEADS.index("II")] = _resample(S[3][: nr["II"]], 5000)
        np.savez(a.out / f"{key}.npz", leads=np.array(LEADS), signals=out, layout=np.array(a.layout))
        done += 1
    print(f"{done} of {len(n_rows)} images converted", flush=True)


if __name__ == "__main__":
    main()
