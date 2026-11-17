"""PTB-XL records that are also in the Kaggle competition (PhysioNet - Digitization of ECG Images).

The row network is trained on real PTB-XL signals. The Kaggle images are rendered from PTB-XL
records, and they are our evaluation set, so those records must stay out of training. The
Kaggle ids are not PTB-XL ids: the match is made on the signal itself (lead II, 100 Hz,
z-normalised correlation over the 10 s).

    python hybrid/ptbxl/dedupe.py --ptbxl PTBXL_DIR --kaggle KAGGLE_DIR --out exclude_ptbxl.txt
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def _z(x):
    x = x - np.nanmean(x, axis=-1, keepdims=True)
    x = np.nan_to_num(x)
    return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-9)


def ptbxl_lead2(root: Path):
    import wfdb
    rows = list(csv.DictReader(open(root / "ptbxl_database.csv")))
    ids, X = [], []
    for r in rows:
        sig, f = wfdb.rdsamp(str(root / r["filename_lr"]))
        X.append(sig[:1000, f["sig_name"].index("II")])
        ids.append(int(float(r["ecg_id"])))
    return np.array(ids), np.array(X, np.float32)


def kaggle_lead2(root: Path):
    ids, X = [], []
    for p in sorted(root.rglob("*.csv")):
        if p.name == "train.csv":
            continue
        rows = list(csv.reader(open(p)))
        if "II" not in rows[0]:
            continue
        c = rows[0].index("II")
        v = np.array([float(r[c]) if c < len(r) and r[c] != "" else np.nan for r in rows[1:]], np.float32)
        n = len(v)   # 10 s at 250-1025 Hz -> 100 Hz
        t = np.linspace(0, n - 1, 1000)
        ok = np.isfinite(v)
        if ok.sum() < n * 0.9:
            continue
        X.append(np.interp(t, np.flatnonzero(ok), v[ok]))
        ids.append(p.stem)
    return ids, np.array(X, np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ptbxl", type=Path, required=True)
    ap.add_argument("--kaggle", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--th", type=float, default=0.95)
    a = ap.parse_args()
    pid, P = ptbxl_lead2(a.ptbxl)
    kid, K = kaggle_lead2(a.kaggle)
    print(f"PTB-XL {len(pid)} records, Kaggle {len(kid)} records", flush=True)
    C = _z(K) @ _z(P).T
    best = C.argmax(1)
    cb = C[np.arange(len(kid)), best]
    excl = sorted({int(pid[b]) for b, c in zip(best, cb) if c >= a.th})
    a.out.write_text("\n".join(map(str, excl)) + "\n")
    hist = np.histogram(cb, bins=[-1, 0.5, 0.8, 0.9, 0.95, 0.99, 1.01])[0].tolist()
    stats = dict(kaggle=len(kid), ptbxl=len(pid), matched=int((cb >= a.th).sum()), excluded=len(excl),
                 best_corr_hist=dict(zip(["<0.5", "0.5-0.8", "0.8-0.9", "0.9-0.95", "0.95-0.99", ">=0.99"], hist)))
    a.out.with_suffix(".json").write_text(json.dumps(stats, indent=1))
    print(json.dumps(stats), flush=True)


if __name__ == "__main__":
    main()
