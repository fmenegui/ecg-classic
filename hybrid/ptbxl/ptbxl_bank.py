"""Bank of real 12-lead signals for hybrid/rowgen.py, built from PTB-XL (CC-BY 4.0).

    python hybrid/ptbxl/ptbxl_bank.py --out /workspace/bank.npy [--work /workspace/ptbxl]

Downloads PTB-XL 1.0.3 (500 Hz records) from PhysioNet, keeps strat folds 1-9 (fold 10 stays
unseen) and drops every record used for evaluation: the Kaggle training records listed in
exclude_ptbxl.txt (hybrid/ptbxl/dedupe.py, when it has been run) and, always, the records whose
lead II matches (correlation >= 0.95) one of our local evaluation ECGs (eval_lead2.npz: lead II of
the 15 Kaggle records of the 135-image set, 100 Hz, z-normalised) and the 100 PM-ECG-ID ECGs
(PTB-XL records too): by their ecg_id (PM-ECG-ID names them LPAE_<ecg_id>_hr) and by a lead-II
match anywhere in the 10 s record (eval_pm_lead2.npz: the first 2.5 s of the ground-truth cut of lead II,
100 Hz, z-normalised). Saves an (N, 12, 5000) float16
array in mV that the data workers memory-map.
"""
from __future__ import annotations

import argparse
import sys
import csv
import os
import subprocess
import zipfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

URL = ("https://physionet.org/static/published-projects/ptb-xl/"
       "ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3.zip")
ORDER = ["I", "II", "III", "AVR", "AVL", "AVF", "V1", "V2", "V3", "V4", "V5", "V6"]


def _read(path):
    import wfdb
    sig, f = wfdb.rdsamp(path)
    names = [n.upper() for n in f["sig_name"]]
    x = np.stack([sig[:5000, names.index(n)] for n in ORDER]).astype(np.float32)
    if x.shape[1] < 5000:
        x = np.pad(x, ((0, 0), (0, 5000 - x.shape[1])), mode="edge")
    return np.nan_to_num(x).astype(np.float16)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--work", type=Path, default=Path("/workspace/ptbxl"))
    ap.add_argument("--exclude", type=Path, default=Path(__file__).with_name("exclude_ptbxl.txt"))
    ap.add_argument("--folds", default="1,2,3,4,5,6,7,8,9")
    ap.add_argument("--jobs", type=int, default=os.cpu_count() or 4)
    a = ap.parse_args()
    a.work.mkdir(parents=True, exist_ok=True)
    z = a.work / "ptbxl.zip"
    if not z.exists():
        subprocess.run(["wget", "-q", "-O", str(z), URL], check=True)
    with zipfile.ZipFile(z) as zf:
        names = [n for n in zf.namelist() if n.endswith("ptbxl_database.csv") or "/records500/" in n]
        root = names[0].split("/")[0]
        if not (a.work / root / "ptbxl_database.csv").exists():
            zf.extractall(a.work, members=names)
    root = a.work / root
    excl = set()
    if a.exclude.exists():
        excl = {int(s) for s in a.exclude.read_text().split() if s.strip()}
    folds = {int(f) for f in a.folds.split(",")}
    pm = Path(__file__).with_name("eval_pm_lead2.npz")
    if not pm.exists():   # never build a bank that may hold evaluation signals
        sys.exit(f"{pm} is missing: cannot remove the PM-ECG-ID ECGs from the bank")
    pm_ids = set(int(v) for v in np.load(pm)["ids"])
    rows = [r for r in csv.DictReader(open(root / "ptbxl_database.csv"))
            if int(r["strat_fold"]) in folds and int(float(r["ecg_id"])) not in excl
            and int(float(r["ecg_id"])) not in pm_ids]
    print(f"{len(pm_ids)} PM-ECG-ID ecg_ids excluded by id", flush=True)
    print(f"{len(rows)} records (folds {sorted(folds)}, {len(excl)} Kaggle records excluded)", flush=True)
    paths = [str(root / r["filename_hr"]) for r in rows]
    with ProcessPoolExecutor(a.jobs) as ex:
        X = np.stack(list(ex.map(_read, paths, chunksize=64)))
    ev = Path(__file__).with_name("eval_lead2.npz")
    if ev.exists():   # drop the records of our evaluation ECGs (matched on lead II)
        E = np.load(ev)["x"].astype(np.float32)
        L2 = X[:, 1, ::5].astype(np.float32)                       # 100 Hz, 1000 samples
        L2 = L2 - L2.mean(1, keepdims=True)
        L2 /= np.linalg.norm(L2, axis=1, keepdims=True) + 1e-9
        C = L2 @ E.T
        bad = C.max(1) >= 0.95
        print(f"evaluation ECGs found in PTB-XL: {int((C.max(0) >= 0.95).sum())} of {len(E)}; "
              f"{int(bad.sum())} records dropped", flush=True)
        X = X[~bad]
    if pm.exists():   # PM-ECG-ID: 2.5 s of lead II anywhere in the record (FFT cross-correlation)
        E = np.load(pm)["x"].astype(np.float32)                  # (100, 250) at 100 Hz
        L2 = X[:, 1, ::5].astype(np.float32)                       # (N, 1000)
        n = 2048
        drop = np.zeros(len(X), bool)
        FE = np.conj(np.fft.rfft(E, n))
        for i in range(0, len(X), 512):
            Y = L2[i:i + 512]
            FY = np.fft.rfft(Y, n)
            # sliding norm of the 500-sample window
            c2 = np.cumsum(np.pad(Y ** 2, ((0, 0), (1, 0))), 1); c1 = np.cumsum(np.pad(Y, ((0, 0), (1, 0))), 1)
            w = E.shape[1]; s2 = c2[:, w:] - c2[:, :-w]; s1 = c1[:, w:] - c1[:, :-w]
            nrm = np.sqrt(np.maximum(s2 - s1 ** 2 / w, 1e-9))     # (b, 501)
            best = np.zeros(len(Y))
            for e in range(len(E)):
                cc = np.fft.irfft(FY * FE[e], n)[:, :nrm.shape[1]] / nrm
                best = np.maximum(best, cc.max(1))
            drop[i:i + 512] = best >= 0.9
        print(f"PM-ECG-ID ECGs matched on lead II: {int(drop.sum())} records dropped", flush=True)
        X = X[~drop]
    np.save(a.out, X)
    print(f"bank {X.shape} {X.dtype} -> {a.out} ({X.nbytes / 2**30:.1f} GB)", flush=True)


if __name__ == "__main__":
    main()
