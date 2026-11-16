"""A/B between two eval_pm.py runs (same image list): what changed, ECG by ECG.

    python benchmarks/pmcardio/ab.py results/pm_before results/pm_after [--min-delta 1.0]

Reads the results.json of each folder. Shows the summary of both (PMcardio paper metric:
failure = 0 dB / PCC 0), how many ECGs improved or got worse and the list of those that
changed by more than --min-delta dB, with the layout read in each run.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

LEADS = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]


def _load(d: Path):
    rows = json.loads((d / "results.json").read_text())["rows"]
    return {(r["scenario"], r["ecg"], r["image_id"]): r for r in rows}


def _agg(rows):
    snr, pcc, fail, snr_ok = [], [], [], []
    for r in rows:
        for l in LEADS:
            m = r["leads"].get(l)
            snr.append(m["snr"] if m else 0.0); pcc.append(m["pcc"] if m else 0.0); fail.append(m is None)
            if m:
                snr_ok.append(m["snr"])
    return dict(n=len(rows), snr=float(np.mean(snr)), pcc=float(np.mean(pcc)), fail=float(np.mean(fail)),
                snr_ok=float(np.mean(snr_ok)) if snr_ok else 0.0, sec=float(np.mean([r.get("sec", 0.0) for r in rows])))


def _ecg_snr(r):
    return float(np.mean([r["leads"][l]["snr"] if r["leads"].get(l) else 0.0 for l in LEADS]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("a", type=Path); ap.add_argument("b", type=Path)
    ap.add_argument("--min-delta", type=float, default=1.0)
    a = ap.parse_args()
    A, B = _load(a.a), _load(a.b)
    keys = sorted(set(A) & set(B))
    if not keys:
        raise SystemExit("nenhum ECG em comum")
    print(f"ECGs in common: {len(keys)} (A has {len(A)}, B has {len(B)})\n")
    print("| run | SNR (dB) | PCC | failure | SNR without failures | s per ECG |\n|---|---|---|---|---|---|")
    for name, R in (("A " + a.a.name, A), ("B " + a.b.name, B)):
        s = _agg([R[k] for k in keys])
        print(f"| {name} | {s['snr']:.2f} | {s['pcc']:.3f} | {100 * s['fail']:.1f}% | {s['snr_ok']:.2f} | {s['sec']:.0f} |")
    d = np.array([_ecg_snr(B[k]) - _ecg_snr(A[k]) for k in keys])
    print(f"\nimproved (> {a.min_delta} dB): {(d > a.min_delta).sum()}   worse: {(d < -a.min_delta).sum()}   "
          f"same: {(np.abs(d) <= a.min_delta).sum()}   median difference: {np.median(d):+.2f} dB\n")
    print("| scenario | format | layout A | layout B | SNR A | SNR B | diff. |\n|---|---|---|---|---|---|---|")
    for i in np.argsort(-np.abs(d)):
        if abs(d[i]) <= a.min_delta:
            break
        k = keys[i]; ra, rb = A[k], B[k]
        la = "+".join(str(x) for x in ra["layout"]); lb = "+".join(str(x) for x in rb["layout"])
        print(f"| {k[0].replace('augmentation_', '')} | {ra['format']} | {la} | {lb} | {_ecg_snr(ra):.1f} | {_ecg_snr(rb):.1f} | {d[i]:+.1f} |")


if __name__ == "__main__":
    main()
