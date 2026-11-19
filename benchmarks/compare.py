"""Paired comparison of two evaluations, with coverage and both metrics.

    python benchmarks/compare.py pm BASE NEW          # PM-ECG-ID results.json files, dirs or globs
    python benchmarks/compare.py kaggle BASE NEW      # Kaggle results.json files, dirs or globs

Only the images present in both evaluations are compared; the coverage of each is reported
against the whole database (6,000 PM-ECG-ID ECGs; 8,793 Kaggle images of 977 records).
PM-ECG-ID: PMcardio-style SNR (failed leads count 0 dB), PCC, failed leads, the official Kaggle
metric on the same leads, per scenario group, and the mean over the 60 scenarios next to
PMcardio's paper (easy 12 / medium 29 / hard 19 subsets: 16.68 / 15.79 / 11.10 dB; the paper does
not list which scenario is in which group, so only the 60-subset mean, 14.48 dB, compares).
Kaggle: the official metric, raw and with holes filled as submitted, on all records, the
holdout (id % 5 == 0 and the 15 local records, never trained on) and the rest, per image type.
"""
from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import metrics as MT  # noqa: E402

PM_TOTAL, K_TOTAL = 6000, 8793
PMCARDIO_60 = (12 * 16.682 + 29 * 15.785 + 19 * 11.101) / 60
LOCAL15 = set(("1257259783,2587667307,3042142152,426433979,994226251,119693645,1449491391,1962423499,"
               "235672018,2841936136,3574249147,3646789790,3879757040,4050029009,644026427").split(","))


def _files(spec):
    out = []
    for s in spec.split(","):
        p = Path(s)
        if p.is_dir():
            out += [str(f) for f in p.rglob("*results.json")]
        else:
            out += glob.glob(s, recursive=True)
    return sorted(set(out))


def _pm_load(spec):
    d = {}
    for f in _files(spec):
        j = json.load(open(f))
        for r in j.get("rows", []):
            d[(r["scenario"], str(r["ecg"]), str(r["image_id"]))] = r
    return d


def _pm_summ(rows):
    snr, pcc, fail, kag = [], [], [], []
    for r in rows:
        kp = r["leads"].get("_kaggle")
        if kp:
            kag.append(MT.compute_snr(sum(v[0] for v in kp.values()), sum(v[1] for v in kp.values())))
        for L in MT.LEADS:
            m = r["leads"].get(L)
            snr.append(m["snr"] if m else 0.0); pcc.append(m["pcc"] if m else 0.0); fail.append(m is None)
    return (float(np.mean(snr)), float(np.mean(pcc)), 100 * float(np.mean(fail)),
            MT.kaggle_score(kag) if kag else float("nan"))


def _group(s):
    if s.startswith("photos"):
        return s
    if s.startswith("augmentation"):
        return "aug_" + s.split("_")[1]
    return s.split("_")[0]


def pm(base, new):
    B, V = _pm_load(base), _pm_load(new)
    k = sorted(set(B) & set(V))
    if not k:
        return "no common ECG"
    n_ecg = len({(x[0], x[1]) for x in k})
    L = [f"## PM-ECG-ID: {n_ecg} ECGs in both ({100 * n_ecg / PM_TOTAL:.1f}% of {PM_TOTAL})", "",
         "| Group | n | PM SNR (dB) | PCC | Failed leads | Kaggle metric (dB) |", "|---|---|---|---|---|---|"]
    groups = {}
    for x in k:
        groups.setdefault(_group(x[0]), []).append(x)
    for g, ks in [("ALL", k)] + sorted(groups.items()):
        a, b = _pm_summ([B[x] for x in ks]), _pm_summ([V[x] for x in ks])
        L.append(f"| {g} | {len(ks)} | {a[0]:.2f} → **{b[0]:.2f}** ({b[0] - a[0]:+.2f}) | {a[1]:.3f} → {b[1]:.3f} | "
                 f"{a[2]:.1f}% → {b[2]:.1f}% | {a[3]:.2f} → {b[3]:.2f} |")
    sc = {}
    for x in k:
        sc.setdefault(x[0], []).append(x)
    if len(sc) >= 50:
        ma = np.mean([_pm_summ([B[x] for x in v])[0] for v in sc.values()])
        mb = np.mean([_pm_summ([V[x] for x in v])[0] for v in sc.values()])
        L += ["", f"Mean over the {len(sc)} scenarios: {ma:.2f} → **{mb:.2f} dB** "
                  f"(PMcardio, 60 subsets of its paper: {PMCARDIO_60:.2f} dB)."]
    return "\n".join(L)


def _k_load(spec):
    d = {}
    for f in _files(spec):
        if any(p.endswith("-merged") for p in Path(f).parts) and len(_files(spec)) > 1:
            continue
        for r in json.load(open(f)).get("results", []):
            if "kaggle_lin" in r:
                d[(str(r["id"]), str(r["type"]))] = r
    return d


def kaggle(base, new):
    B, V = _k_load(base), _k_load(new)
    k = sorted(set(B) & set(V))
    if not k:
        return "no common image"
    hold = lambda x: int(x[0]) % 5 == 0 or x[0] in LOCAL15  # noqa: E731
    L = [f"## Kaggle training records: {len(k)} images in both ({100 * len(k) / K_TOTAL:.1f}% of {K_TOTAL}), "
         f"{len({x[0] for x in k})} records", ""]
    for M, title in [("kaggle_lin", "official metric"), ("kaggle_fill_lin", "holes filled as submitted")]:
        L += [f"### {title}", "", "| Set | Images | Base (dB) | New (dB) |", "|---|---|---|---|"]
        for nm, sel in [("all", k), ("holdout", [x for x in k if hold(x)]), ("rest", [x for x in k if not hold(x)])]:
            if sel:
                a = MT.kaggle_score([B[x].get(M, B[x]["kaggle_lin"]) for x in sel])
                b = MT.kaggle_score([V[x].get(M, V[x]["kaggle_lin"]) for x in sel])
                L.append(f"| {nm} | {len(sel)} | {a:.2f} | **{b:.2f}** ({b - a:+.2f}) |")
        L.append("")
    L += ["| Type | Images | Base (dB) | New (dB) |", "|---|---|---|---|"]
    for t in sorted({x[1] for x in k}):
        sel = [x for x in k if x[1] == t]
        a = MT.kaggle_score([B[x]["kaggle_lin"] for x in sel]); b = MT.kaggle_score([V[x]["kaggle_lin"] for x in sel])
        L.append(f"| {t} | {len(sel)} | {a:.2f} | {b:.2f} ({b - a:+.2f}) |")
    return "\n".join(L)


if __name__ == "__main__":
    what, base, new = sys.argv[1:4]
    print({"pm": pm, "kaggle": kaggle}[what](base, new))
