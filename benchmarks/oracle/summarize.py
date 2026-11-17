"""Merge the shards of the stage oracle and write the attribution report.

    python benchmarks/oracle/summarize.py shards_dir/ out_dir/
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import metrics as MT  # noqa: E402
READINGS = ["photo", "photo+cal", "rectified", "rectified+cal", "document", "document+cal"]

STEPS = [("geometry (perspective, curl, folds, crop)", "photo", "rectified"),
         ("capture appearance (light, blur, noise, screen)", "rectified", "document"),
         ("layout + calibration on the photo", "photo", "photo+cal"),
         ("layout + calibration on the flat document", "document", "document+cal"),
         ("everything (photo -> clean flat page with known calibration)", "photo", "document+cal")]


def _kaggle(recs, nm):
    v = [r[nm]["kaggle_lin"] for r in recs if nm in r]
    return MT.kaggle_score(v), len(v)


def _row(recs, nm):
    k, n = _kaggle(recs, nm)
    pm = MT.pm_summary([r[nm]["pm"] for r in recs if nm in r])
    med = np.median([r[nm]["kaggle_db"] for r in recs if nm in r]) if n else float("nan")
    return k, med, pm, n


def table(recs, title):
    out = [f"### {title} ({len(recs)} pages)", "",
           "| Reading | n | Kaggle score (dB) | median image (dB) | PM SNR (dB) | PM PCC | PM failed leads |",
           "|---|---|---|---|---|---|---|"]
    rows = {}
    for nm in READINGS:
        k, med, pm, n = _row(recs, nm)
        rows[nm] = (k, pm)
        if n:
            out.append(f"| {nm} | {n} | {k:.2f} | {med:.2f} | {pm['snr']:.2f} | {pm['pcc']:.3f} | {100 * pm['failure']:.1f}% |")
    out += ["", "| Stage replaced by its truth | Kaggle gain (dB) | PM SNR gain (dB) | PM failed leads |", "|---|---|---|---|"]
    for label, a, b in STEPS:
        if a in rows and b in rows and np.isfinite(rows[a][0]) and np.isfinite(rows[b][0]):
            out.append(f"| {label} | {rows[b][0] - rows[a][0]:+.2f} | {rows[b][1]['snr'] - rows[a][1]['snr']:+.2f} | "
                       f"{100 * rows[a][1]['failure']:.1f}% -> {100 * rows[b][1]['failure']:.1f}% |")
    return out + [""]


def main():
    src, dst = Path(sys.argv[1]), Path(sys.argv[2])
    dst.mkdir(parents=True, exist_ok=True)
    recs = []
    for p in sorted(src.rglob("results.json")):
        recs += json.loads(p.read_text())
    bad = [r for r in recs if "error" in r]
    recs = [r for r in recs if "error" not in r]
    (dst / "results.json").write_text(json.dumps(recs))
    md = ["# Stage oracle: where the classical reader loses dB", "",
          f"{len(recs)} pages rendered by ecg-torch-renderer from PTB-XL fold 10 (never used in training), "
          f"{len(bad)} pages failed to render. Each page is read six times, one stage at a time replaced by its "
          "truth (see `benchmarks/oracle/run.py`). Kaggle = official metric (mean of linear image SNRs); "
          "PM = PMcardio style (failure counts as 0 dB / PCC 0).", ""]
    md += table(recs, "All pages")
    st = defaultdict(list)
    for r in recs:
        st[r.get("style", "generic")].append(r)
    for k in sorted(st):
        md += table(st[k], f"Style: {k}")
    by = defaultdict(list)
    for r in recs:
        by[r.get("capture", "?")].append(r)
    for k in sorted(by):
        md += table(by[k], f"Capture: {k}")
    fam = defaultdict(list)
    for r in recs:
        fam[f"{len(r['rows'])} rows"].append(r)
    for k in sorted(fam, key=lambda s: int(s.split()[0])):
        md += table(fam[k], f"Layout with {k}")
    # where the auto reading fails on the clean page: calibration read wrong
    wrong = defaultdict(int)
    for r in recs:
        d = r.get("document", {})
        if d.get("speed") and abs(d["speed"] - r["speed"]) > 0.5:
            wrong["speed"] += 1
        if d.get("gains") and all(abs(g - r["gain"]) > 0.5 for g in d["gains"]):
            wrong["gain"] += 1
        if not d.get("ok", True):
            wrong["no reading"] += 1
    md += ["### Calibration read on the flat document (auto)", "",
           f"speed wrong on {wrong['speed']} pages, gain wrong on {wrong['gain']}, no reading on {wrong['no reading']} "
           f"of {len(recs)}.", ""]
    (dst / "report.md").write_text("\n".join(md))
    print("\n".join(md))


if __name__ == "__main__":
    main()
