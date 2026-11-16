"""Table of ecgdig on PM-ECG-ID next to the numbers published by PMcardio.

    python benchmarks/pmcardio/compare.py <merged results.json> --gt <folder with leads.npz> --out <dir>

PMcardio numbers (Demolder et al., medRxiv 2024.08.31.24312876): whole database PCC > 0.91,
SNR > 12.5 dB, 6.62% of leads failed; groups easy (12 subsets) PCC 0.977 / SNR 16.68 dB,
medium (29) 0.970 / 15.79 dB, hard (19) 0.817 / 11.10 dB. The paper does not list which
scenarios form each group, it only names scale 1/32, blur 31/41, brightness -160 and extreme
zoom-out as hard; so the per-group comparison here uses scenario families, not the paper's groups.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

LEADS = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
PM = dict(pcc=0.91, snr=12.5, fail=0.0662)
PM_GROUPS = {"easy (12)": (0.977, 16.682), "medium (29)": (0.970, 15.785), "hard (19)": (0.817, 11.101)}


def family(s):
    if s.startswith("photos_"):
        return "photos"
    if s.startswith("digital_data_"):
        return "digital: " + ("noise" if "noise" in s else "opacity")
    k = s.replace("augmentation_", "")
    for f in ("JPEG_compression", "blur_kernel", "brightness", "contrast_gamma", "padding", "perspective",
              "resolution_scale_down", "rotation", "zoom"):
        if k.startswith(f):
            return f
    return k


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results", type=Path)
    ap.add_argument("--gt", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    rows = json.loads(a.results.read_text())["rows"]
    L = np.load(a.gt / "leads.npz")
    agg = defaultdict(lambda: defaultdict(list))
    for r in rows:
        g = L[r["ecg"]]
        for j, name in enumerate(LEADS):
            mt = r["leads"].get(name)
            gg = g[:, j].astype(float)
            rz = float(np.sqrt(np.mean((gg - np.median(gg)) ** 2)))   # RMSE of a null signal
            for key in ("ALL", "fam:" + family(r["scenario"]), r["scenario"]):
                d = agg[key]
                d["snr"].append(mt["snr"] if mt else 0.0)
                d["pcc"].append(mt["pcc"] if mt else 0.0)
                d["rmse"].append(mt["rmse"] if mt else rz)
                d["fail"].append(mt is None)
                if mt:
                    d["snr_ok"].append(mt["snr"]); d["pcc_ok"].append(mt["pcc"]); d["rmse_ok"].append(mt["rmse"])
                d["ecg"].append((r["scenario"], r["ecg"]))

    def line(k, label=None):
        d = agg[k]
        n = len({e for e in d["ecg"]})
        f = lambda v: float(np.mean(v)) if v else float("nan")
        return (f"| {label or k} | {n} | {f(d['snr']):.2f} | {f(d['pcc']):.3f} | {f(d['rmse']):.3f} | "
                f"{100 * f(d['fail']):.1f}% | {f(d['snr_ok']):.2f} | {f(d['pcc_ok']):.3f} | {f(d['rmse_ok']):.3f} |")

    hdr = ["| set | ECGs | SNR | PCC | RMSE | failure | SNR (no failures) | PCC (no failures) | RMSE (no failures) |",
           "|---|---|---|---|---|---|---|---|---|"]
    md = ["# ecgdig (classical) on PM-ECG-ID", "",
          "Failure = lead missing or < 50% covered; it counts as SNR 0, PCC 0 and the RMSE of a null signal.", "",
          "## Whole database", ""] + hdr + [line("ALL", "ecgdig")]
    md += [f"| PMcardio (published) | 100 | > {PM['snr']} | > {PM['pcc']} | — | {100 * PM['fail']:.2f}% | — | — | — |", ""]
    md += ["Groups of the PMcardio paper (no scenario list): " +
           "; ".join(f"{k}: PCC {v[0]:.3f}, SNR {v[1]:.2f} dB" for k, v in PM_GROUPS.items()), ""]
    md += ["## By scenario family", ""] + hdr + [line(k, k[4:]) for k in sorted(k for k in agg if k.startswith("fam:"))]
    md += ["", "## By scenario", ""] + hdr + [line(k) for k in sorted(k for k in agg if k != "ALL" and not k.startswith("fam:"))]
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "comparison.md").write_text("\n".join(md) + "\n")
    summ = {k: dict(snr=float(np.mean(v["snr"])), pcc=float(np.mean(v["pcc"])), rmse=float(np.mean(v["rmse"])),
                    fail=float(np.mean(v["fail"])), snr_ok=float(np.mean(v["snr_ok"])) if v["snr_ok"] else None)
            for k, v in agg.items()}
    (a.out / "comparison.json").write_text(json.dumps(summ, indent=1))
    print("\n".join(md[:12]))


if __name__ == "__main__":
    main()
