"""Gallery figures for the README: PM-ECG-ID pages with the ground truth.

    python scripts/make_gallery.py --root <folder with the scenario subfolders> \
        --gt <folder with leads.npz> --meta <metadata.csv> --out docs/gallery \
        <relative/path/img_N_page_0.jpeg>[:LEAD] ...

Each figure has the photo, the page rectified by the grid with the extracted trace (blue),
and one lead: ground truth (blue) vs ecgdig (red), aligned with the PM-ECG-ID metric.
The images come from PM-ECG-ID (Zenodo 13617673, GPL-3.0-or-later).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "benchmarks" / "pmcardio"))

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from ecgdig import digitize, Config, LEADS  # noqa: E402
import eval_pm as E  # noqa: E402


def _thumb(img, side=1400):
    s = side / max(img.shape[:2])
    return cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA) if s < 1 else img


def figure(path, rel, lead, L, meta, out):
    row = meta.loc[rel]
    ecg = row["ECG ID"]
    r = digitize(str(path), Config(), return_debug=True)
    o = dict(ok=True, leads=r["leads"], signals=r["signals"].astype(np.float32), layout=r.get("layout"),
             seg={k: [np.nonzero(np.isfinite(m))[0][[0, -1]] if np.isfinite(m).any() else None for m in v]
                  for k, v in r.get("per_segment", {}).items()})
    per, _ = E.evaluate_ecg(L[ecg], np.zeros((5000, 0)), [o])
    snr = np.mean([per[l]["snr"] if per[l] else 0.0 for l in LEADS])
    pcc = np.mean([per[l]["pcc"] if per[l] else 0.0 for l in LEADS])
    fail = np.mean([per[l] is None for l in LEADS])
    if lead is None:   # the lead closest to the page mean
        ok = [l for l in LEADS if per[l]]
        lead = min(ok, key=lambda l: abs(per[l]["snr"] - snr)) if ok else "II"
    g = L[ecg][:, LEADS.index(lead)].astype(float)
    F = o["signals"][o["leads"].index(lead)].astype(float)
    segs = [s for s in o["seg"].get(lead, []) if s is not None]
    p = None
    if segs:
        seg = min(segs, key=lambda s: abs((s[1] - s[0] + 1) - len(g)))
        p = E._best_shift(g, F, int(seg[0]), span=int(seg[1] - seg[0] + 1 - len(g)))
        if p is not None:
            m = np.isfinite(p)
            p = p + np.median(g[m] - p[m])

    img = cv2.cvtColor(_thumb(cv2.imread(str(path))), cv2.COLOR_BGR2RGB)
    dbg = cv2.cvtColor(_thumb(r["debug_image"]), cv2.COLOR_BGR2RGB)
    fig = plt.figure(figsize=(14, 8.6), dpi=110)
    scen = rel.split("/")[0].replace("augmentation_", "")
    fig.text(0.015, 0.985, f"{scen} · {Path(rel).stem}", fontsize=14, weight="bold", va="top")
    fig.text(0.015, 0.95, f"printed format {row['ECG format']} · read as {o['layout']} · "
             f"mean over 12 leads: SNR {snr:.1f} dB, PCC {pcc:.3f}, failed leads {100 * fail:.0f}%",
             fontsize=11, va="top", color="#333")
    ax = fig.add_axes([0.01, 0.35, 0.48, 0.52]); ax.imshow(img); ax.axis("off")
    ax.set_title("Input image", fontsize=10, loc="left")
    ax = fig.add_axes([0.51, 0.35, 0.48, 0.52]); ax.imshow(dbg); ax.axis("off")
    ax.set_title("Rectified by the grid + extracted trace (blue)", fontsize=10, loc="left")
    ax = fig.add_axes([0.05, 0.06, 0.92, 0.25])
    t = np.arange(len(g)) / 500.0
    ax.plot(t, g, color="#1f5fbf", lw=1.4, label="ground truth")
    if p is not None:
        ax.plot(t, p, color="#d1342f", lw=1.0, label="ecgdig")
    ls = f"SNR {per[lead]['snr']:.1f} dB, PCC {per[lead]['pcc']:.3f}" if per[lead] else "not read"
    ax.set_title(f"Lead {lead}: {ls}", fontsize=10, loc="left")
    ax.set_xlabel("s"); ax.set_ylabel("mV"); ax.grid(alpha=0.3); ax.legend(loc="upper right", fontsize=9)
    out.mkdir(parents=True, exist_ok=True)
    name = f"{scen}__{Path(rel).stem}.png".replace("/", "_")
    fig.savefig(out / name); plt.close(fig)
    print(name, f"SNR {snr:.2f} PCC {pcc:.3f} layout {o['layout']}", flush=True)
    return name, snr, pcc


def main():
    import pandas as pd
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--gt", type=Path, required=True)
    ap.add_argument("--meta", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=ROOT / "docs" / "gallery")
    ap.add_argument("items", nargs="+")
    a = ap.parse_args()
    meta = pd.read_csv(a.meta).set_index("Image relative path")
    L = np.load(a.gt / "leads.npz")
    for it in a.items:
        rel, _, lead = it.partition(":")
        figure(a.root / rel, rel, lead or None, L, meta, a.out)


if __name__ == "__main__":
    main()
