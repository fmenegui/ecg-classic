"""Probe pages: where do the Kaggle winners stop reading? One factor changed at a time.

    python benchmarks/topsol/probe_pages.py --out /root/probe --n 20

The hengck23-based winners (2nd, 3rd; the 1st used the same rectification) read the Kaggle sheet
as a template: nine lead-name keypoints are mapped by a homography onto their positions in ONE
Kaggle image (640106434-0001), a 44 x 57 lattice of 5 mm grid points covering the whole page is
snapped onto that image's lattice, and the signal is then read at fixed places of the 2200 x 1700
result: zero-mV lines at y = 703.5 / 987.5 / 1271.5 / 1531.5 px, 79 px/mV, time from x = 118 to
2080 px split in four 2.5 s columns (stage0_common.make_ref_point, stage1_common.rectify_image,
3rd place configs/stage2.py). These pages start from that exact geometry, drawn by us with real
PTB-XL ECGs (fold 10, never a PM-ECG-ID ECG or patient), and change one thing per variant:

  K0  the template geometry itself (full-page 5 mm grid on the reference lattice, labels at the
      reference keypoints, 3x4 + lead II rhythm strip)
  K1  rows closer together (25 mm instead of 36.4: PM-ECG-ID's 3x4+1R pages)
  K2  grid only in a box under the traces, white margins (most printed ECGs)
  K3  the whole page drawn 8% smaller in the same image (another paper size or scan scale)
  K4  no lead labels
  K5  6x2 layout on the template page
  K6  our print-kit page (A4, its own margins and row spacing), 3x4+1R

Writes OUT/test/<variant>_<page>.png (200 dpi, 2200 x 1700, like the Kaggle images),
OUT/test.csv (the competition's request: 12 leads, 2.5 s each, II 10 s) and OUT/gt/<key>.npz
(signals 12 x 5000 mV at 500 Hz, the printed segments). benchmarks/topsol/score_probe.py scores
any reader on them.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "kit"))
import make_kit as K  # noqa: E402

W, H = 2200, 1700
PPMM = 7.8                       # the reference lattice: 39 px per 5 mm
ZERO = [703.5, 987.5, 1271.5, 1531.5]
X0, X1 = 118.0, 2080.0           # 10 s
GAIN = 79.0                      # px per mV
ROWS_3x4 = [["I", "aVR", "V1", "V4"], ["II", "aVL", "V2", "V5"], ["III", "aVF", "V3", "V6"]]
ROWS_6x2 = [["I", "V1"], ["II", "V2"], ["III", "V3"], ["aVR", "V4"], ["aVL", "V5"], ["aVF", "V6"]]
VARIANTS = ["K0", "K1", "K2", "K3", "K4", "K5", "K6"]


def _lattice():
    xs = np.linspace(0, 2199, 57)
    ys = np.linspace(7, 1699, 44)
    return xs, ys


def draw(ax, sig, v):
    """One page of variant v on an axes in pixels (x right, y down). Returns the printed segments."""
    minor, major = "#f3c4c4", "#e07a7a"
    xs, ys = _lattice()
    if v == "K2":
        box = (X0 - 60, ZERO[0] - 160, X1 + 40, ZERO[3] + 120)
    else:
        box = (0, ys[0], 2199, ys[-1])
    step = 39.27 / 5
    gx = np.arange(box[0], box[2] + 1e-6, step)
    gy = np.arange(box[1], box[3] + 1e-6, step)
    for k, x in enumerate(gx):
        ax.plot([x, x], [box[1], box[3]], color=major if k % 5 == 0 else minor, lw=0.9 if k % 5 == 0 else 0.4, zorder=1)
    for k, y in enumerate(gy):
        ax.plot([box[0], box[2]], [y, y], color=major if k % 5 == 0 else minor, lw=0.9 if k % 5 == 0 else 0.4, zorder=1)
    if v == "K5":
        rows = ROWS_6x2
        zero = list(np.linspace(ZERO[0] - 300, ZERO[3], 6))
    else:
        rows = ROWS_3x4 + [["R:II"]]
        zero = list(ZERO)
        if v == "K1":
            zero = [ZERO[0] + i * 25 * PPMM for i in range(4)]
    segs = []
    for i, row in enumerate(rows):
        y0 = zero[i]
        # calibration pulse: 0.2 s, 1 mV, before the trace
        ax.plot([X0 - 52, X0 - 44, X0 - 44, X0 - 8, X0 - 8, X0 - 2], [y0, y0, y0 - GAIN, y0 - GAIN, y0, y0],
                color="k", lw=1.2, zorder=3)
        n = len(row)
        for c, name in enumerate(row):
            rhythm = name.startswith("R:")
            lead = name[2:] if rhythm else name
            t0, t1 = (0.0, 10.0) if rhythm else (10.0 * c / n, 10.0 * (c + 1) / n)
            a, b = int(round(t0 * 500)), int(round(t1 * 500))
            t = np.arange(a, b) / 500.0
            x = X0 + t / 10.0 * (X1 - X0)
            y = y0 - sig[K.LEADS.index(lead), a:b] * GAIN
            ax.plot(x, y, color="k", lw=1.1, zorder=3)
            if v != "K4":
                # the reference keypoints sit ~10 px right of the column start, ~68 px below the zero line
                ax.text(x[0] + 12, y0 + 68, lead, fontsize=13, family="DejaVu Sans", weight="bold",
                        ha="left", va="center", zorder=4)
            segs.append(dict(lead=lead, rhythm=rhythm, t0=t0, t1=t1))
    ax.text(40, 40, "25mm/s  10mm/mV", fontsize=11, family="DejaVu Sans")
    return segs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--folds", default="10")
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--variants", default=",".join(VARIANTS))
    ap.add_argument("--db", type=Path, default=None)
    a = ap.parse_args()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import cv2
    (a.out / "test").mkdir(parents=True, exist_ok=True)
    (a.out / "gt").mkdir(parents=True, exist_ok=True)
    cache = a.out / ".records"; cache.mkdir(exist_ok=True)
    db = a.db or (a.out / "ptbxl_database.csv")
    if not db.exists():
        db.write_bytes(K._get(K.BASE + "ptbxl_database.csv"))
    lo, _, hi = a.folds.partition("-")
    recs = K.choose_records(db, set(range(int(lo), int(hi or lo) + 1)), a.n, a.seed, cache)
    rows = []
    for v in a.variants.split(","):
        for p, rec in enumerate(recs):
            key = f"{v}_{p:03d}"
            png = a.out / "test" / f"{key}.png"
            if v == "K6":
                # our print-kit page (A4 landscape, 3x4+1R), drawn at 200 dpi then fitted in 2200 x 1700
                fig = plt.figure(figsize=(K.PAGE_W / 25.4, K.PAGE_H / 25.4), dpi=200)
                ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, K.PAGE_W); ax.set_ylim(K.PAGE_H, 0); ax.axis("off")
                plan = dict(layout="3x4+1R", grid="red", pulse="left", trace_color="#000000", trace_mm=0.3,
                            labels=True, separators=False)
                gt = K.draw_page(ax, 200 + p, rec, plan, "probe")   # marker ids 800+ (pages < 250)
                fig.savefig(png, dpi=200); plt.close(fig)
                im = cv2.imread(str(png)); h, w = im.shape[:2]; k = min(W / w, H / h)
                r = cv2.resize(im, (int(w * k), int(h * k)), interpolation=cv2.INTER_AREA)
                outi = np.full((H, W, 3), 255, np.uint8); outi[(H - r.shape[0]) // 2:(H - r.shape[0]) // 2 + r.shape[0],
                                                                (W - r.shape[1]) // 2:(W - r.shape[1]) // 2 + r.shape[1]] = r
                cv2.imwrite(str(png), outi)
                segs = [dict(lead=s["lead"], rhythm=s["rhythm"], t0=s["t0"], t1=s["t1"]) for s in gt["segments"]]
            else:
                fig = plt.figure(figsize=(W / 200, H / 200), dpi=200)
                ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, W); ax.set_ylim(H, 0); ax.axis("off")
                if v == "K3":   # the whole page 8% smaller, centred
                    s = 0.92
                    ax.set_xlim(-(W / s - W) / 2, W + (W / s - W) / 2); ax.set_ylim(H + (H / s - H) / 2, -(H / s - H) / 2)
                segs = draw(ax, rec["signal"], v)
                fig.savefig(png, dpi=200); plt.close(fig)
            np.savez_compressed(a.out / "gt" / f"{key}.npz", signals=rec["signal"], leads=np.array(K.LEADS),
                                segments=json.dumps(segs), ecg_id=int(float(rec["ecg_id"])), variant=v)
            n = 1250
            rows += [(key, L, 500, 4 * n if L == "II" else n) for L in K.LEADS]
        print(v, "done", flush=True)
    with open(a.out / "test.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["id", "lead", "fs", "number_of_rows"]); w.writerows(rows)
    print(len(rows) // 12, "pages in", a.out)


if __name__ == "__main__":
    main()
