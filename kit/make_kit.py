"""Print kit: real PTB-XL ECGs on A4 pages with fiducial markers, to photograph.

    python kit/make_kit.py --out kit_out --pages 40 --folds 1-9 --first-page 0
    python kit/make_kit.py --out kit_test --pages 20 --folds 10 --first-page 100

Each page is an ECG printed at 25 mm/s and 10 mm/mV in one of several layouts, grid colours and
pulse positions, with four ArUco markers (4x4 dictionary, ids 4p..4p+3 for page p) at known
positions. A photo of a printed page then has an exact ground truth: kit/read_photo.py finds the
markers, the page and the homography from page millimetres to photo pixels, so every trace sample
(time, lead, mV) maps to a photo position.

Output in --out:
- kit.pdf: the pages, vector, A4 landscape; print at 100% (no "fit to page");
- gt/page_XXXX.json: record, layout, every printed segment (lead, time window, x and baseline
  in mm), the markers' corners in mm, gain and speed;
- gt/page_XXXX.npz: the 12 x 5000 signals (mV, 500 Hz) of the record;
- manifest.csv.

Records: PTB-XL 1.0.3 (CC-BY 4.0), 500 Hz, from the chosen strat folds. Never a PM-ECG-ID ECG nor
another record of the same patient, nor a record whose lead II matches an evaluation ECG
(hybrid/ptbxl/eval_*_lead2.npz, the same rule as the training bank).
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import os
import random
import urllib.request
from pathlib import Path

import numpy as np

BASE = "https://physionet.org/files/ptb-xl/1.0.3/"
LEADS = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
PAGE_W, PAGE_H = 297.0, 210.0          # A4 landscape, mm
SPEED, GAIN = 25.0, 10.0               # mm/s, mm/mV
FS = 500
MARKER_MM, MARKER_OFF = 14.0, 7.0      # marker side and its distance to the page edges
# page grid area (mm)
GX0, GX1, GY0, GY1 = 12.0, 285.0, 26.0, 186.0

LAYOUTS = {
    # rows of (lead, column); columns split the 10 s; "R:<lead>" is a 10 s rhythm strip
    "3x4+1R": [["I", "aVR", "V1", "V4"], ["II", "aVL", "V2", "V5"], ["III", "aVF", "V3", "V6"], ["R:II"]],
    "3x4": [["I", "aVR", "V1", "V4"], ["II", "aVL", "V2", "V5"], ["III", "aVF", "V3", "V6"]],
    "3x4+3R": [["I", "aVR", "V1", "V4"], ["II", "aVL", "V2", "V5"], ["III", "aVF", "V3", "V6"],
               ["R:V1"], ["R:II"], ["R:V5"]],
    "6x2": [["I", "V1"], ["II", "V2"], ["III", "V3"], ["aVR", "V4"], ["aVL", "V5"], ["aVF", "V6"]],
    "6x2+1R": [["I", "V1"], ["II", "V2"], ["III", "V3"], ["aVR", "V4"], ["aVL", "V5"], ["aVF", "V6"], ["R:II"]],
    "12x1": [[l] for l in LEADS],
}
GRIDS = {   # (minor colour, major colour)
    "red": ("#f4b6b6", "#e06666"),
    "pink": ("#f8cde0", "#e58bb4"),
    "green": ("#b9e0c0", "#5fae6e"),
    "blue": ("#bcd3f0", "#6f99d8"),
    "grey": ("#d0d0d0", "#8a8a8a"),
    "orange": ("#f7d3b0", "#e69a52"),
}


def _get(url: str) -> bytes:
    with urllib.request.urlopen(url, timeout=120) as r:
        return r.read()


def _record(filename_hr: str, cache: Path) -> np.ndarray:
    """(12, 5000) mV of one 500 Hz record, downloaded once into cache."""
    import wfdb
    stem = cache / Path(filename_hr).name
    for ext in (".hea", ".dat"):
        p = stem.with_suffix(ext)
        if not p.exists():
            p.write_bytes(_get(BASE + filename_hr + ext))
    sig, f = wfdb.rdsamp(str(stem))
    names = [n.upper() for n in f["sig_name"]]
    x = np.stack([sig[:5000, names.index(n.upper())] for n in LEADS]).astype(np.float32)
    return np.nan_to_num(x)


def _lead2_matches(x: np.ndarray, here: Path) -> bool:
    """Lead II of the record matches an evaluation ECG (same test as hybrid/ptbxl/ptbxl_bank.py)."""
    l2 = x[1, ::5].astype(np.float32)
    l2 = l2 - l2.mean()
    l2 /= np.linalg.norm(l2) + 1e-9
    ev = here / "eval_lead2.npz"
    if ev.exists() and (np.load(ev)["x"].astype(np.float32) @ l2).max() >= 0.95:
        return True
    pm = here / "eval_pm_lead2.npz"
    E = np.load(pm)["x"].astype(np.float32)                       # (100, 250): 2.5 s at 100 Hz
    w = E.shape[1]
    win = np.lib.stride_tricks.sliding_window_view(x[1, ::5].astype(np.float32), w)
    win = win - win.mean(1, keepdims=True)
    win /= np.linalg.norm(win, axis=1, keepdims=True) + 1e-9
    return bool((win @ E.T).max() >= 0.95)


def choose_records(db_csv: Path, folds: set[int], n: int, seed: int, cache: Path) -> list[dict]:
    here = Path(__file__).resolve().parents[1] / "hybrid" / "ptbxl"
    pm_ids = {int(v) for v in np.load(here / "eval_pm_lead2.npz")["ids"]}
    rows = list(csv.DictReader(open(db_csv)))
    pm_patients = {r["patient_id"] for r in rows if int(float(r["ecg_id"])) in pm_ids}
    pool = [r for r in rows if int(r["strat_fold"]) in folds and int(float(r["ecg_id"])) not in pm_ids
            and r["patient_id"] not in pm_patients]
    random.Random(seed).shuffle(pool)
    out, patients = [], set()
    for r in pool:
        if len(out) == n:
            break
        if r["patient_id"] in patients:   # one record per patient
            continue
        x = _record(r["filename_hr"], cache)
        if np.abs(x).max() > 8 or _lead2_matches(x, here):
            continue
        r = dict(r, signal=x)
        out.append(r)
        patients.add(r["patient_id"])
    return out


def page_plan(p: int, rng: random.Random) -> dict:
    layout = list(LAYOUTS)[p % len(LAYOUTS)]
    return dict(layout=layout, grid=rng.choice(list(GRIDS)),
                pulse=rng.choice(["left", "left", "right", "both"]),
                trace_color=rng.choice(["#000000", "#000000", "#1a1a80"]),
                trace_mm=rng.choice([0.25, 0.3, 0.4]),
                labels=rng.random() < 0.85, separators=rng.random() < 0.4)


def draw_page(ax, p: int, rec: dict, plan: dict, kit: str) -> dict:
    """Draw page p on a matplotlib axes in mm (x right, y down). Returns its ground truth."""
    import cv2
    rows = LAYOUTS[plan["layout"]]
    minor, major = GRIDS[plan["grid"]]
    # grid
    for k, x in enumerate(np.arange(GX0, GX1 + 1e-6, 1.0)):
        ax.plot([x, x], [GY0, GY1], color=major if k % 5 == 0 else minor, lw=(0.7 if k % 5 == 0 else 0.3), zorder=1)
    for k, y in enumerate(np.arange(GY0, GY1 + 1e-6, 1.0)):
        ax.plot([GX0, GX1], [y, y], color=major if k % 5 == 0 else minor, lw=(0.7 if k % 5 == 0 else 0.3), zorder=1)
    # rows: baselines evenly spaced in the grid area
    n = len(rows)
    step = (GY1 - GY0) / (n + 0.5)
    pl = 5.0 if plan["pulse"] in ("left", "both") else 0.0      # pulse width (0.2 s) + gap
    x_start = GX0 + 8.0 + (pl + 2.0 if pl else 0.0)
    lw_pt = plan["trace_mm"] / 25.4 * 72
    segs = []
    sig = rec["signal"]
    for i, row in enumerate(rows):
        yb = GY0 + step * (i + 0.75)
        if plan["pulse"] in ("left", "both"):
            xs = [x_start - 7.0, x_start - 6.0, x_start - 6.0, x_start - 1.0, x_start - 1.0, x_start]
            ys = [yb, yb, yb - GAIN, yb - GAIN, yb, yb]
            ax.plot(xs, ys, color=plan["trace_color"], lw=lw_pt, zorder=3, solid_joinstyle="miter")
        ncol = len(row)
        for c, name in enumerate(row):
            rhythm = name.startswith("R:")
            lead = name[2:] if rhythm else name
            t0, t1 = (0.0, 10.0) if rhythm else (10.0 * c / ncol, 10.0 * (c + 1) / ncol)
            a, b = int(round(t0 * FS)), int(round(t1 * FS))
            v = sig[LEADS.index(lead), a:b]
            t = np.arange(a, b) / FS
            x = x_start + t * SPEED
            y = yb - v * GAIN
            ax.plot(x, y, color=plan["trace_color"], lw=lw_pt, zorder=3, solid_joinstyle="round")
            if plan["labels"]:
                ax.text(x[0] + 1.0, yb - 6.5, lead, fontsize=7, family="DejaVu Sans", weight="bold", zorder=4)
            if plan["separators"] and c > 0:
                ax.plot([x[0], x[0]], [yb - 3, yb + 3], color=plan["trace_color"], lw=lw_pt, zorder=3)
            segs.append(dict(row=i, lead=lead, rhythm=rhythm, t0=t0, t1=t1, sample0=a, sample1=b,
                             x0_mm=float(x[0]), x1_mm=float(x[-1]), baseline_mm=float(yb)))
        if plan["pulse"] in ("right", "both"):
            xe = x_start + 10.0 * SPEED + 1.0
            xs = [xe, xe + 1.0, xe + 1.0, xe + 6.0, xe + 6.0, xe + 7.0]
            ys = [yb, yb, yb - GAIN, yb - GAIN, yb, yb]
            ax.plot(xs, ys, color=plan["trace_color"], lw=lw_pt, zorder=3, solid_joinstyle="miter")
    # header, print check, footer
    ax.text(GX0 + 16, 12.0, f"{kit}  page {p:04d}", fontsize=9, family="DejaVu Sans", weight="bold")
    ax.text(GX0 + 16, 17.0, "25 mm/s   10 mm/mV   print at 100% (no fit to page)", fontsize=7, family="DejaVu Sans")
    ax.plot([200, 250], [14, 14], color="black", lw=1.0)
    for x in (200, 250):
        ax.plot([x, x], [12.5, 15.5], color="black", lw=1.0)
    ax.text(225, 12.0, "50 mm", fontsize=6, ha="center", family="DejaVu Sans")
    ax.text(PAGE_W / 2, 197.0, f"PTB-XL ecg_id {int(float(rec['ecg_id']))} (CC-BY 4.0)", fontsize=6,
            ha="center", family="DejaVu Sans", color="#555555")
    # markers: ids 4p + (0 top-left, 1 top-right, 2 bottom-right, 3 bottom-left)
    d = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_1000)
    pos = [(MARKER_OFF, MARKER_OFF), (PAGE_W - MARKER_OFF - MARKER_MM, MARKER_OFF),
           (PAGE_W - MARKER_OFF - MARKER_MM, PAGE_H - MARKER_OFF - MARKER_MM),
           (MARKER_OFF, PAGE_H - MARKER_OFF - MARKER_MM)]
    markers = []
    for k, (mx, my) in enumerate(pos):
        mid = 4 * p + k
        img = cv2.aruco.generateImageMarker(d, mid, 6)           # 6x6 cells (4x4 + border)
        cell = MARKER_MM / 6
        for r_ in range(6):
            for c_ in range(6):
                if img[r_, c_] == 0:
                    ax.add_patch(__import__("matplotlib.patches", fromlist=["Rectangle"]).Rectangle(
                        (mx + c_ * cell, my + r_ * cell), cell, cell, color="black", lw=0, zorder=5))
        markers.append(dict(id=mid, corners_mm=[[mx, my], [mx + MARKER_MM, my],
                                                [mx + MARKER_MM, my + MARKER_MM], [mx, my + MARKER_MM]]))
    return dict(kit=kit, page=p, ecg_id=int(float(rec["ecg_id"])), patient_id=int(float(rec["patient_id"])),
                strat_fold=int(rec["strat_fold"]), page_mm=[PAGE_W, PAGE_H], speed_mm_s=SPEED,
                gain_mm_mv=GAIN, fs=FS, plan=plan, segments=segs, markers=markers,
                grid_mm=[GX0, GY0, GX1, GY1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--pages", type=int, default=40)
    ap.add_argument("--folds", default="1-9", help="PTB-XL strat folds, e.g. 1-9 (training) or 10 (test)")
    ap.add_argument("--first-page", type=int, default=0, help="page numbers (and marker ids) start here")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--kit", default="ecgdig-kit")
    ap.add_argument("--db", type=Path, default=None, help="ptbxl_database.csv (downloaded when missing)")
    a = ap.parse_args()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    lo, _, hi = a.folds.partition("-")
    folds = set(range(int(lo), int(hi or lo) + 1))
    if 4 * (a.first_page + a.pages) > 1000:
        raise SystemExit("marker ids run out after page 249")
    (a.out / "gt").mkdir(parents=True, exist_ok=True)
    cache = a.out / ".records"
    cache.mkdir(exist_ok=True)
    db = a.db or (a.out / "ptbxl_database.csv")
    if not db.exists():
        db.write_bytes(_get(BASE + "ptbxl_database.csv"))
    recs = choose_records(db, folds, a.pages, a.seed, cache)
    rng = random.Random(a.seed + 1)
    man = []
    with PdfPages(a.out / "kit.pdf") as pdf:
        for k, rec in enumerate(recs):
            p = a.first_page + k
            fig = plt.figure(figsize=(PAGE_W / 25.4, PAGE_H / 25.4))
            ax = fig.add_axes([0, 0, 1, 1])
            ax.set_xlim(0, PAGE_W); ax.set_ylim(PAGE_H, 0); ax.axis("off")
            plan = page_plan(p, rng)
            gt = draw_page(ax, p, rec, plan, a.kit)
            pdf.savefig(fig)
            plt.close(fig)
            (a.out / "gt" / f"page_{p:04d}.json").write_text(json.dumps(gt, indent=1))
            np.savez_compressed(a.out / "gt" / f"page_{p:04d}.npz", signals=rec["signal"], leads=np.array(LEADS))
            man.append(dict(page=p, ecg_id=gt["ecg_id"], layout=plan["layout"], grid=plan["grid"],
                            pulse=plan["pulse"], fold=gt["strat_fold"]))
            print(f"page {p:04d}: ecg {gt['ecg_id']} {plan['layout']} {plan['grid']} pulse {plan['pulse']}", flush=True)
    with open(a.out / "manifest.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(man[0]))
        w.writeheader(); w.writerows(man)
    print(f"{len(man)} pages -> {a.out / 'kit.pdf'}")


if __name__ == "__main__":
    main()
