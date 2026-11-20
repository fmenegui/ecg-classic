"""Inputs for a Kaggle top solution: our evaluation images as a Kaggle-style test set.

    python benchmarks/topsol/prepare.py --out DIR [--pm-root R --pm-list SEL.csv] [--kaggle-dir D --kaggle-ids IDS]

Writes DIR/test/<key>.png and DIR/test.csv (id, lead, fs, number_of_rows), the format the
competition's test set has, so the solutions' submission notebooks run unchanged. Keys:
  PM-ECG-ID  "<scenario>__<file stem>"   (fs 500 Hz: lead II 10 s, the other leads 2.5 s)
  Kaggle     "<record>-<type>"           (fs of the record, from its CSV)
The solutions assume the Kaggle sheet (3x4 plus a lead II rhythm strip, 10 s); every image gets
the same 12-lead request, as in the competition.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2

LEADS = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]


def rows_for(key, fs):
    n = int(round(fs * 2.5))
    return [(key, L, fs, 4 * n if L == "II" else n) for L in LEADS]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--pm-root", type=Path)
    ap.add_argument("--pm-list", type=Path)
    ap.add_argument("--kaggle-dir", type=Path)
    ap.add_argument("--kaggle-ids", default="")
    a = ap.parse_args()
    (a.out / "test").mkdir(parents=True, exist_ok=True)
    rows = []
    if a.pm_root and a.pm_list:
        sel = [r[0] for r in csv.reader(open(a.pm_list))][1:]
        for rel in sel:
            p = a.pm_root / rel
            img = cv2.imread(str(p))
            if img is None:
                continue
            key = f"{Path(rel).parent.name}__{Path(rel).stem}"
            cv2.imwrite(str(a.out / "test" / f"{key}.png"), img)
            rows += rows_for(key, 500)
    if a.kaggle_dir:
        ids = [i for i in a.kaggle_ids.split(",") if i]
        for rid in ids:
            d = a.kaggle_dir / rid                      # kaggle_zip.py layout: <dir>/<record>/<record>-<type>.png
            if not d.exists():
                d = a.kaggle_dir / "train" / rid
            csv_p = d / f"{rid}.csv"
            if not csv_p.exists():
                continue
            n_rows = sum(1 for _ in open(csv_p)) - 1          # 10 s at the record's fs
            fs = n_rows / 10.0
            for png in sorted(d.glob(f"{rid}-*.png")):
                key = png.stem
                (a.out / "test" / f"{key}.png").write_bytes(png.read_bytes())
                rows += rows_for(key, fs)
    with open(a.out / "test.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "lead", "fs", "number_of_rows"])
        w.writerows(rows)
    print(f"{len(rows) // 12} images in {a.out}", flush=True)


if __name__ == "__main__":
    main()
