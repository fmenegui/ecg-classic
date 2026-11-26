"""Synthetic pages to calibrate the choice between the classical and the page network's reading.

    python benchmarks/selector/make_pages.py --bank bank.npy --out DIR --n 600 [--seed 0]

The default choice (digitize._reading_score with a margin) keeps the wrong reading on ~28% of
the PM-ECG-ID pages where exactly one of the two has the printed layout. A better rule cannot be
fitted on PM-ECG-ID (no training on a benchmark), so it is fitted on these pages: PTB-XL signals
from the bank (strat folds 1-9, evaluation records removed), every layout of ecgdig.layouts that
printed ECGs use, rendered by ecgdig.synth, then ONE generic degradation of the kinds a phone,
a scanner or an image pipeline adds (blur, low resolution, faded pen, brightness, contrast gamma,
JPEG, perspective, rotation, padding), at random strength, or none.

Writes DIR/<key>.png and DIR/<key>.npz (sig 12 x N mV, shown 12 x N, layout, degradation).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from ecgdig.synth import render  # noqa: E402

LAYOUTS = ["3x4", "3x4+1", "3x4+3", "6x2", "6x2+1", "12x1", "3x4+1-cabrera", "6x2-cabrera"]
WEIGHTS = [0.12, 0.28, 0.10, 0.16, 0.12, 0.10, 0.06, 0.06]


def degrade(img, rng):
    """One generic degradation (name, image)."""
    kind = rng.choice(["none", "blur", "resolution", "opacity", "brightness", "contrast", "jpeg",
                       "perspective", "rotation", "padding"],
                      p=[0.10, 0.12, 0.12, 0.10, 0.10, 0.10, 0.09, 0.09, 0.09, 0.09])
    H, W = img.shape[:2]
    if kind == "blur":
        k = int(rng.choice([5, 9, 15, 21, 31, 41])) | 1
        img = cv2.GaussianBlur(img, (k, k), 0)
        return f"blur{k}", img
    if kind == "resolution":
        f = float(rng.choice([2, 4, 8, 16]))
        sm = cv2.resize(img, (max(8, int(W / f)), max(8, int(H / f))), interpolation=cv2.INTER_AREA)
        return f"res{int(f)}", sm
    if kind == "opacity":
        a = float(rng.uniform(0.12, 0.6))
        return f"opacity{a:.2f}", np.clip(255 - a * (255.0 - img), 0, 255).astype(np.uint8)
    if kind == "brightness":
        b = float(rng.choice([-160, -120, -80, -40, 40, 80, 120, 160]))
        return f"bright{int(b)}", np.clip(img.astype(np.float32) + b, 0, 255).astype(np.uint8)
    if kind == "contrast":
        g = float(rng.choice([0.125, 0.25, 0.5, 2.0, 4.0, 8.0]))
        return f"gamma{g}", np.clip(255 * (img / 255.0) ** g, 0, 255).astype(np.uint8)
    if kind == "jpeg":
        q = int(rng.choice([2, 5, 10, 20]))
        enc = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, q])[1]
        return f"jpeg{q}", cv2.imdecode(enc, cv2.IMREAD_COLOR)
    if kind == "perspective":
        d = float(rng.uniform(0.03, 0.15)) * min(H, W)
        src = np.float32([[0, 0], [W, 0], [W, H], [0, H]])
        dst = src + rng.uniform(-d, d, (4, 2)).astype(np.float32)
        M = cv2.getPerspectiveTransform(src, dst)
        return "perspective", cv2.warpPerspective(img, M, (W, H), borderValue=(255, 255, 255))
    if kind == "rotation":
        a = float(rng.choice([-30, -15, -5, 5, 15, 30]))
        p = int(0.2 * max(H, W))
        big = cv2.copyMakeBorder(img, p, p, p, p, cv2.BORDER_CONSTANT, value=(255, 255, 255))
        M = cv2.getRotationMatrix2D((big.shape[1] / 2, big.shape[0] / 2), a, 1.0)
        return f"rot{int(a)}", cv2.warpAffine(big, M, (big.shape[1], big.shape[0]), borderValue=(255, 255, 255))
    if kind == "padding":
        p = int(rng.uniform(0.2, 1.0) * max(H, W))
        big = cv2.copyMakeBorder(img, p, p, p, p, cv2.BORDER_CONSTANT, value=(255, 255, 255))
        return "padding", cv2.resize(big, (W, H), interpolation=cv2.INTER_AREA)
    return "none", img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bank", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--n", type=int, default=600)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    bank = np.load(a.bank, mmap_mode="r")
    a.out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(a.seed)
    for p in range(a.n):
        sig = np.nan_to_num(np.asarray(bank[int(rng.integers(0, len(bank)))], np.float64))
        lay = str(rng.choice(LAYOUTS, p=WEIGHTS))
        img, shown, meta = render(sig, rng, degrade="none", layout=lay)
        name, img = degrade(img, rng)
        key = f"s{p:04d}"
        cv2.imwrite(str(a.out / f"{key}.png"), img)
        np.savez_compressed(a.out / f"{key}.npz", sig=sig.astype(np.float32), shown=shown,
                            layout=lay, degradation=name, meta=json.dumps({k: v for k, v in meta.items()
                                                                          if isinstance(v, (int, float, str))}))
    print(a.n, "pages in", a.out)


if __name__ == "__main__":
    main()
