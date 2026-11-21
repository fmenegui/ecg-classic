"""DIAGNOSTIC ONLY (Nov 21, 2026): row-network strips from PM-ECG-ID pages, to measure how much
training on the benchmark's own kind of capture would give. PM-ECG-ID is never used to train a
model we report or release; the model trained on these strips is tested only on the other half of
the ECGs (Image ID parity: train even, test odd) and never becomes a default.

    python hybrid/pmstrips_build.py --root PM/visual_data --gt PM/final_data/data \
        --meta PM/final_data/metadata.csv --list sel.csv --out DIR [--workers N]

Same labelling as hybrid/realstrips.py (the classical reader places the TRUE signal; a segment is
kept when reading and truth correlate >= 0.9), except where the truth sits in time: PM-ECG-ID gives
each lead's printed cut (n, 12) and the rhythm strips (5000, k), not a 10 s timeline, so each cut
is slid along the reader's segment to its best correlation first.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
LEADS = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
MIN_CORR = 0.9


def _slide(read, g, min_frac=0.8):
    """Best offset s (truth sample i drawn at reading sample i + s) and its correlation."""
    g = g[np.isfinite(g)]
    n, m = len(read), len(g)
    if m < 100:
        return -2.0, 0
    need = int(min_frac * min(n, m))
    best = (-2.0, 0)
    for s in range(-(m - need), n - need + 1, 2):
        a0, a1 = max(0, s), min(n, s + m)
        a = read[a0:a1]; b = g[a0 - s:a1 - s]
        ok = np.isfinite(a)
        if ok.sum() < need or a[ok].std() < 1e-6 or b[ok].std() < 1e-6:
            continue
        c = float(np.corrcoef(a[ok], b[ok])[0, 1])
        if c > best[0]:
            best = (c, s)
    return best


def page_labels(img_bgr, cuts, rhythms):
    import realstrips as RS
    from ecgdig import Config, digitize
    r = digitize(img_bgr, Config(), return_debug=True)
    rect = r["rectified"]
    segs, corrs = [], []
    for j in r["jobs"]:
        name = j["lead"].lstrip("-")
        if name not in LEADS:
            continue
        sign = -1.0 if j["lead"].startswith("-") else 1.0
        xn, yn = np.asarray(j["xn"], float), np.asarray(j["yn"], float)
        k = float(j["gain"]) * RS.R
        read = (float(j["base"]) - yn) / k
        cands = [cuts[:, LEADS.index(name)]]
        if len(read) > 0.7 * 5000:                       # a rhythm row: any of the rhythm strips
            cands += [rhythms[:, i] for i in range(rhythms.shape[1])]
        best = (-2.0, 0, None)
        for g in cands:
            c, s = _slide(read, sign * np.asarray(g, float))
            if c > best[0]:
                best = (c, s, sign * np.asarray(g, float))
        c, s, g = best
        corrs.append(c)
        if c < MIN_CORR or g is None:
            continue
        g = g[np.isfinite(g)]
        gt = np.full(len(read), np.nan)
        a0, a1 = max(0, s), min(len(read), s + len(g))
        gt[a0:a1] = g[a0 - s:a1 - s]
        c2, lag = RS._best_lag(read, gt)
        lag, sc = RS._refine(read, gt, lag)
        idx = np.arange(len(read), dtype=float) - lag
        okx = (idx >= 0) & (idx <= len(xn) - 1) & np.isfinite(gt)
        if okx.sum() < 100:
            continue
        x = np.interp(idx[okx], np.arange(len(xn)), xn)
        yr = np.interp(idx[okx], np.arange(len(yn)), np.where(np.isfinite(yn), yn, np.nan))
        mm = np.isfinite(yr)
        off = np.nanmedian((yr + sc * gt[okx] * k)[mm])
        y = off - sc * gt[okx] * k
        segs.append(dict(lead=j["lead"], row=int(j["row"]), base=float(j["base"]), x=x, y=y, corr=c, lag=lag))
    info = dict(n_jobs=len(corrs), kept=len(segs), corr_median=float(np.median(corrs)) if corrs else 0.0,
                layout=r.get("layout"))
    return rect, segs, info


def _work(args):
    path, cuts, rhythms, seed = args
    import cv2
    import realstrips as RS
    try:
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        rect, segs, info = page_labels(img, cuts, rhythms)
        if info["n_jobs"] == 0 or info["kept"] < 0.5 * info["n_jobs"]:
            return path, None, info
        X, Y = RS.strips(rect, segs, np.random.default_rng(seed), per_row=2)
        if not len(X):
            return path, None, info
        jp = [cv2.imencode(".jpg", x, [cv2.IMWRITE_JPEG_QUALITY, 92])[1].tobytes() for x in X]
        S = Y.shape[1]
        rows = np.arange(S, dtype=np.float32)[:, None]
        w = Y.sum(1)
        h = np.where(w > 0.5, (Y * rows[None]).sum(1) / np.maximum(w, 1e-6), np.nan).astype(np.float16)
        return path, (jp, h), info
    except Exception as e:  # noqa: BLE001
        return path, None, dict(error=f"{type(e).__name__}: {e}"[:200])


def main():
    import pandas as pd
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--gt", type=Path, required=True)
    ap.add_argument("--meta", type=Path, required=True)
    ap.add_argument("--list", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--workers", type=int, default=max(2, (os.cpu_count() or 4) - 1))
    ap.add_argument("--chunk", type=int, default=400)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    m = pd.read_csv(a.meta)
    sel = set(pd.read_csv(a.list).iloc[:, 0])
    m = m[m["Image relative path"].isin(sel)]
    assert (m["Image ID"].astype(int) % 2 == 0).all(), "diagnostic strips come from even Image IDs only"
    L = np.load(a.gt / "leads.npz"); RH = np.load(a.gt / "rhythms.npz")
    rng = np.random.default_rng(0)
    jobs = [(str(a.root / p), L[str(e)], RH[str(e)], int(rng.integers(1 << 30)))
            for p, e in zip(m["Image relative path"], m["ECG ID"]) if (a.root / p).exists()]
    print(f"pm strips: {len(jobs)} pages (even Image IDs)", flush=True)
    buf_x, buf_h, k, done, kept = [], [], 0, 0, 0

    def flush():
        nonlocal buf_x, buf_h, k
        if not buf_x:
            return
        tmp = a.out / f".chunk_{k:04d}.npz"
        np.savez(tmp, jpg=np.array(buf_x, dtype=object), h=np.stack(buf_h))
        os.replace(tmp, a.out / f"chunk_{k:04d}.npz")
        print(f"pm strips: chunk {k} ({len(buf_x)} strips), pages {done}/{len(jobs)}, kept {kept}", flush=True)
        buf_x, buf_h, k = [], [], k + 1

    t0 = time.time()
    with ProcessPoolExecutor(a.workers) as ex:
        futs = [ex.submit(_work, j) for j in jobs]
        for f in as_completed(futs):
            path, got, info = f.result()
            done += 1
            if got is not None:
                kept += 1
                buf_x += got[0]; buf_h += list(got[1])
                if len(buf_x) >= a.chunk:
                    flush()
            if done % 50 == 0:
                print(f"pm strips: {done} pages in {time.time() - t0:.0f} s, kept {kept}", flush=True)
    flush()
    (a.out / "DONE").write_text(str(done))
    print(f"pm strips: done, {kept} of {done} pages kept", flush=True)


if __name__ == "__main__":
    main()
