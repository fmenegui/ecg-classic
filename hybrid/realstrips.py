"""Row-network training strips from REAL pages with known signals (e.g. the Kaggle training set).

The rendered and synthetic strips do not look like real scans and photos (the stage oracle and
the LITFL check show the gap), so the row network also learns from real pages. The label is the
TRUE signal, placed with the classical reader's own geometry:

  1. ecgdig reads the page (rectified at 10 px/mm, rows, columns, time axis, gain);
  2. for every segment, the true signal is aligned to the reading: time lag (+-40 samples, by
     correlation) and vertical offset (median), so only where the signal is drawn comes from
     the reader, never its shape;
  3. a segment is kept only when the reading and the truth agree (correlation >= MIN_CORR), i.e.
     the page geometry is right; a page needs most of its segments kept.

Strips are cut around each row exactly as at inference (512 x 512 at 10 px/mm, centred on the
row's baseline with jitter) and labelled with the row's own true trace only. Nothing about the
page format is used: a strip is a window on one row, at a random horizontal position, so the
network cannot learn a number of columns or a page layout.

    python hybrid/realstrips.py IMAGE GT.csv OUT_DIR   # debug picture of the strips
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

LEADS = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
FS = 500
R = 10.0            # rectified px per mm
MIN_CORR = 0.9
MAX_LAG = 40
SUBSAMPLE = [True]   # sub-sample lag and amplitude scale of the label (see _refine)


def load_gt(csv_path, n_out=5000, duration=10.0):
    """Kaggle training CSV (one column per lead, NaN outside the lead's window, any fs) ->
    (12, n_out) at 500 Hz."""
    import pandas as pd
    df = pd.read_csv(csv_path)
    n = len(df)
    t_in = np.arange(n) * duration / n
    t_out = np.arange(n_out) / FS
    G = np.full((12, n_out), np.nan)
    for i, L in enumerate(LEADS):
        if L not in df:
            continue
        v = df[L].to_numpy(np.float64)
        ok = np.isfinite(v)
        if ok.sum() < 10:
            continue
        g = np.interp(t_out, t_in[ok], v[ok])
        lo, hi = t_in[ok][0], t_in[ok][-1]
        g[(t_out < lo) | (t_out > hi)] = np.nan
        G[i] = g
    return G


def _best_lag(read, gt):
    best = (-2.0, 0)
    for lag in range(-MAX_LAG, MAX_LAG + 1):
        a = read[max(0, -lag):len(read) - max(0, lag)]
        b = gt[max(0, lag):len(gt) - max(0, -lag)]
        m = np.isfinite(a) & np.isfinite(b)
        if m.sum() < 100 or a[m].std() < 1e-6 or b[m].std() < 1e-6:
            continue
        c = float(np.corrcoef(a[m], b[m])[0, 1])
        if c > best[0]:
            best = (c, lag)
    return best


def _refine(read, gt, lag):
    """Sub-sample lag and amplitude scale of the truth against the reading, around the integer
    lag. At 500 Hz one sample is 0.5 mm at 25 mm/s (5 px at 10 px/mm): an integer lag left
    labels up to 2.5 px off on every QRS, which taught the network a blurred trace (the row
    network read clean renders worse than the classical reader)."""
    t = np.arange(len(gt), dtype=float)
    mg = np.isfinite(gt)
    best = (np.inf, float(lag), 1.0)
    for lf in np.arange(lag - 1.0, lag + 1.0001, 0.05):
        # reading sample i shows the truth at i + lf
        g_ = np.interp(t + lf, t[mg], gt[mg], left=np.nan, right=np.nan)
        m = np.isfinite(g_) & np.isfinite(read)
        if m.sum() < 100:
            continue
        a, b = g_[m] - g_[m].mean(), read[m] - read[m].mean()
        sc = float(np.clip((a * b).sum() / max((a * a).sum(), 1e-9), 0.8, 1.25))
        v = float(((b - sc * a) ** 2).mean())
        if v < best[0]:
            best = (v, float(lf), sc)
    return best[1], best[2]


def page_labels(img_bgr, G, cfg=None):
    """Read the page and place the true signal of every segment on the rectified image.
    Returns (rectified BGR, segments, info). A segment: dict(lead, row, base, x, y) with x, y the
    true trace in rectified pixel-centre coordinates."""
    from ecgdig import Config, digitize
    r = digitize(img_bgr, cfg or Config(), return_debug=True)
    rect = r["rectified"]
    segs, corrs = [], []
    for j in r["jobs"]:
        if j["lead"].lstrip("-") not in LEADS:
            continue
        li = LEADS.index(j["lead"].lstrip("-"))
        sign = -1.0 if j["lead"].startswith("-") else 1.0
        n0, n1 = int(j["n0"]), int(j["n1"])
        xn, yn = np.asarray(j["xn"], float), np.asarray(j["yn"], float)
        k = float(j["gain"]) * R                                       # px per mV
        read = (float(j["base"]) - yn) / k
        gt = sign * G[li, n0:n1]
        if np.isfinite(gt).sum() < 100:
            continue
        c, lag = _best_lag(read, gt)
        corrs.append(c)
        if c < MIN_CORR:
            continue
        if SUBSAMPLE[0]:
            lag, sc = _refine(read, gt, lag)
        else:
            sc = 1.0
        idx = np.arange(n1 - n0, dtype=float) - lag                   # GT sample i is drawn where the reading's i - lag is
        okx = (idx >= 0) & (idx <= len(xn) - 1) & np.isfinite(gt)
        x = np.interp(idx[okx], np.arange(len(xn)), xn)
        yr = np.interp(idx[okx], np.arange(len(yn)), np.where(np.isfinite(yn), yn, np.nan))
        m = np.isfinite(yr)
        off = np.nanmedian((yr + sc * gt[okx] * k)[m])
        y = off - sc * gt[okx] * k
        segs.append(dict(lead=j["lead"], row=int(j["row"]), base=float(j["base"]), x=x, y=y, corr=c, lag=lag))
    info = dict(n_jobs=len(corrs), kept=len(segs), corr_median=float(np.median(corrs)) if corrs else 0.0,
                layout=r.get("layout"))
    return rect, segs, info


def strips(rect, segs, rng, per_row=2, size=512, jitter=8.0):
    """Training strips: (n, size, size, 3) uint8 BGR and (n, size, size) float32 sparse labels."""
    from rowgen import _sparse_label
    H, W = rect.shape[:2]
    rows = {}
    for s in segs:
        rows.setdefault(s["row"], []).append(s)
    X, Y = [], []
    for row, ss in rows.items():
        base = float(np.median([s["base"] for s in ss]))
        xs = np.concatenate([s["x"] for s in ss]); ys = np.concatenate([s["y"] for s in ss])
        o = np.argsort(xs); xs, ys = xs[o], ys[o]
        lo, hi = xs[0], xs[-1]
        for _ in range(per_row):
            yc = base + rng.uniform(-jitter, jitter)
            y0 = int(round(yc - size / 2))
            x0 = int(rng.uniform(lo - size * 0.3, max(lo - size * 0.3, hi - size * 0.7)))
            crop = np.full((size, size, 3), 255, np.uint8)
            ya, yb = max(0, y0), min(H, y0 + size); xa, xb = max(0, x0), min(W, x0 + size)
            if yb <= ya or xb <= xa:
                continue
            crop[ya - y0:yb - y0, xa - x0:xb - x0] = rect[ya:yb, xa:xb]
            # columns between segments (lead switches) carry no label: gap > 1.5 px between samples
            lab = np.zeros((size, size), np.float32)
            cols = np.arange(size) + x0
            ok = (cols >= lo) & (cols <= hi)
            gaps = np.flatnonzero(np.diff(xs) > 1.5)
            for g in gaps:
                ok &= ~((cols > xs[g]) & (cols < xs[g + 1]))
            yv = np.interp(cols, xs, ys) - y0
            i = np.floor(yv).astype(int); f = (yv - i).astype(np.float32)
            c = np.flatnonzero(ok & (i >= 0) & (i + 1 < size))
            lab[i[c], c] = 1.0 - f[c]
            lab[i[c] + 1, c] = f[c]
            X.append(crop); Y.append(lab)
    if not X:
        return np.zeros((0, size, size, 3), np.uint8), np.zeros((0, size, size), np.float32)
    return np.stack(X), np.stack(Y)


if __name__ == "__main__":
    import cv2
    img = cv2.imread(sys.argv[1]); G = load_gt(sys.argv[2])
    rect, segs, info = page_labels(img, G)
    print(info, [(s["lead"], round(s["corr"], 3), s["lag"]) for s in segs])
    X, Y = strips(rect, segs, np.random.default_rng(0), per_row=1)
    out = Path(sys.argv[3]); out.mkdir(parents=True, exist_ok=True)
    tiles = []
    for x, y in zip(X[:4], Y[:4]):
        t = x.copy(); t[y > 0.3] = (0, 0, 255)
        tiles.append(np.hstack([x, t]))
    if tiles:
        cv2.imwrite(str(out / "realstrips.jpg"), np.vstack(tiles))
