#!/usr/bin/env python3
"""Error decomposition: how much SNR each kind of error costs.

For each image (signals saved in sig/*.npz by the evaluation), apply cumulative "oracle"
corrections, each using the ground truth only for its own kind of error, and measure the SNR
after each step (same metric as the evaluation: sum over the 12 leads):

  0 metric      : competition alignment (±0.2 s and mean per lead)
  1a labels     : each ground-truth lead may use the predicted channel that explains it best
                  (wrong layout/name: swapped lead or row assigned to the wrong lead)
  1 coverage    : holes inside predicted leads get the ground truth (a whole missing lead
                  does not count here: it is a layout error, it stays in the remainder)
  2 origin      : shift common to all leads, up to ±1.5 s
  3 drift       : per lead, time scale (±3 %) + shift
  3b local drift: 0.5 s windows with their own shift up to ±80 ms (non-linear horizontal
                  distortion: crease, scanner edge, curved paper)
  4 gain        : per lead, optimal amplitude factor
  5 baseline    : slow difference (< 0.7 Hz) removed
  remainder     : shape (tracing, clipped peaks, merges)

The gain of each step is the loss attributable to that error (in the order above).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy.ndimage import uniform_filter1d

LEADS = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
STEPS = ["metric", "labels", "coverage", "origin", "drift", "local drift", "gain", "baseline"]


def _snr(G, P):
    sp = npw = 0.0
    for g, p in zip(G, P):
        m = np.isfinite(g)
        if m.sum() < 10:
            continue
        gg = g[m]
        pp = np.nan_to_num(p[m], nan=0.0)
        sp += float((gg ** 2).sum())
        npw += float(((gg - pp) ** 2).sum())
    return 10 * np.log10(sp / max(npw, 1e-12)) if sp > 0 else np.nan


def _shift(p, s):
    out = np.full_like(p, np.nan)
    if s >= 0:
        out[s:] = p[:len(p) - s] if s else p
    else:
        out[:s] = p[-s:]
    return out


def _best_lag(g, p, maxs, offset=True):
    m = np.isfinite(g)
    best = (np.inf, 0)
    for s in range(-maxs, maxs + 1):
        q = _shift(p, s)[m]
        gg = g[m]
        k = np.isfinite(q)
        if k.sum() < 0.6 * m.sum():
            continue
        q = np.where(k, q, 0.0)
        if offset:
            q = q - q[k].mean() + gg[k].mean()
        e = float(((gg - q) ** 2).mean())
        if e < best[0]:
            best = (e, s)
    return best[1]


def _align1(g, p, fs):
    m = np.isfinite(g)
    s = _best_lag(g, p, int(0.2 * fs))
    q = _shift(p, s)
    k = m & np.isfinite(q)
    if k.any():
        q = q - np.nanmean(q[k]) + g[k].mean()
    return q


def _metric_align(G, P, fs):
    """Competition alignment: per lead, lag ±0.2 s and mean."""
    ms = int(0.2 * fs)
    Q = np.full_like(P, np.nan)
    for i in range(12):
        g = G[i]
        m = np.isfinite(g)
        if m.sum() < 10:
            continue
        s = _best_lag(g, P[i], ms)
        q = _shift(P[i], s)
        k = m & np.isfinite(q)
        if k.any():
            q = q - np.nanmean(q[k]) + g[k].mean()
        Q[i] = q
    return Q


def decompose(G, P, fs, return_signals=False):
    G = G.astype(np.float64)
    P = P.astype(np.float64)
    out = {}
    Q = _metric_align(G, P, fs)
    out["metric"] = _snr(G, Q)
    # 1a labels: best predicted channel for each ground-truth lead
    Pl = P.copy()
    base = [_snr(G[i:i + 1], Q[i:i + 1]) for i in range(12)]
    for i in range(12):
        if not np.isfinite(G[i]).sum() > 10:
            continue
        best = (base[i] if np.isfinite(base[i]) else -np.inf, i)
        for j in range(12):
            if j == i or np.isfinite(P[j][np.isfinite(G[i])]).mean() < 0.5:
                continue
            v = _snr(G[i:i + 1], _align1(G[i], P[j], fs)[None])
            if v > best[0] + 0.5:
                best = (v, j)
        Pl[i] = P[best[1]]
    out["labels"] = _snr(G, _metric_align(G, Pl, fs))
    P = Pl
    # 1 coverage (only holes in leads with a prediction)
    has = np.array([np.isfinite(P[i][np.isfinite(G[i])]).mean() >= 0.2 if np.isfinite(G[i]).any() else False
                    for i in range(12)])
    P1 = np.where(np.isfinite(P) | ~np.isfinite(G) | ~has[:, None], P, G)
    out["coverage"] = _snr(G, _metric_align(G, P1, fs))
    # 2 common origin (up to ±1.5 s)
    ms = int(1.5 * fs)
    best = (np.inf, 0)
    for s in range(-ms, ms + 1, max(1, int(fs / 100))):
        e = 0.0
        for i in range(12):
            m = np.isfinite(G[i])
            if m.sum() < 10:
                continue
            q = _shift(P1[i], s)[m]
            q = np.where(np.isfinite(q), q, 0.0)
            gg = G[i][m]
            e += float(((gg - gg.mean()) - (q - q.mean())) @ ((gg - gg.mean()) - (q - q.mean())))
        if e < best[0]:
            best = (e, s)
    P2 = np.stack([_shift(P1[i], best[1]) for i in range(12)])
    out["origin"] = _snr(G, _metric_align(G, P2, fs))
    out["origin_s"] = best[1] / fs
    # 3 drift: time scale per lead anchored at the start of the shown stretch
    P3 = np.full_like(P2, np.nan)
    for i in range(12):
        g = G[i]
        m = np.isfinite(g)
        if m.sum() < 10:
            continue
        idx = np.nonzero(m)[0]
        t0 = idx[0]
        best = (np.inf, None)
        x = np.arange(len(g), dtype=float)
        p = P2[i]
        okp = np.isfinite(p)
        if okp.sum() < 10:
            continue
        for a in np.linspace(0.97, 1.03, 25):
            xs = t0 + (x - t0) * a
            q = np.interp(xs, x[okp], p[okp], left=np.nan, right=np.nan)
            s = _best_lag(g, q, int(0.05 * fs))
            q = _shift(q, s)[m]
            k = np.isfinite(q)
            if k.sum() < 0.6 * m.sum():
                continue
            q = np.where(k, q, 0.0)
            e = float(((g[m] - g[m].mean()) - (q - q.mean())) @ ((g[m] - g[m].mean()) - (q - q.mean())))
            if e < best[0]:
                best = (e, (a, s))
        if best[1] is None:
            P3[i] = p
            continue
        a, s = best[1]
        q = np.interp(t0 + (x - t0) * a, x[okp], p[okp], left=np.nan, right=np.nan)
        P3[i] = _shift(q, s)
    out["drift"] = _snr(G, _metric_align(G, P3, fs))
    Q3 = _metric_align(G, P3, fs)
    # 3b local drift: shift per 0.5 s window
    P3b = Q3.copy()
    win, ms = int(0.5 * fs), int(0.08 * fs)
    for i in range(12):
        m = np.isfinite(G[i])
        if m.sum() < win or not np.isfinite(Q3[i]).any():
            continue
        idx = np.nonzero(m)[0]
        for a0 in range(idx[0], idx[-1] + 1, win):
            sl = np.arange(a0, min(a0 + win, idx[-1] + 1))
            g = G[i][sl]
            best = (float(((g - np.nan_to_num(Q3[i][sl], nan=0.0)) ** 2).mean()), 0)  # no shift
            for s_ in range(-ms, ms + 1):
                j = sl - s_
                ok = (j >= 0) & (j < len(G[i]))
                q = np.full(len(sl), np.nan)
                q[ok] = Q3[i][j[ok]]
                if np.isfinite(q).sum() < 0.6 * len(sl):
                    continue
                e = float(((g - np.nan_to_num(q, nan=0.0)) ** 2).mean())  # a hole counts as 0, as in the metric
                if e < best[0]:
                    best = (e, s_)
            j = sl - best[1]
            ok = (j >= 0) & (j < len(G[i]))
            P3b[i][sl[ok]] = Q3[i][j[ok]]
    out["local drift"] = _snr(G, P3b)
    Q3 = P3b
    # 4 gain per lead
    P4 = Q3.copy()
    for i in range(12):
        m = np.isfinite(G[i]) & np.isfinite(Q3[i])
        if m.sum() < 10:
            continue
        g = G[i][m] - G[i][m].mean()
        q = Q3[i][m] - Q3[i][m].mean()
        k = float(g @ q / max(q @ q, 1e-12))
        P4[i][m] = q * np.clip(k, 0.3, 3.0) + G[i][m].mean()
    out["gain"] = _snr(G, P4)
    # 5 baseline: slow difference
    P5 = P4.copy()
    w = int(1.4 * fs)
    for i in range(12):
        m = np.isfinite(G[i]) & np.isfinite(P4[i])
        if m.sum() < w:
            continue
        d = np.where(m, G[i] - P4[i], 0.0)
        cnt = uniform_filter1d(m.astype(float), w)
        low = uniform_filter1d(d, w) / np.maximum(cnt, 1e-6)
        P5[i] = np.where(m, P4[i] + low, P4[i])
    out["baseline"] = _snr(G, P5)
    if return_signals:
        return out, P5
    return out


def main(art: Path):
    res = json.loads((art / "results.json").read_text())["results"]
    meta = {(r["id"], r["type"]): r for r in res}
    rows = []
    for f in sorted((art / "sig").glob("*.npz")):
        rid, typ = f.stem.split("-")
        d = np.load(f)
        G, P, fs = d["gt"].astype(float), d["pred"].astype(float), float(d["fs"])
        dec = decompose(G, P, fs)
        dec.update(id=rid, type=typ, layout=meta.get((rid, typ), {}).get("layout"))
        rows.append(dec)
        dec = {k: (float(v) if isinstance(v, (float, np.floating)) else v) for k, v in dec.items()}
        print(rid, typ, " ".join(f"{k}={v:.2f}" for k, v in dec.items() if isinstance(v, float)), flush=True)
    (art / "decomposition.json").write_text(json.dumps(rows, indent=1, default=float))


if __name__ == "__main__":
    main(Path(sys.argv[1]))
