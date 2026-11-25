"""Where the error of a reading of the probe pages is: cumulative oracle corrections per segment.

    python benchmarks/topsol/decompose_probe.py --probe DIR --pred PRED_DIR [--variants E0]

PRED_DIR holds one <key>.npz per page (leads, signals 12 x 5000 at 500 Hz), as
score_probe.py --save-pred writes them. Every printed segment is aligned as score_probe.py does
(integer shift up to +-0.5 s, median offset), then corrected with the truth, one stage after the
other:

  shift   sub-sample time shift (+-1 sample, linear interpolation)
  scale   time scale (+-2%) about the segment centre
  gain    amplitude scale and offset (least squares)

and compared with two ceilings computed from the truth alone: the signal sampled every 4 ms
(one value per column at 10 px/mm and 25 mm/s) or every 2 ms (20 px/mm), linearly interpolated
back to 500 Hz. What remains after every stage is the shape of the trace. Scores are the
PMcardio-style mean of segment dB and the Kaggle-style 10 log10 of the mean page linear SNR.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "pmcardio"))
import eval_pm as E  # noqa: E402

LEADS = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
STAGES = ["read", "shift", "scale", "gain"]


def _snr(g, p):
    m = np.isfinite(p) & np.isfinite(g)
    if m.sum() < 0.5 * len(g):
        return None
    gg, pp = g[m], p[m]
    e = gg - (pp + np.median(gg - pp))
    return 10 * np.log10(max(float(gg @ gg), 1e-12) / max(float(e @ e), 1e-12))


def _resample(p, t):
    """p sampled at integer positions, read at positions t (linear, NaN outside)."""
    i = np.arange(len(p), dtype=float)
    ok = np.isfinite(p)
    if ok.sum() < 2:
        return np.full(len(t), np.nan)
    out = np.interp(t, i[ok], p[ok], left=np.nan, right=np.nan)
    # keep gaps as gaps
    gap = ~ok
    if gap.any():
        near = np.interp(t, i, gap.astype(float)) > 0.5
        out[near] = np.nan
    return out


def ceiling(g, step):
    """Truth sampled every `step` samples (phase averaged), linear interpolation back."""
    vals = []
    n = len(g)
    i = np.arange(n, dtype=float)
    for ph in range(step):
        idx = np.arange(ph, n, step)
        vals.append(_snr(g, np.interp(i, idx, g[idx])))
    return float(np.mean(vals))


def decompose(g, p):
    """g, p aligned (same length). Returns dict stage -> snr (cumulative corrections)."""
    out = {"read": _snr(g, p)}
    if out["read"] is None:
        return None
    n = len(g)
    t0 = np.arange(n, dtype=float)
    best = (out["read"], 0.0)
    for d in np.arange(-1.0, 1.0001, 0.05):
        s = _snr(g, _resample(p, t0 + d))
        if s is not None and s > best[0]:
            best = (s, d)
    out["shift"], d = best
    c = (n - 1) / 2
    best = (out["shift"], 1.0, d)
    for k in np.arange(0.98, 1.0201, 0.001):     # scale with its own best shift (joint search)
        for d_ in np.arange(-1.5, 1.5001, 0.1):
            s = _snr(g, _resample(p, c + (t0 - c) * k + d_))
            if s is not None and s > best[0]:
                best = (s, k, d_)
    k0, d0 = best[1], best[2]
    for k in np.arange(k0 - 0.001, k0 + 0.00101, 0.00025):
        for d_ in np.arange(d0 - 0.1, d0 + 0.1001, 0.02):
            s = _snr(g, _resample(p, c + (t0 - c) * k + d_))
            if s is not None and s > best[0]:
                best = (s, k, d_)
    out["scale"], k, d = best
    q = _resample(p, c + (t0 - c) * k + d)
    m = np.isfinite(q)
    A = np.stack([q[m], np.ones(m.sum())], 1)
    coef, *_ = np.linalg.lstsq(A, g[m], rcond=None)
    q2 = np.full(n, np.nan); q2[m] = A @ coef
    out["gain"] = max(out["scale"], _snr(g, q2))
    out["ceil10"] = ceiling(g, 2)
    out["ceil20"] = ceiling_half(g)
    return out


def ceiling_half(g):
    """20 px/mm at 25 mm/s = 2 ms per column = the 500 Hz grid itself: sample at half-sample
    offsets (column centres between samples) and interpolate back."""
    n = len(g)
    i = np.arange(n, dtype=float)
    mid = (g[:-1] + g[1:]) / 2   # a column centred between two samples sees their mean
    return _snr(g, np.interp(i, i[:-1] + 0.5, mid))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe", type=Path, required=True)
    ap.add_argument("--pred", type=Path, required=True)
    ap.add_argument("--variants", default="")
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args()
    keys = sorted(p.stem for p in (a.probe / "gt").glob("*.npz"))
    if a.variants:
        keys = [k for k in keys if k.split("_")[0] in a.variants.split(",")]
    rows = []
    for key in keys:
        f = a.pred / f"{key}.npz"
        if not f.exists():
            continue
        d = np.load(f, allow_pickle=True)
        S = np.asarray(d["signals"], float)
        names = [str(x) for x in d["leads"]]
        gt = np.load(a.probe / "gt" / f"{key}.npz")
        sig = gt["signals"].astype(float)
        for s in json.loads(str(gt["segments"])):
            i0, i1 = int(round(s["t0"] * 500)), int(round(s["t1"] * 500))
            g = sig[LEADS.index(s["lead"]), i0:i1]
            F = S[names.index(s["lead"])] if s["lead"] in names else None
            p = E._best_shift(g, F, i0) if F is not None else None
            r = decompose(g, p) if p is not None else None
            rows.append(dict(key=key, lead=s["lead"], rhythm=bool(s["rhythm"]), e=float(g @ g), **(r or {})))
    stages = STAGES + ["ceil10", "ceil20"]
    by = defaultdict(list)
    for r in rows:
        by[r["key"]].append(r)
    print("| stage | segment dB (mean) | Kaggle-style (dB) |")
    print("|---|---|---|")
    summ = {}
    for st in stages:
        seg = [r.get(st, 0.0) if r.get(st) is not None else 0.0 for r in rows]
        lin = []
        for k, rs in by.items():
            sp = sum(r["e"] for r in rs)
            npow = sum(r["e"] / 10 ** ((r.get(st) or 0.0) / 10) for r in rs)
            lin.append(sp / max(npow, 1e-12))
        summ[st] = dict(seg=float(np.mean(seg)), kaggle=float(10 * np.log10(np.mean(lin))))
        print(f"| {st} | {summ[st]['seg']:.2f} | {summ[st]['kaggle']:.2f} |")
    if a.out:
        a.out.write_text(json.dumps(dict(summary=summ, segments=rows)))


if __name__ == "__main__":
    main()
