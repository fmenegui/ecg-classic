"""Per-lead quality score, without ground truth.

The digitizer always returns a signal; this score says how much it can be trusted. It only
uses what a real ECG must have, nothing of the clinical content:

  coverage    fraction of the samples of the printed stretch that were read
  beats       leads printed at the same instant are simultaneous: the QRS of one must fall
              at the page QRS instants (median over the other leads); a trace that followed
              the grid, text or the neighbouring row has its energy elsewhere
  repetition  beats of one stretch look alike (median correlation with the median beat);
              noise and swapped traces do not repeat
  jumps       steps > 0.5 mV in 2 ms in more than 5% of the samples (tracker switching traces)
  amplitude   peak to peak outside 0.05–8 mV
  limb        Einthoven/Goldberger residual BEFORE fusion, where 3 limb leads are
              simultaneous (fusion zeroes the residual, so it is measured before)

The score goes from 0 to 1: coverage x repetition (per lead) x layout confidence (page,
applied in digitize). The other parts stay in the report for diagnosis. It is not a
calibrated probability.

Measured against ground truth (PM-ECG-ID, 60 ECGs, 15/11/2026): AUC 0.75 for separating leads
with PCC < 0.5; layout confidence alone gives 0.76. Too little to refuse safely (refusing the
worst 20% raises the SNR of the accepted leads from 8.7 to 9.8 dB and still lets 18% of bad
leads through): most of the garbage comes from a swapped layout, where each trace is a
well-formed ECG with the wrong name. `Config(refuse_below=...)` exists but is off.
"""
from __future__ import annotations

import numpy as np

LIMB = ("I", "II", "III", "aVR", "aVL", "aVF")


def _slope_energy(v: np.ndarray, fs: int) -> np.ndarray:
    """Derivative energy over a ~80 ms window (emphasizes the QRS); NaN where there is no signal."""
    d = np.diff(v, prepend=np.nan)
    e = d * d
    k = max(3, int(0.08 * fs) | 1)
    ok = np.isfinite(e)
    num = np.convolve(np.where(ok, e, 0.0), np.ones(k), mode="same")
    den = np.convolve(ok.astype(float), np.ones(k), mode="same")
    out = np.where(den >= 0.5 * k, num / np.maximum(den, 1.0), np.nan)
    return out


def _page_beats(sig: np.ndarray, fs: int):
    """Page QRS instants: peaks of the median, over the leads present at each instant, of the
    per-lead normalized slope energy. Returns (indices, profile)."""
    n = sig.shape[1]
    E = np.full(sig.shape, np.nan)
    for i, v in enumerate(sig):
        if np.isfinite(v).sum() < 0.3 * fs:
            continue
        e = _slope_energy(v.astype(np.float64), fs)
        s = np.nanpercentile(e, 98) if np.isfinite(e).any() else 0.0
        if s > 0:
            E[i] = np.clip(e / s, 0, 1)
    cnt = np.isfinite(E).sum(0)
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        prof = np.where(cnt >= 2, np.nanmedian(E, axis=0), np.where(cnt == 1, np.nanmax(E, axis=0), 0.0))
    prof = np.nan_to_num(prof)
    # peaks >= 0.25 s apart (up to 240 bpm) and above 30% of the largest
    thr = 0.3 * prof.max() if prof.max() > 0 else np.inf
    gap = int(0.25 * fs)
    order = np.argsort(-prof)
    taken = np.zeros(n, bool)
    beats = []
    for j in order:
        if prof[j] < thr:
            break
        if not taken[j]:
            beats.append(int(j))
            taken[max(0, j - gap):j + gap + 1] = True
    return np.array(sorted(beats), int), prof


def _spans(ok: np.ndarray, min_len: int):
    d = np.diff(np.r_[0, ok.astype(np.int8), 0])
    st, en = np.nonzero(d == 1)[0], np.nonzero(d == -1)[0]
    return [(a, b) for a, b in zip(st, en) if b - a >= min_len]


def lead_quality(v: np.ndarray, fs: int, beats: np.ndarray, others_prof: np.ndarray | None = None) -> dict:
    """Score parts of one lead (v in mV, NaN outside the printed stretch)."""
    ok = np.isfinite(v)
    out = dict(coverage=0.0, beat_sync=np.nan, repeat=np.nan, jumps=0.0, p2p=0.0, score=0.0, flags=[])
    if ok.sum() < 0.3 * fs:
        out["flags"].append("ausente")
        return out
    idx = np.nonzero(ok)[0]
    a, b = int(idx[0]), int(idx[-1]) + 1
    out["coverage"] = float(ok[a:b].mean())
    x = v.astype(np.float64)
    seg = x[a:b]
    out["p2p"] = float(np.nanpercentile(seg, 99.5) - np.nanpercentile(seg, 0.5))
    d = np.abs(np.diff(seg))
    d = d[np.isfinite(d)]
    out["jumps"] = float((d > 0.5 * (500.0 / fs)).mean()) if d.size else 0.0

    # beats: slope energy near the page QRS vs away from them
    e = _slope_energy(x, fs)
    w = int(0.07 * fs)
    near = np.zeros(len(x), bool)
    bs = [int(t) for t in beats if a - w <= t < b + w]
    for t in bs:
        near[max(0, t - w):t + w + 1] = True
    fin = np.isfinite(e)
    if bs and (fin & near).sum() > 5 and (fin & ~near).sum() > 5:
        tot = float(e[fin].sum())
        frac_t = float((fin & near).sum()) / float(fin.sum())
        frac_e = float(e[fin & near].sum()) / tot if tot > 0 else 0.0
        # 1 = all energy in the QRS; 0 = spread like time (trace unrelated to them)
        out["beat_sync"] = float(np.clip((frac_e - frac_t) / max(1e-6, 1.0 - frac_t), 0.0, 1.0))
    # repetition: correlation of each beat with the median beat of the stretch
    h = int(0.12 * fs)
    cuts = [x[t - h:t + h] for t in bs if t - h >= a and t + h <= b]
    cuts = [c - np.nanmedian(c) for c in cuts if np.isfinite(c).mean() > 0.9]
    if len(cuts) >= 2:
        M = np.vstack([np.nan_to_num(c) for c in cuts])
        med = np.median(M, axis=0)
        cs = []
        for c in M:
            if c.std() > 1e-6 and med.std() > 1e-6:
                cs.append(float(np.corrcoef(c, med)[0, 1]))
        if cs:
            out["repeat"] = float(np.median(cs))
    return out


def _combine(q: dict, limb_rel: float | None) -> float:
    """Lead score. Only the parts that ranked leads well against PM-ECG-ID ground truth
    (60 ECGs, 660 read leads, "bad" = PCC < 0.5) enter: coverage and beat repetition
    (AUC 0.72), without saturating (saturating at 1 ties almost everything and hurts ranking).
    QRS energy (AUC 0.59), steps (0.55) and the limb residual (0.74, but it only exists for
    half the leads and mixing it in dropped the whole to 0.63) stay in the report only."""
    s = float(np.clip((q["coverage"] - 0.5) / 0.4, 0.0, 1.0))     # < 50% covered scores 0; >= 90% scores 1
    rep = q["repeat"]
    s *= float(np.clip(rep, 0.05, 1.0)) if np.isfinite(rep) else 0.8   # fewer than 2 beats in the stretch: neutral
    if q["p2p"] < 0.05 or q["p2p"] > 8.0:
        s *= 0.3
    return float(s)


def assess(sig: np.ndarray, leads: list, fs: int, prefusion: np.ndarray | None = None) -> dict:
    """sig: (L, N) in mV after fusion; prefusion: the same BEFORE Einthoven fusion
    (for the limb residual). Returns {"leads": {name: parts + score}, "global": float}."""
    sig = np.asarray(sig, float)
    beats, _ = _page_beats(sig, fs)
    limb_rel = {}
    if prefusion is not None and all(l in leads for l in LIMB):
        P = np.asarray(prefusion, float)
        ix = {l: leads.index(l) for l in LIMB}
        for trio, coef in ((("I", "II", "III"), (1, -1, 1)), (("aVR", "aVL", "aVF"), (1, 1, 1))):
            X = np.stack([P[ix[l]] for l in trio])
            m = np.isfinite(X).all(0)
            if m.sum() >= 0.2 * fs:
                r = sum(c * X[k, m] for k, c in enumerate(coef))
                den = float(np.sqrt(np.mean(X[:, m] ** 2)))
                rel = float(np.sqrt(np.mean(r ** 2)) / max(den, 1e-6))
                for l in trio:
                    limb_rel[l] = rel
    out = {}
    for i, l in enumerate(leads):
        q = lead_quality(sig[i], fs, beats)
        q["limb_residual"] = limb_rel.get(l)
        q["score"] = _combine(q, limb_rel.get(l)) if "ausente" not in q["flags"] else 0.0
        if q["score"] < 0.4 and "ausente" not in q["flags"]:
            q["flags"].append("suspeita")
        out[l] = q
    present = [q["score"] for q in out.values() if "ausente" not in q["flags"]]
    return dict(leads=out, beats=len(beats), score=float(np.mean(present)) if present else 0.0)
