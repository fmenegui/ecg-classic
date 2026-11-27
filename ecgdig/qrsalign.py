"""Beat-by-beat time alignment of leads recorded at the same time (off by default).

On blurred or low-resolution pages the row network places each QRS 20-40 ms early or late,
differently from beat to beat and from lead to lead (experiments log, Nov 22: PM ECG 14, blur 21).
Leads drawn over the same time window (one column of a 3x4 page, the rhythm strip under it) were
recorded together, so their QRS complexes happen at the same instants. Here each lead's QRS times
are found from its own steepest slopes, matched across the leads of the window, and each lead is
warped (piecewise linearly, only near the beats) onto the median beat time of the leads that agree.
Clean pages, where the leads already agree to a few ms, are left unchanged."""
from __future__ import annotations

import numpy as np

MIN_SHIFT_MS = 4.0     # offsets below this are left alone (detection noise on clean pages)
MAX_SHIFT_MS = 60.0    # a beat further off than this is not the same beat
MIN_LEADS = 3          # consensus needs at least this many leads with the beat


def _qrs_times(x: np.ndarray, fs: float) -> np.ndarray:
    """Sample indices of the steepest-slope peak of each QRS (NaN-free stretch of one lead)."""
    if x.size < int(0.6 * fs):
        return np.zeros(0, int)
    d = np.abs(np.diff(x, prepend=x[0]))
    w = max(1, int(0.02 * fs))
    e = np.convolve(d, np.ones(w) / w, mode="same")
    if not np.isfinite(e).all() or e.max() <= 0:
        return np.zeros(0, int)
    thr = 0.35 * np.percentile(e, 99.5)
    gap = int(0.25 * fs)
    out = []
    i = 0
    while i < e.size:
        if e[i] > thr:
            j = min(e.size, i + gap)
            k = i + int(np.argmax(e[i:j]))
            out.append(k)
            i = k + gap
        else:
            i += 1
    return np.asarray(out, int)


def _windows(sig: np.ndarray):
    """Finite stretch (start, end) of each lead."""
    out = []
    for s in sig:
        f = np.isfinite(s)
        if f.sum() < 10:
            out.append(None)
        else:
            idx = np.nonzero(f)[0]
            out.append((int(idx[0]), int(idx[-1]) + 1))
    return out


def align(sig: np.ndarray, fs: float) -> tuple[np.ndarray, dict]:
    """sig (L, n): returns the aligned copy and a small report."""
    sig = np.array(sig, float, copy=True)
    W = _windows(sig)
    L = sig.shape[0]
    beats = [None] * L
    for li in range(L):
        if W[li] is None:
            continue
        a, b = W[li]
        x = sig[li, a:b]
        if not np.isfinite(x).all():
            x = np.where(np.isfinite(x), x, np.nanmedian(x))
        beats[li] = _qrs_times(x, fs) + a
    tol = int(MAX_SHIFT_MS / 1000 * fs)
    min_s = MIN_SHIFT_MS / 1000 * fs
    moved, beats_used = 0, 0
    shifts = {li: [] for li in range(L)}
    # every detected beat of every lead is a candidate instant; cluster them
    allb = sorted((int(t), li) for li in range(L) if beats[li] is not None for t in beats[li])
    used = set()
    for t, li in allb:
        if (t, li) in used:
            continue
        group = {}
        for t2, l2 in allb:
            if abs(t2 - t) <= tol and l2 not in group and (t2, l2) not in used:
                group[l2] = t2
        # leads whose window covers t but missed the beat do not vote
        if len(group) < MIN_LEADS:
            continue
        for l2, t2 in group.items():
            used.add((t2, l2))
        med = float(np.median(list(group.values())))
        beats_used += 1
        for l2, t2 in group.items():
            shifts[l2].append((t2, med - t2))
    for li, sh in shifts.items():
        sh = [(t, d) for t, d in sh if abs(d) >= min_s]
        if not sh or W[li] is None:
            continue
        a, b = W[li]
        n = np.arange(a, b, dtype=float)
        # time map: output sample n takes the input at n - d(n); d is the beat offset near each beat
        # (within +-120 ms of it) and 0 elsewhere, linear in between
        half = int(0.12 * fs)
        knots_t, knots_d = [a, b - 1], [0.0, 0.0]
        for t, d in sh:
            knots_t += [t - half, t + d, t + half]
            knots_d += [0.0, d, 0.0]
        o = np.argsort(knots_t)
        kt, kd = np.asarray(knots_t, float)[o], np.asarray(knots_d, float)[o]
        dn = np.interp(n, kt, kd)
        src = np.clip(n - dn, a, b - 1)
        x = sig[li, a:b]
        f = np.isfinite(x)
        if f.sum() < 2:
            continue
        sig[li, a:b] = np.interp(src, n[f], x[f])
        sig[li, a:b][~f] = np.nan
        moved += len(sh)
    return sig, dict(beats=beats_used, moved=moved)
