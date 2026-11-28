"""Cross-lead consistency of one reading, with no reference signal (LITFL diagnostics).

Leads printed in the same column of a page are recorded at the same instants, and a full-length
rhythm strip runs under every column. That gives checks that need no truth:
  - closure: I - II + III = 0 (column of the limb leads) and aVR + aVL + aVF = 0 (Goldberger),
    per sample, on the measurements before fusion; `bursts` are 0.2 s windows where the residual
    peaks above 0.3 mV and 40% of the trio's largest value (a lead that lost its trace or jumped to a neighbour);
  - blame: within a burst, the lead whose replacement by the value the other two predict removes
    most of the residual;
  - timing: QRS instants per lead (peaks of the smoothed slope); each lead is compared with the
    other leads of its column and with the rhythm strip under it: median offset (ms) and beats
    missing or extra;
  - rhythm lag: best time shift of the grid II against the rhythm-strip II over the same stretch.
Input: `per_segment` of digitize(..., return_debug=True): lead -> list of full-length arrays (NaN
outside the segment), the first one from the grid, a second II from the rhythm strip.
"""
from __future__ import annotations

import numpy as np

FS = 500
TRIOS = {"limb": (("I", 1), ("II", -1), ("III", 1)), "goldberger": (("aVR", 1), ("aVL", 1), ("aVF", 1))}


def _seg(ps, name, k=0):
    lst = ps.get(name) or []
    return np.asarray(lst[k], float) if len(lst) > k else None


def _slope(x, fs=FS):
    y = np.asarray(x, float)
    f = np.isfinite(y)
    if not f.any():
        return np.full(len(y), np.nan)
    z = np.where(f, y, np.nanmedian(y))
    e = np.convolve(np.abs(np.gradient(z)) * fs, np.ones(20) / 20, "same")
    e[~f] = np.nan
    return e


def qrs_times(x, fs=FS):
    """QRS instants (s) of one segment: peaks of the slope smoothed over 40 ms."""
    from scipy.signal import find_peaks
    f = np.isfinite(x)
    if f.sum() < 0.5 * fs:
        return None, None
    idx = np.where(f)[0]
    a, b = idx[0], idx[-1] + 1
    y = np.where(f[a:b], x[a:b], np.nan)
    y = np.where(np.isfinite(y), y, np.nanmedian(y))
    d = np.abs(np.gradient(y)) * fs
    e = np.convolve(d, np.ones(20) / 20, "same")
    h = max(float(np.percentile(e, 99)) * 0.4, 3.0)    # mV/s: T waves stay well below a QRS
    p, _ = find_peaks(e, height=h, distance=int(0.28 * fs))
    return (p + a) / fs, (a / fs, b / fs)


def _match(t, ref, tol=0.08):
    """Offsets (ms) of beats t to the nearest beat in ref, the beats with none within tol."""
    if t is None or ref is None or len(ref) == 0:
        return np.array([]), len(t) if t is not None else 0
    d = np.array([ref[np.argmin(np.abs(ref - x))] - x for x in t])
    ok = np.abs(d) <= tol
    return d[ok] * 1000, int((~ok).sum())


def analyse(ps, fs=FS):
    out = dict(closure={}, bursts=[], timing={}, rhythm_lag_ms=None)
    for nm, trio in TRIOS.items():
        X = [_seg(ps, l) for l, _ in trio]
        if any(x is None for x in X):
            continue
        V = np.stack(X)
        m = np.isfinite(V).all(0)
        if m.sum() < 0.4 * fs:
            continue
        V = V - np.nanmedian(V[:, m], axis=1, keepdims=True)
        c = np.array([s for _, s in trio], float)[:, None]
        e = (c * V).sum(0)
        rms = float(np.sqrt(np.nanmean(V[:, m] ** 2)))
        out["closure"][nm] = round(float(np.sqrt(np.mean(e[m] ** 2)) / max(rms, 1e-6)), 3)
        w = int(0.2 * fs)
        idx = np.where(m)[0]
        E = [_slope(x, fs) for x in X]
        Eref = []
        for ej in E:   # a lead's typical QRS slope: the 99th percentile of its smoothed slope
            Eref.append(float(np.nanpercentile(ej, 99)) if np.isfinite(ej).any() else 0.0)
        for s0 in range(idx[0], idx[-1] - w, w):
            sl = slice(s0, s0 + w)
            if not m[sl].all():
                continue
            r = float(np.max(np.abs(e[sl])))
            loc = float(np.max(np.abs(V[:, sl])))
            if r > 0.3 and r > 0.4 * loc:
                # blame: one closure cannot say which of three leads is wrong (each one alone would
                # absorb the whole residual); the usual failure is a lead that lost its trace, so
                # blame the lead whose slope in the window falls furthest below its own beats
                ratio = []
                for j in range(3):
                    ej = E[j][sl]
                    ratio.append(float(np.nanmax(ej)) / max(Eref[j], 1e-6) if np.isfinite(ej).any() else 0.0)
                j = int(np.argmin(ratio))
                if ratio[j] > 0.5 or sorted(ratio)[1] < 2 * ratio[j]:
                    j = None
                out["bursts"].append(dict(trio=nm, t=round(s0 / fs, 2), res_mv=round(r, 2),
                                          lead=trio[j][0] if j is not None else "?"))
    # lag: the time shift of each lead (others fixed) that best closes its trio; a row read
    # early or late shows up as a biphasic residual at every QRS, removed by the shift
    out["lag_ms"] = {}
    for nm, trio in TRIOS.items():
        X = [_seg(ps, l) for l, _ in trio]
        if any(x is None for x in X):
            continue
        c = [s_ for _, s_ in trio]
        base = sum(ci * x for ci, x in zip(c, X))
        mb = np.isfinite(base)
        if mb.sum() < fs:
            continue
        r0 = float(np.sqrt(np.mean(base[mb] ** 2)))
        for j, (l, _) in enumerate(trio):
            best = (r0, 0)
            for L in range(-25, 26):          # +-50 ms
                Y = list(X)
                Y[j] = np.roll(X[j], L)
                e = sum(ci * y for ci, y in zip(c, Y))
                m = mb & np.isfinite(e)
                m[:30] = m[-30:] = False
                if m.sum() < fs:
                    continue
                rr = float(np.sqrt(np.mean(e[m] ** 2)))
                if rr < best[0]:
                    best = (rr, L)
            out["lag_ms"][l] = dict(lag=best[1] * 1000 / fs, res=round(r0, 3), res_shifted=round(best[0], 3))
    # timing: every lead against the other leads printed over the same seconds and the rhythm strip
    beats = {}
    for l, lst in ps.items():
        for k, x in enumerate(lst):
            t, span = qrs_times(np.asarray(x, float), fs)
            if t is not None:
                beats[(l, k)] = (t, span)
    for key, (t, span) in beats.items():
        others = [tt[(tt >= span[0]) & (tt <= span[1])] for k2, (tt, sp2) in beats.items()
                  if k2 != key and sp2[0] < span[1] and sp2[1] > span[0]]
        others = [o for o in others if len(o)]
        if not others or len(t) == 0:
            continue
        ref = np.concatenate(others)
        off, missing = _match(t, ref)
        # beats the others agree on (in at least half of them) that this lead does not have
        cons = []
        for x in np.unique(np.round(ref, 2)):
            if sum(np.any(np.abs(o - x) <= 0.06) for o in others) >= max(1, len(others) / 2) and \
               span[0] + 0.1 < x < span[1] - 0.1:
                cons.append(x)
        cons = np.array(sorted(set(np.round(cons, 1))))
        # lost: at a beat the others agree on, this lead's slope stays far below its own beats
        # (a missing or flattened QRS), not merely under the detector's threshold
        x_ = np.asarray(ps[key[0]][key[1]], float)
        e_ = _slope(x_, fs)
        own = [np.nanmax(e_[max(0, int((u - 0.04) * fs)):int((u + 0.04) * fs)]) for u in t]
        ref_e = float(np.nanmedian(own)) if own else 0.0
        lost_t = []
        for x in cons:
            if np.any(np.abs(t - x) <= 0.08):
                continue
            seg_ = e_[max(0, int((x - 0.06) * fs)):int((x + 0.06) * fs)]
            if ref_e > 10 and (not np.isfinite(seg_).any() or np.nanmax(seg_) < 0.3 * ref_e):   # mV/s
                lost_t.append(round(float(x), 2))
        lost = len(lost_t)
        name = key[0] if key[1] == 0 else f"{key[0]}#{key[1]}"
        out["timing"][name] = dict(beats=int(len(t)), med_ms=round(float(np.median(off)), 1) if len(off) else None,
                                   extra=missing, lost=lost, lost_t=lost_t)
    a, b = _seg(ps, "II", 0), _seg(ps, "II", 1)
    if a is not None and b is not None:
        m = np.isfinite(a) & np.isfinite(b)
        if m.sum() > fs:
            idx = np.where(m)[0]
            best, lag = -2.0, 0
            for L in range(-60, 61, 2):   # +-120 ms
                i0, i1 = max(idx[0], idx[0] + L), min(idx[-1], idx[-1] + L)
                xa = a[i0:i1]; xb = b[i0 - L:i1 - L]
                mm = np.isfinite(xa) & np.isfinite(xb)
                if mm.sum() < fs // 2:
                    continue
                cc = np.corrcoef(xa[mm], xb[mm])[0, 1]
                if cc > best:
                    best, lag = cc, L
            out["rhythm_lag_ms"] = lag * 1000 / fs
            out["rhythm_corr"] = round(float(best), 3)
    return out


def flags(cl, lag_ms=30, closure=0.5):
    """Short list of human-readable problems."""
    f = []
    for nm, v in cl["closure"].items():
        if v > closure:
            f.append(f"{nm} closure {v}")
    by = {}
    for b in cl["bursts"]:
        by.setdefault((b["trio"], b["lead"]), []).append(b["t"])
    for (trio, l), ts in by.items():
        at = ", ".join(f"{t:g}" for t in ts[:4])
        f.append(f"{l} lost its trace at {at} s ({trio} closure)" if l != "?" else f"{trio} closure broken at {at} s")
    for l, v in cl["timing"].items():
        if v["med_ms"] is not None and abs(v["med_ms"]) > lag_ms:
            f.append(f"{l} beats {v['med_ms']:+.0f} ms vs the simultaneous leads")
        if v["lost"]:
            f.append(f"{l} lost the QRS at {', '.join(f'{u:g}' for u in v['lost_t'][:4])} s")
    for l, v in cl.get("lag_ms", {}).items():
        if 10 <= abs(v["lag"]) < 50 and v["res_shifted"] < 0.7 * v["res"]:   # 50: the search edge
            f.append(f"{l} read {v['lag']:+.0f} ms off its column (closure {v['res']} -> {v['res_shifted']} mV shifted)")
    if cl.get("rhythm_lag_ms") and abs(cl["rhythm_lag_ms"]) > lag_ms:
        f.append(f"grid II vs rhythm II {cl['rhythm_lag_ms']:+.0f} ms")
    return f


def plot(ps, cl, path, title=""):
    """Limb and Goldberger trios with their closure residual, and the QRS instants of every lead."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    t = np.arange(5000) / FS
    fig, ax = plt.subplots(3, 1, figsize=(16, 9))
    for a, (nm, trio) in zip(ax[:2], TRIOS.items()):
        X = [_seg(ps, l) for l, _ in trio]
        for (l, _), x in zip(trio, X):
            if x is not None:
                a.plot(t, x, lw=0.9, label=l)
        if all(x is not None for x in X):
            e = sum(c * x for (_, c), x in zip(trio, X))
            a.plot(t, e, "k", lw=1.2, label="closure residual")
            f = np.isfinite(e)
            if f.any():
                a.set_xlim(t[f][0] - 0.05, t[f][-1] + 0.05)
        for b in cl["bursts"]:
            if b["trio"] == nm:
                a.axvspan(b["t"], b["t"] + 0.2, color="r", alpha=0.15)
                a.text(b["t"], a.get_ylim()[1] * 0.9, b["lead"], color="r", fontsize=8)
        a.legend(loc="upper right", fontsize=8)
        a.grid(alpha=0.3)
        a.set_ylabel(f"{nm} (mV) {cl['closure'].get(nm, '')}")
    a = ax[2]
    names = sorted(ps, key=lambda l: ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"].index(l)
                   if l in ("I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6") else 99)
    yl = []
    for l in names:
        for k, x in enumerate(ps[l]):
            tt, span = qrs_times(np.asarray(x, float))
            lab = l if k == 0 else f"{l}#{k}"
            y = len(yl)
            yl.append(lab)
            if span:
                a.plot(span, [y, y], color="0.85", lw=6)
            if tt is not None:
                a.plot(tt, [y] * len(tt), "|", ms=10, mew=2, color="C3")
    a.set_yticks(range(len(yl)))
    a.set_yticklabels(yl, fontsize=8)
    a.invert_yaxis()
    a.set_xlim(0, 10)
    a.grid(axis="x", alpha=0.3)
    a.set_xlabel("s  (bars: QRS instants per lead; leads stacked; rhythm strips marked #1)")
    fig.suptitle(title + "   " + "; ".join(flags(cl))[:200], fontsize=9)
    plt.tight_layout()
    fig.savefig(path, dpi=60)
    plt.close(fig)
