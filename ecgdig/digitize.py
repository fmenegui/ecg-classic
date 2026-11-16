"""
Full pipeline: image -> 12 leads (mV, fs Hz).

    from ecgdig import digitize
    res = digitize("ecg.jpg")            # automatic layout
    res["signals"]  # (12, N) float32, NaN where the lead is not shown
"""
from __future__ import annotations

from dataclasses import dataclass
import dataclasses
import os
import numpy as np
import cv2

from .geometry import rectify
from .extract import (ink_map, binarize, find_rows, row_extent, detect_pulse, pulse_rows,
                      column_runs, viterbi_track, selected_envelopes, envelope_edges,
                      centerline_from_envelopes, calibrate_pen, mark_occlusions, horizontal_halfwidth,
                      trace_likeness, refine_steep)

from .layouts import LEADS, LIBRARY as LAYOUTS, is_limb
from .layout_engine import Ctx as _LayoutCtx


@dataclass
class Config:
    R: float = 10.0            # px/mm of the rectified frame
    fs: int = 500
    duration: float | None = None  # seconds shown per row; None = measured (trace ends)
    speed: float | None = None     # mm/s; None = read "NN mm/s" on the page, else QRS/RR
    gain: float | None = None      # mm/mV for every lead; None = measured on the pulses
    layout: str | list = "auto"    # "auto", a library name or a list of rows of names
    einthoven: bool = True     # fusion with Einthoven/Goldberger constraints
    fill_limb: bool = False    # rebuild limb leads where possible
    ppmm: float | None = None  # px/mm of the original image, if known (DPI/25.4)
    pen: str = "measure"       # horizontal pen width: measure | calibrate | isotropic
    # beat registration only corrects shear above this (mm at 25 mm/s): the QRS instant
    # differs between leads for physiological reasons (p90 ~12 ms), which mimics a shear of
    # up to ~0.5 mm; below that the registration hurts more than it helps
    reg_min_shift_mm: float = 0.65
    # non-flat paper: "poly" (global cubic polynomial), "cell" (+ local field on 5 mm cells,
    # follows creases and folds) or "none"
    nonrigid: str = "cell"
    major_snap: bool = True   # fit on the major lines (5 mm)
    fine_snap: bool = True    # fit on the minor lines (1 mm)
    steep_subpx: bool = False  # sub-pixel instant on steep stretches (experimental)
    synth: bool = True         # analysis-by-synthesis fit: Gaussian pen + blur (thin trace)
    bridge: bool = True        # Viterbi: a vertical jump linked by weak ink costs little
    synth_max_shift: float = 1.5
    synth_order: str = "sort"
    fit_kw: dict = None
    rigid_edges: bool = True     # Viterbi: a steep stroke translating as a whole does not pay edge continuity
    pulse_retry: bool = True     # pulse also searched from the common left margin of the rows
    row_beats: bool = True       # edge rows without QRS simultaneous with the others are dropped
    ink_model: str = ""         # path of the trace-segmenter ONNX (hybrid mode)
    ink_mix: float = 1.0        # 1 = network only; <1 mixes with the classical map
    prewarp: bool = False          # coarse perspective from the outline of the gridded area
    adaptive_threshold: bool = True  # light trace: redo the binarization with a lower threshold
    pulse_rows: bool = True       # baselines from the calibration pulses when there are >= 3
    row_chain: bool = True        # row extent crosses the gap between columns
    paper_crop: bool = True       # crop to the sheet when it fills < 30% of the frame
    max_side: int = 5000          # downscale images larger than this (memory)
    band_sp: float = 1.6        # trace search band: +-band_sp row spacings
    right_anchor: bool = True   # non-standard duration with agreeing ends: anchor at the end
    extrap_cap: bool = True     # extrapolation ramp at lead switches limited to 2 mm
    pulse_col: bool = True      # pulse column detached from the trace: erase it and use it for gain
    col_gap: bool = True        # common gap between equal columns: removed from the image
    col_align: bool = True      # columns with their own baselines: realign each band
    limb_log: bool = True       # physics in the layout as a log-likelihood of the residual
    fit: str = "none"          # "lsf": inverse rendering with the measured pen profile
    refuse_below: float | None = None  # leads with a quality score below this come out as NaN
    orient_check: bool = True     # 0 vs 180 degrees decided by the layout when pulses and labels do not break the tie


def _pick_rows(peaks, cov, n_expected):
    if len(peaks) <= n_expected:
        return list(peaks)
    # keep the subset of n rows with the highest coverage, then sort
    best = sorted(peaks, key=lambda p: -cov[p])[:n_expected]
    return sorted(best)


def _row_quick_trace(B, D, rows, R, x0, span_px, full=False):
    """Coarse trace (centroid of the segment chosen by Viterbi) of each full row.
    full=True returns dicts with absolute cen/top/bot and the window's x0."""
    H, W = B.shape
    out = []
    xa, xb = max(0, int(x0)), min(W, int(x0 + span_px) + 1)
    for r, y in enumerate(rows):
        up = (y - rows[r - 1]) if r > 0 else 40 * R
        dn = (rows[r + 1] - y) if r + 1 < len(rows) else 40 * R
        ya, yb = max(0, int(y - 0.9 * up)), min(H, int(y + 0.9 * dn))
        runs = column_runs(B, D, ya, yb, xa, xb)
        others = [float(v) for i2, v in enumerate(rows) if i2 != r]
        sel = viterbi_track(runs, float(y), R, others=others)
        if full:
            cen = np.array([runs[i][k][2] if k is not None else np.nan for i, k in enumerate(sel)])
            top = np.array([runs[i][k][0] if k is not None else np.nan for i, k in enumerate(sel)], float)
            bot = np.array([runs[i][k][1] if k is not None else np.nan for i, k in enumerate(sel)], float)
            out.append(dict(x0=xa, cen=cen, top=top, bot=bot))
        else:
            out.append(np.array([y - runs[i][k][2] if k is not None else np.nan for i, k in enumerate(sel)]))
    return out


def _corr(a, b, win=151):
    """Correlation after removing the baseline (~0.6 s moving median) and filling gaps."""
    from scipy.ndimage import median_filter
    m = ~np.isnan(a) & ~np.isnan(b)
    if m.sum() < 50:
        return 0.0
    x = np.arange(len(a))
    out = []
    for v in (a, b):
        v = np.interp(x, x[~np.isnan(v)], v[~np.isnan(v)])
        out.append((v - median_filter(v, size=win, mode="nearest"))[m])
    a, b = out
    if a.std() == 0 or b.std() == 0:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])




def _runs_bool(m):
    """True runs of a boolean vector: (starts, exclusive ends)."""
    d = np.diff(np.r_[0, np.asarray(m, np.int8), 0])
    return np.nonzero(d == 1)[0], np.nonzero(d == -1)[0]


def _tick_span(xs, top, bot, xb, R):
    """The separator mark at boundary xb: "tall" columns (top-bottom > 1.2 mm) within ±0.4 mm
    of xb, extended while top AND bottom stay where they were (±0.3 mm). The mark is a
    rectangle; a QRS touching it has a moving top/bottom and is not included. Returns (x_start, x_end) or None."""
    if not len(xs):
        return None
    hh = bot - top
    tall = hh > 1.2 * R
    c = int(np.argmin(np.abs(xs - xb)))
    win = int(0.4 * R)
    cand = [q for q in range(max(0, c - win), min(len(xs), c + win + 1)) if tall[q]]
    if not cand:
        return None
    q0 = min(cand, key=lambda q: abs(xs[q] - xb))
    t0, b0 = top[q0], bot[q0]
    tol = 0.3 * R

    def same(q):
        return tall[q] and abs(top[q] - t0) < tol and abs(bot[q] - b0) < tol

    a_ = b_ = q0
    while a_ - 1 >= 0 and same(a_ - 1):
        a_ -= 1
    while b_ + 1 < len(xs) and same(b_ + 1):
        b_ += 1
    if xs[b_] - xs[a_] > 1.2 * R:
        return None
    return float(xs[a_]), float(xs[b_])


def _cut_leading_pulse(v, fs):
    """Calibration pulse the detector missed that entered the trace: at the start of the
    row, a flat plateau of ~1 mV (0.5-2.5 mV) for >= 40 ms followed by a sharp drop.
    Everything up to the drop (+ 10 ms) becomes unknown."""
    n = min(len(v), int(0.4 * fs))
    if n < int(0.06 * fs):
        return v
    w = v[:n]
    if not np.isfinite(w).any():
        return v
    k = int(0.04 * fs)
    best = None
    for i in range(0, n - k):
        seg = w[i:i + k]
        if not np.isfinite(seg).all():
            continue
        m = float(np.median(seg))
        if 0.5 <= abs(m) <= 2.5 and np.std(seg) < 0.04:
            # end of the plateau: first sample that leaves it by more than half the height
            j = i + k
            while j < n and np.isfinite(w[j]) and abs(w[j] - m) < 0.5 * abs(m):
                j += 1
            drop = j < n and np.isfinite(w[min(n - 1, j + int(0.02 * fs))]) and \
                abs(w[min(n - 1, j + int(0.02 * fs))]) < 0.35 * abs(m)
            if drop:
                best = j
                break
    if best is None:
        return v
    out = v.copy()
    out[:min(len(v), best + int(0.01 * fs))] = np.nan
    return out

def _box_cols(xa, n, boxes, pad=2):
    """For each label box crossing [xa, xa+n): (i0, i1, y0, y1) in local indices."""
    out = []
    for (bx0, by0, bx1, by1) in boxes:
        i0, i1 = max(0, bx0 - pad - xa), min(n, bx1 + pad + 1 - xa)
        if i1 > i0:
            out.append((i0, i1, by0 - pad, by1 + pad))
    return out


def _drop_label_runs(runs, xa, boxes):
    """Ink segments entirely inside a label are letters, not trace."""
    for i0, i1, y0, y1 in _box_cols(xa, len(runs), boxes):
        for i in range(i0, i1):
            runs[i] = [c for c in runs[i] if not (c[0] >= y0 and c[1] <= y1)]


def _mask_label_edges(job, boxes):
    """An envelope edge falling inside a label is unknown (letter touching the trace); the
    edge on the outside still counts (e.g. the top of an R above the label)."""
    top, bot = job["top"], job["bot"]
    for i0, i1, y0, y1 in _box_cols(job["xa"], len(top), boxes):
        sl = slice(i0, i1)
        with np.errstate(invalid="ignore"):
            t, b = top[sl], bot[sl]
            t[(t >= y0) & (t <= y1)] = np.nan
            b[(b >= y0) & (b <= y1)] = np.nan

def _beat_xs(B, y, half, R, ext=None):
    """x positions of the complexes (QRS) of a row: columns where the band ink extends
    vertically well beyond the typical trace (fast upstroke/downstroke)."""
    H, W = B.shape
    a, b = max(0, int(y - half)), min(H, int(y + half))
    band = B[a:b]
    if ext:
        band = band[:, max(0, int(ext[0])):min(W, int(ext[1]))]
        x_off = max(0, int(ext[0]))
    else:
        x_off = 0
    has = band.any(0)
    if has.sum() < 20 * R:
        return np.array([])
    idx = np.arange(band.shape[0])[:, None]
    top = np.where(band, idx, 10 ** 6).min(0)
    bot = np.where(band, idx, -1).max(0)
    span = np.where(has, bot - top, 0).astype(float)
    base = float(np.median(span[has]))
    thr = max(2.0 * R, 3.0 * base)
    sp = span > thr
    runs = np.diff(np.r_[0, sp.astype(np.int8), 0])
    st, en = np.nonzero(runs == 1)[0], np.nonzero(runs == -1)[0]
    xs, last = [], -10 ** 9
    for s_, e_ in zip(st, en):
        if e_ - s_ > 6 * R:      # too wide for a QRS (text block, border)
            continue
        c = 0.5 * (s_ + e_)
        if c - last > 5 * R:
            xs.append(c + x_off)
            last = c
    return np.array(xs)


def _beat_coherence_drop(B, peaks, ext, half, R):
    """Rows at the edges of the block whose complexes do not coincide in time with those of
    the other rows: in a multichannel ECG the rows are simultaneous, so the QRS fall in the
    same columns. Header, on-screen menu bar, frame and text ruler do not follow that rhythm.
    Agreement between two rows = F1 of the matched complexes (+-2.5 mm): a noisy rhythm row
    (moiré) has extra complexes but contains the other's (high recall)."""
    n = len(peaks)
    keep = np.ones(n, bool)
    if n < 4:
        return keep
    bx = [_beat_xs(B, y, half, R, e) for y, e in zip(peaks, ext)]

    def f1(i, j):
        if len(bx[i]) < 2 or len(bx[j]) < 2:
            return 0.0
        d = np.abs(bx[i][:, None] - bx[j][None, :])
        p_ = float(np.mean(d.min(1) < 2.5 * R)); r_ = float(np.mean(d.min(0) < 2.5 * R))
        return 2 * p_ * r_ / max(p_ + r_, 1e-9)

    F = np.array([[f1(i, j) if i != j else np.nan for j in range(n)] for i in range(n)])
    cand = [i for i in range(n) if len(bx[i]) >= 3]
    if len(cand) < 3:
        return keep
    # consensus: rows that agree (F1 >= 0.5) with most of the other candidates
    cons = [i for i in cand if np.nanmedian([F[i, j] for j in cand if j != i]) >= 0.5]
    if len(cons) < 3:
        return keep
    sp_c = float(np.median(np.diff(np.sort(peaks[cons]))))
    for order in (range(n), range(n - 1, -1, -1)):
        for i in order:
            if i in cons:
                break
            sc = np.nanmedian([F[i, j] for j in cons])
            if _ROW_BEAT_DEBUG:
                print("beat-row", int(peaks[i]), round(float(sc), 2), len(bx[i]))
            # only dropped if, besides being off rhythm, it is off the block spacing (a
            # legitimate lead with small QRS sits at the normal row step)
            kp = [j for j in range(n) if keep[j] and j != i]
            gap = float(np.min(np.abs(peaks[kp] - peaks[i]))) if kp else np.inf
            odd = gap < 0.75 * sp_c or gap > 1.5 * sp_c
            # no complex at an edge, with every consensus row full of QRS: grid band, stain or
            # frame (a real ECG row almost always has some)
            odd = odd or (len(bx[i]) == 0 and min(len(bx[j]) for j in cons) >= 5)
            if sc < _ROW_BEAT_THR[0] and odd and keep.sum() > 3:
                keep[i] = False
            else:
                break
    return keep


def _excursions(B, y, half, R, dev_mm=3.0):
    """How many stretches of the row have ink >= dev_mm from the axis (inside the row band)."""
    H, W = B.shape
    a, b = max(0, int(y - half)), min(H, int(y + half))
    d = int(dev_mm * R)
    band = B[a:b]
    yy = np.arange(a, b)[:, None]
    far = (band & (np.abs(yy - y) >= d)).any(0)
    near = band[max(0, y - a - int(1.5 * R)):y - a + int(1.5 * R)].any(0)
    far &= cv2.dilate(near.astype(np.uint8).reshape(1, -1), np.ones((1, int(2 * R) | 1), np.uint8)).ravel() > 0
    runs = np.diff(np.r_[0, far.astype(np.int8), 0])
    st, en = np.nonzero(runs == 1)[0], np.nonzero(runs == -1)[0]
    # stretches less than 2 mm apart count as one
    n, last = 0, -10 ** 9
    for s_, e_ in zip(st, en):
        if s_ - last > 2 * R:
            n += 1
        last = e_
    return n


def _shift_pulse(p, keep_x):
    """x coordinates of a pulse after removing pixel columns from the image."""
    cum = np.cumsum(keep_x) - 1
    q = dict(p)
    for k in ("x_start", "x_end", "x_fall"):
        if k in q and q[k] is not None:
            q[k] = type(q[k])(cum[int(min(max(q[k], 0), len(cum) - 1))])
    return q


def _column_gaps(B, rows, ext, half, R, min_mm=2.0, max_mm=15.0):
    """Gaps common to all rows (no ink in any row band), min_mm to max_mm wide, that split the
    trace into k equal-width columns (±3%). Returns [(x0, x1)] or None."""
    ex = [e for e in ext if e]
    if len(ex) < len(rows) or len(rows) < 2:
        return None
    L, Rr = int(np.median([e[0] for e in ex])), int(np.median([e[1] for e in ex]))
    if Rr - L < 60 * R:
        return None
    occ = np.zeros(Rr - L, bool)
    Bo = B
    if GAP_IGNORE_TEXT[0]:
        # labels ("V1".."V6") printed inside the gap: letters are small loose components;
        # the trace is one long component. Only the trace occupies the gap
        n_, lab_, st_, _ = cv2.connectedComponentsWithStats(B.astype(np.uint8), connectivity=8)
        small = (st_[:, cv2.CC_STAT_WIDTH] < 5 * R) & (st_[:, cv2.CC_STAT_HEIGHT] < 5 * R)
        small[0] = False
        if small.any():
            Bo = B & ~small[lab_]
    occ_raw = np.zeros(Rr - L, bool)
    for y in rows:
        a, b = max(0, int(y - half)), min(B.shape[0], int(y + half))
        occ |= Bo[a:b, L:Rr].any(0)
        occ_raw |= B[a:b, L:Rr].any(0)
    d = np.diff(np.concatenate([[1], occ.astype(np.int8), [1]]))
    s_, e_ = np.nonzero(d == -1)[0], np.nonzero(d == 1)[0]
    def _text_only(a, b):
        # raw ink in the gap = letters: in each row <= 6 small components, 1.2 to 5 mm tall;
        # slivers of a broken trace are many, short and scattered
        if not (GAP_LABELS[0] and GAP_IGNORE_TEXT[0]):
            return False
        for y in rows:
            ya_, yb_ = max(0, int(y - half)), min(B.shape[0], int(y + half))
            ids = np.unique(lab_[ya_:yb_, L + a:L + b])
            ids = ids[ids > 0]
            if len(ids) > 6 or not small[ids].all():
                return False
            hs = st_[ids, cv2.CC_STAT_HEIGHT]
            if len(ids) and (hs.max() < 1.2 * R or hs.max() > 5 * R):
                return False
        return True
    gaps = [(L + a, L + b) for a, b in zip(s_, e_) if min_mm * R <= b - a <= max_mm * R and a > 0 and b < len(occ)
            # a label fills part of the gap; a trace broken into pieces (low contrast) fills almost all of it
            and (occ_raw[a:b].mean() <= 0.5 or _text_only(a, b))]
    if not gaps or len(gaps) > 5:
        return None
    # columns between the gaps of the same width
    edges = [L] + [x for g in gaps for x in g] + [Rr]
    widths = [edges[2 * i + 1] - edges[2 * i] for i in range(len(gaps) + 1)]
    if len(gaps) >= 2:
        # the middle columns decide the width; the first and last may have extra from a
        # pulse, border or end mark (up to 30 mm), never less
        inner = widths[1:-1]
        w = float(np.median(inner))
        if (max(inner) - min(inner)) > 0.03 * w:
            return None
        if any(not (0.97 * w <= x <= w + 30 * R) for x in (widths[0], widths[-1])):
            return None
    elif (max(widths) - min(widths)) > 0.03 * np.median(widths):
        # 2 columns: the first may have the calibration pulse extra (as in the 3+ case)
        if not (GAP_PULSE_SLACK[0] and 0 <= widths[0] - widths[1] <= 15 * R
                and widths[0] - widths[1] - 0.03 * widths[1] <= 15 * R):
            return None
    if min(widths) < 20 * R:
        return None
    return gaps


def _grid_clean(B, R):
    """B without the straight grid pieces (row extents must not run along a leaked grid line
    to the frame); B itself when the switch is off."""
    if not EXT_GRIDCLEAN[0]:
        return B
    from .extract import GRID_SEGS, clean_grid_segments
    return clean_grid_segments(B, R) if GRID_SEGS[0] else B


def _mid_pulse_cuts(B, rows, ext, half, R):
    """Calibration pulses printed again at the start of each column (a pulse between the
    columns of every row): [(x0, x1)] stretches to remove (pulse + blank margins), or None.
    Requires the pulses of one column at the same x (±2 mm) on at least half the rows (and
    >= 2), in the middle 20-80% of the rows, and columns of equal width after the cut (the
    first may carry the start pulse, up to 15 mm more)."""
    from .extract import find_pulses
    ex = [e for e in ext if e]
    if len(rows) < 2 or len(ex) < len(rows):
        return None
    L, Rr = float(np.median([e[0] for e in ex])), float(np.median([e[1] for e in ex]))
    if Rr - L < 60 * R:
        return None
    ys = np.asarray(rows, float)
    cand = []
    for p in find_pulses(B, R):
        if not (L + 0.2 * (Rr - L) <= p["x_start"] <= Rr - 0.2 * (Rr - L)):
            continue
        r_ = int(np.argmin(np.abs(ys - p["foot"])))
        if abs(ys[r_] - p["foot"]) <= half:
            cand.append((p["x_start"], p["x_end"], r_))
    cand.sort()
    groups = []
    for c in cand:
        if groups and c[0] - groups[-1][0][0] <= 2 * R:
            groups[-1].append(c)
        else:
            groups.append([c])
    need = max(2, int(np.ceil(0.5 * len(rows))))
    groups = [g for g in groups if len({c[2] for c in g}) >= need]
    if not groups or len(groups) > 3:
        return None
    occ = np.zeros(B.shape[1], bool)
    for y in rows:
        occ |= B[max(0, int(y - half)):int(y + half)].any(0)
    cuts = []
    for g in groups:
        a_ = int(min(c[0] for c in g) - 0.3 * R); b_ = int(max(c[1] for c in g) + 0.3 * R)
        while a_ > L and not occ[a_ - 1]:
            a_ -= 1
        while b_ < Rr and not occ[b_]:
            b_ += 1
        cuts.append((a_, b_))
    edges = [L] + [x for c in cuts for x in c] + [Rr]
    widths = [edges[2 * i + 1] - edges[2 * i] for i in range(len(cuts) + 1)]
    w = float(np.median(widths[1:]))
    if w < 20 * R or any(abs(x - w) > 0.04 * w for x in widths[1:]) or \
            not (-0.04 * w <= widths[0] - w <= 0.04 * w + 15 * R):
        return None
    return cuts


def _rows_from_hint(B, R, rows, cov):
    """Rows already known (column structure realigned): only extents and pulses, without the
    rules that drop rows."""
    peaks = np.asarray(rows, dtype=int)
    spacing = float(np.median(np.diff(peaks))) if len(peaks) > 1 else 40 * R
    half = 0.45 * float(np.min(np.diff(peaks))) if len(peaks) > 1 else 0.45 * spacing
    ext = [row_extent(B, y, half, R) for y in peaks]
    pulses = []
    for i, (y, e) in enumerate(zip(peaks, ext)):
        up_sp = (y - peaks[i - 1]) if i > 0 else spacing
        try:
            p_ = detect_pulse(B, y, half, e[0], R, up=max(half, min(0.95 * up_sp, 25 * R))) if e else None
        except Exception:  # noqa: BLE001
            p_ = None
        pulses.append(p_)
    ok = [p for p in pulses if p is not None]
    if len(ok) >= 2:   # pulses count if they agree in height and column
        h0 = float(np.median([p["foot"] - p["top"] for p in ok])); x0_ = float(np.median([p["x_start"] for p in ok]))
        pulses = [p if (p is not None and abs((p["foot"] - p["top"]) - h0) <= 0.2 * h0 and abs(p["x_start"] - x0_) <= 3 * R)
                  else None for p in pulses]
    else:
        pulses = [None] * len(pulses)
    return peaks, cov, ext, pulses, spacing


def _rows_pulses(B, pap, R, rows_hint=None):
    """Candidate trace rows (no text/borders), extents and pulses."""
    H, W = B.shape
    _prow_set = [None]
    from .extract import GRID_SEGS, clean_grid_segments
    # monochrome grid broken in the binary map: removed to find rows (horizontal) and to count
    # beats (vertical too); pulses and extents use B
    Br = clean_grid_segments(B, R, axes=(1,)) if GRID_SEGS[0] else B
    Bb = clean_grid_segments(Br, R, axes=(0,)) if GRID_SEGS[0] else B
    peaks, cov = find_rows(Br, R)
    if rows_hint is not None:
        return _rows_from_hint(B, R, rows_hint, cov)
    if len(peaks) == 0:
        return peaks, cov, [], [], 40 * R
    tl, raw = trace_likeness(Br, peaks, R, return_raw=True)
    if len(tl) and tl.max() > 0:
        # trace: long after joining breaks (tl) OR with long/tall pieces without joining
        # (raw; a weak scan trace may have low tl but high raw)
        ok = tl >= 0.4 * tl.max()
        weak = ~ok & (raw >= 0.3 * max(raw.max(), 1e-9))
        # "weak" rows only enter if they continue the regular spacing of the strong ones
        # (a header with a text table also has tall pieces, but off the step)
        if ok.sum() >= 2 and weak.any():
            sp0 = float(np.median(np.diff(peaks[ok])))
            changed = True
            while changed:
                changed = False
                acc = peaks[ok]
                for i in np.nonzero(weak & ~ok)[0]:
                    dmin = np.min(np.abs(acc - peaks[i]))
                    if 0.8 * sp0 <= dmin <= 1.25 * sp0:
                        ok[i] = True; changed = True
        # light/broken trace row (low tl) at the regular block step and with QRS in the same
        # columns as the accepted rows: it is ECG (the rows are simultaneous)
        if ok.sum() >= 3 and (~ok).any():
            sp0 = float(np.median(np.diff(peaks[ok])))
            hb = 0.45 * sp0
            bx_ok = [_beat_xs(Bb, y, hb, R) for y in peaks[ok]]
            for i in np.nonzero(~ok)[0]:
                dmin = np.min(np.abs(peaks[ok] - peaks[i]))
                if not (0.8 * sp0 <= dmin <= 1.25 * sp0):
                    continue
                bi = _beat_xs(Bb, peaks[i], hb, R)
                if len(bi) < 2:
                    continue
                f1s = []
                for bj in bx_ok:
                    if len(bj) < 2:
                        continue
                    d_ = np.abs(bi[:, None] - bj[None, :])
                    p_ = float(np.mean(d_.min(1) < 2.5 * R)); r_ = float(np.mean(d_.min(0) < 2.5 * R))
                    f1s.append(2 * p_ * r_ / max(p_ + r_, 1e-9))
                if f1s and np.median(f1s) >= 0.5:
                    ok[i] = True
        # text footer/header: only "long" after dilation (raw << tl), sits at the end of the
        # block and closer to its neighbour than the normal row spacing.
        # (a broken scan trace also has low raw, but sits in the middle or well spaced)
        idx = np.nonzero(ok)[0]
        if len(idx) >= 3:
            sp = float(np.median(np.diff(peaks[idx])))
            for e, nb in ((idx[0], idx[1]), (idx[-1], idx[-2])):
                if raw[e] < 0.55 * tl[e] and abs(peaks[e] - peaks[nb]) < 0.85 * sp:
                    ok[e] = False
        peaks = peaks[ok]
    if _PULSE_ROWS[0]:
        # rows from the calibration pulses: robust to giant QRS crossing the rows and to a
        # light trace (the horizontal projection gets confused); projection rows that match no
        # pulse and are far from all of them stay (row without a pulse)
        prow, _pp = pulse_rows(B, R)
        if prow is not None and PULSE_ROWS_GUARD[0] and len(peaks) >= 4:
            # the projection found a regular block of rows (constant step within 12%) and the
            # pulse "feet" do not fall on them (a pulse eaten by a dark grid gives feet at any
            # height) or they are few: the projection wins. Without this, 4 wrong feet replaced
            # 12 correct rows by 3
            d_ = np.diff(np.sort(np.asarray(peaks, float)))
            sp_ = float(np.median(d_))
            if sp_ > 0 and np.all(np.abs(d_ - sp_) <= 0.12 * sp_):
                hit = float(np.mean([np.min(np.abs(np.asarray(peaks, float) - q)) <= 0.25 * sp_ for q in prow]))
                if hit < 0.8 or len(prow) < 0.5 * len(peaks):
                    prow = None
        _prow_set[0] = prow
        if prow is not None and PULSE_HALF[0] and len(prow) >= 2:
            # pulses found only on alternate rows (pulse erased/weak on the others): projection
            # rows midway between the pulse rows are real rows; the real step is half. Pulse
            # ends off the new step are removed.
            spp = float(np.median(np.diff(prow)))
            # (only with a wide pulse step: half of it is still a plausible row step)
            mids = [p for p in peaks if prow.min() < p < prow.max()
                    and 0.35 * spp <= np.min(np.abs(prow - p)) <= 0.65 * spp] if spp >= 40 * R else []
            if mids:
                prow = np.sort(np.concatenate([prow, np.array(mids, float)]))
                sp2 = float(np.median(np.diff(prow)))
                while len(prow) > 2 and prow[-1] - prow[-2] > 1.5 * sp2 and \
                        np.min(np.abs(peaks - prow[-1])) > 0.3 * sp2:
                    prow = prow[:-1]
                while len(prow) > 2 and prow[1] - prow[0] > 1.5 * sp2 and \
                        np.min(np.abs(peaks - prow[0])) > 0.3 * sp2:
                    prow = prow[1:]
        if prow is not None:
            spp = float(np.median(np.diff(prow)))
            xp_ = float(np.median([q["x_start"] for q in _pp]))
            extra = []
            for p in peaks:
                if np.min(np.abs(prow - p)) <= 0.6 * spp:
                    continue
                # a row without a pulse only counts if it starts at the others' margin: a
                # stretch starting mid-page is a shifted column of an existing row
                e_ = row_extent(B, p, 0.3 * spp, R)
                dn = float(np.min(np.abs(prow - p)))
                regular = (0.85 * spp <= dn <= 1.15 * spp) and (p < prow.min() or p > prow.max())
                if e_ is not None and (e_[0] <= xp_ + 30 * R or (regular and e_[1] - e_[0] > 60 * R)):
                    # (or it continues the regular block step at an edge: that row's pulse was lost)
                    extra.append(p)
            peaks = np.sort(np.concatenate([prow, np.array(extra, float)])).astype(int)
    ys = np.nonzero(pap.any(1))[0]
    if len(ys):  # sheet borders / frame are not trace rows
        near = ~((peaks > ys[0] + 5 * R) & (peaks < ys[-1] - 5 * R))
        if near.any():
            # ...but the first ECG row may sit right against the grid border: it stays if it has
            # complexes (the border is straight, without QRS)
            sp_ = float(np.median(np.diff(peaks))) if len(peaks) > 2 else 40 * R
            for i in np.nonzero(near)[0]:
                if peaks[i] > ys[0] + 1.5 * R and peaks[i] < ys[-1] - 1.5 * R and \
                        len(_beat_xs(Bb, peaks[i], 0.45 * sp_, R)) >= 3:
                    near[i] = False
            peaks = peaks[~near]
    spacing = float(np.median(np.diff(peaks))) if len(peaks) > 1 else 40 * R
    half = 0.45 * spacing
    ext = [row_extent(Bb if EXT_GRIDCLEAN[0] else B, y, half, R) for y in peaks]
    pulses = []
    starts = [e[0] for e in ext if e]
    xs_med = float(np.median(starts)) if starts else None
    for i, (y, e) in enumerate(zip(peaks, ext)):
        up_sp = (y - peaks[i - 1]) if i > 0 else spacing
        p_ = detect_pulse(B, y, half, e[0], R, up=max(half, min(0.95 * up_sp, 25 * R))) if e else None
        # weak/broken trace at the start of the row (photo, screen): the extent starts after
        # the pulse. Rows start at the same margin: try again from the common margin
        if p_ is None and e and xs_med is not None and _PULSE_RETRY[0] and 10 * R < e[0] - xs_med < 60 * R:
            p_ = detect_pulse(B, y, half, xs_med, R, up=max(half, min(0.95 * up_sp, 25 * R)))
        if p_ is None and e and _PULSE_RETRY[0]:
            # light, broken vertical pulse edges (screen photo, thin trace): close larger
            # vertical breaks
            p_ = detect_pulse(B, y, half, min(e[0], xs_med) if xs_med is not None else e[0], R,
                              up=max(half, min(0.95 * up_sp, 25 * R)), close_mm=2.5)
            # on the second attempt only the simple rectangle: the wide closing may join the
            # pulse to the start of the trace and create a "step" (false split gain)
            if p_ is not None and (p_.get("step") or p_.get("half")):
                p_ = None
            if p_ is not None:
                p_ = dict(p_, retry=True)
        pulses.append(p_)
    # a pulse found only on the second attempt counts if it agrees with those found on the
    # first (height and column); with none from the first, it counts alone
    first = [p for p in pulses if p is not None and not p.get("retry")]
    if first:
        h0 = float(np.median([p["foot"] - p["top"] for p in first]))
        x0_ = float(np.median([p["x_start"] for p in first]))
        pulses = [None if (p is not None and p.get("retry") and
                           (abs((p["foot"] - p["top"]) - h0) > 0.2 * h0 or abs(p["x_start"] - x0_) > 3 * R))
                  else p for p in pulses]
    else:
        rt = [p for p in pulses if p is not None]
        # second attempt only: at least two are needed, with the same height and column
        ok_ = False
        # and on at least half the rows: the pulse is printed on all of them; two "pulses" in
        # six rows are QRS joined by the wide closing
        if len(rt) >= max(2, int(np.ceil(0.5 * len(pulses)))):
            hs = np.array([p["foot"] - p["top"] for p in rt]); xs_ = np.array([p["x_start"] for p in rt])
            ok_ = np.ptp(hs) <= 0.2 * np.median(hs) and np.ptp(xs_) <= 3 * R
        if not ok_:
            pulses = [None] * len(pulses)
    # row at the block edges without a pulse, when others have one: isolated header, text
    # ruler stuck to the block, edge of the printed area/sheet shadow in a photo (almost flat)
    has = [p is not None for p in pulses]
    if sum(has) >= 2:
        # a "pulse" far from the others' column (QR code, header box) does not count
        xs_p = np.array([p["x_start"] for p in pulses if p is not None], float)
        med_x = float(np.median(xs_p))
        has = [p is not None and abs(p["x_start"] - med_x) < 3 * R for p in pulses]
        if EDGE_PULSE_HEIGHT[0] and sum(has) >= 2:
            # ...nor a half pulse of another height (end marks and steps of a tall QRS next to a
            # letter pass for "half" pulses of any height). A full rectangle stays: a rhythm
            # strip may carry its own gain
            hs_ = [p["foot"] - p["top"] for p, h in zip(pulses, has) if h]
            h0_ = float(np.median(hs_))
            has = [h and (not p.get("half") or abs((p["foot"] - p["top"]) - h0_) <= 0.15 * h0_)
                   for p, h in zip(pulses, has)]
    if sum(has) >= 2 and len(peaks) > sum(has) and len(peaks) >= 3:
        exc = [_excursions(B, int(y), half, R) for y in peaks]
        ref = float(np.median([x for x, h in zip(exc, has) if h]))
        len_p = float(np.median([e[1] - e[0] for e, h in zip(ext, has) if h and e]))
        sp_all = float(np.median(np.diff(peaks)))
        keep = np.ones(len(peaks), bool)
        for order in (range(len(peaks)), range(len(peaks) - 1, -1, -1)):
            for i in order:
                if has[i]:
                    break
                others = peaks[keep & (np.arange(len(peaks)) != i)]
                gap = float(np.min(np.abs(others - peaks[i]))) if len(others) else np.inf
                short = not ext[i] or (ext[i][1] - ext[i][0]) < 0.6 * len_p
                flat = exc[i] < 0.4 * ref
                if gap > 1.6 * sp_all or gap < 0.7 * sp_all or (sum(has) >= 3 and (flat or short)):
                    keep[i] = False
                else:
                    break
        if not keep.all() and keep.sum() >= 2:
            peaks = peaks[keep]
            ext = [e for e, k in zip(ext, keep) if k]
            pulses = [p for p, k in zip(pulses, keep) if k]
            spacing = float(np.median(np.diff(peaks))) if len(peaks) > 1 else spacing
    if _ROW_BEAT_RULE[0] and len(peaks) >= 4:
        keep = _beat_coherence_drop(Bb, peaks, ext, half, R)
        # a row with a calibration pulse is an ECG row, with or without visible QRS
        if _prow_set[0] is not None:
            for i, y in enumerate(peaks):
                if np.min(np.abs(_prow_set[0] - y)) < 3 * R:
                    keep[i] = True
        # a pulse that disagrees with the others' consensus (height +-15%, width +-1.5 mm) does
        # not protect the row: in a QR code or stamp any rectangle passes for a pulse
        prot = [p is not None for p in pulses]
        okp = [p for p in pulses if p is not None]
        if PULSE_CONSENSUS[0] and len(okp) >= 3:
            hs = np.array([p["foot"] - p["top"] for p in okp]); ws = np.array([p["x_end"] - p["x_start"] for p in okp])
            h0, w0 = float(np.median(hs)), float(np.median(ws))
            agree = [abs(h - h0) <= 0.15 * h0 and abs(w - w0) <= 1.5 * R for h, w in zip(hs, ws)]
            if sum(agree) >= 2:
                it = iter(agree)
                prot = [p is not None and next(it) for p in pulses]
        keep = keep | np.array(prot, bool)
        if not keep.all():
            peaks = peaks[keep]
            ext = [e for e, k in zip(ext, keep) if k]
            pulses = [p for p, k in zip(pulses, keep) if k]
            spacing = float(np.median(np.diff(peaks))) if len(peaks) > 1 else spacing
    return peaks, cov, ext, pulses, spacing


_ROW_BEAT_RULE = [True]
EDGE_PULSE_HEIGHT = [True]
PULSE_COL_LEFT = [True]
HALF_ALONE = [True]
MID_PULSE_CUT = [True]
# row extents measured without the straight grid pieces left in the binary map
EXT_GRIDCLEAN = [False]
_PULSE_RETRY = [True]
RIGHT_ANCHOR = [True]
ORIENT_T = [True]
PULSE_CONSENSUS = [True]
EXTRAP_CAP = [True]
PULSE_COL = [True]
ADAPT_MORE = [True]
GLEV_RETRY = [True]
PULSE_HALF = [True]
COL_GAP = [True]
_PULSE_ROWS = [True]
_ROW_BEAT_THR = [0.3]
_ROW_BEAT_DEBUG = False


def _t_after_qrs(Bc, pk, ext, R):
    # physiology: the T wave comes AFTER the QRS (and is larger than the P, which comes
    # before). On an upside-down page time runs backwards. Measure, in each beat, the
    # deviation from the baseline 6-10 mm after and 6-10 mm before the QRS
    if len(pk) < 2:
        return 0.0
    half_ = 0.45 * float(np.median(np.diff(pk)))
    votes = []
    for y, e in zip(pk, ext):
        if not e:
            continue
        a_, b_ = max(0, int(y - half_)), min(Bc.shape[0], int(y + half_))
        band = Bc[a_:b_]
        idx = np.arange(band.shape[0])[:, None]
        cnt = band.sum(0)
        cen = np.where(cnt > 0, (band * idx).sum(0) / np.maximum(cnt, 1), np.nan)
        base = np.nanmedian(cen[e[0]:e[1]]) if np.isfinite(cen[e[0]:e[1]]).any() else np.nan
        if not np.isfinite(base):
            continue
        dev = np.abs(cen - base)
        for xb in _beat_xs(Bc, y, half_, R, e):
            xb = int(xb)
            aft = dev[xb + int(6 * R):xb + int(10 * R)]
            bef = dev[max(0, xb - int(10 * R)):max(0, xb - int(6 * R))]
            if len(aft) and len(bef) and np.isfinite(aft).mean() > 0.5 and np.isfinite(bef).mean() > 0.5:
                da, db = np.nanmean(aft), np.nanmean(bef)
                if max(da, db) > 0.3 * R:
                    votes.append(np.sign(da - db))
    if len(votes) < 6:
        return 0.0
    return float(np.mean(votes))



def _orient(arrs, R):
    """Pick the rotation (0/90/180/270) in which the traces are horizontal and the
    calibration pulses on the left."""
    rect, pap, D, B = arrs
    # traces = long structures (> 50 mm) in one direction; tall QRS do not exceed ~40 mm.
    # Close small breaks (thin/weak trace) before measuring.
    k = int(1.5 * R) | 1
    Bc = cv2.dilate(B.astype(np.uint8), np.ones((k, k), np.uint8))
    n, lab, st, _ = cv2.connectedComponentsWithStats(Bc, connectivity=8)
    w, h = st[1:, cv2.CC_STAT_WIDTH], st[1:, cv2.CC_STAT_HEIGHT]
    sh = w[(w >= 50 * R) & (w > 1.5 * h)].sum()
    sv = h[(h >= 50 * R) & (h > 1.5 * w)].sum()
    cands = [cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_90_COUNTERCLOCKWISE] if sv > sh else [None, cv2.ROTATE_180]

    def rot(a, c):
        if c is None:
            return a
        if a.dtype == bool:
            return cv2.rotate(a.astype(np.uint8), c) > 0
        return cv2.rotate(np.ascontiguousarray(a), c)

    from .ocr import find_words, read_label

    def upright_labels(Bc):
        # readable asymmetric labels (aVR, V1...): I/II/III look the same upside down
        n_ = 0
        tm = text_like_mask(Bc, R)
        for (x0, y0, x1, y1) in find_words(tm, R):
            d = read_label(tm[y0:y1, x0:x1])
            if d:
                k_ = max(d, key=d.get)
                if d[k_] >= 0.6 and k_ not in ("I", "II", "III"):
                    n_ += 1
        return n_

    best, best_score = None, -1
    scores, hard = [], []
    for c in cands:
        Bc, pc = rot(B, c), rot(pap, c)
        pk, _, ext_c, pulses, _ = _rows_pulses(Bc, pc, R)
        # pulses on the left + readable labels (a page printed upside down has pulses on both
        # sides on some devices; the text breaks the tie)
        score = sum(p is not None for p in pulses) + upright_labels(Bc)
        hard.append(float(score))
        if ORIENT_T[0]:
            # only breaks ties (worth less than a pulse or a label) and only with strong
            # agreement: in ~1 of 10 upright pages the T is smaller than what comes before (high HR, large P)
            tq = _t_after_qrs(Bc, pk, ext_c, R)
            score += 0.9 * float(np.sign(tq)) if abs(tq) > 0.3 else 0.0
        scores.append(float(score))
        if score > best_score:
            best, best_score = c, score
    # decision margin (so the caller knows whether it was weak) and forced choice
    _orient.margin = abs(scores[0] - scores[1]) if len(scores) == 2 else float("inf")
    # "weak" decision: pulses and labels differ by at most 2 between the two orientations (a
    # false pulse or label happens; and the T wave, the tie-breaker, is wrong on ~1 of 10 pages)
    _orient.weak = bool(cands[0] is None and len(hard) == 2 and abs(hard[0] - hard[1]) <= 2.0)
    if ORIENT_FORCE[0] is not None and cands[0] is None:
        best = None if ORIENT_FORCE[0] == "keep" else cv2.ROTATE_180
    return [rot(a, best) for a in arrs], best


PULSE_ROWS_GUARD = [True]
ORIENT_FORCE = [None]   # "keep" / "flip": fixes the 0 vs 180 degree choice (internal use)


def _edge_cands(D, B, rows, ext, pulses, half, lev, R, rad=0.0):
    """Sub-pixel left (trace start) and right (end) edges: lists without NaN."""
    xl, xe = _edge_rows(D, B, rows, ext, pulses, half, lev, R, rad)
    return [v for v in xl if not np.isnan(v)], [v for v in xe if not np.isnan(v)]


def _origin_cands(D, B, rows, ext, pulses, half, lev, R, rad):
    """Time-origin candidates: trace starts (strong), row ends and, on rows without a clear
    start, the end of the pulse's falling edge (weak: it picks the group, it does not give the
    value). Without the weak ones, only the row ends or a spurious edge would decide, and the
    dark grid, noise or the sheet border in a photo may be tens of mm away from the trace."""
    xl, xe, wk = _edge_rows(D, B, rows, ext, pulses, half, lev, R, rad, return_weak=True)
    xl = [float(v) for v in xl if not np.isnan(v)]
    # the trace does not start before the calibration pulse: a "start" left of the other
    # rows' pulses is the sheet/grid border in a photo
    xs_p = [p["x_start"] for p in pulses if p is not None]
    if xs_p:
        xl = [v for v in xl if v > float(np.median(xs_p))]
    xe = [float(v) for v in xe if not np.isnan(v)]
    return xl, xe, [float(v) for v in wk if np.isfinite(v)]


def text_like_mask(B, R):
    """Small isolated components (labels, numbers, marks): they do not belong to the trace,
    which is one long component. Used so that edges are not measured on text."""
    n, lab, st, _ = cv2.connectedComponentsWithStats(B.astype(np.uint8), connectivity=8)
    small = (st[:, cv2.CC_STAT_WIDTH] < 25 * R) & (st[:, cv2.CC_STAT_HEIGHT] < 9 * R)
    small[0] = False
    return small[lab]


def _edge_rows(D, B, rows, ext, pulses, half, lev, R, rad=0.0, return_weak=False):
    """Sub-pixel left (trace start) and right (end) edges, one per row (NaN if missing)."""
    tm = getattr(_edge_rows, "text_mask", None)
    if tm is not None and tm.shape == D.shape:
        D = np.where(tm, 0.0, D).astype(np.float32)
    if getattr(_edge_rows, "speckle", False):
        # dark grid (B&W scan): the dotted pattern passes the level on the max profile and
        # "extends" the trace to the border; only the ink that survived binarization (and its surroundings) counts
        Bd = cv2.dilate(B.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
        D = np.where(Bd, D, 0.0).astype(np.float32)
    H, W = D.shape
    lev_rad_hint = [rad]
    weak = []
    xl, xe = [], []
    for y, e, p in zip(rows, ext, pulses):
        xl.append(np.nan); xe.append(np.nan); weak.append(np.nan)
        if not e:
            continue
        a_, b_ = max(0, int(y - half)), min(H, int(y + half))
        # right
        x1 = max(0, e[1] - 12)
        blk = D[a_:b_, x1:min(W, e[1] + 4)]
        if blk.size:
            prof = blk.max(0)
            idx = np.nonzero(prof >= lev)[0]
            if len(idx) and idx[-1] + 1 < len(prof):
                k_ = idx[-1]
                xe[-1] = x1 + k_ + (prof[k_] - lev) / max(prof[k_] - prof[k_ + 1], 1e-6)
        # left: after the gap following the pulse; without a pulse, the first ink of the row
        if p is not None:
            x_s = p["x_end"]
            prof = D[a_:b_, x_s:min(W, x_s + int(6 * R))].max(0)
            below = np.nonzero((prof < lev) & (np.r_[prof[1:], 0] < lev))[0]
            after = np.nonzero(prof[below[0]:] >= lev)[0] if len(below) and below[0] <= 4 * R else []
            if not len(after):
                # trace stuck to the pulse (no gap): t = 0 probably on the pulse's falling edge,
                # a WEAK anchor (in other formats the pulse has a flat tail before the trace and
                # the gap may vanish with blur)
                weak[-1] = p.get("x_fall", p["x_end"]) - lev_rad_hint[0]
                continue
            k_ = below[0] + after[0]
        else:
            x_s = max(0, e[0] - 6)
            prof = D[a_:b_, x_s:min(W, e[0] + 12)].max(0)
            idx = np.nonzero(prof >= lev)[0]
            if not len(idx) or idx[0] == 0:
                continue
            k_ = idx[0]
        xl[-1] = x_s + k_ - 1 + (lev - prof[k_ - 1]) / max(prof[k_] - prof[k_ - 1], 1e-6)
    if return_weak:
        return np.array(xl), np.array(xe), np.array(weak)
    return np.array(xl), np.array(xe)


def _beats(trace, x_off, R):
    """Positions (x, px) of the QRS complexes: peaks of the smoothed absolute derivative (~80 ms)."""
    from scipy.ndimage import uniform_filter1d
    from scipy.signal import find_peaks
    t = np.asarray(trace, float)
    ok = ~np.isnan(t)
    if ok.sum() < 50:
        return np.array([])
    x = np.arange(len(t))
    t = np.interp(x, x[ok], t[ok])
    a = uniform_filter1d(np.abs(np.gradient(t)), max(3, int(2 * R)))
    pk, _ = find_peaks(a, height=0.35 * a.max(), distance=int(12 * R))  # >= 480 ms
    # centroid of the QRS energy (+-60 ms): less lead-dependent than the peak
    g = np.abs(np.gradient(t))
    h = int(1.5 * R)
    out = []
    for k in pk:
        lo, hi = max(0, k - h), min(len(g), k + h + 1)
        w = g[lo:hi] ** 2
        out.append(float((w * np.arange(lo, hi)).sum() / max(w.sum(), 1e-9)))
    return np.array(out) + x_off


def _register_rows(beats, R, width_px, tol_mm=2.5, min_shift_px=4.0, rows=None):
    """Per-row affine registration against the common QRS reference (median across rows).
    All leads are simultaneous: the same beat has the same instant in every row.
    Returns (a_r, b_r) with x_row = a_r + b_r * x_ref.
    Only corrects when the predicted shift exceeds min_shift_px (otherwise identity: the QRS
    phase bias between leads would be larger than the correction)."""
    n = len(beats)
    a = np.zeros(n); b = np.ones(n)
    nonempty = [bb for bb in beats if len(bb)]
    if n < 2 or not nonempty:
        return a, b
    order = np.sort(np.concatenate(nonempty))
    if len(order) < 4:
        return a, b
    groups, cur = [], [order[0]]
    for v in order[1:]:
        if v - cur[-1] <= tol_mm * R:
            cur.append(v)
        else:
            groups.append(cur); cur = [v]
    groups.append(cur)
    refs = np.array([np.median(g) for g in groups if len(g) >= max(2, n // 2)])
    if len(refs) < 2:
        return a, b
    lo, hi = refs.min(), refs.max()
    valid = np.zeros(n, bool)
    for r, bb in enumerate(beats):
        if len(bb) < 2:
            continue
        d = np.abs(bb[:, None] - refs[None, :])
        j = np.argmin(d, 1)
        m = d[np.arange(len(bb)), j] <= tol_mm * R
        if m.sum() < 2:
            continue
        xo, xr = bb[m], refs[j[m]]
        ar, br = float(np.median(xo - xr)), 1.0
        if m.sum() >= 4 and np.ptp(xr) > 0.4 * width_px:
            A = np.stack([np.ones_like(xr), xr], 1)
            c = np.linalg.lstsq(A, xo, rcond=None)[0]
            res = np.abs(xo - A @ c)
            keep = res <= max(2.5 * np.median(res), 2.0)
            if keep.sum() >= 3:
                c = np.linalg.lstsq(A[keep], xo[keep], rcond=None)[0]
            if 0.97 <= c[1] <= 1.03:
                ar, br = float(c[0]), float(c[1])
        a[r], b[r] = ar, br
        valid[r] = True
    # page distortion varies smoothly: shift and scale linear in the row height (averages the
    # noise of the QRS instant between leads, ~2-3 px)
    if rows is None or valid.sum() < 4:
        # with <= 3 rows (e.g. 3x4 without a rhythm strip) each row has different leads and
        # the noise of the QRS instant between leads cannot be told apart from shear
        return np.zeros(n), np.ones(n)
    if True:
        y = np.asarray(rows, float)
        A = np.stack([np.ones(valid.sum()), y[valid]], 1)
        ca = np.linalg.lstsq(A, a[valid], rcond=None)[0]
        cb = np.linalg.lstsq(A, b[valid], rcond=None)[0]
        a = ca[0] + ca[1] * y; b = cb[0] + cb[1] * y
    shift = np.maximum(np.abs(a + (b - 1) * lo), np.abs(a + (b - 1) * hi))
    if shift.max() < min_shift_px:
        return np.zeros(n), np.ones(n)
    return a, b


def _row_time_maps(rows, xl, rad, x0g, R, reg, tol_mm=4.0, weak=None, both_sided=True):
    """Time axis per row: x(t) = x0_r + sc_r * t * v * R.
    reg = (a_r, b_r) from the beat registration (relative geometry between rows).
    The absolute origin comes from the left anchors (end of pulse / start of trace), taken
    to the common frame; if they disagree with the global vote by > 0.8 mm, they win."""
    a_r, b_r = reg
    if weak is not None and not both_sided:
        xl = np.where(np.isnan(xl), weak, xl)
    Lref = (xl + rad - a_r) / b_r
    ok = ~np.isnan(Lref) & (np.abs(Lref - x0g) < tol_mm * R)
    x0_ref = x0g
    # a vote with edges on both sides is the strongest evidence: keep it. Only without it
    # can the left anchors (weak ones included) correct the origin.
    if not both_sided and ok.sum() >= 1:
        med = float(np.median(Lref[ok]))
        if abs(med - x0g) > 0.8 * R:
            x0_ref = med
    return a_r + b_r * x0_ref, b_r.copy()


def _estimate_speed(B, D, rows, R, x_left, width_px, return_scores=False):
    """25 or 50 mm/s from the QRS width in mm (25 mm/s: ~1.5-4 mm; 50 mm/s: ~3-8 mm)
    combined with the RR interval (40-150 bpm)."""
    q = _row_quick_trace(B, D, rows, R, max(0, int(x_left)), width_px)
    widths, rrs = [], []
    from scipy.ndimage import uniform_filter1d
    for t in q:
        ok = ~np.isnan(t)
        if ok.sum() < 100:
            continue
        x = np.arange(len(t)); t = np.interp(x, x[ok], t[ok])
        g = np.abs(np.gradient(t))
        a = uniform_filter1d(g, max(3, int(1.5 * R)))
        b = _beats(t, 0, R)
        if len(b) >= 2:
            rrs += list(np.diff(b) / R)
        thr = 0.5 * np.percentile(g, 99.5)
        act = a > 0.25 * a.max()
        s_, e_ = np.nonzero(np.diff(np.r_[0, act.astype(int), 0]) == 1)[0], np.nonzero(np.diff(np.r_[0, act.astype(int), 0]) == -1)[0]
        for i0, i1 in zip(s_, e_):
            if g[i0:i1].max() > thr:
                widths.append((i1 - i0) / R)
    if not widths:
        return (25.0, {}) if return_scores else 25.0
    wq = float(np.median(widths))
    rr = float(np.median(rrs)) if rrs else None
    sc = {}
    for v in (12.5, 25.0, 50.0):
        qrs_ms = wq / v * 1000
        s_ = -((np.log(qrs_ms / 100.0)) / 0.35) ** 2
        if rr:
            hr = 60.0 / (rr / v)
            s_ += -((np.log(hr / 75.0)) / 0.4) ** 2
        sc[v] = s_
    # 25 mm/s is the overwhelming default: 50 only with clear evidence
    v = 50.0 if sc[50.0] - sc[25.0] > 1.0 else 25.0
    return (v, sc) if return_scores else v


def _vote_x0(xl, xe, rad, span_px, R, weak=()):
    """Vote: each edge proposes an x0; the group with most support wins (+-0.8 mm).
    With edges on both sides, the mean cancels the bias of the trace end.
    weak: weak anchors (end of the pulse's falling edge) vote on the group choice, but only
    give the value when the group has no real edge (they have a ~0.1 mm bias)."""
    c = [(v + rad, "L") for v in xl] + [(v - rad - span_px, "R") for v in xe] + \
        [(v + rad, "W") for v in weak]
    if weak:
        # with pulses, the trace starts at the end of the pulse or a few mm after: a row end
        # that puts the origin 8 mm beyond (or before the pulse) joined something else
        wm = float(np.median(weak)) + rad
        c = [(v, t) for v, t in c if t != "R" or -1.0 * R <= v - wm <= 8.0 * R]
    _vote_x0.both_sided = False
    if not c:
        return None
    vals = np.array([v for v, _ in c])
    # tie: the start wins (left edge or end of pulse); the row end may have an end-of-trace
    # mark or a leftover column
    sup = [(np.abs(vals - v) < 0.8 * R).sum() + 0.1 * len({t for (u, t) in c if abs(u - v) < 0.8 * R})
           + (0.05 if t0 != "R" else 0.0) for v, (_, t0) in zip(vals, c)]
    v0 = vals[int(np.argmax(sup))]
    grp = [(v, t) for v, t in c if abs(v - v0) < 0.8 * R]
    Ls = [v for v, t in grp if t == "L"]; Rs = [v for v, t in grp if t == "R"]
    Ws = [v for v, t in grp if t == "W"]
    _vote_x0.both_sided = bool(Ls and Rs)
    if Ls and Rs:
        return 0.5 * (np.median(Ls) + np.median(Rs))
    if Ls or Rs:
        return float(np.median(Ls or Rs))
    return float(np.median(Ws))



BASELINE_GUARD = [True]
GAP_IGNORE_TEXT = [True]
SPAN_SANITY = [True]
GAP_PULSE_SLACK = [True]
# column gap taken by a label ("aVR" printed in the gap, touching both sides): accepted if the
# gap ink in each row is only text (few small components, letter height)
GAP_LABELS = [True]
# margins between columns up to 25 mm (some printers leave ~18 mm between the two halves of a 6x2)
GAP_WIDE = [True]
RPULSE_RARE = [True]
# a loose step at the end of the row (one edge only, no return to baseline) is an end-of-trace
# mark in several generators, not a calibration: it only stays if it gives the near-universal gain (10 mm/mV)
RPULSE_HALF = [True]


def _baseline_guard(sig, fs, win_s=0.6, thr_mv=1.5):
    """A baseline that wanders > 1.5 mV within a continuous stretch: the tracker switched
    rows or followed a border/crease for a while (crumpled photo, curved sheet). An ECG has
    no such slow drift; remove the 0.6 s moving median (longer than QRS and T) only on those
    stretches. The drift is measured away from the edges, where the window is truncated and
    the median follows the QRS or an initial jump, which fired the guard on clean ECGs (PhysioNet). Normal stretches are untouched."""
    from scipy.ndimage import median_filter
    out = sig.copy()
    k = max(3, int(win_s * fs) | 1)
    for i in range(out.shape[0]):
        s_ = out[i].astype(np.float64)
        idx = np.nonzero(np.isfinite(s_))[0]
        if len(idx) < k:
            continue
        for seg in np.split(idx, np.nonzero(np.diff(idx) > 1)[0] + 1):
            if len(seg) < k:
                continue
            x = s_[seg]
            b = median_filter(x, size=k, mode="nearest")
            bi = b[k // 2:len(b) - k // 2]   # core only: at the edges the median picks up QRS/initial jumps
            if len(bi) and np.ptp(bi) > thr_mv:
                s_[seg] = x - b + np.median(b)
        out[i] = s_.astype(out.dtype)
    return out

def digitize(image, cfg: Config | None = None, return_debug: bool = False):
    """Image -> signals. When the 0 vs 180 degree choice was weak (no pulse or label to break
    the tie), the page is also read in the other orientation and the better layout wins:
    upside down, the limb leads do not close Einthoven/Goldberger and boundaries and labels
    fall out of place. Without clear evidence, the page stays as it came."""
    cfg = cfg or Config()
    _orient.weak = False
    res = _digitize_once(image, cfg, return_debug)
    if not (cfg.orient_check and _orient.weak and ORIENT_FORCE[0] is None):
        if ORIENT_FORCE[0] is None:
            _PRE[0] = None
        return res
    flipped = res.get("rotation") not in ("None", None)
    # the other orientation only up to the layout, reusing the rectification and the
    # binarization (which do not depend on orientation): it costs a fraction of a page
    ORIENT_FORCE[0] = "keep" if flipped else "flip"
    _PRE_USE[0], _LAYOUT_ONLY[0] = True, True
    try:
        alt = _digitize_once(image, cfg, return_debug)
    except Exception:  # noqa: BLE001
        alt = None
    finally:
        _LAYOUT_ONLY[0] = False
    if alt is None:
        ORIENT_FORCE[0] = None; _PRE_USE[0] = False; _PRE[0] = None; _orient.weak = False
        return res

    def sc(r):
        li = r.get("layout_info") or {}
        return float(li.get("score", -np.inf))
    def limb(r):
        li = r.get("layout_info") or {}
        return float((li.get("parts") or {}).get("limb", 0.0))
    keep, flip = (alt, res) if flipped else (res, alt)
    # the first decision is only changed when the other orientation is clearly better in the
    # layout. Flipping also requires the physics: 12 correctly read leads close Einthoven/
    # Goldberger at ~0.8, and three neighbouring precordial leads upside down mimic I, II, III
    # at ~0.35 (floor 0.5); the upright reading must not close
    if flipped:
        out = keep if (np.isfinite(sc(keep)) and sc(keep) > sc(flip) + ORIENT_MARGIN[0]) else flip
    else:
        out = flip if (np.isfinite(sc(keep)) and sc(flip) > sc(keep) + ORIENT_MARGIN[0]
                       and limb(flip) >= 0.5 and limb(keep) <= 0.1) else keep
    chk = dict(keep=sc(keep), flip=sc(flip))
    try:
        if out is not res:
            out = _digitize_once(image, cfg, return_debug)   # the other one won: now in full
    except Exception:  # noqa: BLE001
        out = res
    finally:
        ORIENT_FORCE[0] = None; _PRE_USE[0] = False; _PRE[0] = None; _orient.weak = False
    out["orientation_checked"] = chk
    return out


ORIENT_MARGIN = [5.0]


_PRE = [None]            # (cfg, rect, pap, D, B, glev, ginfo) of the last page, before orientation
_PRE_USE = [False]       # reuse _PRE instead of rectifying and binarizing again
_LAYOUT_ONLY = [False]   # stop right after choosing the layout (no signal extraction)


def _digitize_once(image, cfg: Config | None = None, return_debug: bool = False):
    cfg = cfg or Config()
    if _PRE_USE[0] and _PRE[0] is not None:
        return _digitize_core(*_PRE[0], return_debug)
    img = cv2.imread(image, cv2.IMREAD_COLOR) if isinstance(image, str) else image
    if img is None:
        raise FileNotFoundError(image)
    if cfg.paper_crop:
        # small sheet on a wide background (photo from far): crop before downscaling, so the
        # resolution is not spent on the background
        from .geometry import paper_crop
        img, _ = paper_crop(img)
    if cfg.max_side and max(img.shape[:2]) > cfg.max_side:
        # huge image (600 dpi render): downscale first; rectification works at R px/mm and an
        # A4 sheet at 4000 px still has ~13 px/mm. Avoids running out of memory.
        f = cfg.max_side / max(img.shape[:2])
        img = cv2.resize(img, None, fx=f, fy=f, interpolation=cv2.INTER_AREA)
        if cfg.ppmm:
            cfg = dataclasses.replace(cfg, ppmm=cfg.ppmm * f)
    if cfg.prewarp:
        from .geometry import prewarp_quad
        img, _ = prewarp_quad(img)
    R = cfg.R
    _ROW_BEAT_RULE[0] = cfg.row_beats
    _PULSE_RETRY[0] = cfg.pulse_retry
    _PULSE_ROWS[0] = cfg.pulse_rows
    RIGHT_ANCHOR[0] = cfg.right_anchor
    EXTRAP_CAP[0] = cfg.extrap_cap
    PULSE_COL[0] = cfg.pulse_col
    COL_GAP[0] = cfg.col_gap
    from . import layout_engine as _le
    _le.LIMB_LOG[0] = cfg.limb_log
    from . import extract as _ex
    _ex.ROW_CHAIN[0] = cfg.row_chain
    rect, pap, ginfo = rectify(img, R, cfg.ppmm, cfg.nonrigid, major_snap=cfg.major_snap,
                               fine_snap=cfg.fine_snap)
    D, glev = ink_map(rect, pap, R, return_grid_level=True)
    if cfg.ink_model:
        # hybrid mode: the network says where the trace ink is; everything else stays classical
        from .inkmodel import predict
        Pn = predict(rect, cfg.ink_model)
        Pn[pap == 0] = 0.0
        D = Pn if cfg.ink_mix >= 1.0 else (cfg.ink_mix * Pn + (1 - cfg.ink_mix) * D).astype(np.float32)
        glev = 0.0 if cfg.ink_mix >= 1.0 else glev
    B = binarize(D, R, glev)
    _PRE[0] = (cfg, rect, pap, D, B, glev, ginfo)
    return _digitize_core(cfg, rect, pap, D, B, glev, ginfo, return_debug)


def _digitize_core(cfg, rect, pap, D, B, glev, ginfo, return_debug=False):
    """From the rectified and binarized image on (everything that depends on orientation)."""
    R = cfg.R
    rect, pap, D, B = rect.copy(), pap.copy(), D.copy(), B.copy()   # _PRE stays intact
    (rect, pap, D, B), rot = _orient([rect, pap, D, B], R)
    from .colalign import col_align, ENABLED as _CA
    _CA[0] = cfg.col_align
    rect, pap, D, B, col_info = col_align(rect, pap, D, B, R)
    # column structure found by col_align with stacked rows: the (realigned) rows and the
    # number of columns are known; they are not searched again
    stacked = col_info and (col_info.get("how") == "tracks" or col_info.get("how", "").endswith("+stacked"))
    rows_hint = col_info["ref"] if stacked else None
    col_hint = len(col_info["cols"]) if rows_hint is not None else None

    rows_all, cov, ext_all, pulses_all, spacing = _rows_pulses(B, pap, R, rows_hint)

    def _weak(rows_, ext_):
        # almost no trace: < 2 rows or short rows (< 40 mm)
        lens = [e[1] - e[0] for e in ext_ if e]
        return len(rows_) < 2 or not lens or np.median(lens) < 40 * R

    def _nlong(ext_):
        return sum(1 for e in ext_ if e and e[1] - e[0] >= 40 * R)

    if cfg.adaptive_threshold and _weak(rows_all, ext_all):
        for sc_ in (0.5, 0.3):
            B2 = binarize(D, R, glev, scale=sc_)
            r2 = _rows_pulses(B2, pap, R)
            if not _weak(r2[0], r2[2]):
                B = B2
                rows_all, cov, ext_all, pulses_all, spacing = r2
                break
        if GLEV_RETRY[0] and _weak(rows_all, ext_all):
            # blurred/light trace (0.2-0.4): its halo entered as "grid leakage" and the threshold
            # floor ended up above the trace itself. Redo the grid level excluding ink relative to
            # the trace contrast, and try again
            from . import extract as _ex2
            old_ = _ex2.GLEV_RELATIVE[0]
            _ex2.GLEV_RELATIVE[0] = True
            try:
                _, glev2 = ink_map(rect, pap, R, return_grid_level=True)
            finally:
                _ex2.GLEV_RELATIVE[0] = old_
            if glev2 < glev:
                for sc_ in (1.0, 0.5, 0.3):
                    B2 = binarize(D, R, glev2, scale=sc_)
                    r2 = _rows_pulses(B2, pap, R)
                    if not _weak(r2[0], r2[2]):
                        B, glev = B2, glev2
                        rows_all, cov, ext_all, pulses_all, spacing = r2
                        break
    elif cfg.adaptive_threshold and ADAPT_MORE[0] and _nlong(ext_all) <= 3:
        # few long rows (light trace: only the large-amplitude leads passed the threshold): a
        # lower threshold that reveals >= 2 more long rows, at a regular step, wins
        B2 = binarize(D, R, glev, scale=0.5)
        r2 = _rows_pulses(B2, pap, R)
        n1, n2 = _nlong(ext_all), _nlong(r2[2])
        if n2 >= n1 + 2 and n2 <= 16 and len(r2[0]) >= 3:
            d_ = np.diff(np.sort(np.asarray(r2[0], float)))
            if d_.min() >= 0.5 * np.median(d_):
                B = B2
                rows_all, cov, ext_all, pulses_all, spacing = r2
    H, W = B.shape
    if len(rows_all) == 0:
        raise RuntimeError("no trace row found")

    forced = None
    if cfg.layout != "auto":
        forced = LAYOUTS[cfg.layout] if isinstance(cfg.layout, str) else cfg.layout
        idx = [list(rows_all).index(r) for r in _pick_rows(rows_all, cov, len(forced))]
    else:
        idx = list(range(len(rows_all)))
    rows = [rows_all[i] for i in idx]
    ext = [ext_all[i] for i in idx]
    pulses = [pulses_all[i] for i in idx]
    spacing = float(np.median(np.diff(rows))) if len(rows) > 1 else spacing
    half = 0.45 * spacing
    # pulses printed apart (on the nominal row, with the trace offset by the DC level): they do
    # not join the trace and no row found them. They are not trace: removed from B and D
    # (otherwise the row start falls on the pulse and the time axis stretches); their height gives the default gain
    pcol = None
    if PULSE_COL[0] and all(p is None for p in pulses_all):
        from .extract import pulse_column
        pcol = pulse_column(B, R)
        # only the column left of the trace (it is the one that spoils the time origin); pulses
        # at the end of the row (GE) stay for the gain through the mirrored search
        ex_ = [e for e in ext if e]
        if pcol is not None and ex_ and pcol["x_start"] > float(np.median([e[0] for e in ex_])) + 30 * R:
            pcol = None
        # a pulse printed apart sits before the trace; with the rows already running well to
        # its left, the "column" is the first QRS, simultaneous in rows packed 3-8 mm apart
        if PULSE_COL_LEFT[0] and pcol is not None and ex_ and \
                float(np.median([e[0] for e in ex_])) < pcol["x_start"] - 5 * R:
            pcol = None
        if pcol is not None:
            B = B.copy(); D = D.copy()
            xa_, xb_ = max(0, int(pcol["x_start"] - 1 * R)), int(pcol["x_end"] + 1 * R)
            h_ = pcol["height"]
            for f_ in pcol["feet"]:
                # the pulse rectangle and, at the foot, the little leg/square coming out to its right
                ya_, yb_ = max(0, int(f_ - h_ - 1 * R)), int(f_ + 1 * R)
                B[ya_:yb_, xa_:xb_] = False
                D[ya_:yb_, xa_:xb_] = 0
                ya_, yb_ = max(0, int(f_ - 1.5 * R)), int(f_ + 3 * R)
                B[ya_:yb_, xa_:int(xb_ + 3 * R)] = False
                D[ya_:yb_, xa_:int(xb_ + 3 * R)] = 0
            ext = [row_extent(_grid_clean(B, R), y, half, R) for y in rows]
    # common gap between columns: every row without ink in the same x range, at positions that
    # split the trace into equal columns. Time is continuous from one column to the next, so
    # the gap is removed from the image (B, D and the rectified one) and the rest sees a continuous trace
    col_cut = None
    if COL_GAP[0] and len(rows) >= 2:
        cg = _column_gaps(B, rows, ext, half, R, max_mm=25.0 if GAP_WIDE[0] else 15.0)
        if not cg and MID_PULSE_CUT[0]:
            cg = _mid_pulse_cuts(B, rows, ext, half, R)
        if cg:
            keep_x = np.ones(W, bool)
            for a_, b_ in cg:
                keep_x[a_:b_] = False
            B, D, rect = B[:, keep_x], D[:, keep_x], rect[:, keep_x]
            H, W = B.shape
            cum_ = np.cumsum(keep_x)
            col_cut = [float(cum_[max(0, a_ - 1)]) for a_, _ in cg]  # cut x in the new image
            ext = [row_extent(_grid_clean(B, R), y, half, R) for y in rows]
            pulses = [None if p is None else _shift_pulse(p, keep_x) for p in pulses]
    # pulse at the END of the row (several devices print the calibration on the right, or on
    # both sides): searched in the mirrored image; used only for gain, not for the time axis
    rpulses = [None] * len(rows)
    Bf = B[:, ::-1]
    for i_, (y, e) in enumerate(zip(rows, ext)):
        if e:
            up_sp = (y - rows[i_ - 1]) if i_ > 0 else spacing
            try:
                rpulses[i_] = detect_pulse(Bf, y, half, W - 1 - e[1], R,
                                           up=max(half, min(0.95 * up_sp, 25 * R)))
            except Exception:
                rpulses[i_] = None

    # provisional thickness (segment heights in columns)
    lw_guess = []
    for y, e in zip(rows, ext):
        if not e:
            continue
        mid = (e[0] + e[1]) // 2
        sub = B[max(0, int(y - half)):int(y + half), max(0, mid - int(20 * R)):mid + int(20 * R)]
        for j in range(0, sub.shape[1], 3):
            d = np.diff(np.concatenate([[0], sub[:, j].astype(np.int8), [0]]))
            lw_guess += list(np.nonzero(d == -1)[0] - np.nonzero(d == 1)[0])
    lw0 = float(np.percentile(lw_guess, 30)) if lw_guess else 3.0
    band_vals = np.concatenate([D[max(0, int(y - half)):int(y + half)].ravel() for y in rows])
    lev0 = 0.5 * float(np.percentile(band_vals[band_vals > 0.1], 90)) if (band_vals > 0.1).any() else 0.35

    _edge_rows.text_mask = text_like_mask(B, R)
    _edge_rows.speckle = bool(getattr(ink_map, "dark_grid", False))

    # ---------------------------------------------------- speed and duration
    xlr0, xer0, wkr0 = _edge_rows(D, B, rows, ext, pulses, half, lev0, R, lw0 / 2, return_weak=True)
    # start per row: trace edge, or the end of the pulse when the trace leaves stuck to it
    xs_p0 = [p["x_start"] for p in pulses if p is not None]
    if xs_p0:  # a start left of the pulse is not trace (sheet border)
        xlr0 = np.where(xlr0 <= float(np.median(xs_p0)), np.nan, xlr0)
    xlr0 = np.where(np.isnan(xlr0), wkr0 + lw0 / 2, xlr0)
    xl_ok = xlr0[~np.isnan(xlr0)]
    xr_ok = xer0[~np.isnan(xer0)]
    left = float(np.median(xl_ok)) if len(xl_ok) else float(np.median([e[0] for e in ext if e]))
    right = float(np.median(xr_ok)) if len(xr_ok) else float(np.median([e[1] for e in ext if e]))
    span_mm = (right - left - lw0) / R
    speed = cfg.speed
    speed_src = "config"
    if speed is None:
        from .ocr import read_speed
        sp_ocr, sp_sc = read_speed(_edge_rows.text_mask, R)
        v_phys, sc_phys = _estimate_speed(B, D, rows, R, left, right - left, return_scores=True)
        # the text wins, unless the physiology strongly contradicts it (a weak reading of a
        # degraded "25mm/s" becomes "12.5" or "50": a 50 ms QRS / 30 bpm HR give it away)
        ok_txt = sp_ocr is not None and (
            sp_sc >= 0.7 or not sc_phys or sc_phys.get(sp_ocr, -np.inf) >= max(sc_phys.values()) - 2.0)
        # 25 mm/s is the overwhelming default; "50" read on a photo (a blurred "25") only counts
        # if the physiology (QRS/RR) does not prefer 25
        if ok_txt and sp_ocr != 25.0 and sc_phys and sc_phys.get(25.0, -np.inf) > sc_phys.get(sp_ocr, -np.inf) + 0.5:
            ok_txt = False
        if ok_txt:
            speed, speed_src = sp_ocr, "texto"
        else:
            speed, speed_src = v_phys, "QRS/RR"
    speed_nominal = speed
    duration_assumed = False
    if cfg.duration:
        duration = float(cfg.duration)
    else:
        duration = span_mm / speed
        for d_ in (1.25, 2.5, 5.0, 10.0, 12.0, 15.0, 20.0):
            if abs(duration - d_) / d_ < 0.06:
                duration = d_
                break
        else:
            # the median of the row ends may fall on an artefact (border, shadow) on half the
            # rows: look for the standard duration that the most ends confirm
            sup = {}
            for d_ in (2.5, 5.0, 10.0):
                exp_ = left + lw0 + d_ * speed * R
                sup[d_] = int((np.abs(xr_ok - exp_) < 1.5 * R).sum())
            d_best = max(sup, key=sup.get) if sup else None
            n_pulses = sum(p is not None for p in pulses)
            # no end confirms a standard duration from the measured start, but the row ends
            # agree: the start caught a deformed pulse, a bar or the frame. Anchor at the end:
            # the longest standard duration that starts 2 to 30 mm after the start
            anch = None
            ends_ = [e[1] for e in ext if e]
            n_end = sum(abs(e_ - right) < 3 * R for e_ in ends_)
            sup_ok = d_best is not None and sup[d_best] >= max(1, int(np.ceil(0.25 * len(xr_ok))))
            if RIGHT_ANCHOR[0] and not sup_ok and n_end >= max(2, int(np.ceil(0.5 * len(rows)))):
                # only the standard 10 s page: with short durations, a wrong-scale gap (grid read
                # at another step) fell into the window by chance
                l_ = right - lw0 - 10.0 * speed * R
                if left + 2 * R <= l_ <= left + 30 * R:
                    anch = 10.0
            if anch is not None:
                duration = anch
                left = right - lw0 - anch * speed * R
            elif d_best is not None and sup[d_best] >= max(1, int(np.ceil(0.25 * len(xr_ok)))):
                duration = d_best
            elif n_pulses >= 2 and 10.0 < duration <= 12.5 and len(rows) >= 3:
                # start anchored on the pulses and rows "too long": the ends joined the sheet/grid
                # border (photo). The grid gives the mm; the standard 10 s page wins
                duration = 10.0
            elif len(rows) >= 3:
                # multi-row page outside any standard duration: almost always the horizontal
                # scale is wrong (web image resized without keeping the aspect ratio, grid with
                # major lines only). The standard page has 10 s; the time axis adapts to the
                # measured width (the vertical scale does not change)
                duration = 10.0
                duration_assumed = True
                speed = span_mm / duration
            else:
                duration = round(duration, 1)
    if SPAN_SANITY[0] and not cfg.duration and duration > 12.5 and span_mm > 320 and len(rows) >= 2:
        # row longer than 32 cm: does not fit on A4/Letter. The grid scale is wrong (2 mm step
        # taken for 1 mm, dotted grid); the pulse also measures double, so the amplitude stays
        # right, and time goes back to the standard 10 s page
        duration = 10.0
        duration_assumed = True
        speed = span_mm / duration
    N = int(round(duration * cfg.fs))
    span_px = (N - 1) / cfg.fs * speed * R
    if os.environ.get("ECGDIG_DBG_SPEED"):
        print("SPEED", dict(left=left, right=right, R=R, span_mm=span_mm, speed=speed, src=speed_src,
                            dur=duration, xl=list(np.round(xlr0)), xr=list(np.round(xer0))))

    # ------------------------------------------------------- time origin
    xl, xe, wk_ = _origin_cands(D, B, rows, ext, pulses, half, lev0, R, lw0 / 2)
    x0 = _vote_x0(xl, xe, lw0 / 2, span_px, R, weak=wk_)
    if x0 is None:
        x0 = float(np.median([e[0] for e in ext if e])) if any(ext) else 0.0
    # beat registration: quick trace of each full row -> QRS -> affine per row
    xq0 = max(0, int(x0 - 2 * R))
    qfull = _row_quick_trace(B, D, rows, R, xq0, span_px + 4 * R, full=True)
    qtr = [float(y) - q["cen"] for y, q in zip(rows, qfull)]
    reg = _register_rows([_beats(t, xq0, R) for t in qtr], R, span_px, rows=rows,
                         min_shift_px=cfg.reg_min_shift_mm * R * max(1.0, speed / 25.0))
    xlr, _, wkr = _edge_rows(D, B, rows, ext, pulses, half, lev0, R, lw0 / 2, return_weak=True)
    x0_r, sc_r = _row_time_maps(rows, xlr, lw0 / 2, x0, R, reg, weak=wkr,
                                both_sided=getattr(_vote_x0, "both_sided", True))

    # ------------------------------------- gains per row (1 or 2 levels) and baseline
    def _snap(g):
        # gains are always standardized; a photo in perspective shortens the pulse by up to ~20%
        gg = min((2.5, 5.0, 10.0, 20.0, 40.0), key=lambda v: abs(np.log(g / v)))
        return gg if abs(g - gg) / gg < 0.22 else float(g)
    std_g = (2.5, 5.0, 10.0, 20.0, 40.0)

    def _std_ok(g, tol=0.22):
        return g is not None and min(abs(g - gg) / gg for gg in std_g) < tol

    def _lv(p):
        if p is None:
            return None, None
        g1_ = (p["foot"] + 0.5 - (p["top"] - 0.5) - lw0) / R
        g2_ = (p["foot"] + 0.5 - (p["top2"] - 0.5) - lw0) / R if p.get("step") else None
        return g1_, g2_
    lv1, lv2, src = [], [], []
    for p, rp in zip(pulses, rpulses):
        a1, a2 = _lv(p)
        b1, _ = _lv(rp)
        if RPULSE_HALF[0] and rp is not None and rp.get("half") and _std_ok(b1) and _snap(b1) != 10.0:
            b1 = None
        # the left pulse counts if it gives a standard gain; otherwise try the right one
        if _std_ok(a1) or not _std_ok(b1):
            lv1.append(a1); lv2.append(a2); src.append("L" if a1 is not None else None)
        else:
            lv1.append(b1); lv2.append(None); src.append("R")  # split gain only from the left pulse
    # Page consensus: gains are standardized (2.5/5/10/20/40 mm/mV). A deformed pulse, cut at
    # the border or stuck to the trace gives 3.7 or 7 mm/mV; that level does not count and the
    # row inherits the most voted standard gain among the reliable rows (or 10, if none is).
    # 2.5 and 40 mm/mV are rare; only the left pulse (printed as calibration) supports them.
    # The right "pulse" is the end of a blur-flattened trace and gives 2.5 without any real pulse
    if RPULSE_RARE[0]:
        for r_, g in enumerate(lv1):
            if g is not None and src[r_] == "R" and _std_ok(g) and _snap(g) in (2.5, 40.0):
                lv1[r_] = None; lv2[r_] = None; src[r_] = None
    if HALF_ALONE[0]:
        # a half pulse (only the plateau and the falling edge) is weak evidence: a deep S wave
        # after a flat baseline looks the same. Away from the 10 mm/mV default it needs
        # two other rows that agree
        sn = [_snap(g) if _std_ok(g) else None for g in lv1]
        for r_, p in enumerate(pulses):
            if p is not None and p.get("half") and src[r_] == "L" and sn[r_] is not None and sn[r_] != 10.0 \
                    and sum(1 for k, v in enumerate(sn) if v == sn[r_] and k != r_) < 2:
                lv1[r_] = None; lv2[r_] = None; src[r_] = None
    reliable = [_snap(g) for g in lv1 if _std_ok(g)]
    if reliable:
        vals, cnts = np.unique(reliable, return_counts=True)
        g1def = float(vals[np.argmax(cnts)])
    else:
        g1def = 10.0  # no reliable pulse: 10 mm/mV is the default of almost every device
        # only when the pulse feet follow the row step (it is the calibration column)
        pc_ok = pcol is not None and len(rows) > 1 and \
            abs(float(np.median(np.diff(pcol["feet"]))) - spacing) <= 0.15 * spacing
        if pc_ok and _std_ok((pcol["height"] + 1 - lw0) / R, 0.1):
            g1def = _snap((pcol["height"] + 1 - lw0) / R)
    gain_flags = []
    for r_, g in enumerate(lv1):
        # a right pulse (less reliable: it tends to touch the end of the trace) that disagrees
        # with the page consensus also yields
        if g is not None and src[r_] == "R" and len(reliable) >= 2 and _snap(g) != g1def:
            lv1[r_] = None
            gain_flags.append(r_)
            continue
        if g is not None and not _std_ok(g):
            lv1[r_] = None
            lv2[r_] = None
            gain_flags.append(r_)
    g2s = [g for g in lv2 if _std_ok(g)]
    g2def = _snap(float(np.median(g2s))) if g2s else None
    levels = []
    for r_ in range(len(rows)):
        g1 = _snap(lv1[r_]) if lv1[r_] is not None else g1def
        g2 = _snap(lv2[r_]) if (lv2[r_] is not None and _std_ok(lv2[r_])) else (
            g2def if pulses[r_] is None else None)
        if cfg.gain:
            g1, g2 = float(cfg.gain), None
        levels.append((g1, g2))
    # 0 mV comes from the left pulse plateau only if it is reliable
    pulses_base = [p if (p is not None and src[r_] == "L" and lv1[r_] is not None) else None
                   for r_, p in enumerate(pulses)]
    base_q = []
    for r_, (y, p) in enumerate(zip(rows, pulses_base)):
        if p is not None:
            base_q.append(p["top"] - 0.5 + lw0 / 2 + levels[r_][0] * R)
        else:
            q = qfull[r_]["cen"]; q = q[~np.isnan(q)]
            if len(q):
                hist, edges = np.histogram(q, bins=np.arange(q.min() - 1, q.max() + 2, 1.0))
                hist = cv2.GaussianBlur(hist.astype(np.float32).reshape(-1, 1), (0, 0), 2).ravel()
                base_q.append(float(edges[int(np.argmax(hist))] + 0.5))
            else:
                base_q.append(float(y))

    # ------------------------------------------------------------- layout
    layout_info = None
    label_boxes = []
    if forced is not None:
        rows_spec = [list(r) for r in forced]
        layout = cfg.layout if isinstance(cfg.layout, str) else "custom"
    else:
        ctx = _LayoutCtx(B, D, rows, pulses, R, x0_r, sc_r, speed, duration, qfull, base_q, levels)
        ctx.col_hint = col_hint
        if col_cut:
            # the gap removed from the image was the strongest column boundary there is
            for r_ in range(len(rows)):
                for xc in col_cut:
                    q_ = (xc - x0_r[r_]) / (sc_r[r_] * speed * R * duration)
                    for qq in ctx.bnd[r_]:
                        if abs(qq - q_) < 0.015:
                            ctx.bnd[r_][qq] = 1.0
        layout_info = ctx.choose()
        # boxes of labels read with confidence: a trace crossing a label (e.g. an R passing over
        # "V4") must not be confused with the letters
        label_boxes = [w["box"] for w in ctx.words if w["best"] >= 0.5 and not w["single"]
                       and (w["box"][2] - w["box"][0]) < 15 * R and (w["box"][3] - w["box"][1]) < 8 * R]
        if layout_info is None:
            rows_spec = [[f"row{i + 1}"] for i in range(len(rows))]
            layout = "rows"
        else:
            rows_spec, layout = layout_info["spec"], layout_info["name"]
    if _LAYOUT_ONLY[0]:
        return dict(layout=layout, layout_info=layout_info, rotation=str(rot))

    def seg_gain(r_, name):
        g1, g2 = levels[r_]
        return g2 if (g2 is not None and not is_limb(name) and not name.startswith("row")) else g1

    names = LEADS if all((nm in LEADS or nm == "-aVR") for row in rows_spec for nm in row) else \
        [nm for row in rows_spec for nm in row]

    # ------------------------------------------------------- per-segment extraction
    meas = {l: [] for l in names}  # list of arrays (N) with NaN
    wts = {l: [] for l in names}   # matching confidence weights
    # reference ink level of the trace (for the bridge between segments in the Viterbi)
    _bref = None
    if cfg.bridge:
        vb = D[B]
        _bref = float(np.percentile(vb, 75)) if vb.size > 100 else None

    def br_kw(xa_):
        kw_ = dict(D=D, xa=xa_, bridge_ref=_bref, bridge_floor=min(glev, 0.6 * _bref)) if _bref else {}
        kw_['rigid'] = cfg.rigid_edges
        return kw_

    traces_dbg = []
    jobs_dbg = []
    seg_jobs = []
    for r, (y, spec) in enumerate(zip(rows, rows_spec)):
        base0 = base_q[r]
        up = (y - rows[r - 1]) if r > 0 else spacing
        dn = (rows[r + 1] - y) if r + 1 < len(rows) else spacing
        # generous search band: large waves (R of 3-5 mV) invade the neighbouring row; the
        # first/last row may reach the border. Viterbi + baseline prior keep the right trace
        # even with the neighbours inside the band.
        ya = 0 if r == 0 else max(0, int(y - cfg.band_sp * up))
        yb = H if r == len(rows) - 1 else min(H, int(y + cfg.band_sp * dn))
        k = len(spec)
        for c, lead in enumerate(spec):
            n0, n1 = c * N // k, (c + 1) * N // k
            xa = int(np.floor(x0_r[r] + sc_r[r] * n0 / cfg.fs * speed * R)) - 1
            xb = int(np.ceil(x0_r[r] + sc_r[r] * (n1 - 1) / cfg.fs * speed * R)) + 2
            xa, xb = max(0, xa), min(W, xb)
            if xb - xa < 10:
                continue
            runs = column_runs(B, D, ya, yb, xa, xb)
            _drop_label_runs(runs, xa, label_boxes)
            others = [float(v) for i2, v in enumerate(rows) if i2 != r]
            # in the final extraction, edge continuity (label letters stuck to the trace); the
            # quick traces used for the layout stay as before
            sel = viterbi_track(runs, base0, R, others=others, lw=lw0, w_edge=0.3, **br_kw(xa))
            # physiological constraint: the ECG returns to the baseline every beat. A path that
            # spends > 0.8 s entirely more than 4 mm from it followed something else (sheet
            # border, text ruler, neighbouring trace): those segments go and tracking is redone
            far_len = int(0.8 * speed * R)
            cen = np.array([runs[i][k][2] if k is not None else np.nan for i, k in enumerate(sel)])
            far = np.abs(cen - base0) > 4 * R
            st_, en_ = _runs_bool(far)
            # ... and is FLAT (no QRS): a lead with a real baseline offset still has complexes;
            # a sheet border or ruler does not
            win_ = max(10, int(1.0 * speed * R))

            def _flat(a_, b_):  # in >= 75% of the 1 s windows the local excursion stays < 2.5 mm
                fl = []
                for w0 in range(a_, b_, win_):
                    seg = cen[w0:min(b_, w0 + win_)]
                    seg = seg[np.isfinite(seg)]
                    if len(seg) > 5:
                        fl.append(np.ptp(np.percentile(seg, [5, 95])) < 2.5 * R)
                return bool(fl) and np.mean(fl) >= 0.75
            bad = [(a_, b_) for a_, b_ in zip(st_, en_) if b_ - a_ >= far_len and _flat(a_, b_)]
            if bad:
                for a_, b_ in bad:
                    for i in range(a_, b_):
                        runs[i] = [rr_ for rr_ in runs[i] if abs(rr_[2] - base0) <= 4 * R]
                sel = viterbi_track(runs, base0, R, others=others, lw=lw0, w_edge=0.3, **br_kw(xa))
            seg_jobs.append(dict(lead=lead, row=r, n0=n0, n1=n1, xa=xa, runs=runs, sel=sel))

    # "full ink" level and trace radius (half thickness) on flat stretches
    pks = []
    for j in seg_jobs:
        tb, pk = selected_envelopes(D, j["runs"], j["sel"], j["xa"])
        h = tb[:, 1] - tb[:, 0]
        flat = np.abs(np.diff(tb[:, 0], prepend=np.nan)) + np.abs(np.diff(tb[:, 1], append=np.nan)) < 1
        pks.append(pk[flat & (h >= 1)])
    pks = np.concatenate(pks) if pks else np.array([0.8])
    dfull = float(np.nanpercentile(pks, 75)) if pks.size else 0.8
    th = []
    for j in seg_jobs:
        top, bot = envelope_edges(D, j["runs"], j["sel"], j["xa"], dfull)
        j["top"], j["bot"] = top, bot
        dt = np.abs(np.gradient(top)) + np.abs(np.gradient(bot))
        th.append((bot - top)[dt < 0.3])
    if not seg_jobs:
        raise RuntimeError("no lead segment to track")
    th = np.concatenate(th)
    for j in seg_jobs:
        _mask_label_edges(j, label_boxes)
    mark_occlusions(seg_jobs, float(np.median(th)) if th.size else lw0)
    th = th[~np.isnan(th)]
    rad = float(np.percentile(th, 20)) / 2 if th.size else lw0 / 2
    rad_h = horizontal_halfwidth(D, seg_jobs, 0.5 * dfull, rad) if cfg.pen == "measure" else (
        calibrate_pen([(j["top"], j["bot"]) for j in seg_jobs], rad) if cfg.pen == "calibrate" else rad)

    # confidence per column: the chosen segment is shared with the trace of ANOTHER row
    # (large waves that cross / merge) -> the shape there is ambiguous
    for j in seg_jobs:
        n_ = len(j["sel"])
        tb = np.full((n_, 2), np.nan)
        for i_, k_ in enumerate(j["sel"]):
            if k_ is not None:
                tb[i_] = j["runs"][i_][k_][0], j["runs"][i_][k_][1]
        j["tb"] = tb
        j["shared"] = np.zeros(n_, bool)
    for j in seg_jobs:
        for o in seg_jobs:
            if o is j or o["row"] == j["row"]:
                continue
            lo, hi = max(j["xa"], o["xa"]), min(j["xa"] + len(j["sel"]), o["xa"] + len(o["sel"]))
            if hi <= lo:
                continue
            a_ = j["tb"][lo - j["xa"]:hi - j["xa"]]; b_ = o["tb"][lo - o["xa"]:hi - o["xa"]]
            with np.errstate(invalid="ignore"):
                ov = (a_[:, 0] <= b_[:, 1]) & (b_[:, 0] <= a_[:, 1])
            j["shared"][lo - j["xa"]:hi - j["xa"]] |= ov
    for j in seg_jobs:  # dilate ~1 mm: the surroundings of a merge are suspicious too
        k_ = int(R) | 1
        j["shared"] = cv2.dilate(j["shared"].astype(np.uint8).reshape(1, -1), np.ones((1, k_), np.uint8)).ravel() > 0

    # final time origin with the measured 50% level and radius
    lev = 0.5 * dfull
    xl, xe, wk_ = _origin_cands(D, B, rows, ext, pulses, half, lev, R, rad)
    x0f = _vote_x0(xl, xe, rad, span_px, R, weak=wk_)
    if x0f is not None:
        x0 = float(x0f)
    xlr, _, wkr = _edge_rows(D, B, rows, ext, pulses, half, lev, R, rad, return_weak=True)
    x0_r, sc_r = _row_time_maps(rows, xlr, rad, x0, R, reg, weak=wkr,
                                both_sided=getattr(_vote_x0, "both_sided", True))

    # final baseline with the measured radius: 0 mV = centre of plateau 1 + level-1 gain
    for r_, p in enumerate(pulses_base):
        if p is not None:
            base_q[r_] = p["top"] - 0.5 + rad + levels[r_][0] * R

    # typical extent of the separator mark (left/right of the boundary) on this page
    ext_l, ext_r = [], []
    for j in seg_jobs:
        rr = j["row"]
        xs_ = np.arange(j["xa"], j["xa"] + len(j["top"]), dtype=np.float64)
        if j["n0"] > 0:   # start of a lead: how far the mark reaches after the boundary
            xb = x0_r[rr] + sc_r[rr] * j["n0"] / cfg.fs * speed * R
            sp_ = _tick_span(xs_, j["top"], j["bot"], xb, R)
            if sp_ is not None:
                ext_r.append(sp_[1] - xb)
        if j["n1"] < N:   # end of a lead: how early the mark starts before the boundary
            xb = x0_r[rr] + sc_r[rr] * (j["n1"] - 1) / cfg.fs * speed * R
            sp_ = _tick_span(xs_, j["top"], j["bot"], xb, R)
            if sp_ is not None:
                ext_l.append(xb - sp_[0])
    # 25th percentile: a QRS touching the boundary only widens the measure, never narrows it
    tick_l = float(np.percentile(ext_l, 25)) + 1.0 if len(ext_l) >= 3 else 0.0
    tick_r = float(np.percentile(ext_r, 25)) + 1.0 if len(ext_r) >= 3 else 0.0
    for j in seg_jobs:
        yc, valid = centerline_from_envelopes(j["top"], j["bot"], rad, rh=rad_h)
        if valid.sum() < 5:
            continue
        p = pulses_base[j["row"]]
        gain = seg_gain(j["row"], j["lead"])
        if p is not None:
            base = base_q[j["row"]]
        else:  # no pulse: mode (isoelectric) of the row
            vv = yc[valid]
            vv = vv[np.isfinite(vv)]
            if len(vv) < 5:
                continue
            hist, edges = np.histogram(vv, bins=np.arange(vv.min() - 1, vv.max() + 2, 1.0))
            hist = cv2.GaussianBlur(hist.astype(np.float32).reshape(-1, 1), (0, 0), 2).ravel()
            base = float(edges[int(np.argmax(hist))] + 0.5)
            j["base_from_mode"] = True
        xs = np.arange(j["xa"], j["xa"] + len(yc), dtype=np.float64)
        # segment edges: trace ends / separator marks / end of the neighbouring trace
        # -> drop ~radius+1 px and extrapolate linearly from the interior
        tr = np.ceil(rad_h + 1.0)
        rr = j["row"]
        xs_s = x0_r[rr] + sc_r[rr] * j["n0"] / cfg.fs * speed * R
        xs_e = x0_r[rr] + sc_r[rr] * (j["n1"] - 1) / cfg.fs * speed * R
        lo_cut = xs_s + tr if j["n0"] > 0 else -np.inf          # internal boundary on the left
        hi_cut = xs_e - tr if j["n1"] < N else np.inf           # internal boundary on the right
        if j["n0"] == 0 and pulses[rr] is not None:
            # start of the row: nothing before the end of the calibration pulse (the pulse's
            # falling edge and plateau are not signal)
            pe = pulses[rr].get("x_fall", pulses[rr].get("x_end"))
            if pe is not None:
                lo_cut = max(lo_cut, float(pe) + tr)
        # separator mark at the lead switch: vertical bar wider than the pen in scans and
        # photos (blur). The mark columns have a "tall" trace (top-bottom of several mm)
        # against the boundary; cut to its end, so it does not become a 0.3 mV step.
        # Separator mark: cut to the typical extent of the mark on this page (median over all
        # boundaries), so that a QRS touching the boundary is not cut with it
        if j["n0"] > 0:
            sp_ = _tick_span(xs, j["top"], j["bot"], xs_s, R)
            if sp_ is not None:
                lo_cut = max(lo_cut, xs_s + min(sp_[1] - xs_s, tick_r) + 1.0)
        if j["n1"] < N:
            sp_ = _tick_span(xs, j["top"], j["bot"], xs_e, R)
            if sp_ is not None:
                hi_cut = min(hi_cut, xs_e - min(xs_e - sp_[0], tick_l) - 1.0)
        good = valid & (xs > lo_cut) & (xs < hi_cut)
        if good.sum() < 10:
            good = valid
        xg, yg = xs[good], yc[good]
        if cfg.steep_subpx and len(xg) > 4:
            xg, yg = refine_steep(D, xg, yg, rad_h)
        if cfg.synth and len(xg) > 20:
            from .synthesis import synth_refine
            xg, yg = synth_refine(D, xg, yg, rad, max_shift=cfg.synth_max_shift, order=cfg.synth_order)
        if cfg.fit == "lsf" and len(xg) > 30:
            from .fit import lsf_refine
            xg, yg = lsf_refine(D, xg, yg, **(cfg.fit_kw or {}))
        tn = np.arange(j["n0"], j["n1"]) / cfg.fs
        xn = x0_r[rr] + sc_r[rr] * tn * speed * R
        yn = np.interp(xn, xg, yg)
        # limited linear extrapolation (damped slope), otherwise keep the value
        K = min(6, len(xg))
        a0 = np.polyfit(xg[:K], yg[:K], 1); a1 = np.polyfit(xg[-K:], yg[-K:], 1)
        yn = np.where(xn < xg[0], yg[0] + 0.5 * a0[0] * (xn - xg[0]), yn)
        yn = np.where(xn > xg[-1], yg[-1] + 0.5 * a1[0] * (xn - xg[-1]), yn)
        # extrapolate only what the separator mark hides (~1 pen radius); beyond that the
        # sample is unknown: an invented ramp spoils the first beat and Einthoven
        ext_max = 2.0 * tr
        yn = np.where((xn < xg[0] - ext_max) | (xn > xg[-1] + ext_max), np.nan, yn)
        # at a lead switch the stretch hidden by the mark is short (a few ms): the first/last
        # clean value, constant, is used (better than empty, which the metric reads as zero)
        # extrapolation with half the slope of the first clean samples (a QRS touching the
        # boundary continues under the mark, but the curve usually turns there)
        K4 = min(4, len(xg))
        # the ramp only counts near the last clean sample (2 mm); a trace lost far from the
        # boundary keeps the last value (a 0.5 s ramp would reach several mV)
        ramp = 2.0 * R if EXTRAP_CAP[0] else np.inf
        if j["n0"] > 0 and K4 >= 2:
            sl = np.polyfit(xg[:K4], yg[:K4], 1)[0]
            yn = np.where((xn < xg[0]) & (xn >= xs_s - 1.0),
                          yg[0] + 0.5 * sl * np.maximum(xn - xg[0], -ramp), yn)
        if j["n1"] < N and K4 >= 2:
            sl = np.polyfit(xg[-K4:], yg[-K4:], 1)[0]
            yn = np.where((xn > xg[-1]) & (xn <= xs_e + 1.0),
                          yg[-1] + 0.5 * sl * np.minimum(xn - xg[-1], ramp), yn)
        v = (base - yn) / (gain * R)
        if j["n0"] == 0:
            v = _cut_leading_pulse(v, cfg.fs)
        lead = j["lead"]
        if lead == "-aVR":  # Cabrera: inverted lead
            lead, v = "aVR", -v
        arr = np.full(N, np.nan, np.float32)
        arr[j["n0"]:j["n1"]] = v
        meas[lead].append(arr)
        w = np.full(N, np.nan, np.float32)
        sh = np.interp(xn, xs, j["shared"].astype(float)) > 0.5
        w[j["n0"]:j["n1"]] = np.where(sh, 0.05, 1.0)
        wts[lead].append(w)
        traces_dbg.append((xs, yc, valid))
        if return_debug:
            jobs_dbg.append(dict(lead=lead, row=rr, n0=j["n0"], n1=j["n1"], xn=xn, yn=(base - v * gain * R) if lead == j["lead"] else None,
                                 xg=xg, yg=yg, base=base, gain=gain, top=j["top"], bot=j["bot"], xa=j["xa"]))
    lw = 2 * rad

    sig = np.full((len(names), N), np.nan, np.float32)
    nmeas = np.zeros((len(names), N), np.float32)
    for li, l in enumerate(names):
        if meas[l]:
            st = np.stack(meas[l])
            wt = np.where(np.isnan(st), 0.0, np.nan_to_num(np.stack(wts[l]), nan=0.0))
            sw = wt.sum(0)
            sig[li] = np.where(sw > 0, np.nansum(st * wt, 0) / np.maximum(sw, 1e-9), np.nan)
            nmeas[li] = sw  # "effective number of measurements" (total weight) -> variance in the fusion
    sig_pre = sig.copy()   # before fusion: the Einthoven residual only exists here
    if cfg.einthoven and names is LEADS:
        sig = einthoven_fuse(sig, nmeas, fill=cfg.fill_limb)
    if BASELINE_GUARD[0]:
        sig = _baseline_guard(sig, cfg.fs)
    # quality score per lead (does not change the signal unless refuse_below asks for it)
    quality = None
    try:
        from .quality import assess
        quality = assess(sig, list(names), cfg.fs, prefusion=sig_pre if names is LEADS else None)
        n_rows = max(1, len(rows))
        page = dict(layout_confidence=(layout_info or {}).get("confidence"),
                    layout_warning=(layout_info or {}).get("warning"),
                    pulses_ratio=sum(p is not None for p in pulses) / n_rows)
        # layout confidence is the part that best separates good from bad pages (see quality.py)
        ps = 1.0
        if page["layout_confidence"] is not None:
            ps = float(np.clip(page["layout_confidence"], 0.05, 1.0))
        page["score"] = ps
        quality["page"] = page
        for q_ in quality["leads"].values():
            q_["score"] = float(q_["score"] * ps)
        quality["score"] = float(quality["score"] * ps)
        if cfg.refuse_below is not None:
            for li, l in enumerate(names):
                if quality["leads"][l]["score"] < cfg.refuse_below:
                    sig[li] = np.nan
                    quality["leads"][l]["flags"].append("recusada")
    except Exception as e:  # the score never brings the digitization down
        quality = dict(error=f"{type(e).__name__}: {e}")

    gains_seg = {f"{r_}:{nm}": seg_gain(r_, nm) for r_, row in enumerate(rows_spec[:len(levels)]) for nm in row}
    # rows the layout hypothesis left out (frame, ruler) do not appear in the result
    res = dict(signals=sig, fs=cfg.fs, leads=list(names), layout=layout, layout_rows=[r_ for r_ in rows_spec if r_],
               layout_info=layout_info, rotation=str(rot),
               gain_mm_per_mv=levels[0][0] if levels else None, gains=gains_seg, duration_s=duration,
               speed_mm_per_s=speed_nominal, speed_source=speed_src, duration_assumed=duration_assumed,
               horizontal_scale=speed / speed_nominal, ppmm=ginfo["ppmm"], angle=ginfo["angle"],
               grid_residual_mm=ginfo["residual_rms_mm"], grid_residual_cell_mm=ginfo.get("residual_cell_rms_mm"),
               grid_snap_mm=ginfo.get("major_snap_mm"), grid_fine_mm=ginfo.get("fine_snap_mm"), line_width_px=lw, line_width_h_px=2 * rad_h,
               rows_found=len(rows_all), pulses_found=sum(p is not None for p in pulses),
               pulses_right=sum(p is not None for p in rpulses), gain_rows_inferred=gain_flags,
               quality=quality, col_struct=(dict(cols=col_info["cols"], how=col_info.get("how")) if col_info else None))
    if return_debug:
        dbg = rect.copy()
        for xs, yi, valid in traces_dbg:
            pts = np.stack([xs, yi], 1).astype(np.int32)
            cv2.polylines(dbg, [pts], False, (255, 120, 0), 1, cv2.LINE_AA)
        for y in rows:
            cv2.line(dbg, (0, int(y)), (W, int(y)), (0, 160, 0), 1)
        cv2.line(dbg, (int(x0), 0), (int(x0), H), (0, 160, 0), 1)
        res["debug_image"] = dbg
        res["per_segment"] = {l: [m.copy() for m in v] for l, v in meas.items() if v}
        res["per_segment_weight"] = {l: [m.copy() for m in v] for l, v in wts.items() if v}
        res["internals"] = dict(x0=x0, x0_rows=list(map(float, x0_r)), scale_rows=list(map(float, sc_r)),
                                reg_a=list(map(float, reg[0])), reg_b=list(map(float, reg[1])),
                                xl=xl, xe=xe, ext=ext,
                                rows=list(map(int, rows)), pulses=pulses, rad=rad, rad_h=rad_h)
        res["rectified"] = rect
        res["jobs"] = jobs_dbg
        res["ink"] = D
        res["bin"] = B
    return res


def einthoven_fuse(sig, nmeas, fill=False):
    """Least-squares projection onto the exact linear constraints:
       I - II + III = 0   and   aVR + aVL + aVF = 0   (and, with fill, the Goldberger relations)."""
    s = sig.copy()
    I, II, III, aVR, aVL, aVF = range(6)
    var = 1.0 / np.maximum(nmeas, 0.02)  # occluded samples (low weight) absorb the correction

    def project(idx, coef):
        idx = list(idx)
        X = s[idx]
        ok = ~np.isnan(X).any(0)
        if not ok.any():
            return
        a = np.array(coef, np.float32)[:, None]
        v = var[idx][:, ok]
        e = (a * X[:, ok]).sum(0)
        den = (a * a * v).sum(0)
        for j, k in enumerate(idx):
            s[k, ok] = X[j, ok] - a[j, 0] * v[j] * e / den

    project([I, II, III], [1, -1, 1])
    project([aVR, aVL, aVF], [1, 1, 1])
    if fill:
        # complete using the Goldberger definitions when I and II exist, etc.
        has = ~np.isnan(s)
        m = has[I] & has[II]
        for k, f in ((III, lambda: s[II] - s[I]), (aVR, lambda: -(s[I] + s[II]) / 2),
                     (aVL, lambda: s[I] - s[II] / 2), (aVF, lambda: s[II] - s[I] / 2)):
            val = f()
            fillm = m & ~has[k]
            s[k, fillm] = val[fillm]
        has = ~np.isnan(s)
        m = has[aVR] & has[aVL] & ~has[I]
        Iv = (s[aVL] - s[aVR]) / 1.5
        IIv = -2 * s[aVR] - Iv
        s[I, m] = Iv[m]
        mm = m & ~has[II]
        s[II, mm] = IIv[mm]
        mm = m & ~has[III]
        s[III, mm] = (IIv - Iv)[mm]
    return s
