"""Per-column realignment: pages where each column of leads has its own baselines.

Some generators (and devices that draw each lead at its own DC level) do not align rows
across columns: in one column the three leads sit close together, in another a deep S wave
pushes the lower ones down. The "page rows" model reads that as extra or swapped rows.

The columns are found by blank vertical gaps, by equal-width column hypotheses, or by
baseline tracks that start and end at the column changes. Each (column, row) band is then
shifted vertically only, to a reference height per row (the median across columns, or a
stacked layout when the columns have different row spacings). Shifting changes neither
amplitude nor time: the rest of the program sees an ordinary page.
"""
from __future__ import annotations

import numpy as np
import cv2
from scipy.signal import find_peaks

ENABLED = [True]
LONG_EXTENT = [True]
ROW_DIST6 = [True]   # rows of one column >= 6 mm apart (was 8): tight aVL/aVF
STRUCT = [True]     # structure from baseline tracks when there are no blank gaps between columns
STRUCT_FIRST = [False]   # tracks before the equal-width column hypothesis
TRACKS_MANY_ROWS = [True]   # the tracks structure needs >= max(2, n/3) rows that change height


def _gaps(B, R, ya, yb):
    P = B[ya:yb].sum(0).astype(np.float64)
    ink = P > 0
    xs = np.nonzero(ink)[0]
    if len(xs) < 100 * R:
        return None, None
    L, Rr = int(xs[0]), int(xs[-1])
    if LONG_EXTENT[0]:
        # narrow islands (< 20 mm) at the ends, separated from the body by >= 3 mm with almost
        # no ink, are corner marks, dark grid dashes or text: they do not stretch the columns
        for _ in range(4):
            seg = P[L:Rr + 1]
            b0 = np.median(seg[seg > 0]) if (seg > 0).any() else 0
            d_ = np.diff(np.concatenate([[0], (seg <= max(1.0, 0.03 * b0)).astype(np.int8), [0]]))
            runs = [(a, b) for a, b in zip(np.nonzero(d_ == 1)[0], np.nonzero(d_ == -1)[0]) if b - a >= 3 * R]
            if not runs:
                break
            L0, R0 = L, Rr
            a, b = runs[-1]
            if (Rr - L + 1) - b < 20 * R and a >= 100 * R:
                Rr = L + a - 1
            a, b = runs[0]
            if a < 20 * R and Rr - (L + b) >= 100 * R:
                L = L + b
            if (L, Rr) == (L0, R0):
                break
    base = np.median(P[L:Rr][P[L:Rr] > 0]) if (P[L:Rr] > 0).any() else 0
    blank = P[L:Rr] <= max(1.0, 0.03 * base)
    d = np.diff(np.concatenate([[0], blank.astype(np.int8), [0]]))
    s_, e_ = np.nonzero(d == 1)[0], np.nonzero(d == -1)[0]
    gaps = [(L + a, L + b) for a, b in zip(s_, e_) if 2 * R <= b - a <= 15 * R]
    return gaps, (L, Rr)


def _equal_columns(gaps, L, Rr, R):
    if not gaps or len(gaps) > 5:
        return False
    edges = [L] + [x for g in gaps for x in g] + [Rr]
    widths = [edges[2 * i + 1] - edges[2 * i] for i in range(len(gaps) + 1)]
    if min(widths) < 40 * R:
        return False
    if len(gaps) >= 2:
        inner = widths[1:-1]
        w = float(np.median(inner))
        if max(inner) - min(inner) > 0.03 * w:
            return False
        return all(0.97 * w <= x <= w + 30 * R for x in (widths[0], widths[-1]))
    return max(widths) - min(widths) <= 0.03 * np.median(widths) + 30 * R and \
        min(widths) >= 0.9 * max(widths) - 30 * R


ROW_FILTER = [True]


def _keep_trace_rows(B, R, cols, pks, lr):
    """Peaks that are not trace (QR code, footer text: ink on < 50% of the column width inside
    the trace extent) are dropped before comparing row counts."""
    if not ROW_FILTER[0] or lr is None:
        return pks
    H = B.shape[0]
    out = []
    for (xa, xb), p in zip(cols, pks):
        a, b = max(xa, lr[0]), min(xb, lr[1] + 1)
        if b - a < 20 * R:
            out.append(p)
            continue
        sub = B[:, a:b]
        out.append(np.array([y for y in p if sub[max(0, int(y - 3 * R)):min(H, int(y + 3 * R))].any(0).mean() >= 0.5], int))
    return out


def _col_rows(B, R, xa, xb):
    k = int(2 * R) | 1
    Bd = cv2.dilate(B[:, xa:xb].astype(np.uint8), np.ones((k, 1), np.uint8))
    cov = Bd.mean(1).astype(np.float64)
    cov = cv2.GaussianBlur(cov.reshape(-1, 1), (0, 0), R).ravel()
    if cov.max() <= 0:
        return np.array([], int)
    pk, _ = find_peaks(cov, distance=int((6 if ROW_DIST6[0] else 8) * R), height=0.35 * cov.max(), prominence=0.15 * cov.max())
    return pk


EQUAL_SPLIT = [True]
# columns with rows at their own heights (each column with its own spacing): the
# (column, row) bands move to stacked new rows instead of the median across columns
RELAYOUT = [True]


def _rows_are_traces(B, R, cols, pks, lr):
    """Every row of every column is trace (ink near the height over most of the column width,
    inside the trace extent; >= 70%): a header or footer in a single column does not pass."""
    H = B.shape[0]
    for (xa, xb), p in zip(cols, pks):
        a, b = max(xa, lr[0]), min(xb, lr[1] + 1)
        if b - a < 20 * R:
            return False
        sub = B[:, a:b]
        a, b = 0, b - a
        for y in p:
            band = sub[max(0, int(y - 3 * R)):min(H, int(y + 3 * R)), a:b]
            if band.any(0).mean() < 0.7:
                return False
    return True


def _equal_split(B, R, ya, yb, lr):
    L, Rr = lr
    P = B[ya:yb].sum(0).astype(np.float64)
    for k in (4, 2):
        w = (Rr - L) / k
        if w < 40 * R:
            continue
        cuts = [0]
        for i in range(1, k):
            x = L + i * w
            a, b = int(x - 4 * R), int(x + 4 * R)
            cuts.append(a + int(np.argmin(P[a:b])))
        cuts.append(B.shape[1])
        cols = [(cuts[i], cuts[i + 1]) for i in range(k)]
        pks = _keep_trace_rows(B, R, cols, [_col_rows(B, R, a, b) for a, b in cols], lr)
        n = len(pks[0])
        if n >= 2 and all(len(p) == n for p in pks):
            P_ = np.array(pks, float)
            dl = np.median(P_, 0)[None, :] - P_
            sp = float(np.median(np.diff(np.median(P_, 0)))) if n > 1 else 0.0
            # with no gap confirming the columns, the shift has to be a block shift: some
            # whole column off (median >= 3 mm), no ambiguous matching (> half a row: a header
            # in one column and a footer in another swap the row indices) and >= 5 mm
            if sp > 0 and np.abs(dl).max() >= 5 * R and np.abs(dl).max() <= 0.5 * sp and \
                    np.median(np.abs(dl), 1).max() >= 3 * R:
                return cols
            if RELAYOUT[0] and sp > 0 and np.abs(dl).max() > 0.5 * sp and _rows_are_traces(B, R, cols, pks, lr):
                return cols
    return None


def _win_peaks(B, R, xa, xb, ya, yb, dist_mm=3.5, with_height=False):
    """Baselines in a narrow window: peaks of the horizontal coverage (trace dilated 2 mm)."""
    k = int(2 * R) | 1
    Bd = cv2.dilate(B[ya:yb, xa:xb].astype(np.uint8), np.ones((k, 1), np.uint8))
    cov = cv2.GaussianBlur(Bd.mean(1).astype(np.float64).reshape(-1, 1), (0, 0), 0.6 * R).ravel()
    if cov.max() <= 0.05:
        return np.array([], int)
    pk, pr = find_peaks(cov, distance=int(dist_mm * R), height=max(0.3 * cov.max(), 0.25), prominence=0.12 * cov.max())
    if with_height:
        return pk + ya, pr["peak_heights"]
    return pk + ya


def col_struct(B, R, win_mm=20.0, step_mm=5.0, link_mm=1.5, min_len_mm=30.0):
    """Column structure without relying on blank gaps.

    The page is swept in 20 mm windows; in each one, the peaks of the horizontal coverage are
    the local baselines. Peaks that repeat at the same height from window to window form a
    track (a lead stretch: height, start, end). On an ordinary page the tracks cross the whole
    width. When each column has its own heights, tracks start and end at the column changes:
    the points where several tracks end and others start are the boundaries, and each column
    keeps its own rows.

    Returns (cols, P) with cols = [(xa, xb), ...] and P (columns, rows) = heights, or None
    when the page is ordinary (or the structure is not clear)."""
    H, W = B.shape
    prof = B.mean(1)
    ys = np.nonzero(prof > 0.05 * prof.max())[0] if prof.max() > 0 else []
    xsum = B.sum(0)
    xs = np.nonzero(xsum > 0)[0]
    if len(ys) < 10 or len(xs) < 100 * R:
        return None
    # 5 mm slack: the first/last baseline must not fall on the band edge (a peak on the edge
    # is not a peak for find_peaks)
    ya, yb = max(0, int(ys[0] - 5 * R)), min(H, int(ys[-1]) + 1 + int(5 * R))
    L, Rr = int(xs[0]), int(xs[-1]) + 1
    win, step, tol = int(win_mm * R), int(step_mm * R), link_mm * R
    x0s = list(range(L, max(L + 1, Rr - win + 1), step))
    if len(x0s) < 8:
        return None
    peaks = [_win_peaks(B, R, x, x + win, ya, yb) for x in x0s]
    # tracks: chain peaks of consecutive windows (tolerates 1 window without the peak)
    tracks = []   # dict(ys=[...], j0, j1)
    open_ = []
    for j, pk in enumerate(peaks):
        used = set()
        for t in open_:
            if j - t["j1"] > 2:
                continue
            y_ = float(np.median(t["ys"][-4:]))
            cand = [(abs(float(v) - y_), i) for i, v in enumerate(pk) if i not in used and abs(float(v) - y_) <= tol]
            if cand:
                _, i = min(cand)
                used.add(i); t["ys"].append(float(pk[i])); t["j1"] = j
        open_ = [t for t in open_ if j - t["j1"] <= 2]
        for i, v in enumerate(pk):
            if i not in used:
                t = dict(ys=[float(v)], j0=j, j1=j)
                tracks.append(t); open_.append(t)
    nmin = max(3, int(round((min_len_mm - win_mm) / step_mm)) + 1)
    tr = [dict(y=float(np.median(t["ys"])), xa=x0s[t["j0"]], xb=x0s[t["j1"]] + win, n=t["j1"] - t["j0"] + 1)
          for t in tracks if t["j1"] - t["j0"] + 1 >= nmin]
    # only tracks of at least 45 mm define the structure (footer, QR and S/T plateaus give
    # short tracks)
    tr = [t for t in tr if t["xb"] - t["xa"] >= 45 * R]
    if len(tr) < 4:
        return None
    # trace extent from the tracks themselves (loose ink in the margins does not count)
    L, Rr = min(t["xa"] for t in tr), max(t["xb"] for t in tr)
    span = Rr - L
    full = [t for t in tr if t["xb"] - t["xa"] >= 0.85 * span]
    part = [t for t in tr if t["xb"] - t["xa"] < 0.85 * span]
    if len(part) < 4 or len(part) < len(full):
        return None        # ordinary page: the rows cross the whole width
    # boundaries: interior points where partial tracks end/start, grouped
    ev = sorted([(t["xb"] - win / 2, 0, t["y"]) for t in part if t["xb"] < Rr - 0.08 * span] +
                [(t["xa"] + win / 2, 1, t["y"]) for t in part if t["xa"] > L + 0.08 * span])
    if not ev:
        return None
    groups = [[ev[0]]]
    for e in ev[1:]:
        if e[0] - groups[-1][-1][0] <= 0.75 * win:
            groups[-1].append(e)
        else:
            groups.append([e])
    # a boundary has a track ending AND a track starting
    def _same(g):
        """Tracks that end at the boundary and restart at the SAME height (± 2 mm) on the other
        side: the row was only interrupted (pulse, label), it did not change height."""
        e0 = [y for _, k, y in g if k == 0]; e1 = [y for _, k, y in g if k == 1]
        return sum(1 for y in e0 if e1 and min(abs(y - z) for z in e1) <= 2 * R), min(len(e0), len(e1))
    cand = [(float(np.median([e[0] for e in g])), len(g), _same(g)) for g in groups
            if len(g) >= 2 and any(e[1] == 0 for e in g) and any(e[1] == 1 for e in g)]
    if not cand or len(cand) > 5:
        return None
    # the columns of an ECG have the same width: keep the boundary set (1 or 3: two or four
    # columns) that splits the extent into equal parts (±20%) with the most tracks changing.
    # A track that changes height in the middle of a lead (ST, wide T wave) gives a misplaced
    # "boundary", and cutting there slices the trace
    from itertools import combinations
    best = None
    for k in (1, 3):
        for sub in combinations(cand, k):
            ed = [L] + [z[0] for z in sub] + [Rr]
            wd = np.diff(ed)
            if wd.min() < 35 * R or np.abs(wd - wd.mean()).max() > 0.2 * wd.mean():
                continue
            key = (sum(z[1] for z in sub), k)
            if best is None or key > best[0]:
                best = (key, [z[0] for z in sub], max(z[1] for z in sub),
                        sum(z[2][0] for z in sub), sum(z[2][1] for z in sub))
    if best is None:
        return None
    cuts = best[1]
    # ordinary page with the row interrupted at the column change: half or more of the rows
    # continue at the same height on the other side -> no per-column heights
    if best[3] >= 0.5 * max(best[4], 1):
        return None
    # rows crossing the page weigh against: the strongest boundary needs at least twice as
    # many events (an ordinary page with plateaus and a footer gives loose partial tracks)
    if best[2] < max(4, 2 * len(full)):
        return None
    edges = [L] + cuts + [Rr]
    y_lo = max(0, int(min(t["y"] for t in tr) - 3 * R)); y_hi = min(H, int(max(t["y"] for t in tr) + 3 * R))

    def rows_of(ca, cb, frac, n=None):
        m = 0.5 * (1.0 - frac) * (cb - ca)
        pk, hh = _win_peaks(B, R, int(ca + m), int(cb - m), y_lo, y_hi, with_height=True)
        if n is not None and len(pk) > n:
            # a long ST/S-wave plateau gives an extra peak: keep the n heights with most coverage
            pk = np.sort(pk[np.argsort(-hh)[:n]])
        return pk.astype(float)

    # rows of each column measured in its core (away from the boundaries, still imprecise)
    rows_c = [rows_of(edges[i], edges[i + 1], 0.5) for i in range(len(edges) - 1)]
    cnt = [len(r_) for r_ in rows_c]
    n = int(np.bincount(cnt).argmax())
    if n < 2 or min(cnt) < n or sum(c == n for c in cnt) < 0.5 * len(cnt):
        return None
    # only structures that exist: 2 columns of 3 to 9 rows (6x2 and rhythm variants) or
    # 4 columns of 3 to 6 (3x4 and variants). Anything else is a confusing trace, not columns
    K = len(cnt)
    if not ((K == 2 and 3 <= n <= 9) or (K == 4 and 3 <= n <= 6)):
        return None
    rows_c = [rows_of(edges[i], edges[i + 1], 0.5, n) for i in range(len(edges) - 1)]
    P = np.array(rows_c, float)
    if np.abs(P - np.median(P, 0)[None, :]).max() < 3 * R:
        return None        # same heights in every column: nothing to realign
    cols_c = [(edges[i], edges[i + 1]) for i in range(len(edges) - 1)]
    # fine cut of each boundary: the x where the ink moves from the heights of one column to
    # those of the other (sum, over the rows that change height, of ink in this band minus that one)
    fine = []
    for c in range(len(cols_c) - 1):
        xc = int(cuts[c]); lo, hi = max(0, int(xc - 1.5 * win)), min(W, int(xc + 1.5 * win))
        sc = np.zeros(hi - lo)
        hb = int(1.0 * R)
        for i in range(n):
            y1, y2 = int(P[c, i]), int(P[c + 1, i])
            if abs(y1 - y2) < 2.5 * R:
                continue
            a_ = B[max(0, y1 - hb):y1 + hb + 1, lo:hi].any(0).astype(float)
            b_ = B[max(0, y2 - hb):y2 + hb + 1, lo:hi].any(0).astype(float)
            sc += a_ - b_
        if not sc.any():
            fine.append(xc); continue
        # cut k maximizes sum(sc[:k]) - sum(sc[k:])
        cs = np.cumsum(sc)
        obj = 2 * np.r_[0.0, cs] - cs[-1]
        fine.append(lo + int(np.argmax(obj)))
    bounds = [0] + fine + [W]
    cols = [(int(bounds[i]), int(bounds[i + 1])) for i in range(len(bounds) - 1)]
    # final heights with the fine boundaries (70% core); if the count changes, keep the previous ones
    e2 = [L] + fine + [Rr]
    if np.diff(e2).min() >= 35 * R:
        r2 = [rows_of(e2[i], e2[i + 1], 0.7, n) for i in range(len(e2) - 1)]
        if all(len(r_) == n for r_ in r2):
            P = np.array(r2, float)
    return cols, P


def _fits_plain_rows(B, R, P):
    """The usual rows (projection of the whole page) are already within 40% of the row step
    of every column: no realignment needed."""
    from .extract import find_rows
    pk0 = np.asarray(find_rows(B, R)[0], float)
    if len(pk0) != P.shape[1] or len(pk0) < 2:
        return False
    sp0 = float(np.median(np.diff(pk0)))
    return bool(np.abs(P - pk0[None, :]).max() <= 0.4 * sp0)


def col_align(rect, pap, D, B, R):
    """Return realigned (rect, pap, D, B, info), or the inputs and None.

    Three ways to find the columns, in this order: blank gaps over the full height ("gaps"),
    the hypothesis of 4 or 2 equal-width columns ("equal") and baseline tracks that start and
    end at the column changes ("tracks", `col_struct`). STRUCT_FIRST puts the tracks before
    the equal widths."""
    if not ENABLED[0]:
        return rect, pap, D, B, None
    H, W = B.shape
    prof = B.mean(1)
    ys = np.nonzero(prof > 0.05 * prof.max())[0]
    if len(ys) < 10:
        return rect, pap, D, B, None
    ya, yb = int(ys[0]), int(ys[-1]) + 1
    gaps, lr = _gaps(B, R, ya, yb)

    def _plain():
        """Gaps and equal widths: (cols, P, how) or None."""
        cols, how = None, "gaps"
        if gaps and _equal_columns(gaps, lr[0], lr[1], R):
            cuts = [0] + [int(0.5 * (a + b)) for a, b in gaps] + [W]
            cols = [(cuts[i], cuts[i + 1]) for i in range(len(cuts) - 1)]
            pks = _keep_trace_rows(B, R, cols, [_col_rows(B, R, a, b) for a, b in cols], lr)
            n = len(pks[0])
            if n < 2 or any(len(p) != n for p in pks):
                cols = None
        if cols is None and EQUAL_SPLIT[0] and lr is not None:
            # no gap over the full height (trace joining across columns, rhythm strip below):
            # hypotheses of equal-width columns (4 and 2), cut at the ink minimum near the
            # nominal position; valid if every column has the same number of rows
            cols = _equal_split(B, R, ya, yb, lr)
            how = "equal"
            if cols is None:
                return None
            pks = _keep_trace_rows(B, R, cols, [_col_rows(B, R, a, b) for a, b in cols], lr)
        if cols is None:
            return None
        P = np.array(pks, float)              # (columns, rows)
        ref = np.median(P, 0)
        dl = ref[None, :] - P
        sp = float(np.median(np.diff(ref)))
        if np.abs(dl).max() < 3 * R or sp < 8 * R:
            return None
        if how == "equal" and _fits_plain_rows(B, R, P):
            return None
        if RELAYOUT[0] and np.abs(dl).max() > 0.5 * sp and len(ref) >= 2 and _rows_are_traces(B, R, cols, pks, lr):
            # each column with its own spacing: the bands go to new stacked heights
            # (as with tracks), instead of the median across columns
            return cols, P, how + "+stacked"
        if np.abs(dl).max() > 1.5 * sp:       # shift larger than one row: doubtful matching
            return None
        return cols, P, how

    def _tracks():
        """No blank gaps (dark grid, label or QRS in the gap): structure from the tracks."""
        if not STRUCT[0]:
            return None
        st = col_struct(B, R)
        if st is None:
            return None
        cols, P = st
        if TRACKS_MANY_ROWS[0]:
            # own heights per column means several rows move: one row off (a header text line
            # in one column and another one in the next) is not a column structure
            dev = np.abs(P - np.median(P, 0)[None, :]).max(0)
            if int((dev >= 2.5 * R).sum()) < max(2, int(np.ceil(P.shape[1] / 3))):
                return None
        # small difference between columns (< 40% of the step) and the usual rows already come
        # out right: extraction copes alone (wide band + per-stretch baseline), and realigning
        # would only risk slicing a large QRS at the band seams
        if _fits_plain_rows(B, R, P):
            return None
        return cols, P, "tracks"

    got = None
    for f in ((_tracks, _plain) if STRUCT_FIRST[0] else (_plain, _tracks)):
        got = f()
        if got is not None:
            break
    if got is None:
        return rect, pap, D, B, None
    cols, P, how = got
    n = P.shape[1]
    if how == "gaps" or how == "equal":
        ref = np.median(P, 0)
        H2 = H
    else:
        # reference heights: the first row stays where it is (median) and each interval is the
        # LARGEST across columns, so no band overlaps its neighbour after the shift (in a
        # "tight" column rows sit 4-10 mm apart; in the other, 25-35 mm)
        gaps_ = np.max(np.diff(P, axis=1), axis=0)
        ref = np.concatenate([[float(np.median(P[:, 0]))], float(np.median(P[:, 0])) + np.cumsum(gaps_)])
        H2 = int(max(H, ref[-1] + np.max(H - P[:, -1]) + 1))
    dl = ref[None, :] - P
    rect2 = np.full((H2,) + rect.shape[1:], 255, rect.dtype); pap2 = np.zeros((H2, W), pap.dtype)
    D2 = np.zeros((H2, W), D.dtype); B2 = np.zeros((H2, W), B.dtype)
    for c, (xa, xb) in enumerate(cols):
        p = P[c]
        bounds = [0] + [int(0.5 * (p[i] + p[i + 1])) for i in range(n - 1)] + [H]
        for i in range(n):
            s0, s1 = bounds[i], bounds[i + 1]
            d = int(round(dl[c, i]))
            t0, t1 = s0 + d, s1 + d
            # clip to the frame
            o0, o1 = max(0, t0), min(H2, t1)
            if o1 <= o0:
                continue
            a0, a1 = o0 - d, o1 - d
            sl_t, sl_s = np.s_[o0:o1, xa:xb], np.s_[a0:a1, xa:xb]
            B2[sl_t] |= B[sl_s]
            D2[sl_t] = np.maximum(D2[sl_t], D[sl_s])
            rect2[sl_t] = np.minimum(rect2[sl_t], rect[sl_s])
            pap2[sl_t] = np.maximum(pap2[sl_t], pap[sl_s])
    return rect2, pap2, D2, B2, dict(cols=cols, peaks=P.tolist(), ref=ref.tolist(), how=how)
