"""
Signal extraction on the rectified image (R px/mm, major lines at multiples of 5R).

1. Ink map: per-channel illumination normalization + grid removal by a transmittance
   model (ink over grid = T_grid * T_ink). T_grid comes from (a) a periodic median over the
   phase mod 5 mm and (b) a residual per straight row/column (80th percentile), which covers
   grids of any colour and irregular spacings.
2. Layout: baselines from the horizontal-coverage peaks filtered by "looking like a trace"
   (long components); calibration pulses (gain and 0 mV).
3. Tracking: Viterbi over the ink segments of each column (continuity + baseline prior +
   fragment penalty + failure state).
4. Sub-pixel centre line by morphological inversion: the printed trace is the curve dilated
   by an elliptical pen (rh, rv). Top edge = erosion(y), bottom = dilation(y); opening and
   closing recover y, exact at peaks/valleys. The horizontal pen radius is self-calibrated
   (opening == closing on straight stretches).
5. Occlusions: edges that fall inside the ink of a flat neighbouring trace become NaN.
"""
from __future__ import annotations

import numpy as np
import cv2
from scipy.signal import find_peaks
from skimage.filters import apply_hysteresis_threshold, threshold_otsu


# ------------------------------------------------------------------- ink map

def _bg(ch, R):
    H, W = ch.shape
    f = 2.0 / R  # 2 px/mm
    s = cv2.resize(ch, (max(8, int(W * f)), max(8, int(H * f))), interpolation=cv2.INTER_AREA)
    s = cv2.dilate(s, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))  # ~4.5 mm
    s = cv2.medianBlur(s.astype(np.float32), 5)
    s = cv2.GaussianBlur(s, (0, 0), 3)
    return cv2.resize(s, (W, H), interpolation=cv2.INTER_LINEAR)


TEMPLATE_MM = [10.0]
TEMPLATE_AUTO = [True]
TEMPLATE_THIN = [True]
TEMPLATE_ERODE = [0]
GLEV_RELATIVE = [False]
# coloured trace on an achromatic grid (blue pen on a black/grey grid): the brightest channel
# makes the trace vanish; the chroma (max - min channel) shows it and drops the grid
CHROMA_TRACE = [True]
CHROMA_TH = [0.25]   # chroma above the grid's that counts as coloured ink (photos desaturate the pen)


def _median5_cells(cells: np.ndarray) -> np.ndarray:
    """5x5 median across neighbouring cells, phase by phase: the same as
    scipy.ndimage.median_filter(cells, size=(5, 5, 1, 1), mode="nearest"), but with the OpenCV
    5x5 median on each of the P*P small images (ny, nx); it used to take ~2.5 s per page."""
    ny, nx, P1, P2 = cells.shape
    a = np.ascontiguousarray(cells.reshape(ny, nx, P1 * P2).transpose(2, 0, 1), dtype=np.float32)
    out = np.empty_like(a)
    for k in range(a.shape[0]):
        out[k] = cv2.medianBlur(a[k], 5)
    return out.transpose(1, 2, 0).reshape(ny, nx, P1, P2)


def _chroma_trace(n, t, paper, R):
    """Transmittance from the chroma when the trace is coloured and the grid achromatic.

    The grid is what repeats every 5 mm (median over the cells of each phase); the trace is
    what does not. If the periodic part has no chroma, the grid is visible in the brightness,
    and non-periodic chromatic ink reaches most columns of the page, the trace is the coloured
    pen: t = 1 - chroma (grid and black text vanish)."""
    P = int(round(5.0 * R))
    H, W = t.shape
    ny, nx = H // P, W // P
    if ny < 4 or nx < 4:
        return t
    C = np.clip(np.maximum(np.maximum(n[0], n[1]), n[2]) - np.minimum(np.minimum(n[0], n[1]), n[2]), 0, 1)
    pm = paper[: ny * P, : nx * P].reshape(ny, P, nx, P).min(axis=(1, 3)) > 0
    if pm.sum() < 10:
        return t

    def tiles(X):
        return X[: ny * P, : nx * P].reshape(ny, P, nx, P).transpose(0, 2, 1, 3)[pm]
    Ct = np.median(tiles(C), axis=0)
    Tt = np.median(tiles(t), axis=0)
    if float(Ct.max()) >= 0.1 or float(Tt.min()) > 0.7:
        return t   # coloured grid (it already vanishes in the brightest channel) or no visible grid
    # the trace runs across the page: its chromatic ink covers most of the columns (coloured
    # labels or stamps cover a few)
    reps = (int(np.ceil(H / P)), int(np.ceil(W / P)))
    M = ((C - np.tile(Ct, reps)[:H, :W]) > CHROMA_TH[0]) & (paper > 0)
    xs = paper.any(0)
    if not xs.any() or float(np.mean(M.any(0)[xs])) < 0.5:
        return t
    return np.clip(1.0 - C, 0.0, 1.2).astype(np.float32)


def ink_map(rect: np.ndarray, paper: np.ndarray, R: float, return_grid_level: bool = False, tmin: float | None = None):
    f = rect.astype(np.float32)
    n = [f[..., c] / (_bg(f[..., c], R) + 1e-3) for c in range(3)]
    t = np.clip(np.maximum(np.maximum(n[0], n[1]), n[2]), 0, 1.2)  # transmittance in the brightest channel
    if CHROMA_TRACE[0]:
        t = _chroma_trace(n, t, paper, R)
    # 10 mm template: covers the 5 mm periodicity and also grids with an extra line every
    # 10 mm (5 mm templates left that line as "ink" and created false rows)
    H, W = t.shape

    def _tmpl(P):
        ny, nx = H // P, W // P
        blk = t[: ny * P, : nx * P].reshape(ny, P, nx, P)
        pm = paper[: ny * P, : nx * P].reshape(ny, P, nx, P).min(axis=(1, 3)) > 0
        tiles = blk.transpose(0, 2, 1, 3)[pm]  # (k, P, P)
        tm = np.median(tiles, axis=0) if len(tiles) > 10 else np.ones((P, P), np.float32)
        return ny, nx, blk, pm, tm

    P = int(round(TEMPLATE_MM[0] * R))
    ny, nx, blk, pm, tmpl = _tmpl(P)
    if TEMPLATE_AUTO[0] and TEMPLATE_MM[0] == 10.0:
        # the 10 mm template only counts when the grid really has a 10 mm structure (extra
        # line); otherwise the 5 mm one, with twice the cells and a 25 mm local neighbourhood, is better
        P5 = int(round(5.0 * R))
        r5 = _tmpl(P5)
        if P == 2 * P5:
            d = np.abs(tmpl - np.tile(r5[4], (2, 2)))
            # (a thin 10 mm line darker than the 5 mm one covers ~1% of the tile and the 95th
            # percentile misses it: compare the darkest line of each template)
            thin10 = TEMPLATE_THIN[0] and float(tmpl.min()) < float(r5[4].min()) - 0.08
            if float(np.percentile(d, 95)) < 0.08 and not thin10:
                P = P5
                ny, nx, blk, pm, tmpl = r5
    # dark grid (B&W scan, photocopy): the major lines are as black as the trace. The template
    # has to reach down to them (otherwise the grid line becomes continuous "ink" that
    # stretches the rows to the sheet border and creates false rows)
    dark_grid = float(tmpl.min()) < 0.15
    ink_map.tmpl_min = float(tmpl.min())
    ink_map.dark_grid = dark_grid
    if tmin is None:
        tmin = 0.05 if dark_grid else 0.3
        if TMIN_ADAPT[0] and not dark_grid:
            # dark but not black grid (darkened photo): lines at 0.15-0.3 transmittance would
            # pass the 0.3 floor and become "ink" over the whole page
            tmin = float(np.clip(0.85 * tmpl.min(), 0.05, 0.3))
    tmpl = np.clip(tmpl, tmin, 1.0)
    reps = (int(np.ceil(H / P)), int(np.ceil(W / P)))
    T = np.tile(tmpl, reps)[:H, :W]
    # LOCAL template: median per phase (mod 5 mm) over a neighbourhood of 5x5 cells (25 mm).
    # In photos the sharpness/darkness of the grid varies across the sheet; the global
    # template lets the grid "leak" as ink where it is darker than average.
    if ny >= 5 and nx >= 5:
        cells = blk.transpose(0, 2, 1, 3).copy()               # (ny, nx, P, P)
        cells[~pm] = tmpl                                        # off the paper: global
        loc = _median5_cells(cells)
        loc = np.clip(loc, tmin, 1.0)
        TL = np.ones((reps[0] * P, reps[1] * P), np.float32)
        TL[: ny * P, : nx * P] = loc.transpose(0, 2, 1, 3).reshape(ny * P, nx * P)
        TL = TL[:H, :W]
        # leftover edges (fraction of a cell) keep the global one
        TL[ny * P:, :] = T[ny * P:, :]; TL[:, nx * P:] = T[:, nx * P:]
        T = TL
    # residual per row/column: straight grid lines the periodic template does not explain
    # (irregular grids, extra-dark line every 25 mm...).
    t1 = t / T
    tt = np.where(paper > 0, np.minimum(t1, 1.0), np.nan)
    with np.errstate(all="ignore"):
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # 80th percentile: the trace (even on a long flat baseline) covers far less than
            # 80% of a grid line; a real grid line covers ~100%
            lv = np.nan_to_num(np.nanpercentile(tt, 80, axis=0), nan=1.0)
            lh = np.nan_to_num(np.nanpercentile(tt, 80, axis=1), nan=1.0)
    lv = np.where(lv < 0.92, lv, 1.0); lh = np.where(lh < 0.92, lh, 1.0)
    T = T * np.clip(lh[:, None] * lv[None, :], 0.2, 1.0)
    if dark_grid:
        # dark grid (B&W scan): the template must cover the small position error of the
        # lines (±1 px) or their edges leak as ink
        T = cv2.erode(T.astype(np.float32), np.ones((3, 3), np.uint8))
    elif TEMPLATE_ERODE[0] > 0:
        # photos: the lines wander by a fraction of a mm around the template (lens, paper):
        # the template covers that error, or the line edges leak as ink and raise the threshold
        k_ = 2 * TEMPLATE_ERODE[0] + 1
        T = cv2.erode(T.astype(np.float32), np.ones((k_, k_), np.uint8))
    D = 1.0 - np.clip(t / T, 0, 1)
    pe = cv2.erode(paper.astype(np.uint8), np.ones((int(2 * R), int(2 * R)), np.uint8))
    D[pe == 0] = 0
    D = D.astype(np.float32)
    if return_grid_level:
        # residual grid level: apparent ink on grid-line pixels (the trace occupies a tiny
        # fraction of them, so the 95th percentile measures the grid leakage)
        # relative "strong ink": a blurred/light trace sits at 0.2-0.4 and, if not excluded,
        # its halo becomes "grid leakage" and the threshold rises until it erases the trace itself
        vv = D[D > 0.08]
        s_th = 0.55 if not GLEV_RELATIVE[0] or vv.size < 100 else min(0.55, max(0.15, 0.5 * float(np.percentile(vv, 90))))
        kd = 7 if not GLEV_RELATIVE[0] else max(7, int(1.0 * R) | 1)
        strong = cv2.dilate((D > s_th).astype(np.uint8), np.ones((kd, kd), np.uint8)) > 0
        gp = (T < 0.9) & (pe > 0) & ~strong
        glev = float(np.percentile(D[gp], 95)) if gp.sum() > 1000 else 0.0
        return D, glev
    return D


def binarize(D: np.ndarray, R: float = 10.0, grid_level: float = 0.0, scale: float = 1.0) -> np.ndarray:
    # Otsu on thin structures only: bars, stamps and black text blocks (> 2.5 mm thick)
    # would pull the threshold up and erase light/thin traces
    kb = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (int(2.5 * R) | 1,) * 2)
    blob = cv2.morphologyEx((D > 0.25).astype(np.uint8), cv2.MORPH_OPEN, kb)
    blob = cv2.dilate(blob, np.ones((int(0.8 * R) | 1,) * 2, np.uint8)) > 0
    v = D[(D > 0.08) & ~blob]
    if v.size < 100:
        B = D > 0.4
    else:
        floor = 0.2
        if OTSU_FLOOR_REL[0]:
            # low-contrast page (high gamma, weak toner): the darkest trace does not exceed
            # ~0.2; the fixed 0.2 floor would erase the whole trace. The floor becomes
            # relative to the trace's own contrast (p99), always above the grid leakage.
            floor = min(0.2, max(grid_level + 0.05, 0.35 * float(np.percentile(v, 99))))
        th = float(np.clip(threshold_otsu(v), floor, 0.6))
        if scale != 1.0:
            # very light trace (low opacity, weak toner): Otsu is decided by black text and QR;
            # try lower thresholds, always above the grid leakage
            th = max(grid_level + 0.04, 0.05, th * scale)
        # low hysteresis threshold above the grid leakage (grid lines touch the trace and
        # would be "pulled" along through connectivity)
        lo = max(0.55 * th, min(grid_level + 0.08, 0.85 * th))
        Dn = _local_contrast(D, R, blob, grid_level, lo) if LOCAL_CONTRAST[0] and scale == 1.0 else D
        B = apply_hysteresis_threshold(Dn, lo, max(th, lo + 0.05))
    # leftover straight grid lines (thick black grid, template 1 px off): pixel rows/columns
    # that are almost all "ink" across the page, several, at a regular step. A trace does
    # not cover 70% of a pixel row over the whole page.
    if GRID_LINE_CLEAN[0]:
        B = _clean_grid_lines(B, R)
    if LONG_VLINES[0]:
        B = _remove_long_vertical(B, R)
    # salt-and-pepper removal: tiny components (< ~0.4 mm²) are not trace
    n, lab, st, _ = cv2.connectedComponentsWithStats(B.astype(np.uint8), connectivity=8)
    small = st[:, cv2.CC_STAT_AREA] < max(4, int(0.4 * R * R))
    small[0] = False
    if small.any():
        B = B & ~small[lab]
    return B


GRID_LINE_CLEAN = [True]
LONG_VLINES = [True]
LONG_VLINES_EDGE = [True]   # only lines with trace on one side (border) are removed; QRS chains stay
LONG_VLINES_SIDE = [0.15]
LOCAL_CONTRAST = [True]


def _local_contrast(D, R, blob, glev, lo, tile_mm=50.0):
    """Trace contrast falling across the page (glare, grazing light in a photo): the global
    threshold cuts the trace in the faded part into pieces. Trace level per ~5 cm block (high
    ink percentile); where it is below the page level, the ink is boosted up to the page
    level, at most until the grid leakage touches the low threshold."""
    H, W = D.shape
    T = max(16, int(tile_mm * R))
    ny, nx = max(1, H // T), max(1, W // T)
    Dm = np.where(blob, 0, D)
    lev = np.zeros((ny, nx), np.float32)
    for i in range(ny):
        for j in range(nx):
            t = Dm[i * T:(i + 1) * T if i < ny - 1 else H, j * T:(j + 1) * T if j < nx - 1 else W]
            lev[i, j] = float(np.percentile(t, 99.7)) if t.size else 0.0
    valid = lev >= glev + 0.2
    if valid.sum() < 3:
        return D
    g = float(np.median(lev[valid]))
    if g / max(float(lev[valid].min()), 1e-6) < 1.25:
        return D
    fmax = min(1.6, (lo - 0.03) / max(glev, 0.05))
    if fmax <= 1.05:
        return D
    f = np.where(valid, np.clip(g / np.maximum(lev, 1e-6), 1.0, fmax), 1.0).astype(np.float32)
    F = cv2.resize(f, (W, H), interpolation=cv2.INTER_LINEAR)
    return np.minimum(D * F, 1.0)


# a straight, thin line (centre within 0.12 mm of a straight fit, median width <= 0.5 mm) is a
# ruled line even with trace on both sides; stacked QRS chains wiggle (0.15-0.25 mm) and are
# 0.6-1 mm wide (PM-ECG-ID scans 53, 64 vs PhysioNet 119693645-0005: 15.2 dB with the line removed)
LONG_VLINES_STRAIGHT = [True]


def _straight_thin(lab, V, i, st_i, R):
    x0, y0, w, h = (int(v) for v in st_i[:4])
    m = (lab[y0:y0 + h, x0:x0 + w] == i) & (V[y0:y0 + h, x0:x0 + w] > 0)
    rows = np.nonzero(m.any(1))[0]
    if len(rows) < 0.9 * h:
        return False
    mr = m[rows]
    cols = np.arange(w, dtype=np.float64)
    cnt = mr.sum(1)
    cen = (mr * cols).sum(1) / cnt
    first = mr.argmax(1); last = w - 1 - mr[:, ::-1].argmax(1)
    wid = last - first + 1
    p = np.polyfit(rows.astype(np.float64), cen, 1)
    res = float(np.std(cen - np.polyval(p, rows)))
    return res <= 0.12 * R and float(np.median(wid)) <= 0.5 * R


def _remove_long_vertical(B, R, min_mm=100.0):
    """(Almost) continuous vertical line >= 10 cm: border of the grid area, fold/shadow of the
    sheet, dark grid line left curved by the perspective correction. A deep S reaches 6-7 mV
    (60-70 mm at 10 mm/mV) and crosses neighbouring rows; a pulse is ~10 mm. Without this,
    the tracker climbs the border and the row starts at the grid border."""
    H, W = B.shape
    L = max(5, int(8 * R))
    V = cv2.morphologyEx(B.astype(np.uint8), cv2.MORPH_OPEN, np.ones((L, 1), np.uint8))
    if not V.any():
        return B
    # join pieces of the same curved line (lateral drift of 1-2 px per stretch)
    Vc = cv2.morphologyEx(V, cv2.MORPH_CLOSE, np.ones((int(2 * R) | 1, 3), np.uint8))
    n, lab, st, _ = cv2.connectedComponentsWithStats(Vc, connectivity=8)
    longc = np.zeros(n, bool)
    hh, ww = st[1:, cv2.CC_STAT_HEIGHT], st[1:, cv2.CC_STAT_WIDTH]
    cx = st[1:, cv2.CC_STAT_LEFT] + 0.5 * ww
    xs_ = np.nonzero(B.any(0))[0]
    x_lo, x_hi = (float(np.percentile(xs_, 0.5)), float(np.percentile(xs_, 99.5))) if len(xs_) else (0, W)
    # >= 10 cm anywhere; >= 45 mm only at the margins of the ink block (grid border): deep
    # QRS stay inside the rows
    margin = (cx <= x_lo + 15 * R) | (cx >= x_hi - 15 * R)
    longc[1:] = (ww <= 6 * R) & ((hh >= min_mm * R) | ((hh >= 45 * R) & margin))
    if LONG_VLINES_EDGE[0]:
        # QRS of stacked leads fall in the same column and touch (the S of one touches the R
        # of the one below): on a 6x2 page the chain exceeds 10 cm, and at the margin the
        # page's last QRS exceeds 45 mm. Erasing it takes the R of one beat in every row (PM-ECG-ID
        # scans: -4 to -7 dB). The chain has trace on both sides; the grid border on one only.
        # With trace on both sides the line stays, whatever its length; one shorter than
        # 10 cm is only removed if it spans >= 70% of the block height
        wm, gm = int(13 * R), int(2 * R)
        yy = np.nonzero(B.any(1))[0]
        yspan = float(yy[-1] - yy[0] + 1) if len(yy) else float(H)
        for i in np.nonzero(longc)[0]:
            x0, y0, w_, h_ = (int(v) for v in st[i, :4])
            lft = B[y0:y0 + h_, max(0, x0 - gm - wm):max(0, x0 - gm)]
            rgt = B[y0:y0 + h_, min(W, x0 + w_ + gm):min(W, x0 + w_ + gm + wm)]
            cl = float(lft.any(0).mean()) if lft.shape[1] >= 3 * R else 0.0
            cr = float(rgt.any(0).mean()) if rgt.shape[1] >= 3 * R else 0.0
            if LONG_VLINES_STRAIGHT[0] and _straight_thin(lab, V, i, st[i], R):
                continue   # a ruled line (fold, curved grid line), not a chain of QRS
            if min(cl, cr) >= LONG_VLINES_SIDE[0] or (h_ < min_mm * R and h_ < 0.7 * yspan):
                longc[i] = False
    if not longc.any():
        return B
    rm = longc[lab] & (V > 0)
    rm = cv2.dilate(rm.astype(np.uint8), np.ones((1, 3), np.uint8)) > 0
    return B & ~rm
OTSU_FLOOR_REL = [True]
TMIN_ADAPT = [True]


def _clean_grid_lines(B, R, frac=0.7):
    H, W = B.shape
    ys, xs = np.nonzero(B.any(1))[0], np.nonzero(B.any(0))[0]
    if len(ys) < 10 or len(xs) < 10:
        return B
    out = B.copy()
    for axis in (1, 0):
        span = (xs[-1] - xs[0] + 1) if axis == 1 else (ys[-1] - ys[0] + 1)
        cov = B.sum(axis) / max(span, 1)
        lines = np.nonzero(cov > frac)[0]
        if len(lines) < 3:
            continue
        # group neighbouring pixels into a single line and require a regular step (multiple of ~5 mm)
        grp = np.split(lines, np.nonzero(np.diff(lines) > 1)[0] + 1)
        cen = np.array([g.mean() for g in grp])
        if len(cen) < 3:
            continue
        d = np.diff(cen) / R
        reg = np.mean(np.abs(d / 5.0 - np.round(d / 5.0)) < 0.15)
        if reg < 0.7:
            continue
        # a real grid: most neighbours at 5 mm (or 1 mm). Flat baselines of long traces also
        # cover the page and sit at multiples of 5 mm, but 25-40 mm apart
        near = np.mean((np.abs(d - 5.0) < 0.75) | (np.abs(d - 1.0) < 0.3))
        if len(cen) < 5 or near < 0.6:
            continue
        for g in grp:
            lo, hi = max(0, g[0] - 1), g[-1] + 2
            if axis == 1:
                out[lo:hi, :] = False
            else:
                out[:, lo:hi] = False
    return out


# ------------------------------------------------------------------------ rows

GRID_SEGS = [True]


def clean_grid_segments(B, R, L_mm=8.0, frac=0.03, axes=(1, 0)):
    """A monochrome (black/grey) grid is left in the binary map as straight, broken pieces
    (cleanup by whole line needs >= 70% coverage). Straight pieces >= L_mm in a thin band
    (<= ~0.5 mm) with another such band 5 or 10 mm away are grid and are removed; the trace
    crossing them stays (only the straight piece goes). A flat baseline is straight too, but
    ECG rows sit >= ~15 mm apart."""
    from scipy.signal import find_peaks as _fp
    H, W = B.shape
    ys, xs = np.nonzero(B.any(1))[0], np.nonzero(B.any(0))[0]
    if len(ys) < 10 or len(xs) < 10:
        return B
    L = max(5, int(L_mm * R))
    Bu = B.astype(np.uint8)
    out = B.copy()
    for axis in axes:
        k = np.ones((1, L), np.uint8) if axis == 1 else np.ones((L, 1), np.uint8)
        kc = max(3, int(1.5 * R))
        # dashed line (weak grid binarized in pieces): close the short gaps first
        Bc_ = cv2.morphologyEx(Bu, cv2.MORPH_CLOSE, np.ones((1, kc), np.uint8) if axis == 1
                               else np.ones((kc, 1), np.uint8))
        S = cv2.morphologyEx(Bc_, cv2.MORPH_OPEN, k)
        span = (xs[-1] - xs[0] + 1) if axis == 1 else (ys[-1] - ys[0] + 1)
        cov = S.sum(axis) / max(span, 1)
        pk, pr = _fp(cov, height=frac, width=(None, 0.5 * R + 2), rel_height=0.5)
        if len(pk) < 4:
            continue
        cen = pk / R
        keep = [i for i in range(len(pk))
                if np.any((np.abs(np.abs(cen - cen[i]) - 5.0) < 0.5) | (np.abs(np.abs(cen - cen[i]) - 10.0) < 0.7))]
        if len(keep) < 4:
            continue
        hw = int(0.5 * R) + 1
        for i in keep:
            lo, hi = max(0, pk[i] - hw), pk[i] + hw + 1
            if axis == 1:
                out[lo:hi, :] &= ~(S[lo:hi, :] > 0)
            else:
                out[:, lo:hi] &= ~(S[:, lo:hi] > 0)
    return out


ROWS_CLOSE = [True]
HALF_PULSE_END = [True]


def find_rows(B: np.ndarray, R: float):
    H, W = B.shape
    k = int(2 * R) | 1
    Bd = cv2.dilate(B.astype(np.uint8), np.ones((k, 1), np.uint8))
    cov = Bd.mean(1).astype(np.float64)
    cov = cv2.GaussianBlur(cov.reshape(-1, 1), (0, 0), R).ravel()
    pk, pr = find_peaks(cov, distance=int(8 * R), height=0.3 * cov.max(), prominence=0.1 * cov.max())
    if ROWS_CLOSE[0] and len(pk):
        # rows printed closer than 8 mm (12x1 placed by signal offset: baselines 3-8 mm apart):
        # a second line of almost full coverage, with a deep valley between the two
        m = cov.max()
        p2, _ = find_peaks(cov, distance=int(2.5 * R), height=0.6 * m, prominence=0.4 * m)
        add = [p for p in p2 if np.min(np.abs(pk - p)) >= 2.5 * R]
        if add:
            pk = np.sort(np.concatenate([pk, np.asarray(add, pk.dtype)]))
    return pk, cov


def find_pulses(B: np.ndarray, R: float):
    """All calibration pulses (rectangles) on the page, without knowing where the rows are.
    Vertical edges of 4..25 mm (vertical opening), in pairs 1.5..12 mm apart, with aligned
    tops and feet and an ink plateau between them. Returns a list of dict(x_start, x_end,
    top, foot)."""
    H, W = B.shape
    Bu = B.astype(np.uint8)
    kc = max(3, int(0.8 * R) | 1)
    Bc = cv2.morphologyEx(Bu, cv2.MORPH_CLOSE, np.ones((kc, 1), np.uint8))
    V = cv2.morphologyEx(Bc, cv2.MORPH_OPEN, np.ones((int(4 * R), 1), np.uint8))
    n, lab, st, _ = cv2.connectedComponentsWithStats(V, connectivity=8)
    edges = []
    for i in range(1, n):
        x, y, w, h, a = st[i]
        if h > 26 * R or w > 2.5 * R:
            continue
        edges.append((x + 0.5 * w, float(y), float(y + h - 1), x, w))
    edges.sort()
    out = []
    for i, (xc1, t1, b1, x1, w1) in enumerate(edges):
        for xc2, t2, b2, x2, w2 in edges[i + 1:]:
            sep = (xc2 - xc1) / R
            if sep > 12:
                break
            if sep < 1.5 or abs(t1 - t2) > 1.0 * R:
                continue
            h = min(b1, b2) - 0.5 * (t1 + t2)
            if not (4 * R <= h <= 25 * R) or abs(b1 - b2) > 10 * R:
                continue
            ty = int(round(0.5 * (t1 + t2)))
            band = B[max(0, ty - int(0.5 * R)):ty + int(0.5 * R) + 1, int(x1 + w1):int(x2)]
            if band.size == 0 or band.any(0).mean() < 0.8:
                continue
            # foot: the trace baseline just left of the rise (or the end of the left edge)
            out.append(dict(x_start=int(x1), x_end=int(x2 + w2 - 1), top=float(0.5 * (t1 + t2)),
                            foot=float(b1), x_fall=float(xc2)))
            break
    return out


def pulse_rows(B: np.ndarray, R: float):
    """Baselines of the trace rows from the calibration pulses: the largest group of pulses of
    the same height (±12%) in the same column (±5 mm), one per row. None if < 3."""
    P = find_pulses(B, R)
    H, W = B.shape
    # a real pulse joins the trace: right after it there is ink near the foot for most of the
    # next 25 mm (QR code block, text box and header do not)
    def _cont(p):
        a, b = max(0, int(p["foot"] - 4 * R)), min(H, int(p["foot"] + 4 * R))
        x0, x1 = int(p["x_end"] + 1), min(W, int(p["x_end"] + 25 * R))
        if x1 - x0 < 5 * R:
            return False
        return B[a:b, x0:x1].any(0).mean() >= 0.5
    P = [p for p in P if _cont(p)]
    best = None
    for p in P:
        h = p["foot"] - p["top"]
        grp = [q for q in P if abs(q["x_start"] - p["x_start"]) <= 5 * R and
               abs((q["foot"] - q["top"]) - h) <= 0.12 * h]
        feet = sorted(q["foot"] for q in grp)
        uniq = [f for i, f in enumerate(feet) if i == 0 or f - feet[i - 1] > 3 * R]
        if len(uniq) >= 2 and np.min(np.diff(uniq)) < 8 * R:
            continue  # ECG rows sit >= ~10 mm apart
        # ends off the group step (QR code rectangle or stamp well below/above the block):
        # removed
        while len(uniq) >= 4:
            dd = np.diff(uniq); md = float(np.median(dd))
            if dd[-1] > 1.5 * md:
                uniq = uniq[:-1]
            elif dd[0] > 1.5 * md:
                uniq = uniq[1:]
            else:
                break
        key = (len(uniq), -p["x_start"])
        if best is None or key > best[0]:
            best = (key, uniq, grp)
    if best is None or len(best[1]) < 3:
        return None, P
    return np.array(best[1]), best[2]


def pulse_column(B: np.ndarray, R: float, P=None):
    """Column of calibration pulses that do not join the trace (printed apart, on the
    nominal row, with the trace offset by the lead's DC level). Requires >= 3 pulses with the
    same x (±2 mm), the same height (±8%), the same width (±1 mm) and a regular vertical step
    (±15%): a text block or a QR code does not form that. Returns dict(x_start, x_end,
    height, feet) or None."""
    P = find_pulses(B, R) if P is None else P
    best = None
    for p in P:
        h = p["foot"] - p["top"]; w = p["x_end"] - p["x_start"]
        grp = [q for q in P if abs(q["x_start"] - p["x_start"]) <= 2 * R
               and abs((q["foot"] - q["top"]) - h) <= 0.08 * h
               and abs((q["x_end"] - q["x_start"]) - w) <= 1 * R]
        feet = sorted({round(q["foot"]) for q in grp})
        feet = [f for i, f in enumerate(feet) if i == 0 or f - feet[i - 1] > 3 * R]
        if len(feet) < 3:
            continue
        d = np.diff(feet)
        if d.min() < 8 * R or (d.max() - d.min()) > 0.15 * np.median(d):
            continue
        cand = dict(x_start=float(np.median([q["x_start"] for q in grp])),
                    x_end=float(np.median([q["x_end"] for q in grp])),
                    height=float(np.median([q["foot"] - q["top"] for q in grp])), feet=feet)
        if best is None or len(feet) > len(best["feet"]):
            best = cand
    return best


def _runs(mask_1d):
    d = np.diff(np.concatenate([[0], mask_1d.astype(np.int8), [0]]))
    return np.nonzero(d == 1)[0], np.nonzero(d == -1)[0] - 1


def row_extent(B, y0, half, R, max_gap_mm=3.0):
    """Longest continuous horizontal ink stretch in the row band."""
    H, W = B.shape
    a, b = max(0, int(y0 - half)), min(H, int(y0 + half))
    has = B[a:b].any(0)
    # close small gaps
    k = int(max_gap_mm * R) | 1
    hc = cv2.morphologyEx(has.astype(np.uint8).reshape(1, -1), cv2.MORPH_CLOSE, np.ones((1, k), np.uint8)).ravel()
    s, e = _runs(hc > 0)
    if len(s) == 0:
        return None
    i = int(np.argmax(e - s))
    lo, hi = int(s[i]), int(e[i])
    if ROW_CHAIN[0]:
        # columns separated by a gap (calibration pulse at the start of each column, margin
        # between columns): join long stretches in the same band up to 12 mm apart
        # (the trace of the other column may come broken into pieces; it is enough that the
        # sequence of pieces is dense and reaches a long stretch)
        gap, minlen = 12 * R, 15 * R
        for direction in (-1, 1):
            j = i + direction; acc = []
            edge = lo if direction < 0 else hi
            while 0 <= j < len(s):
                g_ = (edge - e[j]) if direction < 0 else (s[j] - edge)
                # a wide margin between columns (up to 25 mm) only joins a long stretch: the
                # other column itself, not text or a stamp
                if g_ > gap and not (ROW_CHAIN_WIDE[0] and not acc and g_ <= 25 * R
                                     and (e[j] - s[j]) >= 40 * R):
                    break
                acc.append(j)
                edge = s[j] if direction < 0 else e[j]
                if (e[j] - s[j]) >= minlen:  # reached a long stretch: accept the chain
                    if direction < 0:
                        lo = int(s[j])
                    else:
                        hi = int(e[j])
                    acc = []
                j += direction
    return lo, hi


ROW_CHAIN = [True]
ROW_CHAIN_WIDE = [True]


def detect_pulse(B, y0, half, xa, R, up=None, close_mm=0.8):
    """Calibration pulse (rectangle) at the start of the row.
    Requires two tall vertical edges (4..25 mm) 1.5..12 mm apart, with aligned heights, tops
    and feet (< 1 mm), a horizontal top covered with ink between them and a foot near the
    baseline. This rules out QRS (a point, not a plateau) and text.
    Returns dict(top, foot, x_start, x_end) or None."""
    H, W = B.shape
    up = half if up is None else up
    a, b = max(0, int(y0 - up)), min(H, int(y0 + 0.6 * half))
    x0s = max(0, int(xa - 10 * R))
    # 30 mm: in photos the row extent may start at the sheet/grid border, before the pulse
    x1 = min(W, int(xa + 30 * R))
    sub = B[a:b, x0s:x1].astype(np.uint8)
    kv = max(3, int(close_mm * R) | 1)  # close vertical breaks (low resolution / JPEG)
    sub = cv2.morphologyEx(sub, cv2.MORPH_CLOSE, np.ones((kv, 1), np.uint8)) > 0
    n = sub.shape[1]
    tops = np.full(n, np.nan); bots = np.full(n, np.nan)
    for j in range(n):
        s_, e_ = _runs(sub[:, j])
        if len(s_):
            i = int(np.argmax(e_ - s_))
            if e_[i] - s_[i] >= 4 * R:
                tops[j], bots[j] = s_[i] + a, e_[i] + a
    cs, ce = _runs(~np.isnan(tops))
    for i in range(len(cs)):
        for k in range(i + 1, len(cs)):
            sep = (cs[k] - ce[i]) / R
            if sep > 12:
                break
            if sep < 1.5:
                continue
            j1 = np.arange(cs[i], ce[i] + 1); j2 = np.arange(cs[k], ce[k] + 1)
            t1, b1 = np.median(tops[j1]), np.median(bots[j1])
            t2, b2 = np.median(tops[j2]), np.median(bots[j2])
            # the RISING edge goes from 0 to 1 mV; the falling one ends at the ECG value at
            # t = 0 (anything). Foot = left edge; top = plateau (mean of both).
            # height from the shorter edge (the longer may carry shadow/crease or the ECG
            # value at t = 0); the final baseline comes from the plateau + common gain
            h = min(b1, b2) - 0.5 * (t1 + t2)
            if not (4 * R <= h <= 25 * R):
                continue
            if abs(t1 - t2) > 1.0 * R and abs(b1 - b2) < 1.0 * R and min(b1 - t1, b2 - t2) >= 2 * R:
                # STEP pulse (split gain, e.g. 10/5 mm/mV): high plateau on the left,
                # vertical step, low plateau on the right. Check both plateaus.
                foot = min(b1, b2)
                if abs(foot - y0) > 4 * R:
                    continue
                xa_, xb_ = x0s + ce[i] + 1, x0s + cs[k]
                if xb_ - xa_ < 2 * R:
                    continue
                hb = int(0.5 * R)
                cov1 = B[max(0, int(t1) - hb):int(t1) + hb + 1, xa_:xb_].any(0)
                cov2 = B[max(0, int(t2) - hb):int(t2) + hb + 1, xa_:xb_].any(0)
                # plateau 1 covers a prefix and plateau 2 a suffix; together they cover almost everything
                if (cov1 | cov2).mean() < 0.8 or cov1[: len(cov1) // 4].mean() < 0.6 or \
                        cov2[-(len(cov2) // 4):].mean() < 0.6:
                    continue
                return dict(top=float(t1), top2=float(t2), foot=float(foot), x_start=int(x0s + cs[i]),
                            x_end=int(x0s + ce[k]), x_fall=float(x0s + 0.5 * (cs[k] + ce[k])), step=True)
            # the falling edge ends at the ECG value at t = 0 and may join the separator mark:
            # its foot sits well below the rising edge's (up to ~1 mV)
            if abs(t1 - t2) > 1.0 * R or abs(b1 - b2) > 10.0 * R:
                continue
            top = 0.5 * (t1 + t2); foot = min(b1, b2)
            if abs(foot - y0) > 4 * R:
                continue
            # plateau: ink in a narrow band around the top, between the edges
            ty = int(round(top))
            band = B[max(0, ty - int(0.5 * R)):ty + int(0.5 * R) + 1, x0s + ce[i] + 1:x0s + cs[k]]
            if band.size == 0 or band.any(0).mean() < 0.8:
                continue
            return dict(top=float(top), foot=float(foot), x_start=int(x0s + cs[i]), x_end=int(x0s + ce[k]),
                        x_fall=float(x0s + 0.5 * (cs[k] + ce[k])))
    # half pulse (left edge cut by the image margin): plateau >= 1.5 mm ending in a tall
    # falling edge, foot near the baseline
    for i in range(len(cs)):
        j1 = np.arange(cs[i], ce[i] + 1)
        t1, b1 = np.median(tops[j1]), np.median(bots[j1])
        h = b1 - t1
        if not (4 * R <= h <= 25 * R) or abs(b1 - y0) > 4 * R:
            continue
        ty = int(round(t1))
        xl = x0s + cs[i] - int(1.5 * R)
        if xl < 0:
            continue
        band = B[max(0, ty - int(0.5 * R)):ty + int(0.5 * R) + 1, xl:x0s + cs[i]]
        left = B[max(0, ty + int(1.0 * R)):int(b1 - 1.0 * R), max(0, xl - int(10 * R)):xl]
        # after the falling edge the trace runs on at the foot (the baseline); the deep S of a
        # QRS that starts from a flat baseline comes back up at once
        xr = x0s + ce[i] + 1
        fy = int(round(b1))
        after = B[max(0, fy - int(1.0 * R)):fy + int(1.0 * R) + 1, xr + int(0.5 * R):xr + int(3 * R)]
        if HALF_PULSE_END[0] and (after.size == 0 or after.any(0).mean() < 0.6):
            continue
        if band.size and band.any(0).mean() >= 0.8 and (left.size == 0 or left.any(1).mean() < 0.5):
            return dict(top=float(t1), foot=float(b1), x_start=int(x0s + cs[i]), x_end=int(x0s + ce[i]),
                        x_fall=float(x0s + 0.5 * (cs[i] + ce[i])), half=True)
    return None


# ----------------------------------------------------------------- tracking

def column_runs(B, D, ya, yb, xa, xb):
    """Per-column list of segments (top, bot, centroid, mass).

    All columns at once (it used to be a Python loop per column)."""
    sub = B[ya:yb, xa:xb]
    h, w = sub.shape
    if w == 0:
        return []
    if h == 0:
        return [[] for _ in range(w)]
    pad = np.zeros((w, h + 2), np.int8)
    pad[:, 1:-1] = sub.T
    d = np.diff(pad, axis=1)
    cs, ss = np.nonzero(d == 1)          # column, first row of the segment (order: column, row)
    _, ee = np.nonzero(d == -1)
    ee = ee - 1                           # last row of the segment
    if len(ss) == 0:
        return [[] for _ in range(w)]
    # merge fragments of the same column separated by <= 3 px (antialiasing / JPEG halo)
    first = np.r_[True, (cs[1:] != cs[:-1]) | ((ss[1:] - ee[:-1] - 1) > 3)]
    last = np.r_[first[1:], True]
    c_, a_, b_ = cs[first], ss[first], ee[last]
    # mass and centroid of each merged segment (gap pixels included), in float64
    Dt = np.zeros(w * h + 1, np.float64)
    Dt[:-1] = D[ya:yb, xa:xb].T.ravel()
    idx = np.empty(2 * len(a_), np.int64)
    idx[0::2] = c_ * h + a_
    idx[1::2] = c_ * h + b_ + 1
    m = np.add.reduceat(Dt, idx)[0::2] + 1e-9
    yy = np.arange(ya, ya + h, dtype=np.float64)   # yb may go past the end of the image
    Dy = Dt.copy()
    Dy[:-1] *= np.tile(yy, w)
    cen = np.add.reduceat(Dy, idx)[0::2] / m
    # top/bottom stay np.int64, as before: later users of the segments compute with float32,
    # and a Python int would change the precision of the result (float32 instead of float64)
    top = list(a_ + ya); bot = list(b_ + ya); cen = cen.tolist(); m = m.tolist()
    cnt = np.bincount(c_, minlength=w).tolist()
    out = []
    i = 0
    for n in cnt:
        if n == 0:
            out.append([])
            continue
        col = [(top[k], bot[k], cen[k], m[k]) for k in range(i, i + n)]
        i += n
        # drop massless fragments when there is an alternative
        if n > 1:
            big = [c for c in col if c[3] >= 0.6]
            col = big or col
        out.append(col)
    return out


def _bridge_table(D, xa, n, R, floor=0.0):
    """Cumulative sum (in y) of the max ink over 3 columns (x-2..x): evidence of a weak trace
    joining two segments vertically (fast QRS upstroke/downstroke)."""
    from scipy.ndimage import maximum_filter1d
    H, W = D.shape
    a, b = max(0, xa - 2), min(W, xa + n)
    Dm = maximum_filter1d(np.clip(D[:, a:b].astype(np.float32) - floor, 0, None), 3, axis=1, origin=1)
    Dm = Dm[:, xa - a:xa - a + n] if xa - a >= 0 else Dm
    from scipy.ndimage import uniform_filter1d
    return uniform_filter1d(Dm, 3, axis=0)  # 3 px vertical mean: a 1 px hole in a weak trace does not count


def viterbi_track(runs, base, R, w_base=0.004, c_gap=1.0, others=(), lw=None, w_edge=0.0,
                  D=None, xa=0, bridge_ref=None, rigid=False, bridge_floor=0.0):
    """Pick one segment per column (or failure) minimizing a continuity cost.

    With D: a vertical jump between segments that the ink links (even weak, below the
    binarization threshold) costs little. The penalty is continuous: the QRS upstroke and
    downstroke, thin and light in a scan/photo, break apart in the binary map, and without
    this the tracker skipped the R peak or the S valley (the dominant error at the peaks)."""
    INF = 1e18
    n = len(runs)
    if n == 0:
        return []
    base, R = float(base), float(R)
    others = [float(o) for o in others]
    # Python scalars inside the loop (same values; NumPy scalars cost several times more)
    runs = [[(int(t), int(b), float(c), float(m)) for (t, b, c, m) in col] for col in runs]
    cs_br = None
    if D is not None and bridge_ref is not None and bridge_ref > 0 and n > 0:
        cs_br = _bridge_table(D, xa, n, R, bridge_floor)
        bridge_ref = float(bridge_ref - bridge_floor)
        Hb = cs_br.shape[0]
    states = []  # per column: list of (top, bot, center)
    backs = []
    flat_h = float(3.0 * (lw if lw else 0.3 * R))
    w_base, c_gap = float(w_base), float(c_gap)
    mx = [max(m for (_, _, _, m) in col) for col in runs if col]
    m_ref = float(np.median(mx)) if mx else 1.0

    def emis(t, b, c, m):
        # distance from the interval to the baseline (0 if the segment crosses it)
        dm = max(0.0, t - base, base - b) / R
        # saturated prior: 2-4 mV waves are physiological; the penalty only needs to break
        # ties between nearby traces, not push a tall R back to the baseline
        e = min(w_base * dm * dm, 0.1)
        # flat stretch on top of ANOTHER lead's baseline = neighbouring trace.
        # (a tall R crosses the neighbouring row vertically: tall segment, no penalty)
        if others and (b - t) <= flat_h and dm > 6:
            for ob in others:
                if abs(c - ob) < 4 * R:
                    e += 0.5
                    break
        if (b - t) / R > 30:
            e += 5.0
        if m < 0.35 * m_ref:  # weak fragment (halo/noise)
            e += 0.8
        return e

    _we = float(w_edge)
    _rigid = rigid

    def trans(pt, pb, pc, t, b, c, i=None):
        g = max(0.0, t - pb, pt - b) / R
        g0 = g
        if g > 0.2 and cs_br is not None and i is not None and i < cs_br.shape[1]:
            # empty vertical interval between the segments
            y0, y1 = (int(pb) + 1, int(t)) if t > pb else (int(b) + 1, int(pt))
            y0, y1 = max(0, y0), min(Hb, y1)
            if y1 > y0:
                # low quantile along the gap: a weak trace is continuous; scan stains/dots
                # have holes (the mean alone went through dark stains)
                ev = float(cs_br[y0:y1, i].min()) / bridge_ref
                if ev > 0.35:
                    g *= max(0.05, 1.0 - (ev - 0.35) / 0.35)
        # EDGE continuity: in a pen-drawn trace, from one column to the next at least one of
        # the edges (top/bottom) moves little; a letter stuck to the trace (label crossed by
        # an R) sits in the middle and breaks both
        # (steep trace: the whole segment translates from one column to the next, both edges
        # move together; only a stuck letter changes top and bottom unevenly)
        ed = min(abs(t - pt), abs(b - pb))
        if _rigid and g0 == 0.0:  # only with overlap: the same trace continuing
            ed = min(ed, abs((t - pt) - (b - pb)))
        return 2.0 * g + 0.5 * g * g + 0.02 * abs(c - pc) / R + _we * ed / R

    # arithmetic on Python scalars (float/int): same IEEE values, without the cost of NumPy
    # scalars in every sum
    prev_states = [(base, base, base)]
    prev_cost = [0.0]
    last_cost = prev_cost
    for i in range(n):
        col = runs[i]
        cs = []
        bk = []
        for (t, b, c, m) in col:
            best, arg = INF, -1
            k = 0
            for (pt, pb, pc) in prev_states:
                v = prev_cost[k] + trans(pt, pb, pc, t, b, c, i)
                if v < best:
                    best, arg = v, k
                k += 1
            cs.append(best + emis(t, b, c, m))
            bk.append(arg)
        # failure state (last index)
        pmin = min(prev_cost)
        kmin = prev_cost.index(pmin)        # first minimum, like np.argmin
        cs.append(pmin + c_gap)
        bk.append(kmin)
        new_states = [(t, b, c) for (t, b, c, m) in col]
        new_states.append(prev_states[kmin] if kmin < len(prev_states) else prev_states[-1])
        states.append(new_states)
        backs.append(bk)
        prev_states, prev_cost = new_states, cs
        last_cost = cs
    # backtrack
    sel = [None] * n
    k = last_cost.index(min(last_cost))
    for i in range(n - 1, -1, -1):
        is_gap = k == len(states[i]) - 1
        sel[i] = None if is_gap else k
        k = backs[i][k]
    return sel


def selected_envelopes(D, runs, sel, xa):
    """Sub-pixel edges (50% of full ink level) of the chosen segment in each column."""
    n = len(runs)
    tb = np.full((n, 2), np.nan)
    pk = np.full(n, np.nan)
    for i, k in enumerate(sel):
        if k is None:
            continue
        t, b = runs[i][k][0], runs[i][k][1]
        col = D[:, xa + i]
        tb[i] = t, b
        pk[i] = col[t:b + 1].max()
    return tb, pk


def _edge_up(col, t, b, lev):
    k = t
    while k - 1 >= 0 and col[k - 1] >= lev and k > t - 3:
        k -= 1
    while k < b and col[k] < lev:
        k += 1
    if k == 0:
        return k - 0.5
    d0, d1 = col[k - 1], col[k]
    return (k - 1) + (lev - d0) / (d1 - d0) if d1 > d0 and d1 >= lev else k - 0.5


def _edge_dn(col, t, b, lev):
    k = b
    H = len(col)
    while k + 1 < H and col[k + 1] >= lev and k < b + 3:
        k += 1
    while k > t and col[k] < lev:
        k -= 1
    if k + 1 >= H:
        return k + 0.5
    d0, d1 = col[k], col[k + 1]
    return k + (d0 - lev) / (d0 - d1) if d0 > d1 and d0 >= lev else k + 0.5


def envelope_edges(D, runs, sel, xa, dfull):
    lev = 0.5 * dfull
    n = len(runs)
    top = np.full(n, np.nan); bot = np.full(n, np.nan)
    for i, k in enumerate(sel):
        if k is None:
            continue
        t, b = runs[i][k][0], runs[i][k][1]
        col = D[:, xa + i]
        top[i] = _edge_up(col, t, b, lev)
        bot[i] = _edge_dn(col, t, b, lev)
    return top, bot


def _shift_pad(a, d):
    if d == 0:
        return a
    p = np.pad(a, (abs(d), abs(d)), mode="edge")
    return p[abs(d) - d: abs(d) - d + len(a)]


def _gamma_phi(top, bot, r, rh, U):
    n = len(top)
    xs = np.arange(n, dtype=np.float64)
    okt, okb = ~np.isnan(top), ~np.isnan(bot)
    ok = okt | okb
    if okt.sum() < 2 or okb.sum() < 2:
        okt = okb = okt & okb
    t = np.interp(xs, xs[okt], top[okt]); b = np.interp(xs, xs[okb], bot[okb])
    xf = np.arange(0, n - 1 + 1e-9, 1.0 / U)
    tf = np.interp(xf, xs, t); bf = np.interp(xf, xs, b)
    rU = int(np.floor(rh * U))
    gam = np.full_like(tf, -np.inf); phi = np.full_like(bf, np.inf)
    for d in range(-rU, rU + 1):
        h = r * np.sqrt(max(1.0 - (d / U / rh) ** 2, 0.0))
        gam = np.maximum(gam, _shift_pad(tf, d) + h)
        phi = np.minimum(phi, _shift_pad(bf, d) - h)
    return xf, tf, bf, gam, phi, ok


def calibrate_pen(segments, r, U=4):
    """Self-calibration of the pen's horizontal half-width (elliptical pen: thermal printers
    and renderers are rarely isotropic). In the right model, opening (of the top) and closing
    (of the bottom) agree on sloped strokes; pick the rh that minimizes the median
    disagreement, within a plausible range (0.7..1.1 x rv)."""
    best, best_rh = np.inf, r
    for f in np.arange(0.7, 1.11, 0.05):  # plausible range of pen anisotropy
        rh = r * f
        errs = []
        for top, bot in segments:
            ok = ~np.isnan(top) & ~np.isnan(bot)
            if ok.sum() < 20:
                continue
            xf, tf, bf, gam, phi, _ = _gamma_phi(top, bot, r, rh, U)
            # sloped-trace columns (top and bottom edges well apart)
            straight = (bf - tf) > 6 * r
            if straight.sum() > 5:
                errs.append(np.abs(gam - phi)[straight])
        if not errs:
            return r
        e = float(np.median(np.concatenate(errs)))
        if e < best:
            best, best_rh = e, rh
    return best_rh


def centerline_from_envelopes(top, bot, r, U=4, rh=None):
    """Invert the model 'trace = curve (+) elliptical pen (rh, r)':
       top = erosion(y), bottom = dilation(y). Opening the top (<= y) is exact at upward
       peaks; closing the bottom (>= y) is exact at valleys. Pick by curvature."""
    n = len(top)
    xs = np.arange(n, dtype=np.float64)
    ok = ~np.isnan(top) | ~np.isnan(bot)
    if (~np.isnan(top) & ~np.isnan(bot)).sum() < 3:
        return np.full(n, np.nan), ok
    rh = r if rh is None else rh
    xf, tf, bf, gam, phi, ok = _gamma_phi(top, bot, r, rh, U)
    m = 0.5 * (gam + phi)
    k = max(1, int(np.floor(rh * U)))
    c = _shift_pad(m, k) + _shift_pad(m, -k) - 2 * m
    yf = np.where(c > 0, gam, phi)
    agree = np.abs(gam - phi) < 0.5
    yf = np.where(agree, m, yf)
    y = np.interp(xs, xf, yf)
    return y, ok


def refine_steep(D, xs, yc, rh, min_dy=2.0):
    """Posterior mean of the INSTANT on steep stretches (QRS).

    In a column, the almost vertical trace occupies a tall run and its centre only gives x to
    1-column resolution (0.1 mm = 4 ms at 25 mm/s); on a 50 mV/s QRS that is already 0.2 mV
    of error. Here x and y swap roles: on each pixel row the steep trace crosses, the
    horizontal ink centroid (density-weighted, sub-pixel) is the minimum squared-error
    estimate of the instant at which the curve passes that height.
    Flat stretches keep the per-column centre. Returns (x, y) sorted in x."""
    xs = np.asarray(xs, float); yc = np.asarray(yc, float)
    n = len(xs)
    if n < 4:
        return xs, yc
    H, W = D.shape
    dy = np.diff(yc)
    contig = np.diff(xs) == 1
    sgn = np.sign(dy)
    steep = (np.abs(dy) >= min_dy) & contig & np.isfinite(dy)
    # same direction in the neighbours: away from the turn of the peak (where up and down strokes merge)
    same = np.zeros_like(steep)
    same[1:-1] = (sgn[1:-1] == sgn[:-2]) & (sgn[1:-1] == sgn[2:])
    steep &= same
    w = int(np.ceil(rh)) + 2
    px, py = [], []
    for i in np.nonzero(steep)[0]:
        y0, y1 = yc[i], yc[i + 1]
        lo, hi = (y0, y1) if y0 < y1 else (y1, y0)
        rows = np.arange(int(np.floor(lo + 1.0)) + 1, int(np.ceil(hi - 1.0)))
        for Y in rows:
            if Y < 0 or Y >= H:
                continue
            x0 = xs[i] + (Y - y0) / (y1 - y0)
            a, b = max(0, int(np.floor(x0)) - w), min(W, int(np.ceil(x0)) + w + 1)
            prof = D[Y, a:b].astype(float)
            if prof.size < 3 or prof.max() < 0.2:
                continue
            thr = 0.3 * prof.max()
            on = prof > thr
            k = int(np.clip(round(x0) - a, 0, len(prof) - 1))
            if not on[k]:
                cand = np.nonzero(on)[0]
                k = int(cand[np.argmin(np.abs(cand - (x0 - a)))])
            l_, r_ = k, k
            while l_ - 1 >= 0 and on[l_ - 1]:
                l_ -= 1
            while r_ + 1 < len(prof) and on[r_ + 1]:
                r_ += 1
            if l_ == 0 or r_ == len(prof) - 1 or (r_ - l_ + 1) > 2 * rh + 4:
                continue  # touching the window or too wide: two traces/smear
            wt = prof[l_:r_ + 1] - thr
            xc = a + l_ + float((wt * np.arange(r_ - l_ + 1)).sum() / max(wt.sum(), 1e-9))
            if abs(xc - x0) > 1.5:
                continue
            if not (min(xs[i], xs[i + 1]) < xc < max(xs[i], xs[i + 1])):
                continue  # outside the interval between the two columns: would break monotonicity
            px.append(xc); py.append(float(Y))
    if not px:
        return xs, yc
    # the column points stay (at the peak/valley the pen inversion is exact); the row points
    # go between them and give the sub-column instant of the upstrokes and downstrokes
    X = np.concatenate([xs, px]); Y = np.concatenate([yc, py])
    o = np.argsort(X, kind="stable")
    X, Y = X[o], Y[o]
    ok = np.r_[True, np.diff(X) > 1e-3]
    return X[ok], Y[ok]


def mark_occlusions(jobs, lw, margin=1.0):
    """Overlapping traces (neighbouring rows that cross): the top/bottom edge of a trace
    that falls inside the ink of ANOTHER trace is not reliable -> NaN.
    The envelope is then interpolated over the occlusion."""
    for j in jobs:
        j["top_o"], j["bot_o"] = j["top"].copy(), j["bot"].copy()
    for j in jobs:
        xj = j["xa"] + np.arange(len(j["top"]))
        for o in jobs:
            if o is j or o["row"] == j["row"]:
                continue
            xo0, xo1 = o["xa"], o["xa"] + len(o["top"])
            lo, hi = max(xj[0], xo0), min(xj[-1] + 1, xo1)
            if hi <= lo:
                continue
            sj = slice(lo - j["xa"], hi - j["xa"])
            so = slice(lo - xo0, hi - xo0)
            ot, ob = o["top"][so] - margin, o["bot"][so] + margin
            tj, bj = j["top"][sj], j["bot"][sj]
            with np.errstate(invalid="ignore"):
                # only a locally flat neighbouring trace (that did not "swallow" another QRS) occludes
                flat_o = (o["bot"][so] - o["top"][so]) <= 2.5 * lw
                occ_t = flat_o & (tj >= ot) & (tj <= ob)
                occ_b = flat_o & (bj >= ot) & (bj <= ob)
            j["top_o"][sj][occ_t] = np.nan
            j["bot_o"][sj][occ_b] = np.nan
    for j in jobs:
        j["top"], j["bot"] = j.pop("top_o"), j.pop("bot_o")


def trace_likeness(B, peaks, R, min_len_mm=25.0, return_raw=False):
    """For each candidate row: total width covered by long connected components (traces),
    centred near the row. Text/header = many short pieces."""
    k = int(1.5 * R) | 1
    Bc = cv2.dilate(B.astype(np.uint8), np.ones((3, k), np.uint8))  # joins small breaks
    n, lab, st, cen = cv2.connectedComponentsWithStats(Bc, connectivity=8)
    long_ = [i for i in range(1, n) if st[i, cv2.CC_STAT_WIDTH] >= min_len_mm * R]
    scores = []
    for y in peaks:
        cov = 0.0
        for i in long_:
            y0, h = st[i, cv2.CC_STAT_TOP], st[i, cv2.CC_STAT_HEIGHT]
            if y0 <= y <= y0 + h:
                # the component must have ink near the row over much of its width
                x0, w = st[i, cv2.CC_STAT_LEFT], st[i, cv2.CC_STAT_WIDTH]
                band = lab[max(0, int(y - 3 * R)):int(y + 3 * R), x0:x0 + w] == i
                cov += band.any(0).sum()
        scores.append(cov / R)  # in mm
    if not return_raw:
        return np.array(scores)
    # "raw" coverage: long components WITHOUT joining 1.5 mm breaks. Text (footer
    # "50mm/s 10mm/mV ...") becomes one long component after dilation, but raw it is letters
    # of a few mm; a trace stays long.
    B8 = B.astype(np.uint8)
    # remove straight rulers (border of the gridded area etc.): lines > 60 mm horizontally
    lines = cv2.morphologyEx(B8, cv2.MORPH_OPEN, np.ones((1, int(60 * R) | 1), np.uint8))
    lines = cv2.dilate(lines, np.ones((5, 1), np.uint8))
    Br = cv2.morphologyEx(B8 & (1 - lines), cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    n2, lab2, st2, _ = cv2.connectedComponentsWithStats(Br, connectivity=8)
    # straight lines (border of the gridded area, rulers) are not traces: a trace of
    # > 50 mm always has QRS/waves taller than 1.5 mm
    w2, h2 = st2[:, cv2.CC_STAT_WIDTH], st2[:, cv2.CC_STAT_HEIGHT]
    # words: wide but short (< 4 mm); trace pieces have waves or are long
    long2 = np.nonzero(((w2 >= 6.0 * R) & (h2 >= 4.0 * R)) | (w2 >= 25.0 * R))[0]
    long2 = long2[long2 > 0]
    keep = np.zeros(n2, bool); keep[long2] = True
    raw = []
    for y in peaks:
        band = keep[lab2[max(0, int(y - 3 * R)):int(y + 3 * R)]]
        raw.append(band.any(0).sum() / R)
    return np.array(scores), np.array(raw)


def horizontal_halfwidth(D, jobs, lev, rv):
    """Pen horizontal half-width measured directly: in the middle of nearly vertical strokes
    (QRS), horizontal ink profile at the 50% level. Up and down strokes stuck together give
    doubled widths -> use a low percentile."""
    H, W = D.shape
    out = []
    for j in jobs:
        top, bot, xa = j["top"], j["bot"], j["xa"]
        for i in range(2, len(top) - 2):
            if np.isnan(top[i]) or np.isnan(bot[i]) or bot[i] - top[i] < 8 * rv:
                continue
            y = int(round(0.5 * (top[i] + bot[i])))
            x = xa + i
            if x < 12 or x + 12 >= W or not (0 <= y < H):
                continue
            prof = D[y, x - 12:x + 13]
            c = 12 + int(np.argmax(prof[10:15])) - 2
            if prof[c] < lev:
                continue
            l = c
            while l > 0 and prof[l - 1] >= lev:
                l -= 1
            r = c
            while r < 24 and prof[r + 1] >= lev:
                r += 1
            if l == 0 or r == 24:
                continue
            le = (l - 1) + (lev - prof[l - 1]) / max(prof[l] - prof[l - 1], 1e-6)
            re = r + (prof[r] - lev) / max(prof[r] - prof[r + 1], 1e-6)
            out.append(0.5 * (re - le))
    if len(out) < 20:
        return rv
    return float(np.clip(np.percentile(out, 25), 0.7 * rv, 1.0 * rv))
