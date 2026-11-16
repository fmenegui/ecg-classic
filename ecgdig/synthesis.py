"""Analysis-by-synthesis fit of one lead's trace.

Image formation model: the ink seen at pixel p is
    M(p) = 1 - exp(-S(p)),   S(p) = sum_k a * g_sigma(p - c_k)
where c_k are points of the ECG curve sampled every `ds` px of arc length and g_sigma is an
isotropic Gaussian that combines pen and capture blur. The 1 - exp(-S) saturates where the
trace passes several times (narrow peak), like real ink.

The points move along the curve normal (delta_k) to match M to the ink map D (least
squares), with second-order regularization on delta. a and sigma come from the image
itself (vertical trace profile on flat stretches).

No learning: it is the "inverse renderer" with a physical pen + blur model.
"""
from __future__ import annotations

import numpy as np


def _densify(x, y, ds):
    d = np.hypot(np.diff(x), np.diff(y))
    s = np.r_[0.0, np.cumsum(d)]
    if s[-1] < 2 * ds:
        return x.astype(float), y.astype(float)
    t = np.arange(0.0, s[-1], ds)
    return np.interp(t, s, x), np.interp(t, s, y)


def _normals(cx, cy):
    tx = np.gradient(cx); ty = np.gradient(cy)
    nrm = np.hypot(tx, ty) + 1e-9
    return -ty / nrm, tx / nrm


def _pav(y):
    """Isotonic (non-decreasing) regression by pool-adjacent-violators. Returns (value per
    point, block label per point)."""
    vals, wts, cnt = [], [], []
    for v in y:
        vals.append(float(v)); wts.append(1.0); cnt.append(1)
        while len(vals) > 1 and vals[-2] > vals[-1]:
            w = wts[-2] + wts[-1]
            v2 = (vals[-2] * wts[-2] + vals[-1] * wts[-1]) / w
            c2 = cnt[-2] + cnt[-1]
            vals.pop(); wts.pop(); cnt.pop()
            vals[-1], wts[-1], cnt[-1] = v2, w, c2
    out = np.repeat(vals, cnt)
    lab = np.repeat(np.arange(len(vals)), cnt)
    return out, lab


def monotone_order(cx, cy, spread=0.3):
    """Order the curve in x without scrambling arc order: x becomes non-decreasing through
    isotonic regression (mean of stretches that go backwards) and, inside each grouped
    "vertical" stretch, points spread over +-spread px in arc order. Sorting by plain x
    interleaves the up and down strokes of the QRS and turns them jagged."""
    xi, lab = _pav(cx)
    out = xi.copy()
    if len(lab):
        st = np.r_[0, np.nonzero(np.diff(lab))[0] + 1]
        en = np.r_[st[1:], len(lab)]
        for a, b in zip(st, en):
            if b - a > 1:
                lo = xi[a] - spread if a == 0 else max(xi[a] - spread, 0.5 * (xi[a - 1] + xi[a]))
                hi = xi[a] + spread if b == len(xi) else min(xi[a] + spread, 0.5 * (xi[a] + xi[b]))
                out[a:b] = np.linspace(lo, hi, b - a)
    # make it strictly increasing
    o = np.maximum.accumulate(out)
    keep = np.r_[True, np.diff(o) > 1e-4]
    return o[keep], np.asarray(cy)[keep]


def estimate_psf(D, xs, yc, rad, n_max=400):
    """(a, sigma) from the vertical ink profile on flat trace columns."""
    from scipy.optimize import least_squares
    H, W = D.shape
    xs = np.asarray(xs); yc = np.asarray(yc)
    flat = np.r_[False, (np.abs(np.diff(yc)) < 0.3)] & np.r_[(np.abs(np.diff(yc)) < 0.3), False]
    idx = np.nonzero(flat & np.isfinite(yc))[0]
    if len(idx) < 10:
        return None
    idx = idx[np.linspace(0, len(idx) - 1, min(n_max, len(idx))).astype(int)]
    h = int(max(6, 4 * rad + 4))
    prof, off = [], []
    for i in idx:
        x = int(round(xs[i])); y0 = yc[i]
        if not (0 <= x < W):
            continue
        ys = np.arange(int(y0) - h, int(y0) + h + 1)
        ys = ys[(ys >= 0) & (ys < H)]
        prof.append(D[ys, x].astype(float)); off.append(ys - y0)
    if not prof:
        return None
    p = np.concatenate(prof); o = np.concatenate(off)

    def model(th):
        A, s = th
        return 1.0 - np.exp(-A * np.exp(-0.5 * (o / s) ** 2))

    r = least_squares(lambda th: model(th) - p, x0=[2.0, max(0.7, rad)], bounds=([0.05, 0.3], [50.0, 6.0]))
    A, s = r.x
    # linear density: points every ds along the curve -> S of a horizontal line =
    # a/ds * sqrt(2*pi)*s * exp(-y^2/2s^2)  =>  a = A * ds / (sqrt(2*pi) * s)
    return float(A), float(s)


SYNTH_BUDGET = [4.0]
SYNTH_MAXWORK = [3e7]
SYNTH_WORK = [2e8]      # point-pixel pairs x iterations per segment (deterministic limit)


def synth_refine(D, xg, yg, rad, n_iter=60, ds=0.5, lam=2.0, max_shift=1.5, max_rad=2.0, order="sort",
                 time_budget=None):
    """Refine the curve points (xg, yg) by matching the rendered trace to D.
    Returns new (x, y), sorted in x, or the inputs if the fit is not reliable."""
    xg = np.asarray(xg, float); yg = np.asarray(yg, float)
    ok = np.isfinite(xg) & np.isfinite(yg)
    xg, yg = xg[ok], yg[ok]
    if len(xg) < 20:
        return xg, yg
    if rad > max_rad:
        # thick pen with a sharp outline: the morphological model (inversion by the edges) is
        # already exact and the Gaussian gets the flat profile wrong, pulling narrow peaks down.
        # Synthesis is for thin blurred traces (scans, photos, PhysioNet renders)
        return xg, yg
    psf = estimate_psf(D, xg, yg, rad)
    if psf is None:
        return xg, yg
    A, sig = psf
    if sig > 3.0:
        return xg, yg  # blur larger than the pen itself: the model cannot separate curve and smear
    a = A * ds / (np.sqrt(2 * np.pi) * sig)
    H, W = D.shape
    cx0, cy0 = _densify(xg, yg, ds)
    K = len(cx0)
    rk = int(np.ceil(3 * sig)) + 1
    o1 = np.arange(-rk, rk + 1)
    n1 = len(o1)
    npair = n1 * n1
    # work region: rectangle around the curve
    x0 = max(0, int(np.floor(cx0.min())) - rk - 2); x1 = min(W, int(np.ceil(cx0.max())) + rk + 3)
    y0 = max(0, int(np.floor(cy0.min())) - rk - 2); y1 = min(H, int(np.ceil(cy0.max())) + rk + 3)
    Dp = D[y0:y1, x0:x1].astype(float)
    h, w = Dp.shape
    # only pixels near the initial curve count (tube): the rest may be another trace/text
    tube = np.zeros((h, w), bool)
    bx = np.clip(np.round(cx0).astype(int) - x0, 0, w - 1); by = np.clip(np.round(cy0).astype(int) - y0, 0, h - 1)
    tube[by, bx] = True
    from scipy.ndimage import binary_dilation
    tube = binary_dilation(tube, iterations=rk + 1)
    nx, ny = _normals(cx0, cy0)
    delta = np.zeros(K)
    lr = 0.5 / (a * a + 1e-9)

    CH = max(1, 400000 // npair)  # points per block: bounds memory (~400k pairs)
    pat = (o1[:, None] * w + o1[None, :]).ravel()   # index offset of each neighbour (y, x)
    inv2 = 0.5 / (sig * sig)

    def _blocks(fx, fy):
        """Per block of points: pixel index, Gaussian weight and offsets of each pair
        (point, neighbouring pixel). The 2D Gaussian is the product of two 1D ones (n1
        exponentials per axis instead of n1*n1 per point), and the blocks of one iteration
        serve both rendering AND gradient: they used to be recomputed in both passes."""
        out = []
        for c0 in range(0, K, CH):
            fxc, fyc = fx[c0:c0 + CH], fy[c0:c0 + CH]
            ix = np.floor(fxc).astype(np.int64); iy = np.floor(fyc).astype(np.int64)
            px = ix[:, None] + o1[None, :]                                  # (k, n1)
            py = iy[:, None] + o1[None, :]
            dx = px - fxc[:, None]; dy = py - fyc[:, None]
            ex = np.exp(-inv2 * dx * dx); ey = a * np.exp(-inv2 * dy * dy)
            if ix.min() >= rk and ix.max() < w - rk and iy.min() >= rk and iy.max() < h - rk:
                # common case (the region has a margin around the curve): everything inside
                idx = (iy * w + ix)[:, None] + pat[None, :]                 # (k, n1*n1)
            else:
                inx = (px >= 0) & (px < w); iny = (py >= 0) & (py < h)
                ex = np.where(inx, ex, 0.0); ey = np.where(iny, ey, 0.0)    # outside the region: weight 0
                idx = ((np.clip(py, 0, h - 1) * w)[:, :, None] + np.clip(px, 0, w - 1)[:, None, :]).reshape(len(ix), npair)
            gv = (ey[:, :, None] * ex[:, None, :]).reshape(len(ix), npair)
            out.append((c0, idx, gv, dx, dy))
        return out

    if time_budget is None:
        time_budget = SYNTH_BUDGET[0]
    if K * npair > SYNTH_MAXWORK[0]:
        return xg, yg  # very long curve with wide blur: prohibitive cost, uncertain gain
    # deterministic budget (point-pixel pairs per iteration x iterations): the same result on
    # any machine. The wall-clock limit is only an emergency brake.
    n_iter = int(max(8, min(n_iter, SYNTH_WORK[0] // max(1, K * npair))))
    import time as _time
    t_start = _time.time()
    prev = None
    for it in range(n_iter):
        if _time.time() - t_start > time_budget:
            break
        cx = cx0 + delta * nx; cy = cy0 + delta * ny
        blocks = _blocks(cx - x0, cy - y0)
        S = np.zeros(h * w)
        for _, idx, gv, _, _ in blocks:
            S += np.bincount(idx.ravel(), weights=gv.ravel(), minlength=h * w)
        S = S.reshape(h, w)
        M = 1.0 - np.exp(-S)
        r = np.where(tube, M - Dp, 0.0)
        loss = float((r * r).sum())
        if prev is not None and loss > prev * 0.9999 and it > 5:
            lr *= 0.5
            if lr < 1e-3 / (a * a + 1e-9):
                break
        prev = loss
        # dL/dc_k = sum_p 2 r(p) (1 - M(p)) * a g(p - c_k) * (p - c_k)/sig^2
        G2 = (2 * r * (1 - M)).ravel()
        grad = np.empty(K)
        for c0, idx, gv, dx, dy in blocks:
            coef = (G2[idx] * gv / (sig * sig)).reshape(len(dx), n1, n1)   # (k, n1[y], n1[x])
            gx = np.einsum("kji,ki->k", coef, dx)
            gy = np.einsum("kji,kj->k", coef, dy)
            grad[c0:c0 + len(gx)] = gx * nx[c0:c0 + len(gx)] + gy * ny[c0:c0 + len(gx)]
        # second-order regularization (smooth curve at the scale of ~1 px of arc)
        lap = np.r_[0.0, delta[:-2] - 2 * delta[1:-1] + delta[2:], 0.0]
        delta = delta - lr * grad + lr * lam * a * a * lap
        delta = np.clip(delta, -max_shift, max_shift)
    cx = cx0 + delta * nx; cy = cy0 + delta * ny
    if order == "monotone":
        return monotone_order(cx, cy)
    o = np.argsort(cx, kind="stable")
    cx, cy = cx[o], cy[o]
    keep = np.r_[True, np.diff(cx) > 1e-3]
    return cx[keep], cy[keep]
