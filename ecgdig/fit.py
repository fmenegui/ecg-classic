"""Fit by inverse rendering with the measured pen profile (LSF).

Image formation model for an isolated trace: the ink at pixel p depends only on the
distance from p to the ECG curve,

    M(p) = alpha_k * LSF(d(p, C)),

where C is the polyline (x_k, y_k) with x_k on a fixed grid (step h px; the curve is a
function of time), d is the Euclidean distance to the nearest segment (union: ink does not
add up where the pen passes twice, it saturates) and LSF is the cross profile of pen + blur,
measured on the image itself on flat stretches (non-parametric: works for a thin blurred
pen and for a thick sharp one). alpha_k is the ink density of each segment (fast strokes
come out lighter in scans and photos), with a prior at 1.

The unknowns y_k (and alpha_k) come from robust least squares (Cauchy) between M and the
ink map D in a tube around the initial curve, with weak second-order regularization.
Because the curve is parametrized by time, there is no point-ordering problem in the QRS.

No learning: it is the inverse of the renderer, with the renderer measured on the image.
"""
from __future__ import annotations

import numpy as np


def estimate_lsf(D, xg, yg, rsup_max=7.0, step=0.25):
    """Cross profile of the ink as a function of distance to the centre line (flat stretches).
    Returns (d_tab, lsf_tab, rsup) or None."""
    H, W = D.shape
    xg = np.asarray(xg, float); yg = np.asarray(yg, float)
    xi = np.arange(int(np.ceil(xg[0])), int(np.floor(xg[-1])) + 1)
    if len(xi) < 30:
        return None
    yi = np.interp(xi, xg, yg)
    sl = np.abs(np.gradient(yi))
    flat = (sl < 0.15) & (np.r_[sl[1:], 1] < 0.15) & (np.r_[1, sl[:-1]] < 0.15)
    xi, yi = xi[flat], yi[flat]
    if len(xi) < 20:
        return None
    if len(xi) > 600:
        k = np.linspace(0, len(xi) - 1, 600).astype(int)
        xi, yi = xi[k], yi[k]
    R = int(np.ceil(rsup_max)) + 2
    off = np.arange(-R, R + 1)
    yy = np.round(yi)[:, None].astype(int) + off[None, :]
    ok = (yy >= 0) & (yy < H) & (xi[:, None] >= 0) & (xi[:, None] < W)
    vals = np.where(ok, D[np.clip(yy, 0, H - 1), np.clip(xi, 0, W - 1)[:, None]], np.nan)
    # sub-pixel centre per column: centroid of the ink above 20% of the local maximum
    v = np.nan_to_num(vals)
    w = np.clip(v - 0.2 * v.max(1, keepdims=True), 0, None)
    yc = (w * yy).sum(1) / np.maximum(w.sum(1), 1e-9)
    d = np.abs(yy - yc[:, None]).ravel()
    vv = vals.ravel()
    m = np.isfinite(vv)
    d, vv = d[m], vv[m]
    edges = np.arange(0, R + step, step)
    idx = np.digitize(d, edges) - 1
    tab = np.full(len(edges) - 1, np.nan)
    for i in range(len(tab)):
        s = vv[idx == i]
        if s.size >= 5:
            tab[i] = np.median(s)
    dt = 0.5 * (edges[:-1] + edges[1:])
    ok = np.isfinite(tab)
    if ok.sum() < 6:
        return None
    tab = np.interp(dt, dt[ok], tab[ok])
    # background (paper) -> 0; monotone non-increasing profile
    bg = float(np.median(tab[-4:]))
    tab = np.clip(tab - bg, 0, None)
    tab = np.minimum.accumulate(tab)
    if tab[0] < 0.1:
        return None
    below = np.nonzero(tab < 0.04 * tab[0])[0]
    rsup = float(dt[below[0]]) if below.size else float(dt[-1])
    rsup = min(max(rsup, 1.0), rsup_max)
    keep = dt <= rsup + step
    d_tab = np.r_[0.0, dt[keep], rsup + 2 * step]
    l_tab = np.r_[tab[0], tab[keep], 0.0]
    return d_tab, l_tab, rsup


def _seg_dist(px, py, x0, y0, x1, y1):
    """Distance from (px,py) to segments (x0,y0)-(x1,y1) (broadcast arrays) + u + point."""
    vx, vy = x1 - x0, y1 - y0
    L2 = vx * vx + vy * vy + 1e-12
    u = np.clip(((px - x0) * vx + (py - y0) * vy) / L2, 0.0, 1.0)
    cx, cy = x0 + u * vx, y0 + u * vy
    dx, dy = px - cx, py - cy
    return np.sqrt(dx * dx + dy * dy), u, dy


def lsf_refine(D, xg, yg, lsf=None, h=1.0, lam=0.02, lam_a=0.5, c_rob=0.15, max_dev=4.0,
               maxiter=80, use_alpha=True):
    """Refine the trace curve (xg, yg) of one lead. Returns (x, y) on the step-h grid
    (or the inputs if the model cannot be estimated)."""
    import cv2
    from scipy.optimize import minimize
    xg = np.asarray(xg, float); yg = np.asarray(yg, float)
    ok = np.isfinite(xg) & np.isfinite(yg)
    xg, yg = xg[ok], yg[ok]
    if len(xg) < 30:
        return xg, yg
    o = np.argsort(xg, kind="stable"); xg, yg = xg[o], yg[o]
    keep = np.r_[True, np.diff(xg) > 1e-6]; xg, yg = xg[keep], yg[keep]
    if lsf is None:
        lsf = estimate_lsf(D, xg, yg)
    if lsf is None:
        return xg, yg
    d_tab, l_tab, rsup = lsf
    dl_tab = np.gradient(l_tab, d_tab)
    H, W = D.shape
    xk = np.arange(xg[0], xg[-1] + 1e-9, h)
    n = len(xk)
    if n < 10:
        return xg, yg
    y0 = np.interp(xk, xg, yg)
    # tube: pixels within rsup+1.5 px of the initial curve
    m = int(np.ceil(rsup + max_dev)) + 2
    bx0 = max(0, int(np.floor(xk[0])) - m); bx1 = min(W, int(np.ceil(xk[-1])) + m + 1)
    by0 = max(0, int(np.floor(y0.min())) - m); by1 = min(H, int(np.ceil(y0.max())) + m + 1)
    hh, ww = by1 - by0, bx1 - bx0
    S = 4  # rasterize at 1/4 px for the distance
    canvas = np.full((hh * 1, ww * 1), 255, np.uint8)
    pts = np.stack([(xk - bx0), (y0 - by0)], 1)
    cv2.polylines(canvas, [np.round(pts * S).astype(np.int32)], False, 0, 1, cv2.LINE_8, shift=2)
    dist = cv2.distanceTransform(canvas, cv2.DIST_L2, 5)
    tube = dist <= rsup + 1.5
    py, px = np.nonzero(tube)
    if px.size < 50:
        return xg, yg
    px = px.astype(float) + bx0; py = py.astype(float) + by0
    Dv = D[py.astype(int), px.astype(int)].astype(float)
    # candidate segments per pixel: indices whose x covers [px - rsup, px + rsup]
    kspan = int(np.ceil((rsup + 0.5) / h)) + 1
    kc = np.clip(np.round((px - xk[0]) / h).astype(int), 0, n - 2)
    offs = np.arange(-kspan, kspan + 1)
    K = np.clip(kc[:, None] + offs[None, :], 0, n - 2)  # segment K: (K, K+1)
    PX = px[:, None]; PY = py[:, None]
    X0 = xk[K]; X1 = xk[K + 1]
    ns = n - 1

    def unpack(z):
        y = z[:n]
        a = z[n:] if use_alpha else np.ones(ns)
        return y, a

    def fun(z):
        y, a = unpack(z)
        d, u, dy = _seg_dist(PX, PY, X0, y[K], X1, y[K + 1])
        j = np.argmin(d, 1)
        r_ = np.arange(len(px))
        dm = d[r_, j]; um = u[r_, j]; dym = dy[r_, j]; km = K[r_, j]
        L = np.interp(dm, d_tab, l_tab, right=0.0)
        dL = np.interp(dm, d_tab, dl_tab, right=0.0)
        am = a[km]
        M = am * L
        r = M - Dv
        # Cauchy
        rr = r / c_rob
        loss = 0.5 * c_rob ** 2 * np.log1p(rr * rr).sum()
        psi = r / (1 + rr * rr)  # dρ/dr
        # dd/dy0 = -(dy/d) * (1-u) ; dd/dy1 = -(dy/d) * u   (dy = py - cy)
        g_d = psi * am * dL
        inv = np.where(dm > 1e-6, 1.0 / np.maximum(dm, 1e-6), 0.0)
        gy0 = g_d * (-dym * inv) * (1 - um)
        gy1 = g_d * (-dym * inv) * um
        gy = np.bincount(km, gy0, minlength=n) + np.bincount(km + 1, gy1, minlength=n)
        # second-order regularization on y (px)
        d2 = y[:-2] - 2 * y[1:-1] + y[2:]
        loss += 0.5 * lam * (d2 * d2).sum()
        gy[:-2] += lam * d2; gy[1:-1] -= 2 * lam * d2; gy[2:] += lam * d2
        if use_alpha:
            ga = np.bincount(km, psi * L, minlength=ns)
            loss += 0.5 * lam_a * ((a - 1) ** 2).sum()
            ga += lam_a * (a - 1)
            return loss, np.r_[gy, ga]
        return loss, gy

    z0 = np.r_[y0, np.ones(ns)] if use_alpha else y0.copy()
    bounds = [(v - max_dev, v + max_dev) for v in y0]
    if use_alpha:
        bounds += [(0.25, 1.3)] * ns
    res = minimize(fun, z0, jac=True, method="L-BFGS-B", bounds=bounds,
                   options=dict(maxiter=maxiter))
    y, _ = unpack(res.x)
    return xk, y
