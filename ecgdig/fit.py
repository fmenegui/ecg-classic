"""Ajuste por renderização inversa com o perfil de pena medido (LSF).

Modelo de formação da imagem para um traço isolado: a tinta num pixel p depende só da
distância de p à curva do ECG,

    M(p) = alpha_k * LSF(d(p, C)),

onde C é a poligonal (x_k, y_k) com x_k numa grade fixa (passo h px, a curva é função do
tempo), d é a distância euclidiana ao segmento mais próximo (união: a tinta não soma onde a
pena passa duas vezes, ela satura) e LSF é o perfil transversal da pena + borrão, medido
na própria imagem nos trechos planos (não paramétrico: serve para pena fina borrada e para
pena grossa nítida). alpha_k é a densidade de tinta de cada segmento (traços rápidos saem
mais claros em scans e fotos), com prior em 1.

As incógnitas y_k (e alpha_k) saem de mínimos quadrados robustos (Cauchy) entre M e o mapa
de tinta D num tubo em volta da curva inicial, com regularização fraca de segunda ordem.
Como a curva é parametrizada pelo tempo, não há problema de ordem dos pontos no QRS.

Sem aprendizado: é a inversa do renderizador, com o renderizador medido na imagem.
"""
from __future__ import annotations

import numpy as np


def estimate_lsf(D, xg, yg, rsup_max=7.0, step=0.25):
    """Perfil transversal da tinta em função da distância à linha central (trechos planos).
    Devolve (d_tab, lsf_tab, rsup) ou None."""
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
    # centro sub-pixel por coluna: centróide da tinta acima de 20% do máximo local
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
    # fundo (papel) -> 0; perfil monotônico não crescente
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
    """Distância de (px,py) aos segmentos (x0,y0)-(x1,y1) (arrays broadcast) + u + ponto."""
    vx, vy = x1 - x0, y1 - y0
    L2 = vx * vx + vy * vy + 1e-12
    u = np.clip(((px - x0) * vx + (py - y0) * vy) / L2, 0.0, 1.0)
    cx, cy = x0 + u * vx, y0 + u * vy
    dx, dy = px - cx, py - cy
    return np.sqrt(dx * dx + dy * dy), u, dy


def lsf_refine(D, xg, yg, lsf=None, h=1.0, lam=0.02, lam_a=0.5, c_rob=0.15, max_dev=4.0,
               maxiter=80, use_alpha=True):
    """Refina a curva (xg, yg) do traço de uma derivação. Devolve (x, y) na grade de passo h
    (ou os originais se o modelo não puder ser estimado)."""
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
    # tubo: pixels a menos de rsup+1.5 px da curva inicial
    m = int(np.ceil(rsup + max_dev)) + 2
    bx0 = max(0, int(np.floor(xk[0])) - m); bx1 = min(W, int(np.ceil(xk[-1])) + m + 1)
    by0 = max(0, int(np.floor(y0.min())) - m); by1 = min(H, int(np.ceil(y0.max())) + m + 1)
    hh, ww = by1 - by0, bx1 - bx0
    S = 4  # rasteriza com 1/4 px para a distância
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
    # segmentos candidatos por pixel: índices cujo x cobre [px - rsup, px + rsup]
    kspan = int(np.ceil((rsup + 0.5) / h)) + 1
    kc = np.clip(np.round((px - xk[0]) / h).astype(int), 0, n - 2)
    offs = np.arange(-kspan, kspan + 1)
    K = np.clip(kc[:, None] + offs[None, :], 0, n - 2)  # segmento K: (K, K+1)
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
        # regularização de 2ª ordem em y (px)
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
