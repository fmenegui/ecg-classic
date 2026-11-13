"""Ajuste por síntese (análise por síntese) do traço de uma derivação.

Modelo de formação da imagem: a tinta vista num pixel p é
    M(p) = 1 - exp(-S(p)),   S(p) = sum_k a * g_sigma(p - c_k)
onde c_k são pontos da curva do ECG amostrados a cada `ds` px de comprimento de arco e
g_sigma é uma gaussiana isotrópica que junta a pena e o borrão da captura. O 1 - exp(-S)
satura onde o traço passa várias vezes (pico estreito), como a tinta real.

Os pontos são deslocados ao longo da normal da curva (delta_k) para casar M com o mapa de
tinta D (mínimos quadrados), com regularização de segunda ordem em delta. a e sigma vêm da
própria imagem (perfil vertical do traço nos trechos planos).

Sem aprendizado: é o "inverso do renderizador" com um modelo físico da pena + borrão.
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


def estimate_psf(D, xs, yc, rad, n_max=400):
    """(a, sigma) pelo perfil vertical da tinta em colunas planas do traço."""
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
    # densidade linear: pontos a cada ds ao longo da curva -> S de uma linha horizontal =
    # a/ds * sqrt(2*pi)*s * exp(-y^2/2s^2)  =>  a = A * ds / (sqrt(2*pi) * s)
    return float(A), float(s)


def synth_refine(D, xg, yg, rad, n_iter=60, ds=0.5, lam=2.0, max_shift=1.5):
    """Refina os pontos (xg, yg) da curva casando o traço renderizado com D.
    Devolve (x, y) novos, ordenados em x, ou os originais se o ajuste não for confiável."""
    xg = np.asarray(xg, float); yg = np.asarray(yg, float)
    ok = np.isfinite(xg) & np.isfinite(yg)
    xg, yg = xg[ok], yg[ok]
    if len(xg) < 20:
        return xg, yg
    psf = estimate_psf(D, xg, yg, rad)
    if psf is None:
        return xg, yg
    A, sig = psf
    a = A * ds / (np.sqrt(2 * np.pi) * sig)
    H, W = D.shape
    cx0, cy0 = _densify(xg, yg, ds)
    K = len(cx0)
    rk = int(np.ceil(3 * sig)) + 1
    oy, ox = np.mgrid[-rk:rk + 1, -rk:rk + 1]
    oy = oy.ravel(); ox = ox.ravel()
    # região de trabalho: retângulo em volta da curva
    x0 = max(0, int(np.floor(cx0.min())) - rk - 2); x1 = min(W, int(np.ceil(cx0.max())) + rk + 3)
    y0 = max(0, int(np.floor(cy0.min())) - rk - 2); y1 = min(H, int(np.ceil(cy0.max())) + rk + 3)
    Dp = D[y0:y1, x0:x1].astype(float)
    h, w = Dp.shape
    # só pixels perto da curva inicial contam (tubo): o resto pode ser outro traço/texto
    tube = np.zeros((h, w), bool)
    bx = np.clip(np.round(cx0).astype(int) - x0, 0, w - 1); by = np.clip(np.round(cy0).astype(int) - y0, 0, h - 1)
    tube[by, bx] = True
    from scipy.ndimage import binary_dilation
    tube = binary_dilation(tube, iterations=rk + 1)
    nx, ny = _normals(cx0, cy0)
    delta = np.zeros(K)
    lr = 0.5 / (a * a + 1e-9)

    def render(cx, cy):
        fx = cx - x0; fy = cy - y0
        ix = np.floor(fx).astype(int); iy = np.floor(fy).astype(int)
        px = ix[:, None] + ox[None, :]; py = iy[:, None] + oy[None, :]
        dx = px - fx[:, None]; dy = py - fy[:, None]
        gv = a * np.exp(-0.5 * (dx * dx + dy * dy) / (sig * sig))
        inside = (px >= 0) & (px < w) & (py >= 0) & (py < h)
        S = np.bincount((py * w + px)[inside], weights=gv[inside], minlength=h * w)
        return S.reshape(h, w), px, py, dx, dy, gv, inside

    prev = None
    for it in range(n_iter):
        cx = cx0 + delta * nx; cy = cy0 + delta * ny
        S, px, py, dx, dy, gv, inside = render(cx, cy)
        M = 1.0 - np.exp(-S)
        r = np.where(tube, M - Dp, 0.0)
        loss = float((r * r).sum())
        if prev is not None and loss > prev * 0.9999 and it > 5:
            lr *= 0.5
            if lr < 1e-3 / (a * a + 1e-9):
                break
        prev = loss
        # dL/dc_k = sum_p 2 r(p) (1 - M(p)) * gv * (p - c_k)/sig^2
        coef = np.zeros_like(gv)
        flat_idx = np.where(inside, py * w + px, 0)
        rr = (2 * r * (1 - M)).ravel()[flat_idx] * inside
        coef = rr * gv / (sig * sig)
        gx = (coef * dx).sum(1); gy = (coef * dy).sum(1)
        grad = gx * nx + gy * ny
        # regularização de 2ª ordem (curva suave na escala de ~1 px de arco)
        lap = np.r_[0.0, delta[:-2] - 2 * delta[1:-1] + delta[2:], 0.0]
        delta = delta - lr * grad + lr * lam * a * a * lap
        delta = np.clip(delta, -max_shift, max_shift)
    cx = cx0 + delta * nx; cy = cy0 + delta * ny
    o = np.argsort(cx, kind="stable")
    cx, cy = cx[o], cy[o]
    keep = np.r_[True, np.diff(cx) > 1e-3]
    return cx[keep], cy[keep]
