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


def _pav(y):
    """Regressão isotônica (não decrescente) por pool-adjacent-violators. Devolve (valor por
    ponto, rótulo do bloco por ponto)."""
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
    """Ordena a curva em x sem embaralhar a ordem do arco: x vira não decrescente pela
    regressão isotônica (média dos trechos que voltam para trás) e, dentro de cada trecho
    "vertical" agrupado, os pontos se espalham em +-spread px na ordem do arco. Ordenar por x
    puro intercala os pontos da subida e da descida do QRS e vira serrilhado."""
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
    # garante estritamente crescente
    o = np.maximum.accumulate(out)
    keep = np.r_[True, np.diff(o) > 1e-4]
    return o[keep], np.asarray(cy)[keep]


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


SYNTH_BUDGET = [4.0]
SYNTH_MAXWORK = [3e7]
SYNTH_WORK = [2e8]      # pares ponto-pixel x iterações por segmento (limite determinístico)


def synth_refine(D, xg, yg, rad, n_iter=60, ds=0.5, lam=2.0, max_shift=1.5, max_rad=2.0, order="sort",
                 time_budget=None):
    """Refina os pontos (xg, yg) da curva casando o traço renderizado com D.
    Devolve (x, y) novos, ordenados em x, ou os originais se o ajuste não for confiável."""
    xg = np.asarray(xg, float); yg = np.asarray(yg, float)
    ok = np.isfinite(xg) & np.isfinite(yg)
    xg, yg = xg[ok], yg[ok]
    if len(xg) < 20:
        return xg, yg
    if rad > max_rad:
        # pena grossa de contorno nítido: o modelo morfológico (inversão pelas bordas) já é
        # exato e a gaussiana erra o perfil chapado, puxando os picos estreitos para baixo.
        # A síntese serve ao traço fino e borrado (scans, fotos, render do PhysioNet)
        return xg, yg
    psf = estimate_psf(D, xg, yg, rad)
    if psf is None:
        return xg, yg
    A, sig = psf
    if sig > 3.0:
        return xg, yg  # borrão maior que a própria pena: o modelo não separa curva e mancha
    a = A * ds / (np.sqrt(2 * np.pi) * sig)
    H, W = D.shape
    cx0, cy0 = _densify(xg, yg, ds)
    K = len(cx0)
    rk = int(np.ceil(3 * sig)) + 1
    o1 = np.arange(-rk, rk + 1)
    n1 = len(o1)
    npair = n1 * n1
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

    CH = max(1, 400000 // npair)  # pontos por bloco: limita a memória (~400 mil pares)
    pat = (o1[:, None] * w + o1[None, :]).ravel()   # deslocamento do índice de cada vizinho (y, x)
    inv2 = 0.5 / (sig * sig)

    def _blocks(fx, fy):
        """Por bloco de pontos: índice do pixel, peso gaussiano e deslocamentos de cada par
        (ponto, pixel vizinho). A gaussiana 2D é o produto de duas 1D (n1 exponenciais por
        eixo em vez de n1*n1 por ponto), e os blocos de uma iteração servem ao desenho E ao
        gradiente: antes eram recalculados do zero nas duas passadas."""
        out = []
        for c0 in range(0, K, CH):
            fxc, fyc = fx[c0:c0 + CH], fy[c0:c0 + CH]
            ix = np.floor(fxc).astype(np.int64); iy = np.floor(fyc).astype(np.int64)
            px = ix[:, None] + o1[None, :]                                  # (k, n1)
            py = iy[:, None] + o1[None, :]
            dx = px - fxc[:, None]; dy = py - fyc[:, None]
            ex = np.exp(-inv2 * dx * dx); ey = a * np.exp(-inv2 * dy * dy)
            if ix.min() >= rk and ix.max() < w - rk and iy.min() >= rk and iy.max() < h - rk:
                # caso comum (a região tem margem em volta da curva): tudo dentro
                idx = (iy * w + ix)[:, None] + pat[None, :]                 # (k, n1*n1)
            else:
                inx = (px >= 0) & (px < w); iny = (py >= 0) & (py < h)
                ex = np.where(inx, ex, 0.0); ey = np.where(iny, ey, 0.0)    # fora da região: peso 0
                idx = ((np.clip(py, 0, h - 1) * w)[:, :, None] + np.clip(px, 0, w - 1)[:, None, :]).reshape(len(ix), npair)
            gv = (ey[:, :, None] * ex[:, None, :]).reshape(len(ix), npair)
            out.append((c0, idx, gv, dx, dy))
        return out

    if time_budget is None:
        time_budget = SYNTH_BUDGET[0]
    if K * npair > SYNTH_MAXWORK[0]:
        return xg, yg  # curva longuíssima com borrão largo: custo proibitivo, ganho incerto
    # orçamento determinístico (pares ponto-pixel por iteração x iterações): o mesmo resultado em
    # qualquer máquina. O limite por relógio fica só como freio de emergência.
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
        # regularização de 2ª ordem (curva suave na escala de ~1 px de arco)
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
