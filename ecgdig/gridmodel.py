"""Hybrid geometry: the grid seen by a network (continuous grid phase, hybrid/gridgen.py GridPhase).

The network gives, for every pixel of the photo, the cos/sin of its position on the paper's grid
modulo 5 mm and modulo 1 mm, and whether the pixel is grid paper. From it we draw a clean grid
map: the 5 mm lines and the 1 mm lines where the phase says a line passes, nothing elsewhere.
Traces, text, shadows, stains and dot grids do not reach it. It replaces the classical grid map
(geometry.grid_map) feeding rotation, scale, the cell field and the snaps, which stay classical:
the network does not decide the geometry, it only cleans what the geometry measures.
Optional dependency: onnxruntime.
"""
from __future__ import annotations

import numpy as np

from .inkmodel import _session

_CACHE: dict = {}


def predict_phase(img_bgr: np.ndarray, model_path: str, tile: int = 512, margin: int = 48) -> np.ndarray:
    """(9, H, W): cos/sin u5, cos/sin v5, cos/sin u1, cos/sin v1, confidence (0..1). Tiles with overlap."""
    sess = _session(model_path)
    H, W = img_bgr.shape[:2]
    x = img_bgr[..., ::-1].astype(np.float32) / 255.0
    out = np.zeros((9, H, W), np.float32)
    step = tile - 2 * margin
    for y0 in range(0, H, step):
        for x0 in range(0, W, step):
            ya, xa = max(0, y0 - margin), max(0, x0 - margin)
            yb, xb = min(H, y0 + step + margin), min(W, x0 + step + margin)
            t = x[ya:yb, xa:xb]
            ph, pw = (-t.shape[0]) % 32, (-t.shape[1]) % 32
            if ph or pw:
                t = np.pad(t, ((0, ph), (0, pw), (0, 0)), mode="edge")
            o = sess.run(None, {"x": t.transpose(2, 0, 1)[None]})[0][0][:, :yb - ya, :xb - xa]
            oy, ox = y0 - ya, x0 - xa
            hh, ww = min(step, H - y0), min(step, W - x0)
            out[:, y0:y0 + hh, x0:x0 + ww] = o[:, oy:oy + hh, ox:ox + ww]
    out[8] = 1.0 / (1.0 + np.exp(-out[8]))
    return out


def _line(c, s, sharp):
    """Line strength from a (cos, sin) pair: 1 on the line (phase 0), 0 away from it."""
    n = np.sqrt(c * c + s * s) + 1e-6
    return np.clip((c / n - sharp) / (1 - sharp), 0, 1)


def grid_map_from_phase(P: np.ndarray, paper: np.ndarray | None = None) -> np.ndarray:
    """Grid map in the classical convention (0 = background, <= 0.5 = grid line): 5 mm lines 0.5,
    1 mm lines 0.25, weighted by the network's confidence."""
    conf = P[8]
    major = np.maximum(_line(P[0], P[1], 0.97), _line(P[2], P[3], 0.97))
    minor = np.maximum(_line(P[4], P[5], 0.8), _line(P[6], P[7], 0.8))
    G = (0.5 * major + 0.25 * minor * (1 - major)) * conf
    G = np.clip(G, 0, 0.5).astype(np.float32)
    if paper is not None:
        G[paper == 0] = 0
    return G


MIN_GAIN = [0.3]
MAX_SIDE = [1600]
MIN_COH = [0.9]    # agreement of the network's 5 mm phase with the fitted smooth grid, both axes   # line contrast the snapped network grid must reach on both axes to be used


def grid_map(img_bgr: np.ndarray, paper: np.ndarray | None, model_path: str):
    """Clean grid map drawn from the network's smooth geometry, snapped to the image's own lines;
    None when the network's grid does not match the image (the caller keeps the classical map)."""
    key = (model_path, img_bgr.shape, int(img_bgr[::97, ::89].astype(np.int64).sum()))
    if key not in _CACHE:
        if len(_CACHE) > 6:
            _CACHE.clear()
        # network, fit and snap at <= MAX_SIDE px (memory and time; the map is smooth), then the
        # coordinates are resized to the image and the lines drawn at full resolution
        import cv2
        H, W = img_bgr.shape[:2]
        sc = min(1.0, MAX_SIDE[0] / max(H, W))
        small = cv2.resize(img_bgr, (max(32, round(W * sc)), max(32, round(H * sc))),
                           interpolation=cv2.INTER_AREA) if sc < 1 else img_bgr
        P = predict_phase(small, model_path)
        G = None
        if float(P[8].mean()) > 0.1:
            U, V, pap = fit_smooth(P)
            coh = coherence(P, U, V)
            U2, V2, gains = snap_offsets(small, U, V, pap)
            # the image moves an axis only when its own lines clearly agree (faint grids: keep the
            # network's offset); the network is used only when its phase is a smooth grid (a domain
            # gap shows as an incoherent phase, measured on real photos: > 0.95 good, < 0.3 bad)
            U = U2 if gains[0] >= MIN_GAIN[0] else U
            V = V2 if gains[1] >= MIN_GAIN[0] else V
            del P
            if min(coh) >= MIN_COH[0]:
                if sc < 1:
                    U = cv2.resize(U.astype(np.float32), (W, H), interpolation=cv2.INTER_LINEAR)
                    V = cv2.resize(V.astype(np.float32), (W, H), interpolation=cv2.INTER_LINEAR)
                    pap = cv2.resize(pap.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST) > 0
                G = draw_grid(U.astype(np.float32), V.astype(np.float32), pap)
        _CACHE[key] = G
    G = _CACHE[key]
    if G is None:
        return None
    G = G.copy()
    if paper is not None:
        G[paper == 0] = 0
    return G


# --------------------------------------------------------------------------------------------
# Smooth geometry from the phase. Per-pixel phases are noisy (traces, text, a domain gap between
# rendered and real photos), and drawing lines where cos(phase) ~ 1 turns that noise into wavy,
# broken lines. The paper is smooth, so we fit it instead: the phase gradient (local grid
# frequency, read from products z(x+1) conj(z(x)), never unwrapped) is integrated by a weighted,
# outlier-robust least-squares fit of a smooth polynomial u(x, y), v(x, y) in mm; the free
# offset comes from the circular mean of the 5 mm phase and is refined by the 1 mm phase.


def _basis(x, y, deg):
    """Monomials x^i y^j (i + j <= deg) and their x / y derivatives."""
    B, Bx, By = [], [], []
    for i in range(deg + 1):
        for j in range(deg + 1 - i):
            B.append(x ** i * y ** j)
            Bx.append(i * x ** max(i - 1, 0) * y ** j if i else np.zeros_like(x))
            By.append(j * x ** i * y ** max(j - 1, 0) if j else np.zeros_like(x))
    return np.stack(B, -1), np.stack(Bx, -1), np.stack(By, -1)


def _fit_axis(zc, zs, z1c, z1s, w, xs, ys, scale, deg, iters=6):
    """One grid axis: polynomial u (mm) whose 5 mm phase matches the network's."""
    z = (zc + 1j * zs) / (np.abs(zc + 1j * zs) + 1e-6)
    gx = np.angle(z[:, 1:] * np.conj(z[:, :-1]))[:-1, :]          # rad per sample step
    gy = np.angle(z[1:, :] * np.conj(z[:-1, :]))[:, :-1]
    ww = (w[:-1, :-1] * w[:-1, 1:] * w[1:, :-1]).ravel()
    X, Y = xs[:-1, :-1].ravel(), ys[:-1, :-1].ravel()
    _, Bx, By = _basis(X, Y, deg)
    k = 5.0 / (2 * np.pi) / scale                                   # rad per step -> mm per normalised unit
    A = np.concatenate([Bx, By]); b = np.concatenate([gx.ravel(), gy.ravel()]) * k
    W = np.concatenate([ww, ww])
    sel = W > 0.05
    A, b, W = A[sel], b[sel], W[sel]
    coef = np.zeros(A.shape[1])
    for _ in range(iters):                                          # IRLS, Huber
        r = A @ coef - b
        s = 1.4826 * np.median(np.abs(r)) + 1e-9
        hub = np.where(np.abs(r) <= 2 * s, 1.0, 2 * s / np.abs(r))
        ws = W * hub
        M = A.T @ (A * ws[:, None]) + 1e-6 * np.eye(A.shape[1])
        coef = np.linalg.solve(M, A.T @ (ws * b))
    B, _, _ = _basis(xs.ravel(), ys.ravel(), deg)
    u = (B @ coef).reshape(xs.shape)
    # offset from the 5 mm phase, then from the 1 mm phase
    r5 = np.angle(z * np.exp(-2j * np.pi * u / 5.0))
    off = np.angle(np.sum(w * np.exp(1j * r5))) * 5.0 / (2 * np.pi)
    u = u + off
    z1 = (z1c + 1j * z1s) / (np.abs(z1c + 1j * z1s) + 1e-6)
    r1 = np.angle(z1 * np.exp(-2j * np.pi * u))
    c1 = np.sum(w * np.exp(1j * r1))
    if np.abs(c1) > 0.3 * np.sum(w):                                # the 1 mm phase agrees enough
        u = u + np.angle(c1) / (2 * np.pi)
    return coef, off, u


def fit_smooth(P: np.ndarray, deg: int = 4, step: int = 4):
    """Smooth paper coordinates (U, V) in mm (modulo the 5 mm lattice) for every pixel, from the
    network output P (9, H, W); and the paper mask (confidence > 0.5)."""
    H, W = P.shape[1:]
    Ps = P[:, ::step, ::step]
    h, w = Ps.shape[1:]
    sc = max(H, W) / 2.0
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float64)
    xs = (xs * step - W / 2) / sc; ys = (ys * step - H / 2) / sc
    wgt = np.clip(Ps[8].astype(np.float64), 0, 1) ** 2
    out = []
    for c in ((0, 1, 4, 5), (2, 3, 6, 7)):
        coef, off, _ = _fit_axis(*(Ps[i].astype(np.float64) for i in c), wgt, xs, ys, step / sc, deg)
        out.append((coef, off))
    # full resolution: evaluate the polynomials, then the same 1 mm offset refinement
    xf1 = (np.arange(W) - W / 2) / sc
    maps = []
    for (coef, off), c in zip(out, ((0, 1, 4, 5), (2, 3, 6, 7))):
        u = np.empty((H, W), np.float64)
        for y0 in range(0, H, 256):                                 # in row blocks (memory)
            yy, xx = np.meshgrid((np.arange(y0, min(H, y0 + 256)) - H / 2) / sc, xf1, indexing="ij")
            Bf, _, _ = _basis(xx.ravel(), yy.ravel(), deg)
            u[y0:y0 + len(yy)] = (Bf @ coef).reshape(yy.shape)
        u += off
        z1 = (P[c[2]] + 1j * P[c[3]]); z1 = z1 / (np.abs(z1) + 1e-6)
        c1 = np.sum(P[8] * np.exp(1j * np.angle(z1 * np.exp(-2j * np.pi * u))))
        if np.abs(c1) > 0.3 * P[8].sum():
            u = u + np.angle(c1) / (2 * np.pi)
        maps.append(u)
    return maps[0], maps[1], P[8] > 0.5


def grid_map_smooth(P: np.ndarray, deg: int = 4) -> np.ndarray:
    U, V, paper = fit_smooth(P, deg)
    return draw_grid(U, V, paper)


def draw_grid(U, V, paper):
    """Clean grid map drawn from smooth coordinates (mm): 5 mm lines 0.5, 1 mm lines 0.25, about
    one pixel wide, on the paper only."""
    def dist_px(a, period):
        d = np.abs((a + period / 2) % period - period / 2)          # mm to the nearest line
        g = np.hypot(*np.gradient(a))                               # mm per px across the line
        return d / np.maximum(g, 1e-6)
    major = np.maximum(np.exp(-0.5 * (dist_px(U, 5.0) / 0.7) ** 2), np.exp(-0.5 * (dist_px(V, 5.0) / 0.7) ** 2))
    minor = np.maximum(np.exp(-0.5 * (dist_px(U, 1.0) / 0.6) ** 2), np.exp(-0.5 * (dist_px(V, 1.0) / 0.6) ** 2))
    G = (0.5 * major + 0.25 * minor * (1 - major)) * paper
    return G.astype(np.float32)


def snap_offsets(img_bgr: np.ndarray, U: np.ndarray, V: np.ndarray, paper: np.ndarray, step: int = 2):
    """The network gives the shape of the grid (scale, rotation, warp); its absolute phase can be off
    on real photos (domain gap). The image decides it: shift U (and V) by the offset, in [0, 5) mm,
    that puts the most dark pixels on the 1 mm and 5 mm lines it predicts. Returns the shifted maps
    and, per axis, the contrast gain of the best offset over the median offset (0 = no grid seen)."""
    d = 1.0 - cv2_gray(img_bgr)[::step, ::step]
    pm = paper[::step, ::step]
    out, gains = [], []
    for A in (U, V):
        a = A[::step, ::step]
        g = np.hypot(*np.gradient(a)) / step + 1e-6                 # mm per px
        offs = np.arange(0, 5, 0.04)
        sc = []
        for o in offs:
            d5 = np.abs((a + o + 2.5) % 5 - 2.5) / g
            d1 = np.abs((a + o + 0.5) % 1 - 0.5) / g
            m5 = pm & (d5 < 0.8); m1 = pm & (d1 < 0.8) & ~m5
            sc.append((d[m5].mean() if m5.any() else 0) + 0.5 * (d[m1].mean() if m1.any() else 0))
        sc = np.array(sc)
        k = int(np.argmax(sc))
        out.append(A + offs[k])
        gains.append(float((sc[k] - np.median(sc)) / (np.median(sc) + 1e-6)))
    return out[0], out[1], gains


def cv2_gray(img_bgr):
    f = img_bgr.astype(np.float32) / 255.0
    return f.mean(2)


def coherence(P, U, V):
    """Per axis, |weighted mean of exp(i (network 5 mm phase - fitted phase))|: 1 when the network's
    phase is exactly a smooth grid, ~0 when it is noise."""
    w = P[8]
    out = []
    for A, c, s_ in ((U, 0, 1), (V, 2, 3)):
        z = P[c] + 1j * P[s_]
        z = z / (np.abs(z) + 1e-6)
        out.append(float(np.abs(np.sum(w * z * np.exp(-2j * np.pi * A / 5))) / (w.sum() + 1e-6)))
    return out
