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


def grid_map(img_bgr: np.ndarray, paper: np.ndarray | None, model_path: str) -> np.ndarray:
    key = (model_path, img_bgr.shape, int(img_bgr[::97, ::89].astype(np.int64).sum()))
    if key not in _CACHE:
        if len(_CACHE) > 6:
            _CACHE.clear()
        _CACHE[key] = grid_map_from_phase(predict_phase(img_bgr, model_path))
    G = _CACHE[key].copy()
    if paper is not None:
        G[paper == 0] = 0
    return G
