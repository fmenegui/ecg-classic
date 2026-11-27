"""ONNX appearance restorer (hybrid/train_restore.py) run tile by tile on a page.

Gate as the classical restorer: only blurred or tiny pages are touched. The page is brought to
the scale the network was trained at (longer side 1980 px, about 7 px/mm), restored in
overlapping tiles (residual output added to the input) and returned at that scale."""
from __future__ import annotations

import numpy as np
import cv2

from .restore import MIN_SIDE, SIGMA_ON, blur_sigma

_SESS = {}
LONG = 1980
TILE, PAD = 512, 48


def _sess(path):
    if path not in _SESS:
        import onnxruntime as ort
        so = ort.SessionOptions(); so.intra_op_num_threads = 2
        _SESS[path] = ort.InferenceSession(path, so, providers=["CPUExecutionProvider"])
    return _SESS[path]


def run(img: np.ndarray, path: str) -> tuple[np.ndarray, dict]:
    info = dict(kind="model", sigma=0.0, upscale=1.0, restored=False)
    side = max(img.shape[:2])
    s_native = blur_sigma(img.mean(2) if img.ndim == 3 else img)
    info["sigma"] = round(s_native, 2)
    if side >= MIN_SIDE and s_native <= SIGMA_ON:
        return img, info
    f = LONG / side
    x = cv2.resize(img, None, fx=f, fy=f, interpolation=cv2.INTER_AREA if f < 1 else cv2.INTER_LANCZOS4)
    info["upscale"] = round(f, 3)
    rgb = x[..., ::-1].astype(np.float32) / 255.0 if x.ndim == 3 else np.repeat(x[..., None], 3, 2).astype(np.float32) / 255.0
    H, W = rgb.shape[:2]
    out = np.zeros_like(rgb); wsum = np.zeros((H, W, 1), np.float32)
    sess = _sess(path)
    step = TILE - 2 * PAD
    for y0 in range(0, max(1, H - 2 * PAD), step):
        for x0 in range(0, max(1, W - 2 * PAD), step):
            y1, x1 = min(H, y0 + TILE), min(W, x0 + TILE)
            t = rgb[y0:y1, x0:x1]
            h, w = t.shape[:2]
            ph, pw = (-h) % 32, (-w) % 32
            tp = np.pad(t, ((0, ph), (0, pw), (0, 0)), mode="edge")
            r = sess.run(None, {"x": tp.transpose(2, 0, 1)[None]})[0][0].transpose(1, 2, 0)[:h, :w]
            o = np.clip(t + r, 0, 1)
            wy = np.minimum(np.arange(h) + 1, np.arange(h)[::-1] + 1).astype(np.float32)
            wx = np.minimum(np.arange(w) + 1, np.arange(w)[::-1] + 1).astype(np.float32)
            wt = np.minimum(np.minimum(wy[:, None], PAD), np.minimum(wx[None, :], PAD))[..., None]
            out[y0:y1, x0:x1] += o * wt; wsum[y0:y1, x0:x1] += wt
    out = out / np.maximum(wsum, 1e-6)
    info["restored"] = True
    return (out[..., ::-1] * 255).round().clip(0, 255).astype(np.uint8), info
