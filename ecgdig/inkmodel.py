"""Hybrid mode: trace-ink map from a small network (ONNX) instead of the classical ink map.

The network only says where the ECG pen ink is in the rectified image (10 px/mm); grid, rows,
pulses, tracking, synthesis and constraints stay classical. Optional dependency: onnxruntime.
"""
from __future__ import annotations

import numpy as np

_SESS = {}


def _session(path):
    if path not in _SESS:
        import onnxruntime as ort
        so = ort.SessionOptions()
        so.intra_op_num_threads = 2
        _SESS[path] = ort.InferenceSession(path, so, providers=["CPUExecutionProvider"])
    return _SESS[path]


def predict(rect_bgr: np.ndarray, model_path: str, tile: int = 512, margin: int = 32) -> np.ndarray:
    """Probability (0..1) that each pixel is trace ink. Tiles with overlap (no seams)."""
    sess = _session(model_path)
    H, W = rect_bgr.shape[:2]
    x = rect_bgr[..., ::-1].astype(np.float32) / 255.0  # RGB
    out = np.zeros((H, W), np.float32)
    step = tile - 2 * margin
    for y0 in range(0, H, step):
        for x0 in range(0, W, step):
            ya, xa = max(0, y0 - margin), max(0, x0 - margin)
            yb, xb = min(H, y0 + step + margin), min(W, x0 + step + margin)
            t = x[ya:yb, xa:xb]
            ph, pw = (-t.shape[0]) % 32, (-t.shape[1]) % 32   # encoders downsample by 32
            if ph or pw:
                t = np.pad(t, ((0, ph), (0, pw), (0, 0)), mode="edge")
            lg = sess.run(None, {"x": t.transpose(2, 0, 1)[None]})[0][0, 0]
            p = 1.0 / (1.0 + np.exp(-lg))
            p = p[:yb - ya, :xb - xa]
            oy, ox = y0 - ya, x0 - xa
            out[y0:min(H, y0 + step), x0:min(W, x0 + step)] = p[oy:oy + min(step, H - y0), ox:ox + min(step, W - x0)]
    return out


def predict_row(rect_bgr: np.ndarray, yc: float, model_path: str, h: int = 512):
    """Row network: coverage (0..1) of the trace of the row whose baseline is near y = yc, in a
    strip of height h centred there. Returns (y0, P) with P of shape (h, W) and y0 the strip's
    first image row (may be negative: the strip is padded with the image's edge)."""
    sess = _session(model_path)
    H, W = rect_bgr.shape[:2]
    y0 = int(round(yc)) - h // 2
    ya, yb = max(0, y0), min(H, y0 + h)
    strip = rect_bgr[ya:yb]
    strip = np.pad(strip, ((ya - y0, y0 + h - yb), (0, (-W) % 32), (0, 0)), mode="edge")
    x = strip[..., ::-1].astype(np.float32).transpose(2, 0, 1)[None] / 255.0
    lg = sess.run(None, {"x": x})[0][0, 0][:, :W]
    return y0, (1.0 / (1.0 + np.exp(-lg))).astype(np.float32)


def decode_columns(P: np.ndarray, y0: int = 0, half: int = 3):
    """Per-column signal height from a row-network map trained on a sparse signal label (the
    signal's height split over 2 pixels): sub-pixel centroid within +-half px of the column's
    peak. Returns (y, conf) over the strip's columns; y in image rows (strip offset y0 added)."""
    H, W = P.shape
    k = np.argmax(P, axis=0)
    conf = P[k, np.arange(W)]
    rows = k[None, :] + np.arange(-half, half + 1)[:, None]
    ok = (rows >= 0) & (rows < H)
    w = np.where(ok, P[np.clip(rows, 0, H - 1), np.arange(W)[None, :]], 0.0)
    y = (w * rows).sum(0) / np.maximum(w.sum(0), 1e-9)
    # pixel-centre convention (row i is at y = i), as the classical centreline
    return y + y0, conf
