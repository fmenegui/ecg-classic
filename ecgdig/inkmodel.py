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
