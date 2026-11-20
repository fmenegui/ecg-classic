"""Flatten a photographed page with the dense page-flattening network (hybrid/train_dewarp.py).

The network gives, for every pixel of the photo (at 512 x 704), its normalised flat-page
coordinate and whether it shows the page. The flat page is then sampled from the full-resolution
photo: for every pixel of the output page, Newton iterations find the photo position whose
predicted coordinate is that pixel (the scheme of hybrid/rendergen.rectify). The output is an
A4-landscape page image that the classical pipeline reads like a scan (its own grid geometry
still refines scale and residual distortion).
"""
from __future__ import annotations

import numpy as np

IN_HW = (512, 704)
_SESS = {}


def _session(path):
    if path not in _SESS:
        import onnxruntime as ort
        so = ort.SessionOptions()
        so.intra_op_num_threads = 2
        _SESS[path] = ort.InferenceSession(path, so, providers=["CPUExecutionProvider"])
    return _SESS[path]


def predict_field(img_bgr, path):
    """Field (2, 512, 704) of normalised page coordinates and visibility probability (512, 704)."""
    import cv2
    x = cv2.resize(img_bgr, (IN_HW[1], IN_HW[0]), interpolation=cv2.INTER_AREA)[..., ::-1].astype(np.float32) / 255
    out = _session(path).run(None, {"x": x.transpose(2, 0, 1)[None].copy()})[0][0]
    return out[:2], 1 / (1 + np.exp(-out[2]))


def _sample(a, p):
    """Bilinear sample of a (C, h, w) at normalised coordinates p (H, W, 2), like torch's
    grid_sample(align_corners=False, padding_mode="border"); returns (H, W, C)."""
    import cv2
    h, w = a.shape[1:]
    mx = (((p[..., 0] + 1) * w - 1) / 2).astype(np.float32)
    my = (((p[..., 1] + 1) * h - 1) / 2).astype(np.float32)
    return np.stack([cv2.remap(np.ascontiguousarray(c, np.float32), mx, my, cv2.INTER_LINEAR,
                               borderMode=cv2.BORDER_REPLICATE) for c in a], -1)


def flatten(img_bgr, path, out_wh=(2970, 2100), work_wh=(1188, 840), iters=25):
    """The flat page (out_wh, BGR) and the fraction of it the photo shows; None if the network
    sees no page. numpy and OpenCV only (the evaluation hosts have no torch: until Nov 20 the
    dewarp reading failed there with ModuleNotFoundError and was never chosen)."""
    import cv2
    f, vis = predict_field(img_bgr, path)
    if (vis > 0.5).mean() < 0.05:
        return None, 0.0
    f = f.astype(np.float32)
    h, w = f.shape[1:]
    # derivatives of the field (per normalised photo unit) for the Newton step
    fx = np.zeros_like(f); fy = np.zeros_like(f)
    fx[:, :, 1:-1] = (f[:, :, 2:] - f[:, :, :-2]) / (4 / w); fx[:, :, 0] = fx[:, :, 1]; fx[:, :, -1] = fx[:, :, -2]
    fy[:, 1:-1] = (f[:, 2:] - f[:, :-2]) / (4 / h); fy[:, 0] = fy[:, 1]; fy[:, -1] = fy[:, -2]
    stack = np.concatenate([f, fx, fy], 0)                          # (6, h, w)
    Wk, Hk = work_wh
    ys, xs = np.meshgrid(np.arange(Hk, dtype=np.float32), np.arange(Wk, dtype=np.float32), indexing="ij")
    c = np.stack([(xs + 0.5) / Wk * 2 - 1, (ys + 0.5) / Hk * 2 - 1], -1)   # target page coordinates
    # affine start from the visible pixels
    py_, px_ = np.nonzero(vis > 0.5)
    sel = np.random.default_rng(0).permutation(len(py_))[:20000]
    P = np.stack([(px_[sel] + 0.5) / w * 2 - 1, (py_[sel] + 0.5) / h * 2 - 1], 1)
    Q = f[:, py_[sel], px_[sel]].T
    M = np.linalg.lstsq(np.concatenate([P, np.ones((len(P), 1))], 1), Q, rcond=None)[0]   # photo -> page
    A, b = M[:2].T, M[2]
    p = ((c - b) @ np.linalg.inv(A).T).astype(np.float32)
    for _ in range(iters):
        s_ = _sample(stack, p)
        r = c - s_[..., :2]
        det = s_[..., 2] * s_[..., 5] - s_[..., 4] * s_[..., 3]
        det = np.where(np.abs(det) < 1e-6, 1e-6, det)
        dx = (s_[..., 5] * r[..., 0] - s_[..., 4] * r[..., 1]) / det
        dy = (-s_[..., 3] * r[..., 0] + s_[..., 2] * r[..., 1]) / det
        p = p + np.clip(np.stack([dx, dy], -1), -0.2, 0.2).astype(np.float32)
    s_ = _sample(stack, p)
    err = np.linalg.norm((c - s_[..., :2]) * np.array([Wk / 2, Hk / 2], np.float32), axis=-1)
    seen = (err < 1.0) & (np.abs(p) <= 1).all(-1)
    # sample the full-resolution photo on the output page grid
    Wo, Ho = out_wh
    pu = np.stack([cv2.resize(p[..., k], (Wo, Ho), interpolation=cv2.INTER_LINEAR) for k in range(2)], -1)
    su = cv2.resize(seen.astype(np.float32), (Wo, Ho), interpolation=cv2.INTER_LINEAR) > 0.5
    out = _sample(np.ascontiguousarray(img_bgr.transpose(2, 0, 1), np.float32), pu)
    if su.any():
        paper = np.median(out[su], 0)
        out = np.where(su[..., None], out, paper[None, None])
    return np.clip(out, 0, 255).astype(np.uint8), float(seen.mean())
