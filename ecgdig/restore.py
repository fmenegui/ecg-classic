"""Appearance restoration before reading (Config.restore; off by default).

"classical": for pages that are blurred or tiny, a Wiener deconvolution with a Gaussian of the
blur measured on the page itself, after an upscale of very small pages. Only appearance is
restored: no geometry, nothing filled in. Sharp pages of normal size come back unchanged.
A path to an ONNX restorer (hybrid/train_restore.py) runs that network instead, tile by tile."""
from __future__ import annotations

import math

import cv2
import numpy as np

SIGMA_ON = 2.0       # px: blur measured above this is deconvolved
MIN_SIDE = 1600      # px: pages whose longer side is below this are upscaled first
TARGET_SIDE = 2400


def blur_sigma(gray: np.ndarray) -> float:
    """Gaussian blur of the page in px, from the steepest edges: for a blurred step of height C,
    the largest gradient is C / (sigma * sqrt(2 pi))."""
    g = gray.astype(np.float32)
    lo, hi = np.percentile(g, [0.5, 99.0])
    c = float(hi - lo)
    if c < 10:
        return 0.0
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3) / 8.0
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3) / 8.0
    gm = float(np.percentile(np.hypot(gx, gy), 99.95))
    est = c / max(gm * math.sqrt(2 * math.pi), 1e-6)
    # a thin pen line blurred wider than itself never reaches the full contrast, so the raw
    # estimate reads low; calibrated on an ecg-image-kit page blurred by sigma 2-10 px
    return est if est < 1.3 else 1.7 * est - 0.5


def _wiener(ch: np.ndarray, sigma: float, k: float) -> np.ndarray:
    H, W = ch.shape
    pad = int(3 * sigma) + 4
    x = cv2.copyMakeBorder(ch, pad, pad, pad, pad, cv2.BORDER_REFLECT)
    h, w = x.shape
    fy = np.fft.fftfreq(h)[:, None]
    fx = np.fft.rfftfreq(w)[None, :]
    G = np.exp(-2 * (math.pi * sigma) ** 2 * (fx ** 2 + fy ** 2))
    X = np.fft.rfft2(x)
    Y = X * G / (G * G + k)
    y = np.fft.irfft2(Y, s=(h, w))
    return y[pad:pad + H, pad:pad + W]


def classical(img: np.ndarray, k: float = 0.1) -> tuple[np.ndarray, dict]:
    info = dict(kind="classical", upscale=1.0, sigma=0.0, deconvolved=False)
    out = img
    side = max(img.shape[:2])
    if side < MIN_SIDE:
        f = TARGET_SIDE / side
        out = cv2.resize(out, None, fx=f, fy=f, interpolation=cv2.INTER_LANCZOS4)
        info["upscale"] = round(f, 3)
    gray = out.mean(2) if out.ndim == 3 else out
    s = blur_sigma(gray)
    info["sigma"] = round(s, 2)
    if s > SIGMA_ON:
        o = out.astype(np.float32)
        chans = [o[..., c] for c in range(o.shape[2])] if o.ndim == 3 else [o]
        # deconvolve a little less than measured: the estimate includes the pen's own width
        r = [_wiener(c, 0.8 * s, k) for c in chans]
        o = np.stack(r, -1) if o.ndim == 3 else r[0]
        out = np.clip(o, 0, 255).astype(np.uint8)
        info["deconvolved"] = True
    return out, info


def restore(img: np.ndarray, how: str) -> tuple[np.ndarray, dict]:
    if how == "classical":
        return classical(img)
    from .restoremodel import run as _run
    return _run(img, how)
