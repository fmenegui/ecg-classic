"""Photometric normalisation of a page before it is read.

PM-ECG-ID's hardest scenarios are global intensity changes of a readable page: brightness -120 /
-160 (the page is almost black, grey levels 0-56), contrast gamma 0.125 / 0.25 (washed out, grey
levels 200-250) and gamma 8 (crushed). Our reader's thresholds and the row network expect paper near
white and ink near black, and read these pages at -3 to +3 dB. A per-page tone map undoes a global
change: stretch the luminance between robust percentiles, then a gamma that puts the paper at its
usual level. The same lookup table is applied to every channel, so hues (red grid, black ink) keep
their order. Only pages whose grey levels span less than
`max_range` are changed; ordinary photos are left alone, and a white page with faint content
(low ink opacity) is only stretched.
"""
from __future__ import annotations

import numpy as np

PAPER_Q = 75          # the paper is most of the page: its level is the 75th percentile
PAPER_LEVEL = 0.92    # where the paper sits on a normal scan or render (0..1)


def tone_map(img: np.ndarray, max_range: int = 100):
    """img: HxWx3 uint8 (BGR). Returns (image, info). info["applied"] says whether it changed."""
    g = img.mean(axis=2) if img.ndim == 3 else img.astype(np.float32)
    sub = g[::4, ::4]
    lo, hi = np.percentile(sub, [0.5, 99.5])
    paper = np.percentile(sub, PAPER_Q)
    rng = hi - lo
    pn = (paper - lo) / max(rng, 1.0)
    info = dict(lo=float(lo), hi=float(hi), paper=float(paper), applied=False)
    # only a compressed page: ordinary phone photos (paper ~180, range ~130) are left as they are
    if rng >= max_range:
        return img, info
    if paper >= 250:   # white page, faint content (low opacity): stretch only, no gamma
        gamma = 1.0
    else:
        pn = float(np.clip(pn, 0.05, 0.98))
        gamma = float(np.clip(np.log(PAPER_LEVEL) / np.log(pn), 0.2, 5.0))
    x = np.clip((np.arange(256, dtype=np.float32) - lo) / max(rng, 1.0), 0, 1)
    lut = np.clip(255.0 * x ** gamma + 0.5, 0, 255).astype(np.uint8)
    info.update(applied=True, gamma=gamma)
    return lut[img], info
