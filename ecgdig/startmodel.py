"""Crop geometry of the row-start network (trained by hybrid/startnet.py)."""
from __future__ import annotations

import cv2
import numpy as np

CH, CW = 64, 96          # crop at 10 px/mm: 6.4 mm tall, 9.6 mm wide
XUP = 2                  # stretched horizontally for the network (as the row network)


def crop(rect: np.ndarray, xc: float, yc: float) -> np.ndarray:
    """CH x (CW * XUP) crop centred at (xc, yc) (sub-pixel), BGR uint8."""
    c = cv2.getRectSubPix(rect, (CW, CH), (float(xc), float(yc)))
    return cv2.resize(c, (CW * XUP, CH), interpolation=cv2.INTER_CUBIC)
