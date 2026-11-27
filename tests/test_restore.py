"""restore: sharp pages of normal size pass unchanged; blurred ones are measured and deconvolved."""
import cv2
import numpy as np

from ecgdig.restore import blur_sigma, classical


def _page(h=1800, w=2400):
    img = np.full((h, w, 3), 255, np.uint8)
    for x in range(0, w, 20):
        img[:, x] = (200, 200, 255)
    for y in (400, 900, 1400):
        xs = np.arange(w)
        ys = (y + 60 * np.sin(xs / 37.0)).astype(int)
        for d in (-1, 0, 1):
            img[np.clip(ys + d, 0, h - 1), xs] = 0
    return img


def test_sharp_unchanged():
    p = _page()
    out, info = classical(p)
    assert not info["deconvolved"] and info["upscale"] == 1.0 and np.array_equal(out, p)


def test_blur_measured_and_deconvolved():
    b = cv2.GaussianBlur(_page(), (0, 0), 6)
    assert blur_sigma(b.mean(2)) > 3
    out, info = classical(b)
    assert info["deconvolved"]
    # deconvolution steepens the edges again
    assert blur_sigma(out.mean(2)) < blur_sigma(b.mean(2))
