import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "hybrid"))
import realstrips as RS  # noqa: E402


def _height(lab):
    w = lab.sum(0)
    r = np.arange(lab.shape[0])[:, None]
    return np.where(w > 0, (lab * r).sum(0) / np.maximum(w, 1e-9), np.nan)


def test_hires_label_is_the_true_signal_at_its_columns():
    rect = np.full((600, 900, 3), 255, np.uint8)
    x = np.arange(100, 800, 0.5)                    # 500 Hz at 10 px/mm, 25 mm/s: 0.5 px apart
    y = 300 + 40 * np.sin(x / 7.0)
    segs = [dict(lead="II", row=0, base=300.0, x=x, y=y)]
    X1, Y1 = RS.strips(rect, segs, np.random.default_rng(0), per_row=1, size=256, jitter=0)
    X2, Y2 = RS.strips(rect, segs, np.random.default_rng(0), per_row=1, size=256, jitter=0, xup=2)
    assert X2.shape == X1.shape == (1, 256, 256, 3) and Y2.shape == (1, 256, 512)
    h1, h2 = _height(Y1[0]), _height(Y2[0])
    # the stretched label at the base columns' positions equals the base label
    assert np.nanmax(np.abs(h2[1::2][:len(h1)] - h1)) < 2.0
    # and between them it follows the signal, not a straight line between base columns
    x0 = np.flatnonzero(np.isfinite(h1))[0]
    lin = 0.5 * (h1[:-1] + h1[1:])
    mid = h2[2::2][:len(lin)]
    ok = np.isfinite(lin) & np.isfinite(mid)
    assert ok.sum() > 50 and np.abs(mid[ok] - lin[ok]).max() > 0.01 and x0 >= 0
