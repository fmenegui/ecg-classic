import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "hybrid"))
import rowgen  # noqa: E402


def _height(lab):
    w = lab.sum(0)
    r = np.arange(lab.shape[0])[:, None]
    return np.where(w > 0, (lab * r).sum(0) / np.maximum(w, 1e-9), np.nan)


def test_stretched_strip_matches_base_strip():
    """Same draw with xup=1 and 2: the stretched label, read back at the base columns, is the
    base label (up to the curvature between two columns), and the image is twice as wide."""
    im1, l1 = rowgen.sample(np.random.default_rng(5), 128, xup=1)
    im2, l2 = rowgen.sample(np.random.default_rng(5), 128, xup=2)
    assert im2.shape == (128, 256, 3) and l2.shape == (128, 256)
    h1, h2 = _height(l1), _height(l2)
    x2 = (np.arange(256) + 0.5) / 2 - 0.5
    ok = np.isfinite(h1)
    back = np.interp(np.arange(128), x2[np.isfinite(h2)], h2[np.isfinite(h2)])
    assert np.nanmedian(np.abs(back[ok] - h1[ok])) < 0.05


def test_label_is_sparse_per_column():
    _, lab = rowgen.sample(np.random.default_rng(1), 96, xup=2)
    assert (np.count_nonzero(lab, axis=0) <= 2).all()
    assert np.allclose(lab.sum(0)[lab.sum(0) > 0], 1.0, atol=1e-5)
