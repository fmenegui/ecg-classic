"""heavy_aug with the extended limits (blur and low resolution of the hardest PM-ECG-ID scenarios)."""
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "hybrid"))


def test_extended_limits_keep_shape_and_range():
    import train_gpu as T
    T.HEAVY_DOWN_MAX[0], T.HEAVY_BLUR_MAX[0] = 24.0, 10.0
    g = torch.Generator().manual_seed(0)
    x = torch.rand(6, 3, 128, 512, generator=g)
    y = T.heavy_aug(x, g, 1.0)
    assert y.shape == x.shape and float(y.min()) >= 0 and float(y.max()) <= 1
    T.HEAVY_DOWN_MAX[0], T.HEAVY_BLUR_MAX[0] = 6.0, 5.0
