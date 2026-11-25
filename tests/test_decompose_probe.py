import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmarks" / "topsol"))
import decompose_probe as D  # noqa: E402


def _ecg(n=1250, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(n) / 500.0
    x = np.zeros(n)
    for c in np.arange(0.2, t[-1], 0.8):
        x += 1.2 * np.exp(-((t - c) / 0.012) ** 2) + 0.3 * np.exp(-((t - c - 0.25) / 0.05) ** 2)
    return x + 0.001 * rng.standard_normal(n)


def test_stages_recover_shift_scale_gain():
    g = _ecg()
    i = np.arange(len(g), dtype=float)
    c = (len(g) - 1) / 2
    p = np.interp(c + (i - c) * 1.01 + 0.4, i, g) * 0.9 + 0.05   # time scale, sub-sample lag, gain
    r = D.decompose(g, p)
    assert r["read"] < r["shift"] <= r["scale"] <= r["gain"]
    assert r["gain"] > 30


def test_ceilings_order():
    g = _ecg(seed=1)
    assert D.ceiling(g, 2) < D.ceiling_half(g)
    assert np.isfinite(D.ceiling(g, 2))
