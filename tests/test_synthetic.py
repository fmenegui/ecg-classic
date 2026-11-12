"""Regressão rápida: 2 ECGs sintéticos por nível de degradação devem passar de um SNR mínimo."""
import numpy as np
import pytest

from ecgdig.synth import synth_signals, render
from ecgdig.digitize import digitize


def _snr(gt, pr, shown):
    vals = []
    for i in range(12):
        m = shown[i] & ~np.isnan(pr[i])
        e = gt[i][m] - pr[i][m]
        vals.append(10 * np.log10((gt[i][m] ** 2).sum() / (e ** 2).sum()))
    return float(np.mean(vals))


@pytest.mark.parametrize("level,min_snr", [("none", 15.0), ("medium", 12.0), ("heavy", 8.0)])
def test_snr(level, min_snr):
    rng = np.random.default_rng(7)
    for _ in range(2):
        sig = synth_signals(rng)
        img, shown, _ = render(sig, rng, degrade=level)
        res = digitize(img)
        assert res["layout"] == "3x4+1"
        assert _snr(sig, res["signals"], shown) > min_snr
