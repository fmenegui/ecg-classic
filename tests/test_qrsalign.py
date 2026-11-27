"""qrsalign: leads recorded together are put back onto shared beat times; agreeing leads stay."""
import numpy as np

from ecgdig.qrsalign import align


def _ecg(n, fs, beats, shift=0):
    t = np.arange(n)
    x = np.zeros(n)
    for b in beats:
        x += np.exp(-0.5 * ((t - b - shift) / (0.008 * fs)) ** 2)
    return x


def test_agreeing_leads_unchanged():
    fs, n = 500, 1250
    beats = [100, 500, 900]
    sig = np.stack([_ecg(n, fs, beats) * a for a in (1.0, 0.7, 1.2)])
    out, rep = align(sig, fs)
    assert np.allclose(out, sig) and rep["moved"] == 0


def test_shifted_beat_pulled_back():
    fs, n = 500, 1250
    beats = [100, 500, 900]
    sig = np.stack([_ecg(n, fs, beats), _ecg(n, fs, beats), _ecg(n, fs, beats), _ecg(n, fs, beats)])
    true = sig[0].copy()
    bad = _ecg(n, fs, [100, 515, 900])          # one beat 30 ms late in one lead
    sig[3] = bad
    out, rep = align(sig, fs)
    assert rep["moved"] >= 1
    assert np.abs(out[3] - true).max() < 0.5 * np.abs(bad - true).max()
    assert np.allclose(out[:3], sig[:3])
