"""Quality score: high on a well-read ECG, low on garbage."""
import numpy as np

from ecgdig.quality import assess
from ecgdig.synth import synth_signals
from ecgdig.layouts import LEADS


def _page(sig):
    """Crop like a 3x4 + rhythm II: each lead only shows for 2.5 s."""
    out = np.full_like(sig, np.nan)
    n = sig.shape[1] // 4
    for c, grp in enumerate(([0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11])):
        for i in grp:
            out[i, c * n:(c + 1) * n] = sig[i, c * n:(c + 1) * n]
    out[1] = sig[1]
    return out


def test_clean_ecg_scores_high():
    sig = _page(synth_signals(np.random.default_rng(3)).astype(np.float32))
    q = assess(sig, LEADS, 500, prefusion=sig)
    assert q["score"] > 0.8
    assert all(v["score"] > 0.6 for v in q["leads"].values())


def test_trace_replaced_by_noise_scores_low():
    rng = np.random.default_rng(4)
    sig = _page(synth_signals(rng).astype(np.float32))
    bad = LEADS.index("V5")
    m = np.isfinite(sig[bad])
    sig[bad, m] = np.cumsum(rng.normal(0, 0.05, m.sum())).astype(np.float32)   # random walk
    q = assess(sig, LEADS, 500)
    assert q["leads"]["V5"]["score"] < 0.5
    assert q["leads"]["V5"]["score"] < min(v["score"] for k, v in q["leads"].items() if k != "V5")


def test_missing_lead():
    sig = _page(synth_signals(np.random.default_rng(5)).astype(np.float32))
    sig[LEADS.index("V1")] = np.nan
    q = assess(sig, LEADS, 500)
    assert q["leads"]["V1"]["score"] == 0.0 and "ausente" in q["leads"]["V1"]["flags"]
