"""benchmarks/litfl/crosslead.py: closure bursts, blame and lost beats on synthetic leads."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmarks" / "litfl"))
import crosslead  # noqa: E402


def _beats(t0, t1, hr=75, amp=1.0, fs=500):
    x = np.full(5000, np.nan)
    t = np.arange(5000) / fs
    m = (t >= t0) & (t < t1)
    y = np.zeros(5000)
    for b in np.arange(t0 + 0.2, t1, 60 / hr):
        y += amp * np.exp(-((t - b) / 0.012) ** 2)
    x[m] = y[m]
    return x


def _page():
    I_full, II_full = _beats(0, 10, amp=0.4), _beats(0, 10, amp=1.0)

    def cut(x, a, b):
        y = np.full(5000, np.nan)
        y[int(a * 500):int(b * 500)] = x[int(a * 500):int(b * 500)]
        return y
    I, II = cut(I_full, 0, 2.5), cut(II_full, 0, 2.5)
    Ir, IIr = cut(I_full, 2.5, 5), cut(II_full, 2.5, 5)
    return {"I": [I], "II": [II, II_full], "III": [II - I], "aVR": [-(Ir + IIr) / 2],
            "aVL": [Ir - IIr / 2], "aVF": [IIr - Ir / 2]}


def test_consistent_page_has_no_flags():
    cl = crosslead.analyse(_page())
    assert cl["closure"]["limb"] < 0.05 and cl["closure"]["goldberger"] < 0.05
    assert crosslead.flags(cl) == []
    assert cl["rhythm_lag_ms"] == 0


def test_lost_trace_is_blamed_on_the_right_lead():
    ps = _page()
    x = ps["III"][0].copy()
    x[(np.arange(5000) >= 300) & (np.arange(5000) < 700)] = 0.0   # III flat over 0.6-1.4 s (beat at 1.0 s lost)
    ps["III"] = [x]
    cl = crosslead.analyse(ps)
    assert cl["bursts"] and {b["lead"] for b in cl["bursts"]} == {"III"}
    assert cl["timing"]["III"]["lost"] >= 1
