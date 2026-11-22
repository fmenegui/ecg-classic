"""score_probe: a perfect reading scores high, a missing one 0 dB, a shifted one is realigned."""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmarks" / "topsol"))
import score_probe as SP  # noqa: E402


def test_score_page(tmp_path):
    t = np.arange(5000) / 500
    sig = np.stack([np.sin(2 * np.pi * (1 + 0.1 * k) * t) + 0.3 * np.sin(2 * np.pi * 7 * t) for k in range(12)])
    segs = [dict(lead=l, rhythm=False, t0=2.5 * (i % 4), t1=2.5 * (i % 4) + 2.5) for i, l in enumerate(SP.LEADS)]
    np.savez(tmp_path / "K0_000.npz", signals=sig, segments=json.dumps(segs))
    out, lin = SP.score_page(tmp_path / "K0_000.npz", sig + 0.01)
    assert min(s["snr"] for s in out) > 30
    shifted = np.roll(sig, 40, axis=1)
    out2, _ = SP.score_page(tmp_path / "K0_000.npz", shifted)
    assert min(s["snr"] for s in out2) > 20
    out3, lin3 = SP.score_page(tmp_path / "K0_000.npz", None)
    assert all(s["snr"] == 0 and not s["read"] for s in out3) and abs(lin3 - 1) < 1e-9
