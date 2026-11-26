import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmarks" / "selector"))
import fit  # noqa: E402
import importlib  # noqa: E402
D = importlib.import_module("ecgdig.digitize")


def _row(qc, qn, sc, sn, same=False):
    fc = dict(score=qc, quality=qc, rows_found=4.0)
    fn = dict(score=qn, quality=qn, rows_found=4.0)
    return dict(features_classical=fc, features_network=fn, snr_classical=sc, snr_network=sn,
                layout_classical="3x4+1", layout_network="3x4+1" if same else "6x2", chosen="classical",
                degradation="none")


def test_fit_learns_a_separable_choice(tmp_path):
    rng = np.random.default_rng(0)
    rows = []
    for _ in range(200):
        qc, qn = rng.uniform(0, 1, 2)
        better_n = qn > qc + 0.1
        rows.append(_row(qc, qn, 5.0, 12.0 if better_n else 1.0))
    X, names = fit.design(rows)
    y = np.array([r["snr_network"] > r["snr_classical"] for r in rows], float)
    m = fit.logistic(X, y, np.ones(len(y)))
    acc = (fit.predict(m, X) == (y > 0)).mean()
    assert acc > 0.9
    m["features"] = names
    p = tmp_path / "sel.json"; p.write_text(json.dumps(m))
    rc = dict(layout="3x4+1", quality=dict(score=0.2, page=dict(layout_confidence=0.2)), signals=np.zeros((12, 10)))
    rn = dict(layout="6x2", quality=dict(score=0.9, page=dict(layout_confidence=0.9)), signals=np.zeros((12, 10)))
    assert D._selector_prefers_network(str(p), rc, rn)
    assert not D._selector_prefers_network(str(p), rn, rc)
