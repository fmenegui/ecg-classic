"""The Kaggle kernel's time budget: networks while time allows, then classical, then skip."""
import importlib.util
import os

import pytest

pytest.importorskip("pandas")   # the kernel imports pandas (present on Kaggle)


def _load():
    os.environ.setdefault("ECG_COMP_DIR", "/nonexistent")
    spec = importlib.util.spec_from_file_location(
        "kaggle_submission", os.path.join(os.path.dirname(__file__), "..", "kaggle", "submission.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_mode_switches_with_time_left():
    m = _load()
    m.MODELS, m.N_WORKERS, m.CLASSIC_S, m.HYBRID_S = {"row_model": "x"}, 2, 40.0, 150.0
    assert m._mode(1e6, 1000) == "hybrid"
    # 100 images left classically at 20 s each (2 workers) need 2,000 s, plus 150 for this one
    assert m._mode(2200, 101) == "hybrid"
    assert m._mode(2100, 101) == "classical"
    assert m._mode(5, 10) == "skip"


def test_mode_without_models_is_classical():
    m = _load()
    m.MODELS = {}
    assert m._mode(1e6, 1000) == "classical"
    assert m._mode(1, 1) == "skip"
