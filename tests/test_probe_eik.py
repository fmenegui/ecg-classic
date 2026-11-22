"""probe_eik: ecg-image-kit's default sheet with its printed segments (needs a clone in ECG_IMAGE_KIT)."""
import os
import sys
from pathlib import Path

import pytest

EIK = os.environ.get("ECG_IMAGE_KIT", "")
pytestmark = pytest.mark.skipif(not (EIK and Path(EIK).exists()), reason="no ecg-image-kit clone")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmarks" / "topsol"))


def test_render_gives_13_segments(tmp_path):
    import probe_eik as P
    recs = sorted(Path(os.environ["ECG_RECORDS"]).glob("*.hea"))
    png, meta = P.render((Path(EIK) / "codes" / "ecg-image-generator").resolve(), recs[0].with_suffix("").resolve(),
                         tmp_path, seed=0)
    assert png.exists() and len(meta["leads"]) == 13
