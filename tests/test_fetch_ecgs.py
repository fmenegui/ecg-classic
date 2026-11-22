"""fetch.py --ecgs: every page of the listed (scenario, Image ID) pairs."""
import sys
from pathlib import Path

import pytest

pd = pytest.importorskip("pandas")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmarks" / "pmcardio"))
import fetch  # noqa: E402


def test_select_ecgs_keeps_all_pages():
    m = pd.DataFrame({"Image relative path": ["a/img_1_page_0.jpeg", "a/img_1_page_1.jpeg", "b/img_1_page_0.jpeg",
                                              "a/img_2_page_0.jpeg"], "Image ID": [1, 1, 1, 2]})
    out = fetch.select_ecgs(m, "a:1, b:2")
    assert list(out["Image relative path"]) == ["a/img_1_page_0.jpeg", "a/img_1_page_1.jpeg"]
