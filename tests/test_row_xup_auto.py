"""row_xup=0 takes the row network's horizontal scale from its saved metadata."""
import json

from ecgdig.digitize import Config, _resolve_row_xup


def test_default_is_replace_and_auto():
    c = Config()
    assert c.row_mode == "replace" and c.row_xup == 0
    assert _resolve_row_xup(c).row_xup == 1            # no model: 1


def test_sidecar_and_meta(tmp_path):
    m = tmp_path / "row.onnx"; m.write_bytes(b"")
    assert _resolve_row_xup(Config(row_model=str(m))).row_xup == 1      # no metadata: 1
    (tmp_path / "meta.json").write_text(json.dumps({"xup": 2}))
    assert _resolve_row_xup(Config(row_model=str(m))).row_xup == 2      # meta.json beside it
    (tmp_path / "row.json").write_text(json.dumps({"xup": 1}))
    assert _resolve_row_xup(Config(row_model=str(m))).row_xup == 1      # <model>.json wins
    assert _resolve_row_xup(Config(row_model=str(m), row_xup=2)).row_xup == 2   # explicit wins
