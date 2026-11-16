"""Merge the results of the PM-ECG-ID workflow shards and summarize.

    python benchmarks/pmcardio/merge.py <folder with the downloaded artifacts> <output>
"""
from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_pm import summarize  # noqa: E402

src, dst = Path(sys.argv[1]), Path(sys.argv[2])
dst.mkdir(parents=True, exist_ok=True)
rows, cfg = [], None
for f in sorted(glob.glob(str(src / "*" / "results.json"))):
    d = json.loads(Path(f).read_text())
    rows += d["rows"]; cfg = d.get("config")
(dst / "results.json").write_text(json.dumps(dict(config=cfg, rows=rows), default=float))
summarize(rows, dst)
