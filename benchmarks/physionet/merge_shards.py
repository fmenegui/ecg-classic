"""Merge the per-shard results of kaggle-eval.yml and report both metrics.

    python benchmarks/physionet/merge_shards.py ART_DIR OUT_DIR [--subset id1,id2,...]

Writes OUT_DIR/results.json (all images) and OUT_DIR/report.md with the official Kaggle metric and
the PMcardio-style metrics, for all images and for the subset (the 15 records of the local
135-image regression set by default).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from eval_snr import report_both  # noqa: E402

LOCAL15 = ("1257259783,2587667307,3042142152,426433979,994226251,119693645,1449491391,1962423499,"
           "235672018,2841936136,3574249147,3646789790,3879757040,4050029009,644026427")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("art", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--subset", default=LOCAL15)
    a = ap.parse_args()
    res, cfg = [], None
    for f in sorted(a.art.rglob("results.json")):
        d = json.loads(f.read_text())
        cfg = d.get("config", cfg)
        res += d.get("results", [])
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "results.json").write_text(json.dumps(dict(config=cfg, results=res), default=str))
    sub = set(a.subset.split(","))
    rs = [r for r in res if str(r.get("id")) in sub]
    L = [f"# Kaggle training records: {len({r.get('id') for r in res})} records, {len(res)} images",
         "", f"Config: `{json.dumps(cfg)}`", "", "## All records", "", report_both(res), "",
         f"## Local regression subset ({len(rs)} images)", "", report_both(rs)]
    (a.out / "report.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
