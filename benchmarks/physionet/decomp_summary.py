"""Where the Kaggle score is lost: diagnose.py's oracle steps, aggregated per image type.

    python benchmarks/physionet/decomp_summary.py DECOMPOSITION_JSON... [--md out.md]

Each decomposition.json (one per shard, written by diagnose.py next to the evaluation) holds, per
image, the SNR after each cumulative oracle correction (metric, labels, coverage, origin, drift,
local drift, gain, baseline). Scores are aggregated like the competition metric: mean of the
linear SNR over images, then dB. A column is the score after that correction; the gain of a step
over the previous one is what that kind of error costs. "remainder" is what no oracle fixes
(shape: tracing, clipped peaks, merged traces).
"""
from __future__ import annotations

import json
import sys

import numpy as np

STEPS = ["metric", "labels", "coverage", "origin", "drift", "local drift", "gain", "baseline"]


def score(v):
    v = np.asarray([x for x in v if np.isfinite(x)], float)
    return 10 * np.log10(np.mean(10 ** (v / 10))) if len(v) else float("nan")


def table(rows):
    groups = {"all": rows}
    for r in rows:
        groups.setdefault(r["type"], []).append(r)
    L = ["| Type | Images | " + " | ".join(STEPS) + " |", "|---|---|" + "---|" * len(STEPS)]
    for g in ["all"] + sorted(k for k in groups if k != "all"):
        rs = groups[g]
        vals = [score([r.get(s, np.nan) for r in rs]) for s in STEPS]
        cells = [f"{vals[0]:.2f}"] + [f"{b:.2f} ({b - a:+.2f})" for a, b in zip(vals, vals[1:])]
        L.append(f"| {g} | {len(rs)} | " + " | ".join(cells) + " |")
    return "\n".join(L)


def main(argv):
    md = None
    if "--md" in argv:
        i = argv.index("--md"); md = argv[i + 1]; argv = argv[:i] + argv[i + 2:]
    rows = []
    for f in argv:
        rows += json.load(open(f))
    out = (f"Error decomposition, {len(rows)} images (score after each cumulative oracle correction; "
           f"in brackets the gain of that step = what that error costs)\n\n" + table(rows))
    print(out)
    if md:
        open(md, "w").write(out + "\n")


if __name__ == "__main__":
    main(sys.argv[1:])
