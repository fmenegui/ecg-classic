"""Open-ECG-Digitizer output -> one npz per image for our evaluators (ECGDIG_EXTERNAL).

    python benchmarks/topsol/to_npz_oed.py OUTPUT_DIR TEST_CSV OUT_DIR

Its <key>_timeseries_canonical.csv holds 12 columns (I ... V6) of 5,000 samples (10 s at 500 Hz)
in microvolts, each lead on its own time position and NaN elsewhere; ours is the same timeline
in millivolts. digitization_metadata.csv gives the layout it matched.
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np

LEADS = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]


def main(src, test_csv, out):
    src, out = Path(src), Path(out)
    out.mkdir(parents=True, exist_ok=True)
    keys = sorted({r["id"] for r in csv.DictReader(open(test_csv))})
    lay = {}
    for m in src.rglob("digitization_metadata.csv"):
        for r in csv.DictReader(open(m)):
            lay[Path(r["file_path"]).stem] = r.get("lead_layout", "")
    done = 0
    for k in keys:
        f = next(iter(src.rglob(f"{k}_timeseries_canonical.csv")), None)
        if f is None:
            continue
        d = np.genfromtxt(f, delimiter=",", names=True)
        sig = np.full((12, 5000), np.nan, np.float32)
        for i, L in enumerate(LEADS):
            if L in d.dtype.names:
                y = np.asarray(d[L], float) / 1000.0          # uV -> mV
                n = min(len(y), 5000)
                sig[i, :n] = np.interp(np.linspace(0, len(y) - 1, n), np.arange(len(y)), y) if len(y) != n else y[:n]
        np.savez(out / f"{k}.npz", leads=np.array(LEADS), signals=sig, layout=np.array(lay.get(k, "oed")))
        done += 1
    print(f"{done} of {len(keys)} images converted", flush=True)


if __name__ == "__main__":
    main(*sys.argv[1:4])
