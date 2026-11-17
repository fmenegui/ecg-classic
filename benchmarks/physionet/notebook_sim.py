"""Run the Kaggle submission kernel (kaggle/submission.py) on training records, as if they were the
hidden test, and score its output with the official metric.

It answers one question: does the submission path itself lose dB compared with our evaluation
(windows of the short leads, filling of unknown samples, failures and time limits)? Each image
becomes a test id; test.csv asks, for every lead, as many rows as the ground truth shows.

    python benchmarks/physionet/notebook_sim.py DATA_DIR OUT_DIR
DATA_DIR holds train.csv and <id>/<id>.csv + <id>/<id>-<type>.png (kaggle_zip.py shard output).
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent))
import metrics as MT  # noqa: E402
from eval_snr import _load_gt, LEADS  # noqa: E402


def main():
    data, out = Path(sys.argv[1]), Path(sys.argv[2])
    comp = out / "comp"; (comp / "test").mkdir(parents=True, exist_ok=True)
    rows, meta = [], []
    k = 0
    for gt_p in sorted(data.glob("*/*.csv")):
        rid = gt_p.stem
        G = _load_gt(gt_p); fs = G.shape[1] / 10
        for img in sorted(gt_p.parent.glob(f"{rid}-*.png")):
            iid = str(10_000_000 + k); k += 1
            dst = comp / "test" / f"{iid}.png"
            if not dst.exists():
                os.symlink(img.resolve(), dst)
            for i, l in enumerate(LEADS):
                rows.append(dict(id=iid, lead=l, fs=fs, number_of_rows=int(np.isfinite(G[i]).sum())))
            meta.append(dict(iid=iid, rid=rid, type=img.stem.split("-")[1], gt=str(gt_p)))
    pd.DataFrame(rows).to_csv(comp / "test.csv", index=False)
    env = dict(os.environ, ECG_COMP_DIR=str(comp), ECG_OUT=str(out / "submission.csv"))
    subprocess.run([sys.executable, str(HERE.parents[1] / "kaggle" / "submission.py")], env=env, check=True)
    sub = pd.read_csv(out / "submission.csv")
    parts = sub["id"].str.split("_", expand=True)
    sub["iid"], sub["row"], sub["lead"] = parts[0], parts[1].astype(int), parts[2]
    res = []
    for m in meta:
        G = _load_gt(m["gt"]); fs = G.shape[1] / 10
        s = sub[sub.iid == m["iid"]]
        sig = noi = 0.0
        for i, l in enumerate(LEADS):
            g = G[i][np.isfinite(G[i])]
            if not len(g):
                continue
            p = s[s.lead == l].sort_values("row")["value"].to_numpy(float)
            al = MT.align_signals(g, p, int(fs * MT.MAX_TIME_SHIFT))
            a, b = MT.compute_power(g, al); sig += a; noi += b
        lin = MT.compute_snr(sig, noi)
        res.append(dict(id=m["rid"], type=m["type"], kaggle_lin=float(lin), kaggle_db=float(10 * np.log10(max(lin, 1e-12)))))
    (out / "notebook_results.json").write_text(json.dumps(res))
    print(f"{len(res)} images, notebook-path Kaggle score {MT.kaggle_score([r['kaggle_lin'] for r in res]):.2f} dB")


if __name__ == "__main__":
    main()
