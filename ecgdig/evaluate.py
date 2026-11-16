"""Evaluation on the synthetic set: SNR (dB) and correlation per lead, on visible samples only."""
from __future__ import annotations

import glob
import sys
import time
import numpy as np

from .digitize import digitize, LEADS


def snr_db(gt, pr):
    ok = ~np.isnan(pr) & ~np.isnan(gt)
    if ok.sum() < 10:
        return np.nan
    e = gt[ok] - pr[ok]
    return 10 * np.log10((gt[ok] ** 2).sum() / max((e ** 2).sum(), 1e-12))


def evaluate(folder, cfg=None, verbose=True):
    files = sorted(glob.glob(f"{folder}/*.png"))
    rows = []
    for f in files:
        d = np.load(f.replace(".png", ".npz"))
        gt, shown = d["sig"], d["shown"]
        t = time.time()
        try:
            res = digitize(f, cfg)
        except Exception as ex:  # noqa
            print(f, "FALHOU:", ex)
            rows.append([np.nan] * 12 + [np.nan])
            continue
        pr = res["signals"].copy()
        pr[~shown] = np.nan  # same visible window
        s = [snr_db(np.where(shown[i], gt[i], np.nan), pr[i]) for i in range(12)]
        r = [np.corrcoef(gt[i][shown[i] & ~np.isnan(pr[i])], pr[i][shown[i] & ~np.isnan(pr[i])])[0, 1]
             if (shown[i] & ~np.isnan(pr[i])).sum() > 10 else np.nan for i in range(12)]
        rows.append(s + [np.nanmean(r)])
        if verbose:
            print(f"{f.split('/')[-1]}  mean SNR {np.nanmean(s):6.2f} dB  min {np.nanmin(s):6.2f}  "
                  f"r {np.nanmean(r):.4f}  gain {res['gain_mm_per_mv']}  layout {res['layout']}  "
                  f"pulses {res['pulses_found']}  {time.time() - t:.1f}s")
    A = np.array(rows)
    if verbose:
        print("per lead (dB):", " ".join(f"{l}:{v:.1f}" for l, v in zip(LEADS, np.nanmean(A[:, :12], 0))))
        print(f"OVERALL MEAN: SNR {np.nanmean(A[:, :12]):.2f} dB   correlation {np.nanmean(A[:, 12]):.4f}")
    return A


if __name__ == "__main__":
    evaluate(sys.argv[1])
