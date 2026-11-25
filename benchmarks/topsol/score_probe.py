"""Score readings of the probe pages (benchmarks/topsol/probe_pages.py) per variant.

    python benchmarks/topsol/score_probe.py --probe /root/probe --npz EXT_DIR --out probe_results.json
    python benchmarks/topsol/score_probe.py --probe /root/probe --ecgdig [--config k=v ...] --out ...

--npz: one <key>.npz per page (leads, signals 12 x 5000 at 500 Hz: benchmarks/topsol/to_npz*.py);
--ecgdig: read the pages with ecgdig here. Every printed segment of a page is compared with the
reading over the same time window, shifted by up to +-0.5 s and offset vertically (median), like
benchmarks/pmcardio/eval_pm.py; a lead with no reading counts 0 dB (PMcardio style). The official
Kaggle-style score per variant is 10 log10 of the mean linear SNR of the pages.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "pmcardio"))
import eval_pm as E  # noqa: E402

LEADS = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]


def score_page(gt_npz, pred):
    """pred: (12, 5000) at 500 Hz or None. Per-segment metrics and the page's linear SNR."""
    g = np.load(gt_npz)
    sig = g["signals"].astype(float)
    segs = json.loads(str(g["segments"]))
    out, sp, np_ = [], 0.0, 0.0
    for s in segs:
        a, b = int(round(s["t0"] * 500)), int(round(s["t1"] * 500))
        gt = sig[LEADS.index(s["lead"]), a:b]
        m = None
        if pred is not None:
            F = np.asarray(pred[LEADS.index(s["lead"])], float)
            p = E._best_shift(gt, F, a)
            m = E._metrics(gt, p) if p is not None else None
        if m is None:
            out.append(dict(s, snr=0.0, pcc=0.0, read=False))
            sp += float(gt @ gt); np_ += float(gt @ gt)
        else:
            out.append(dict(s, snr=m["snr"], pcc=m["pcc"], read=True))
            e2 = float(gt @ gt) / 10 ** (m["snr"] / 10)
            sp += float(gt @ gt); np_ += e2
    return out, sp / max(np_, 1e-12)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe", type=Path, required=True)
    ap.add_argument("--npz", type=Path)
    ap.add_argument("--ecgdig", action="store_true")
    ap.add_argument("--config", action="append", default=[])
    ap.add_argument("--jobs", type=int, default=2)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--variants", default="", help="only these variants, comma separated")
    ap.add_argument("--save-pred", type=Path, default=None, help="--ecgdig: also write each reading as <key>.npz here")
    a = ap.parse_args()
    keys = sorted(p.stem for p in (a.probe / "gt").glob("*.npz"))
    if a.variants:
        keys = [k for k in keys if k.split("_")[0] in a.variants.split(",")]
    preds = {}
    if a.npz:
        for k in keys:
            f = a.npz / f"{k}.npz"
            if f.exists():
                d = np.load(f, allow_pickle=True)
                names = [str(x) for x in d["leads"]]
                S = np.full((12, 5000), np.nan)
                for i, n in enumerate(names):
                    if n in LEADS:
                        v = np.asarray(d["signals"][i], float)
                        S[LEADS.index(n), :min(5000, len(v))] = v[:5000]
                preds[k] = S
    elif a.ecgdig:
        from concurrent.futures import ProcessPoolExecutor
        cfg = {}
        for kv in a.config:
            k_, v_ = kv.split("=", 1)
            try:
                cfg[k_] = json.loads(v_)
            except ValueError:
                cfg[k_] = v_
        with ProcessPoolExecutor(a.jobs) as ex:
            for k, S in zip(keys, ex.map(_read_ecgdig, [(str(a.probe / "test" / f"{k}.png"), cfg) for k in keys])):
                if S is not None:
                    preds[k] = S
                    if a.save_pred:
                        a.save_pred.mkdir(parents=True, exist_ok=True)
                        np.savez_compressed(a.save_pred / f"{k}.npz", leads=np.array(LEADS), signals=S)
    rows, by = [], defaultdict(list)
    for k in keys:
        segs, lin = score_page(a.probe / "gt" / f"{k}.npz", preds.get(k))
        v = k.split("_")[0]
        rows.append(dict(key=k, variant=v, read=k in preds, lin=lin, segments=segs))
        by[v].append(rows[-1])
    summ = {}
    for v, rs in sorted(by.items()):
        seg = [s for r in rs for s in r["segments"]]
        summ[v] = dict(pages=len(rs), read=sum(r["read"] for r in rs),
                       snr=float(np.mean([s["snr"] for s in seg])), pcc=float(np.mean([s["pcc"] for s in seg])),
                       failed=float(np.mean([not s["read"] for s in seg])),
                       kaggle=float(10 * np.log10(max(np.mean([r["lin"] for r in rs]), 1e-12))))
    a.out.write_text(json.dumps(dict(summary=summ, pages=rows)))
    print("| variant | pages read | SNR (dB) | PCC | failed segments | Kaggle-style (dB) |")
    print("|---|---|---|---|---|---|")
    for v, s in summ.items():
        print(f"| {v} | {s['read']}/{s['pages']} | {s['snr']:.2f} | {s['pcc']:.3f} | {100 * s['failed']:.0f}% | {s['kaggle']:.2f} |")


def _read_ecgdig(args):
    path, cfg = args
    try:
        from ecgdig import Config, digitize
        r = digitize(path, Config(**cfg))
        S = np.full((12, 5000), np.nan)
        for i, n in enumerate(r["leads"]):
            if n in LEADS:
                v = np.asarray(r["signals"][i], float)
                S[LEADS.index(n), :min(5000, len(v))] = v[:5000]
        return S
    except Exception:  # noqa: BLE001
        return None


if __name__ == "__main__":
    main()
