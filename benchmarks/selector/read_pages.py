"""Read the selector pages both ways (classical and page network) and score each reading.

    python benchmarks/selector/read_pages.py --pages DIR --out rows.jsonl --jobs 8 \
        --config row_model=... --config page_model=... [--config k=v ...]

One JSON line per page: its layout and degradation, which reading the current rule chose, both
readings' layouts, their self-describing features (digitize.reading_features) and their scores
against the truth: PMcardio-style mean over the 12 leads of each lead's segment SNR (every shown
run aligned as benchmarks/pmcardio/eval_pm.py does; a lead not read counts 0 dB).
"""
from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "benchmarks" / "pmcardio"))

LEADS = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]


def score(sig, shown, leads, signals):
    """Mean over the 12 leads of the lead SNR (dB, 0 when the lead was not read)."""
    import eval_pm as E
    out = []
    S = {str(n): np.asarray(v, float) for n, v in zip(leads, signals)} if signals is not None else {}
    for li, L in enumerate(LEADS):
        idx = np.flatnonzero(shown[li])
        if not len(idx):
            continue
        br = np.flatnonzero(np.diff(idx) > 1)
        st, en = np.r_[idx[0], idx[br + 1]], np.r_[idx[br], idx[-1]] + 1
        vals = []
        for a, b in zip(st, en):
            g = sig[li, a:b]
            F = S.get(L)
            p = E._best_shift(g, F, a) if F is not None else None
            m = E._metrics(g, p) if p is not None else None
            vals.append(m["snr"] if m is not None else 0.0)
        out.append(float(np.mean(vals)))
    return float(np.mean(out)) if out else 0.0


def _one(args):
    png, cfg = args
    from ecgdig import Config, digitize
    z = np.load(png.with_suffix(".npz"))
    sig, shown = z["sig"].astype(float), z["shown"].astype(bool)
    row = dict(key=png.stem, layout=str(z["layout"]), degradation=str(z["degradation"]))
    try:
        r = digitize(str(png), Config(**cfg, page_select_keep=True))
    except Exception as e:  # noqa: BLE001
        return dict(row, error=f"{type(e).__name__}: {e}"[:200])
    ps = r.get("page_select") or {}
    chosen = ps.get("chosen")
    s_chosen = score(sig, shown, r["leads"], r["signals"])
    o = ps.get("other")
    s_other = score(sig, shown, o["leads"], o["signals"]) if o else 0.0
    sc, sn = (s_chosen, s_other) if chosen == "classical" else (s_other, s_chosen)
    return dict(row, chosen=chosen, snr_classical=sc, snr_network=sn,
                layout_classical=ps.get("layout_classical"), layout_network=ps.get("layout_network"),
                features_classical=ps.get("features_classical"), features_network=ps.get("features_network"),
                errors=ps.get("errors"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pages", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--jobs", type=int, default=2)
    ap.add_argument("--config", action="append", default=[])
    a = ap.parse_args()
    cfg = {}
    for kv in a.config:
        k, v = kv.split("=", 1)
        try:
            cfg[k] = json.loads(v)
        except ValueError:
            cfg[k] = v
    pngs = sorted(a.pages.glob("*.png"))
    done = set()
    if a.out.exists():
        done = {json.loads(l)["key"] for l in a.out.read_text().splitlines() if l.strip()}
    todo = [p for p in pngs if p.stem not in done]
    with open(a.out, "a") as f, ProcessPoolExecutor(a.jobs) as ex:
        for i, row in enumerate(ex.map(_one, [(p, cfg) for p in todo])):
            f.write(json.dumps(row) + "\n"); f.flush()
            if i % 20 == 0:
                print(f"{i + 1}/{len(todo)} read", flush=True)
    print("done", len(todo), "pages ->", a.out)


if __name__ == "__main__":
    main()
