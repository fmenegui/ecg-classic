"""Evaluation of ecgdig on the PMcardio ECG Image Database (PM-ECG-ID, Zenodo 13617673).

Metrics in the style of the PMcardio paper (Demolder et al., medRxiv 2024.08.31.24312876):
per lead, alignment of up to ±0.5 s, vertical offset removed by the median of the
difference; PCC, RMSE (mV) and SNR (dB). Failure (lead missing or < 50% covered): PCC = 0,
SNR = 0 and the RMSE of an all-zero signal, as in the paper for whole-database metrics.

    python benchmarks/pmcardio/eval_pm.py --root <folder with visual_data/ and data/> \
        --meta metadata.csv --list sel.csv --out results/pm --jobs 2 [--config k=v ...]

Multi-page images of the same ECG are merged per lead before the comparison.

The summary also reports the official Kaggle metric (benchmarks/metrics.py) per ECG image: the
12 lead cuts, cross-correlation alignment of +-0.2 s, powers summed over the leads, linear mean
over images, then dB; a lead not read is submitted as zeros.
"""
from __future__ import annotations

import argparse
import ast
import json
import os
import sys
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

LEADS = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
FS = 500
MAXS = int(0.5 * FS)


def _apply_flags(spec: str):
    """Ablation switches "module.FLAG=value;..." (1-element lists in ecgdig.*), to compare
    variants on the whole database without changing the code."""
    import importlib
    for it in [x.strip() for x in spec.split(";") if x.strip()]:
        mod, rest = it.split(".", 1)
        var, val = rest.split("=", 1)
        getattr(importlib.import_module("ecgdig." + mod), var)[0] = ast.literal_eval(val)


# printed format of each page (metadata columns) -> ecgdig library layout, for the layout oracle
def oracle_layout(fmt, n_pages, page, n_cols, n_rows, n_rhythm):
    if n_pages == 1:
        return {"12x1": "12x1", "3x4": "3x4", "3x4+1R": "3x4+1", "3x4+3R": "3x4+3",
                "6x2": "6x2", "6x2+1R": "6x2+1"}[fmt]
    if fmt == "6x2":
        return ["6x1-limb", "6x1-prec"][page]
    if fmt == "3x4" and n_cols == 2:   # 3x2 per page: I-III + aVR-aVF, then V1-V3 + V4-V6
        return [[["I", "aVR"], ["II", "aVL"], ["III", "aVF"]], [["V1", "V4"], ["V2", "V5"], ["V3", "V6"]]][page]
    if fmt == "3x4" and n_cols == 1:
        return ["3x1-limb", "3x1-aug", "3x1-v13", "3x1-v46"][page]
    raise ValueError(f"no oracle layout for {fmt} on {n_pages} pages")


def _diag(r):
    """Per-page structure diagnosis kept in results.json (small, JSON-safe)."""
    li = r.get("layout_info") or {}
    it = r.get("internals") or {}
    cs = r.get("col_struct") or {}
    q = (r.get("quality") or {}).get("page") or {}
    f = lambda v: None if v is None else round(float(v), 3)
    return dict(rows_found=r.get("rows_found"), rows=len(it.get("rows", [])),
                rows_y_mm=[round(y / max(r.get("ppmm") or 1, 1e-6), 1) for y in it.get("rows", [])],
                layout_rows=[len(x) for x in r.get("layout_rows", [])],
                ranking=li.get("ranking"), confidence=f(li.get("confidence")), warning=li.get("warning"),
                parts={k: f(v) for k, v in (li.get("parts") or {}).items() if isinstance(v, (int, float))},
                dropped=li.get("dropped"), col_how=cs.get("how"), n_cols=len(cs.get("cols") or []) or None,
                rotation=r.get("rotation"), pulses=r.get("pulses_found"), pulses_right=r.get("pulses_right"),
                speed=r.get("speed_mm_per_s"), speed_src=r.get("speed_source"), duration=f(r.get("duration_s")),
                gain=f(r.get("gain_mm_per_mv")), ppmm=f(r.get("ppmm")), angle=f(r.get("angle")),
                pulses_ratio=f(q.get("pulses_ratio")),
                hybrid={k: v for k, v in (r.get("hybrid") or {}).items() if k != "errors"} or None)


def _digitize(job):
    path, cfg_kw = job
    import warnings
    warnings.filterwarnings("ignore")
    from ecgdig import Config, digitize
    import signal
    _apply_flags(os.environ.get("ECGDIG_FLAGS", ""))

    def _alarm(*_):
        raise TimeoutError("timed out")
    signal.signal(signal.SIGALRM, _alarm)
    signal.alarm(int(os.environ.get("PM_TIMEOUT", "300")))
    t0 = time.time()
    try:
        r = digitize(path, Config(**cfg_kw), return_debug=True)
        signal.alarm(0)
        out = dict(ok=True, leads=r["leads"], signals=r["signals"].astype(np.float32),
                   seg={k: [np.nonzero(np.isfinite(m))[0][[0, -1]] if np.isfinite(m).any() else None for m in v]
                        for k, v in r.get("per_segment", {}).items()},
                   layout=r.get("layout"), quality=r.get("quality"), sec=time.time() - t0,
                   diag=_diag(r), select=r.get("page_select"), dewarp=r.get("dewarp_select"))
    except Exception as e:  # noqa: BLE001
        out = dict(ok=False, err=f"{type(e).__name__}: {e}", sec=time.time() - t0)
    signal.alarm(0)
    return path, out


def _run_all(jobs, n, cache: Path, root: Path):
    """Run with a per-image cache (resumable) and survive a killed process (memory)."""
    import pickle
    from concurrent.futures.process import BrokenProcessPool
    cache.mkdir(parents=True, exist_ok=True)

    def key(p):
        return cache / (os.path.relpath(p, root).replace("/", "__") + ".pkl")

    res, todo = {}, []
    for j in jobs:
        k = key(j[0])
        if k.exists():
            res[os.path.relpath(j[0], root)] = pickle.loads(k.read_bytes())
        else:
            todo.append(j)
    # n images in flight at all times (a batch that waited for its slowest image kept a 32-vCPU
    # pod at a load of 3-5). Processes are recycled by starting a new executor every n*4
    # submissions (max_tasks_per_child hangs on Python 3.11); the old one finishes its images.
    # Each image has a deadline; past it the image is a failure and its executor is killed.
    from concurrent.futures import wait, FIRST_COMPLETED
    per_img = float(os.environ.get("ECGDIG_IMG_TIMEOUT", "900"))
    todo = list(todo)
    execs, futs, broken = [], {}, []          # futs: future -> (job, executor, start time)
    cur, n_cur = None, 0

    def finish(path, o):
        res[os.path.relpath(path, root)] = o
        key(path).write_bytes(pickle.dumps(o))

    while todo or futs:
        while todo and len(futs) < n:
            if cur is None or n_cur >= n * 4:
                cur, n_cur = ProcessPoolExecutor(n), 0
                execs.append(cur)
            j = todo.pop(0)
            try:
                futs[cur.submit(_digitize, j)] = (j, cur, time.time())
            except BrokenProcessPool:   # a child of this executor died: start a new one
                todo.insert(0, j)
                cur = None
                continue
            n_cur += 1
        done, _ = wait(set(futs), timeout=30, return_when=FIRST_COMPLETED)
        for f in done:
            j, _, _ = futs.pop(f)
            try:
                path, o = f.result()
            except BrokenProcessPool:
                broken.append(j)
                continue
            except Exception as e:  # noqa: BLE001
                path, o = j[0], dict(ok=False, err=f"{type(e).__name__}: {e}", sec=0.0)
            finish(path, o)
        now = time.time()
        late = [f for f, (_, _, t) in futs.items() if now - t > per_img]
        for f in late:
            j, ex, t = futs.pop(f)
            finish(j[0], dict(ok=False, err="timed out", sec=now - t))
            # its executor may hold other images: they are resubmitted
            for g in [g for g, v in futs.items() if v[1] is ex]:
                todo.insert(0, futs.pop(g)[0])
            for pr in list(getattr(ex, "_processes", {}).values()):
                pr.kill()
            ex.shutdown(wait=False, cancel_futures=True)
            if ex is cur:
                cur = None
        live = {v[1] for v in futs.values()}
        for ex in [e for e in execs if e not in live and e is not cur]:
            ex.shutdown(wait=False)
            execs.remove(ex)
    for ex in execs:
        ex.shutdown(wait=False, cancel_futures=True)
    for j in broken:
        # killed process (memory?): run it alone; if it dies again, it becomes a failure
        e1 = ProcessPoolExecutor(1)
        try:
            path, o = e1.submit(_digitize, j).result(timeout=per_img)
        except Exception as e:  # noqa: BLE001
            path, o = j[0], dict(ok=False, err=f"killed process/timeout: {type(e).__name__}", sec=0.0)
            for pr in list(getattr(e1, "_processes", {}).values()):
                pr.kill()
        e1.shutdown(wait=False, cancel_futures=True)
        finish(path, o)
    return res


def _metrics(g, p):
    m = np.isfinite(p) & np.isfinite(g)
    if m.sum() < 0.5 * np.isfinite(g).sum():
        return None
    gg, pp = g[m], p[m]
    pp = pp + np.median(gg - pp)
    e = gg - pp
    snr = 10 * np.log10(max(float(gg @ gg), 1e-12) / max(float(e @ e), 1e-12))
    pcc = float(np.corrcoef(gg, pp)[0, 1]) if gg.std() > 1e-9 and pp.std() > 1e-9 else 0.0
    return dict(snr=snr, pcc=pcc, rmse=float(np.sqrt(np.mean(e * e))))


def _best_shift(g, F, a, span=0):
    """g: ground-truth cut; F: predicted signal (whole page); a: start of the cut in the
    prediction; span: how much longer the predicted strip is than the cut (10 s strip with a
    2.5 s cut: the cut may be anywhere in it)."""
    n = len(g); best = None
    for s in range(-MAXS, MAXS + max(0, span) + 1, 2):
        i0 = a + s
        p = np.full(n, np.nan)
        lo, hi = max(0, i0), min(len(F), i0 + n)
        if hi - lo < n // 2:
            continue
        p[lo - i0:hi - i0] = F[lo:hi]
        m = np.isfinite(p)
        if m.sum() < n // 2:
            continue
        gg, pp = g[m], p[m]
        if pp.std() < 1e-9:
            continue
        c = np.corrcoef(gg, pp)[0, 1]
        if best is None or c > best[0]:
            best = (c, s, p)
    if best is None:
        return None
    # refine with step 1
    c0, s0, _ = best
    for s in (s0 - 1, s0 + 1):
        i0 = a + s; p = np.full(n, np.nan); lo, hi = max(0, i0), min(len(F), i0 + n)
        if hi - lo < n // 2:
            continue
        p[lo - i0:hi - i0] = F[lo:hi]; m = np.isfinite(p)
        if m.sum() < n // 2 or p[m].std() < 1e-9:
            continue
        c = np.corrcoef(g[m], p[m])[0, 1]
        if c > best[0]:
            best = (c, s, p)
    return best[2]


def _kaggle_powers(g, F=None, a=0):
    """Signal and noise power of one lead under the official Kaggle metric: the prediction cut
    at the lead's window (start a in the page prediction F), NaN submitted as 0, aligned by
    cross-correlation (+-0.2 s) and vertical offset (benchmarks/metrics.py)."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import metrics as MT
    n = len(g)
    p = np.zeros(n)
    if F is not None:
        lo, hi = max(0, a), min(len(F), a + n)
        if hi > lo:
            p[lo - a:hi - a] = F[lo:hi]
    p = np.nan_to_num(p, nan=0.0, posinf=0.0, neginf=0.0)
    p[np.abs(p) > 1e3] = 0.0            # a runaway value (no ECG has 1 V) would overflow the correlation
    g = np.nan_to_num(np.asarray(g, np.float64), nan=0.0)
    try:
        al = MT.align_signals(g, p, int(FS * MT.MAX_TIME_SHIFT))
    except ValueError:                  # degenerate correlation: score as a missing lead
        al = np.zeros(n)
    ps, pn = MT.compute_power(g, al)
    return float(ps), float(pn)


def evaluate_ecg(gt, rh, pages):
    """gt: (n, 12) cuts; rh: (5000, k) rhythms; pages: digitize outputs of the pages.
    Each lead's metrics also carry its Kaggle-metric powers ("kp": [signal, noise])."""
    out = {}
    n = gt.shape[0]
    kag = {}
    for j, name in enumerate(LEADS):
        g = gt[:, j].astype(float)
        best = None
        best_src = None
        for pg in pages:
            if not pg.get("ok") or name not in pg["leads"]:
                continue
            F = pg["signals"][pg["leads"].index(name)].astype(float)
            segs = [s for s in pg["seg"].get(name, []) if s is not None]
            if not segs:
                fin = np.nonzero(np.isfinite(F))[0]
                if not len(fin):
                    continue
                segs = [fin[[0, -1]]]
            seg = min(segs, key=lambda s: abs((s[1] - s[0] + 1) - n))
            p = _best_shift(g, F, int(seg[0]), span=int(seg[1] - seg[0] + 1 - n))
            if p is None:
                continue
            mt = _metrics(g, p)
            if mt and (best is None or mt["pcc"] > best["pcc"]):
                best = mt
                best_src = (F, int(seg[0]))
        out[name] = best
        kag[name] = _kaggle_powers(g, *best_src) if best_src else _kaggle_powers(g)
    rhy = []
    for k in range(rh.shape[1]):
        g = rh[:, k].astype(float); best = None
        for pg in pages:
            if not pg.get("ok"):
                continue
            for i, name in enumerate(pg["leads"]):
                F = pg["signals"][i].astype(float)
                fin = np.nonzero(np.isfinite(F))[0]
                if len(fin) < 0.6 * len(g):
                    continue
                p = _best_shift(g, F, int(fin[0]))
                if p is None:
                    continue
                mt = _metrics(g, p)
                if mt and (best is None or mt["pcc"] > best["pcc"]):
                    best = dict(mt, lead=name)
        rhy.append(best)
    out["_kaggle"] = kag
    return out, rhy


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True, help="folder with the scenario subfolders")
    ap.add_argument("--gt", type=Path, required=True, help="folder with leads.npz and rhythms.npz")
    ap.add_argument("--meta", type=Path, required=True)
    ap.add_argument("--list", type=Path, default=None, help="csv with the relative paths to evaluate")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--jobs", type=int, default=2)
    ap.add_argument("--config", action="append", default=[])
    ap.add_argument("--rescore", action="store_true", help="only recompute the metrics from the cache")
    ap.add_argument("--oracle-layout", action="store_true",
                    help="force the printed layout of each page (ceiling of reading the format right)")
    a = ap.parse_args()
    import pandas as pd
    cfg = {}
    for kv in a.config:
        k, v = kv.split("=", 1)
        try:
            cfg[k] = ast.literal_eval(v)
        except Exception:  # noqa: BLE001
            cfg[k] = v
    m = pd.read_csv(a.meta)
    if a.list is not None:
        sel = set(pd.read_csv(a.list).iloc[:, 0])
        m = m[m["Image relative path"].isin(sel)]
    m = m[[(a.root / p).exists() for p in m["Image relative path"]]]
    a.out.mkdir(parents=True, exist_ok=True)
    L = np.load(a.gt / "leads.npz"); RH = np.load(a.gt / "rhythms.npz")
    if a.rescore:
        cfg = json.loads((a.out / "results.json").read_text()).get("config", cfg) if (a.out / "results.json").exists() else cfg
    if a.oracle_layout:
        cols = ["ECG format", "ECG number of pages", "Image page", "ECG number of columns per page",
                "ECG number of rows per page", "ECG number of rhythm leads"]
        jobs = [(str(a.root / p), dict(cfg, layout=oracle_layout(*row)))
                for p, row in zip(m["Image relative path"], m[cols].itertuples(index=False))]
    else:
        jobs = [(str(a.root / p), cfg) for p in m["Image relative path"]]
    res = _run_all(jobs, a.jobs, a.out / "cache", a.root)
    rows = []
    m["scen"] = m["Image relative path"].str.split("/").str[0]
    for (scen, ecg, iid), g in m.groupby(["scen", "ECG ID", "Image ID"]):
        pages = [res[p] for p in g["Image relative path"]]
        per, rhy = evaluate_ecg(L[ecg], RH[ecg], pages)
        rows.append(dict(scenario=scen, ecg=ecg, image_id=int(iid), format=g["ECG format"].iloc[0],
                         pages=len(pages), ok=all(p.get("ok") for p in pages),
                         layout=[p.get("layout") for p in pages], sec=sum(p.get("sec", 0) for p in pages),
                         leads=per, rhythms=rhy, err=[p.get("err") for p in pages if not p.get("ok")],
                         diag=[p.get("diag") for p in pages],
                         select=[p.get("select") for p in pages],
                         dewarp=[p.get("dewarp") for p in pages]))
    (a.out / "results.json").write_text(json.dumps(dict(config=cfg, rows=rows), default=float))
    summarize(rows, a.out)


def summarize(rows, out: Path | None = None):
    """Means per scenario and overall (failure = 0 dB / PCC 0)."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import metrics as MT
    agg = defaultdict(lambda: defaultdict(list))
    for r in rows:
        kp = r["leads"].get("_kaggle")
        if kp:   # Kaggle metric: powers summed over the 12 leads -> one linear SNR per ECG image
            lin = MT.compute_snr(sum(v[0] for v in kp.values()), sum(v[1] for v in kp.values()))
            for key in ("ALL", r["scenario"]):
                agg[key]["kag"].append(lin)
        for name in LEADS:
            mt = r["leads"].get(name)
            for key in ("ALL", r["scenario"]):
                agg[key]["snr"].append(mt["snr"] if mt else 0.0)
                agg[key]["pcc"].append(mt["pcc"] if mt else 0.0)
                agg[key]["fail"].append(mt is None)
                if mt:
                    agg[key]["snr_ok"].append(mt["snr"]); agg[key]["rmse_ok"].append(mt["rmse"])
    lines = ["| scenario | ECGs | SNR (dB) | PCC | failure | SNR without failures | RMSE without failures (mV) | Kaggle score (dB) |",
             "|---|---|---|---|---|---|---|---|"]
    keys = ["ALL"] + sorted(k for k in agg if k != "ALL")
    summ = {}
    for k in keys:
        d = agg[k]
        s = dict(n=len(d["snr"]) // 12, snr=float(np.mean(d["snr"])), pcc=float(np.mean(d["pcc"])),
                 fail=float(np.mean(d["fail"])), snr_ok=float(np.mean(d["snr_ok"])) if d["snr_ok"] else 0.0,
                 rmse_ok=float(np.mean(d["rmse_ok"])) if d["rmse_ok"] else float("nan"),
                 kaggle=MT.kaggle_score(d["kag"]) if d["kag"] else float("nan"))
        summ[k] = s
        lines.append(f"| {k} | {s['n']} | {s['snr']:.2f} | {s['pcc']:.3f} | {100 * s['fail']:.1f}% | {s['snr_ok']:.2f} | "
                     f"{s['rmse_ok']:.3f} | {s['kaggle']:.2f} |")
    txt = "\n".join(lines)
    print(txt)
    if out is not None:
        (out / "summary.md").write_text(txt + "\n")
        (out / "summary.json").write_text(json.dumps(summ))
    return summ


if __name__ == "__main__":
    main()
