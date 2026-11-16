"""Avaliação do ecgdig no PMcardio ECG Image Database (PM-ECG-ID, Zenodo 13617673).

Métricas no estilo do artigo da PMcardio (Demolder et al., medRxiv 2024.08.31.24312876):
por derivação, alinhamento de até ±0,5 s, desvio vertical removido pela mediana da
diferença; PCC, RMSE (mV) e SNR (dB). Falha (derivação ausente ou < 50% coberta): PCC = 0,
SNR = 0 e RMSE do sinal todo zero, como no artigo para as métricas da base inteira.

    python benchmarks/pmcardio/eval_pm.py --root <pasta com visual_data/ e data/> \
        --meta metadata.csv --list sel.csv --out results/pm --jobs 2 [--config k=v ...]

As imagens de várias páginas do mesmo ECG são juntadas por derivação antes da comparação.
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
    """Chaves de ablação "modulo.FLAG=valor;..." (listas de 1 elemento em ecgdig.*), para
    comparar variantes na base inteira sem mudar o código."""
    import importlib
    for it in [x.strip() for x in spec.split(";") if x.strip()]:
        mod, rest = it.split(".", 1)
        var, val = rest.split("=", 1)
        getattr(importlib.import_module("ecgdig." + mod), var)[0] = ast.literal_eval(val)


def _digitize(job):
    path, cfg_kw = job
    import warnings
    warnings.filterwarnings("ignore")
    from ecgdig import Config, digitize
    import signal
    _apply_flags(os.environ.get("ECGDIG_FLAGS", ""))

    def _alarm(*_):
        raise TimeoutError("tempo esgotado")
    signal.signal(signal.SIGALRM, _alarm)
    signal.alarm(int(os.environ.get("PM_TIMEOUT", "300")))
    t0 = time.time()
    try:
        r = digitize(path, Config(**cfg_kw), return_debug=True)
        signal.alarm(0)
        out = dict(ok=True, leads=r["leads"], signals=r["signals"].astype(np.float32),
                   seg={k: [np.nonzero(np.isfinite(m))[0][[0, -1]] if np.isfinite(m).any() else None for m in v]
                        for k, v in r.get("per_segment", {}).items()},
                   layout=r.get("layout"), quality=r.get("quality"), sec=time.time() - t0)
    except Exception as e:  # noqa: BLE001
        out = dict(ok=False, err=f"{type(e).__name__}: {e}", sec=time.time() - t0)
    signal.alarm(0)
    return path, out


def _run_all(jobs, n, cache: Path, root: Path):
    """Roda com cache por imagem (retoma) e sobrevive a um processo morto (memória)."""
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
    # Lotes de n*4 imagens, um executor novo por lote (recicla a memória dos processos sem
    # max_tasks_per_child, que trava o executor no Python 3.11 depois de n*4 tarefas). Cada
    # imagem tem prazo; o que estourar vira falha e os processos do lote são mortos.
    from concurrent.futures import wait, FIRST_COMPLETED
    per_img = float(os.environ.get("ECGDIG_IMG_TIMEOUT", "900"))
    while todo:
        batch, todo = todo[: n * 4], todo[n * 4:]
        ex = ProcessPoolExecutor(n)
        futs = {ex.submit(_digitize, j): j for j in batch}
        t0 = time.time()
        pending = set(futs)
        broken = []
        while pending:
            done, pending = wait(pending, timeout=30, return_when=FIRST_COMPLETED)
            for f in done:
                j = futs[f]
                try:
                    path, o = f.result()
                except BrokenProcessPool:
                    broken.append(j)
                    continue
                except Exception as e:  # noqa: BLE001
                    path, o = j[0], dict(ok=False, err=f"{type(e).__name__}: {e}", sec=0.0)
                res[os.path.relpath(path, root)] = o
                key(path).write_bytes(pickle.dumps(o))
            if pending and time.time() - t0 > per_img * 4:
                for f in pending:
                    j = futs[f]
                    o = dict(ok=False, err="tempo esgotado", sec=time.time() - t0)
                    res[os.path.relpath(j[0], root)] = o
                    key(j[0]).write_bytes(pickle.dumps(o))
                for pr in list(getattr(ex, "_processes", {}).values()):
                    pr.kill()
                pending = set()
        ex.shutdown(wait=False, cancel_futures=True)
        for j in broken:
            # processo morto (memória?): roda sozinha; se morrer de novo, vira falha
            e1 = ProcessPoolExecutor(1)
            try:
                path, o = e1.submit(_digitize, j).result(timeout=per_img)
            except Exception as e:  # noqa: BLE001
                path, o = j[0], dict(ok=False, err=f"processo morto/tempo: {type(e).__name__}", sec=0.0)
                for pr in list(getattr(e1, "_processes", {}).values()):
                    pr.kill()
            e1.shutdown(wait=False, cancel_futures=True)
            res[os.path.relpath(path, root)] = o
            key(path).write_bytes(pickle.dumps(o))
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
    """g: corte do gabarito; F: sinal previsto (página inteira); a: início do corte na previsão;
    span: quanto a tira prevista é mais longa que o corte (tira de 10 s com corte de 2,5 s:
    o corte pode estar em qualquer ponto dela)."""
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
    # refina com passo 1
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


def evaluate_ecg(gt, rh, pages):
    """gt: (n, 12) cortes; rh: (5000, k) ritmos; pages: saídas do digitize das páginas."""
    out = {}
    n = gt.shape[0]
    for j, name in enumerate(LEADS):
        g = gt[:, j].astype(float)
        best = None
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
        out[name] = best
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
    return out, rhy


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True, help="pasta com as subpastas dos cenários")
    ap.add_argument("--gt", type=Path, required=True, help="pasta com leads.npz e rhythms.npz")
    ap.add_argument("--meta", type=Path, required=True)
    ap.add_argument("--list", type=Path, default=None, help="csv com os caminhos relativos a avaliar")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--jobs", type=int, default=2)
    ap.add_argument("--config", action="append", default=[])
    ap.add_argument("--rescore", action="store_true", help="só recalcula as métricas com o cache")
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
                         leads=per, rhythms=rhy, err=[p.get("err") for p in pages if not p.get("ok")]))
    (a.out / "results.json").write_text(json.dumps(dict(config=cfg, rows=rows), default=float))
    summarize(rows, a.out)


def summarize(rows, out: Path | None = None):
    """Médias por cenário e gerais (falha = 0 dB / PCC 0)."""
    agg = defaultdict(lambda: defaultdict(list))
    for r in rows:
        for name in LEADS:
            mt = r["leads"].get(name)
            for key in ("ALL", r["scenario"]):
                agg[key]["snr"].append(mt["snr"] if mt else 0.0)
                agg[key]["pcc"].append(mt["pcc"] if mt else 0.0)
                agg[key]["fail"].append(mt is None)
                if mt:
                    agg[key]["snr_ok"].append(mt["snr"]); agg[key]["rmse_ok"].append(mt["rmse"])
    lines = ["| cenário | ECGs | SNR (dB) | PCC | falha | SNR sem falhas | RMSE sem falhas (mV) |", "|---|---|---|---|---|---|---|"]
    keys = ["ALL"] + sorted(k for k in agg if k != "ALL")
    summ = {}
    for k in keys:
        d = agg[k]
        s = dict(n=len(d["snr"]) // 12, snr=float(np.mean(d["snr"])), pcc=float(np.mean(d["pcc"])),
                 fail=float(np.mean(d["fail"])), snr_ok=float(np.mean(d["snr_ok"])) if d["snr_ok"] else 0.0,
                 rmse_ok=float(np.mean(d["rmse_ok"])) if d["rmse_ok"] else float("nan"))
        summ[k] = s
        lines.append(f"| {k} | {s['n']} | {s['snr']:.2f} | {s['pcc']:.3f} | {100 * s['fail']:.1f}% | {s['snr_ok']:.2f} | {s['rmse_ok']:.3f} |")
    txt = "\n".join(lines)
    print(txt)
    if out is not None:
        (out / "summary.md").write_text(txt + "\n")
        (out / "summary.json").write_text(json.dumps(summ))
    return summ


if __name__ == "__main__":
    main()
