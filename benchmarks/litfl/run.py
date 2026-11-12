#!/usr/bin/env python3
"""Benchmark LITFL: 102 ECGs reais (scans/fotos de várias fontes) sem sinal verdadeiro.

Mede contra uma referência revisada à mão (reference.json):
  - layout impresso, nomes das tiras de ritmo, velocidade, duração e ganho;
e, sem referência nenhuma, a coerência interna do sinal extraído:
  - cobertura: fração das amostras que a página mostra e que saíram extraídas;
  - Einthoven: |I - II + III| / RMS(I, II, III) nas medidas da grade ANTES da fusão (3x4: as três
    derivações de membros estão na mesma coluna, no mesmo instante);
  - duplicata: NRMSE entre a tira de ritmo II e a II da grade no mesmo trecho (só quando a
    tira é simultânea; em montagens do LITFL nem sempre é).

Uso:
  python benchmarks/litfl/run.py                      # baixa (cache em benchmarks/litfl/cache) e roda tudo
  python benchmarks/litfl/run.py --limit 20 --jobs 2
  python benchmarks/litfl/run.py --compare results/litfl_main.json   # aponta regressões
Saída: results/litfl/results.json (por imagem + resumo) e results/litfl/report.md
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
import urllib.request
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))

UA = {"User-Agent": "Mozilla/5.0 (ECG digitization research benchmark)"}


def fetch(entry, cache: Path):
    url = entry["url"]
    ext = os.path.splitext(url.split("?")[0])[1] or ".jpg"
    p = cache / f"{entry['file']}{ext}"
    if not p.exists():
        req = urllib.request.Request(url, headers=UA)
        with urllib.request.urlopen(req, timeout=40) as f:
            data = f.read(25 * 1024 * 1024)
        p.write_bytes(data)
    return p


def _family(name):
    if name is None:
        return None
    return name.replace("-cabrera", "")


def _expected_fraction(layout_rows, fs=500, n=5000):
    """Fração do tempo em que cada derivação aparece no papel, pelo layout."""
    from ecgdig.layouts import LEADS
    frac = np.zeros(len(LEADS))
    for row in layout_rows:
        k = len(row)
        for nm in row:
            base = nm.lstrip("-")
            if base in LEADS:
                frac[LEADS.index(base)] = min(1.0, frac[LEADS.index(base)] + 1.0 / k)
    return frac


def evaluate_one(args):
    key, entry, path, cfg_kw = args
    from ecgdig import digitize, Config
    out = dict(id=key, ok=False)
    t0 = time.monotonic()
    try:
        r = digitize(str(path), Config(**cfg_kw), return_debug=True)
    except Exception as e:  # noqa: BLE001
        out.update(error=repr(e)[:300], trace=traceback.format_exc(limit=2)[-500:], seconds=time.monotonic() - t0)
        return out
    out["seconds"] = round(time.monotonic() - t0, 1)
    rows = r["layout_rows"]
    rhythm = [row[0] for row in rows if len(row) == 1] if any(len(x) > 1 for x in rows) else []
    gains = sorted(set(float(g) for g in r["gains"].values()))
    li = r.get("layout_info") or {}
    out.update(ok=True, layout=r["layout"], rows=rows, rhythm=rhythm, speed=r["speed_mm_per_s"],
               speed_source=r["speed_source"], duration=r["duration_s"], gains=gains,
               confidence=float(li.get("confidence") or 0.0), warning=li.get("warning"),
               rotation=r.get("rotation"), pulses_left=r.get("pulses_found"), pulses_right=r.get("pulses_right"),
               gain_rows_inferred=r.get("gain_rows_inferred"))
    # ---- acertos contra a referência
    ref = entry
    chk = {}
    if ref.get("layout"):
        chk["layout"] = _family(r["layout"]) == ref["layout"]
    if ref.get("rhythm") is not None and ref.get("layout") and ref["layout"] != "rows":
        chk["rhythm"] = rhythm == ref["rhythm"]
        chk["rhythm_strips_ok"] = sum(a == b for a, b in zip(rhythm, ref["rhythm"]))
        chk["rhythm_strips"] = len(ref["rhythm"])
    if ref.get("speed"):
        chk["speed"] = abs(r["speed_mm_per_s"] - ref["speed"]) < 1e-6
    if ref.get("duration"):
        chk["duration"] = abs(r["duration_s"] - ref["duration"]) / ref["duration"] < 0.05
    if ref.get("gain"):
        chk["gain"] = gains == sorted(ref["gain"]) if len(ref["gain"]) > 1 else all(abs(g - ref["gain"][0]) < 1e-6 for g in gains)
    out["check"] = chk
    # ---- coerência interna (sem referência)
    sig = r["signals"]
    exp = _expected_fraction(rows)
    got = np.isfinite(sig).mean(1)
    m = exp > 0
    out["coverage"] = float(np.clip(got[m] / exp[m], 0, 1).mean()) if m.any() else None
    ps = r.get("per_segment", {})

    def first(nm):
        lst = ps.get(nm) or []
        return lst[0] if lst else None
    I, II, III = first("I"), first("II"), first("III")
    if I is not None and II is not None and III is not None:
        mm = np.isfinite(I) & np.isfinite(II) & np.isfinite(III)
        if mm.sum() > 200:
            V = np.stack([I[mm], II[mm], III[mm]])
            V = V - np.median(V, axis=1, keepdims=True)
            scale = np.sqrt(np.mean(V ** 2))
            if scale > 0:
                e = V[0] - V[1] + V[2]
                # normalizado pela amplitude das três (II sozinha pode ser quase plana)
                out["einthoven_ratio"] = float(np.sqrt(np.mean(e ** 2)) / scale)
    lst = ps.get("II") or []
    if len(lst) >= 2:
        a, b = lst[0], lst[1]
        mm = np.isfinite(a) & np.isfinite(b)
        if mm.sum() > 200:
            na, nb = np.linalg.norm(a[mm]), np.linalg.norm(b[mm])
            out["dup_II_nrmse"] = float(np.linalg.norm(a[mm] - b[mm]) / max(na, nb, 1e-9))
    return out


def summarize(results, ref):
    ok = [x for x in results if x["ok"]]
    S = dict(images=len(results), ran=len(ok), errors=len(results) - len(ok))
    for k in ("layout", "rhythm", "speed", "duration", "gain"):
        v = [x["check"][k] for x in ok if k in x.get("check", {})]
        S[f"{k}_acc"] = round(float(np.mean(v)), 3) if v else None
        S[f"{k}_n"] = len(v)
    st = [(x["check"]["rhythm_strips_ok"], x["check"]["rhythm_strips"]) for x in ok if "rhythm_strips" in x.get("check", {})]
    S["rhythm_strip_acc"] = round(sum(a for a, _ in st) / max(1, sum(b for _, b in st)), 3)
    allok = [all(x["check"].get(k, True) for k in ("layout", "speed", "duration", "gain")) for x in ok]
    S["all_structural_ok"] = round(float(np.mean(allok)), 3) if allok else None
    for k in ("coverage", "einthoven_ratio", "dup_II_nrmse"):
        v = [x[k] for x in ok if x.get(k) is not None]
        S[f"{k}_median"] = round(float(np.median(v)), 3) if v else None
        S[f"{k}_n"] = len(v)
    S["seconds_mean"] = round(float(np.mean([x["seconds"] for x in results if "seconds" in x])), 1)
    return S


def report_md(S, results, ref, cmp=None):
    L = ["# Benchmark LITFL", "",
         f"{S['ran']}/{S['images']} imagens processadas ({S['errors']} erros), {S['seconds_mean']} s/imagem.", "",
         "| Métrica | Valor | n |", "|---|---|---|"]
    names = dict(layout_acc="Layout correto", rhythm_acc="Tiras de ritmo (todas certas)", rhythm_strip_acc="Tiras de ritmo (por tira)",
                 speed_acc="Velocidade", duration_acc="Duração", gain_acc="Ganho", all_structural_ok="Layout+velocidade+duração+ganho",
                 coverage_median="Cobertura (mediana)", einthoven_ratio_median="Einthoven |I−II+III|/RMS(I,II,III) (mediana)",
                 dup_II_nrmse_median="Ritmo II × grade II, NRMSE (mediana)")
    for k, lab in names.items():
        n = S.get(k.replace("_acc", "_n").replace("_median", "_n"), "")
        if k in ("rhythm_strip_acc", "all_structural_ok"):
            n = ""
        val = S.get(k)
        old = f" (antes {cmp[k]})" if cmp and cmp.get(k) is not None and cmp.get(k) != val else ""
        L.append(f"| {lab} | {val}{old} | {n} |")
    L += ["", "## Erros por imagem", "", "| Imagem | Falhou em | Detectado | Referência |", "|---|---|---|---|"]
    for x in results:
        if not x["ok"]:
            L.append(f"| {x['id']} | exceção | {x.get('error','')[:60]} | |")
            continue
        bad = [k for k, v in x["check"].items() if v is False]
        if bad:
            rf = ref[x["id"]]
            L.append(f"| {x['id']} | {', '.join(bad)} | {x['layout']} {'/'.join(x['rhythm'])} {x['speed']:g} mm/s "
                     f"{x['duration']:g} s {x['gains']} | {rf.get('layout')} {'/'.join(rf.get('rhythm') or [])} "
                     f"{rf.get('speed')} {rf.get('duration')} {rf.get('gain')} |")
    return "\n".join(L) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reference", type=Path, default=HERE / "reference.json")
    ap.add_argument("--cache", type=Path, default=HERE / "cache")
    ap.add_argument("--out", type=Path, default=Path("results/litfl"))
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2)))
    ap.add_argument("--compare", type=Path, default=None, help="results.json de uma rodada anterior")
    ap.add_argument("--config", action="append", default=[], metavar="CHAVE=VALOR",
                    help="campo do Config (ex.: --config nonrigid=poly)")
    a = ap.parse_args()
    import ast
    cfg_kw = {}
    for kv in a.config:
        k, v = kv.split("=", 1)
        try:
            cfg_kw[k] = ast.literal_eval(v)
        except (ValueError, SyntaxError):
            cfg_kw[k] = v
    ref = json.loads(a.reference.read_text())["images"]
    a.cache.mkdir(parents=True, exist_ok=True)
    a.out.mkdir(parents=True, exist_ok=True)
    todo = []
    for key in sorted(ref):
        if ref[key].get("review") == "indisponivel":
            continue
        try:
            todo.append((key, ref[key], fetch(ref[key], a.cache), cfg_kw))
        except Exception as e:  # noqa: BLE001
            print(f"download falhou {key}: {e}", flush=True)
        if a.limit and len(todo) >= a.limit:
            break
    results = []
    with ProcessPoolExecutor(max_workers=a.jobs) as ex:
        for res in ex.map(evaluate_one, todo):
            results.append(res)
            print(res["id"], "ok" if res["ok"] else "ERRO", {k: v for k, v in res.get("check", {}).items() if v is False}, flush=True)
    S = summarize(results, ref)
    cmp = json.loads(a.compare.read_text())["summary"] if a.compare and a.compare.exists() else None
    (a.out / "results.json").write_text(json.dumps(dict(summary=S, config=cfg_kw, results=results), indent=1, default=str))
    (a.out / "report.md").write_text(report_md(S, results, ref, cmp))
    print(json.dumps(S, indent=1))


if __name__ == "__main__":
    main()
