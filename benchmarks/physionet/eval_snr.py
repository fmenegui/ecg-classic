"""Avaliação no treino da competição do Kaggle "PhysioNet - Digitization of ECG Images".

Formato (conferido com `run.py inspect`):
  train.csv               id, fs, sig_len (fs varia: 250, 500, 512, 1000 Hz; 10 s)
  train/<id>/<id>.csv     12 colunas (I..V6), sig_len linhas; vazio onde a derivação não aparece
  train/<id>/<id>-NNNN.png  9 imagens do mesmo registro (NNNN = tipo de captura: render,
                          scans, fotos de celular etc.)

Métrica: SNR por imagem = 10·log10(Σ sinal² / Σ ruído²) somando as 12 derivações, só nas
amostras que o papel mostra. Duas versões:
  - bruto: sem nenhum ajuste;
  - alinhado: por derivação, desloca até ±0,2 s e remove a diferença de média (o que a
    métrica da competição e a do PhysioNet 2024 toleram). É uma reimplementação: o placar
    oficial pode diferir em detalhes.
"""
from __future__ import annotations

import csv
import json
import random
import time
import traceback
from collections import defaultdict
from pathlib import Path

import numpy as np

LEADS = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]


def _snr_record(gt, pred, fs, align=True, max_shift_s=0.2):
    """gt, pred: (12, n) no fs do gabarito. Retorna (snr_db, por_derivação)."""
    sig_p = noise_p = 0.0
    per = {}
    ms = int(round(max_shift_s * fs))
    for i, l in enumerate(LEADS):
        g = gt[i]
        m = np.isfinite(g)
        if m.sum() < 10:
            continue
        p = np.nan_to_num(pred[i], nan=0.0)
        gi = g[m]
        best = None
        # passo de 1 amostra: com passo grosso (4 ms) o SNR de um render varia ~6 dB com
        # um deslocamento de 1 ms (QRS íngreme), o que torna a comparação entre versões ruído
        shifts = range(-ms, ms + 1) if align else [0]
        idx = np.nonzero(m)[0]
        for s in shifts:
            j = idx + s
            ok = (j >= 0) & (j < len(p))
            if ok.sum() < 0.8 * len(idx):
                continue
            pi = p[j[ok]]
            gg = gi[ok]
            if align:
                pi = pi - pi.mean() + gg.mean()
            e = float(((gg - pi) ** 2).sum())
            if best is None or e < best[0]:
                best = (e, float((gg ** 2).sum()), s)
        if best is None:
            continue
        noise_p += best[0]
        sig_p += best[1]
        per[l] = 10 * np.log10(best[1] / max(best[0], 1e-12))
    if sig_p == 0:
        return float("nan"), per
    return 10 * np.log10(sig_p / max(noise_p, 1e-12)), per


def _to_gt_grid(sig500, fs, n):
    """Reamostra a saída do ecgdig (500 Hz, começa em t=0) para o fs e o tamanho do gabarito."""
    out = np.full((12, n), np.nan)
    t_gt = np.arange(n) / fs
    t_me = np.arange(sig500.shape[1]) / 500.0
    for i in range(12):
        y = sig500[i]
        m = np.isfinite(y)
        if m.sum() < 2:
            continue
        v = np.interp(t_gt, t_me[m], y[m], left=np.nan, right=np.nan)
        # buracos internos (amostras desconhecidas) continuam desconhecidos
        holes = ~np.isfinite(np.interp(t_gt, t_me, np.where(m, 0.0, np.nan)))
        v[holes] = np.nan
        out[i] = v
    return out


TYPES = ("0001", "0003", "0004", "0005", "0006", "0009", "0010", "0011", "0012")


def _load_gt(gt_p):
    gt_rows = list(csv.reader(open(gt_p)))
    hdr = gt_rows[0]
    G = np.full((12, len(gt_rows) - 1), np.nan)
    for k, l in enumerate(LEADS):
        if l in hdr:
            c = hdr.index(l)
            G[k] = [float(r[c]) if c < len(r) and r[c] != "" else np.nan for r in gt_rows[1:]]
    return G


def _work(job):
    """Uma imagem: digitaliza, mede, salva sinais (250 Hz) e sobreposição."""
    from ecgdig import digitize, Config
    import cv2
    rid, typ, fs, img_p, gt_p, out, cfg_kw, keep_raw = job
    out = Path(out)
    rec = dict(id=rid, type=typ, fs=fs)
    t0 = time.monotonic()
    try:
        G = _load_gt(gt_p)
        n = G.shape[1]
        r = digitize(str(img_p), Config(**cfg_kw), return_debug=True)
        S12 = np.full((12, r["signals"].shape[1]), np.nan)
        for k_, nm in enumerate(r["leads"]):  # layout livre pode ter nomes fora dos 12
            if nm in LEADS:
                S12[LEADS.index(nm)] = r["signals"][k_]
        P = _to_gt_grid(S12, fs, n)
        it = r.get("internals", {})
        rec.update(ok=True, layout=r["layout"], rows=r["layout_rows"], speed=r["speed_mm_per_s"],
                   speed_source=r.get("speed_source"), duration=r["duration_s"],
                   duration_assumed=r.get("duration_assumed"), horizontal_scale=r.get("horizontal_scale"),
                   gains=sorted(set(float(g) for g in r["gains"].values())),
                   confidence=float((r.get("layout_info") or {}).get("confidence") or 0),
                   ppmm=r.get("ppmm"), angle=r.get("angle"), rotation=r.get("rotation"),
                   grid_residual=r.get("grid_residual_mm"), grid_residual_cell=r.get("grid_residual_cell_mm"),
                   grid_snap=r.get("grid_snap_mm"), grid_fine=r.get("grid_fine_mm"),
                   pulses_left=r.get("pulses_found"), pulses_right=r.get("pulses_right"),
                   x0_rows=[float(v) for v in it.get("x0_rows", [])], scale_rows=[float(v) for v in it.get("scale_rows", [])],
                   image_shape=list(cv2.imread(str(img_p)).shape[:2]))
        snr_raw, _ = _snr_record(G, P, fs, align=False)
        snr_al, per = _snr_record(G, P, fs, align=True)
        rec.update(snr_raw=snr_raw, snr=snr_al, per_lead=per)
        # sinais a 250 Hz (float16) para o diagnóstico fora do runner
        step = max(1, int(round(fs / 250)))
        (out / "sig").mkdir(exist_ok=True)
        np.savez_compressed(out / "sig" / f"{rid}-{typ}.npz", gt=G[:, ::step].astype(np.float16),
                            pred=P[:, ::step].astype(np.float16), fs=fs / step)
        d = r["debug_image"]
        sc = min(1.0, 900 / d.shape[1])
        cv2.imwrite(str(out / "img" / f"{rid}-{typ}.jpg"), cv2.resize(d, None, fx=sc, fy=sc,
                    interpolation=cv2.INTER_AREA), [cv2.IMWRITE_JPEG_QUALITY, 62])
    except Exception as e:  # noqa: BLE001
        rec.update(ok=False, error=repr(e)[:200], trace=traceback.format_exc(limit=3)[-600:])
    rec["seconds"] = round(time.monotonic() - t0, 1)
    if keep_raw:
        import shutil
        (out / "raw").mkdir(exist_ok=True)
        shutil.copy(img_p, out / "raw" / f"{rid}-{typ}.png")
        shutil.copy(gt_p, out / "raw" / f"{rid}.csv")
    try:
        Path(img_p).unlink()
    except OSError:
        pass
    print(rid, typ, rec.get("layout"), None if rec.get("snr") is None else round(rec["snr"], 2),
          rec.get("error", ""), flush=True)
    return rec


def evaluate(a):
    from run import download  # mesmo módulo de download/CLI
    from concurrent.futures import ProcessPoolExecutor
    import ast

    out: Path = a.out
    out.mkdir(parents=True, exist_ok=True)
    (out / "img").mkdir(exist_ok=True)
    cfg_kw = {}
    for kv in a.config:
        for part in kv.split():
            k, v = part.split("=", 1)
            try:
                cfg_kw[k] = ast.literal_eval(v)
            except (ValueError, SyntaxError):
                cfg_kw[k] = v
    cache = out / "data"
    meta_p = download("train.csv", cache)
    rows = list(csv.DictReader(open(meta_p)))
    random.Random(2026).shuffle(rows)
    ids = [x for x in getattr(a, "ids", "").split(",") if x]
    rows = [r for r in rows if r["id"] in ids] if ids else rows[: a.n]
    keep = getattr(a, "keep_raw", 5)
    results = []
    with ProcessPoolExecutor(max_workers=max(1, a.jobs)) as ex:
        for ri, row in enumerate(rows):
            rid, fs = row["id"], float(row["fs"])
            try:
                gt_p = download(f"train/{rid}/{rid}.csv", cache / rid)
            except Exception as e:  # noqa: BLE001
                print("sem gabarito", rid, e, flush=True)
                continue
            jobs = []
            for typ in TYPES:
                try:
                    p = download(f"train/{rid}/{rid}-{typ}.png", cache / rid)
                except Exception:  # noqa: BLE001
                    continue
                jobs.append((rid, typ, fs, str(p), str(gt_p), str(out), cfg_kw, ri < keep))
            results += list(ex.map(_work, jobs))
            (out / "results.json").write_text(json.dumps(dict(config=cfg_kw, results=results), indent=1, default=str))
    _report(out, results, cfg_kw)


def _report(out, results, cfg_kw):
    ok = [r for r in results if r.get("ok")]
    by = defaultdict(list)
    for r in ok:
        by[r["type"]].append(r)
    L = ["# PhysioNet (Kaggle): SNR do ecgdig", "",
         f"{len({r['id'] for r in results})} registros, {len(results)} imagens, {len(ok)} processadas "
         f"({len(results) - len(ok)} exceções). Config: {cfg_kw or 'padrão'}.", "",
         "SNR por imagem somando as 12 derivações nas amostras mostradas. *Alinhado*: até ±0,2 s e média "
         "removida por derivação (reimplementação; o placar oficial pode diferir).", "",
         "| Tipo | n | SNR alinhado (média) | mediana | SNR bruto (média) | layout mais comum | s/imagem |",
         "|---|---|---|---|---|---|---|"]
    for t in sorted(by):
        v = [r["snr"] for r in by[t] if np.isfinite(r["snr"])]
        vr = [r["snr_raw"] for r in by[t] if np.isfinite(r["snr_raw"])]
        lay = max(set(r["layout"] for r in by[t]), key=[r["layout"] for r in by[t]].count)
        L.append(f"| {t} | {len(by[t])} | {np.mean(v):.2f} dB | {np.median(v):.2f} | {np.mean(vr):.2f} | {lay} | "
                 f"{np.mean([r['seconds'] for r in by[t]]):.1f} |")
    allv = [r["snr"] for r in ok if np.isfinite(r["snr"])]
    if allv:
        L += ["", f"**Todas: {np.mean(allv):.2f} dB (mediana {np.median(allv):.2f}).** Exceções contam como 0 dB "
              f"na média da competição; com elas: {np.mean(allv + [0.0] * (len(results) - len(ok))):.2f} dB."]
    perl = defaultdict(list)
    for r in ok:
        for l, v in r.get("per_lead", {}).items():
            perl[l].append(v)
    if perl:
        L += ["", "| Derivação | SNR médio |", "|---|---|"] + [f"| {l} | {np.mean(perl[l]):.2f} dB |" for l in LEADS if l in perl]
    err = [r for r in results if not r.get("ok")]
    if err:
        L += ["", "## Exceções", ""] + [f"- {r['id']}-{r['type']}: {r.get('error')}" for r in err[:30]]
    (out / "report.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))
