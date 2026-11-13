#!/usr/bin/env python3
"""Decomposição do erro: quanto SNR cada tipo de erro custa.

Para cada imagem (sinais salvos em sig/*.npz pela avaliação), aplica correções "oráculo"
cumulativas, cada uma usando o gabarito só para o seu tipo de erro, e mede o SNR depois de
cada passo (mesma métrica da avaliação: soma nas 12 derivações):

  0 métrica      : alinhamento da competição (±0,2 s e média por derivação)
  1a rótulos     : cada derivação do gabarito pode usar o canal previsto que melhor a explica
                   (layout/nome errado: derivação trocada ou linha atribuída à derivação errada)
  1 cobertura    : buracos dentro de derivações previstas recebem o gabarito (derivação
                   inteira ausente não conta aqui: é erro de layout, fica no resto)
  2 origem       : deslocamento comum a todas as derivações até ±1,5 s
  3 deriva       : por derivação, escala de tempo (±3 %) + deslocamento
  3b deriva local: janelas de 0,5 s com deslocamento próprio até ±80 ms (distorção
                   horizontal não linear: vinco, borda do scanner, papel curvo)
  4 ganho        : por derivação, fator de amplitude ótimo
  5 linha de base: diferença lenta (< 0,7 Hz) removida
  resto          : forma (traçado, picos cortados, fusões)

O ganho de cada passo é a perda atribuível àquele erro (na ordem acima).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy.ndimage import uniform_filter1d

LEADS = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
STEPS = ["métrica", "rótulos", "cobertura", "origem", "deriva", "deriva local", "ganho", "linha de base"]


def _snr(G, P):
    sp = npw = 0.0
    for g, p in zip(G, P):
        m = np.isfinite(g)
        if m.sum() < 10:
            continue
        gg = g[m]
        pp = np.nan_to_num(p[m], nan=0.0)
        sp += float((gg ** 2).sum())
        npw += float(((gg - pp) ** 2).sum())
    return 10 * np.log10(sp / max(npw, 1e-12)) if sp > 0 else np.nan


def _shift(p, s):
    out = np.full_like(p, np.nan)
    if s >= 0:
        out[s:] = p[:len(p) - s] if s else p
    else:
        out[:s] = p[-s:]
    return out


def _best_lag(g, p, maxs, offset=True):
    m = np.isfinite(g)
    best = (np.inf, 0)
    for s in range(-maxs, maxs + 1):
        q = _shift(p, s)[m]
        gg = g[m]
        k = np.isfinite(q)
        if k.sum() < 0.6 * m.sum():
            continue
        q = np.where(k, q, 0.0)
        if offset:
            q = q - q[k].mean() + gg[k].mean()
        e = float(((gg - q) ** 2).mean())
        if e < best[0]:
            best = (e, s)
    return best[1]


def _align1(g, p, fs):
    m = np.isfinite(g)
    s = _best_lag(g, p, int(0.2 * fs))
    q = _shift(p, s)
    k = m & np.isfinite(q)
    if k.any():
        q = q - np.nanmean(q[k]) + g[k].mean()
    return q


def _metric_align(G, P, fs):
    """Alinhamento da competição: por derivação, lag ±0,2 s e média."""
    ms = int(0.2 * fs)
    Q = np.full_like(P, np.nan)
    for i in range(12):
        g = G[i]
        m = np.isfinite(g)
        if m.sum() < 10:
            continue
        s = _best_lag(g, P[i], ms)
        q = _shift(P[i], s)
        k = m & np.isfinite(q)
        if k.any():
            q = q - np.nanmean(q[k]) + g[k].mean()
        Q[i] = q
    return Q


def decompose(G, P, fs):
    G = G.astype(np.float64)
    P = P.astype(np.float64)
    out = {}
    Q = _metric_align(G, P, fs)
    out["métrica"] = _snr(G, Q)
    # 1a rótulos: melhor canal previsto para cada derivação do gabarito
    Pl = P.copy()
    base = [_snr(G[i:i + 1], Q[i:i + 1]) for i in range(12)]
    for i in range(12):
        if not np.isfinite(G[i]).sum() > 10:
            continue
        best = (base[i] if np.isfinite(base[i]) else -np.inf, i)
        for j in range(12):
            if j == i or np.isfinite(P[j][np.isfinite(G[i])]).mean() < 0.5:
                continue
            v = _snr(G[i:i + 1], _align1(G[i], P[j], fs)[None])
            if v > best[0] + 0.5:
                best = (v, j)
        Pl[i] = P[best[1]]
    out["rótulos"] = _snr(G, _metric_align(G, Pl, fs))
    P = Pl
    # 1 cobertura (só buracos em derivações com previsão)
    has = np.array([np.isfinite(P[i][np.isfinite(G[i])]).mean() >= 0.2 if np.isfinite(G[i]).any() else False
                    for i in range(12)])
    P1 = np.where(np.isfinite(P) | ~np.isfinite(G) | ~has[:, None], P, G)
    out["cobertura"] = _snr(G, _metric_align(G, P1, fs))
    # 2 origem comum (até ±1,5 s)
    ms = int(1.5 * fs)
    best = (np.inf, 0)
    for s in range(-ms, ms + 1, max(1, int(fs / 100))):
        e = 0.0
        for i in range(12):
            m = np.isfinite(G[i])
            if m.sum() < 10:
                continue
            q = _shift(P1[i], s)[m]
            q = np.where(np.isfinite(q), q, 0.0)
            gg = G[i][m]
            e += float(((gg - gg.mean()) - (q - q.mean())) @ ((gg - gg.mean()) - (q - q.mean())))
        if e < best[0]:
            best = (e, s)
    P2 = np.stack([_shift(P1[i], best[1]) for i in range(12)])
    out["origem"] = _snr(G, _metric_align(G, P2, fs))
    out["origem_s"] = best[1] / fs
    # 3 deriva: escala de tempo por derivação ancorada no início do trecho mostrado
    P3 = np.full_like(P2, np.nan)
    for i in range(12):
        g = G[i]
        m = np.isfinite(g)
        if m.sum() < 10:
            continue
        idx = np.nonzero(m)[0]
        t0 = idx[0]
        best = (np.inf, None)
        x = np.arange(len(g), dtype=float)
        p = P2[i]
        okp = np.isfinite(p)
        if okp.sum() < 10:
            continue
        for a in np.linspace(0.97, 1.03, 25):
            xs = t0 + (x - t0) * a
            q = np.interp(xs, x[okp], p[okp], left=np.nan, right=np.nan)
            s = _best_lag(g, q, int(0.05 * fs))
            q = _shift(q, s)[m]
            k = np.isfinite(q)
            if k.sum() < 0.6 * m.sum():
                continue
            q = np.where(k, q, 0.0)
            e = float(((g[m] - g[m].mean()) - (q - q.mean())) @ ((g[m] - g[m].mean()) - (q - q.mean())))
            if e < best[0]:
                best = (e, (a, s))
        if best[1] is None:
            P3[i] = p
            continue
        a, s = best[1]
        q = np.interp(t0 + (x - t0) * a, x[okp], p[okp], left=np.nan, right=np.nan)
        P3[i] = _shift(q, s)
    out["deriva"] = _snr(G, _metric_align(G, P3, fs))
    Q3 = _metric_align(G, P3, fs)
    # 3b deriva local: deslocamento por janela de 0,5 s
    P3b = Q3.copy()
    win, ms = int(0.5 * fs), int(0.08 * fs)
    for i in range(12):
        m = np.isfinite(G[i])
        if m.sum() < win or not np.isfinite(Q3[i]).any():
            continue
        idx = np.nonzero(m)[0]
        for a0 in range(idx[0], idx[-1] + 1, win):
            sl = np.arange(a0, min(a0 + win, idx[-1] + 1))
            g = G[i][sl]
            best = (float(((g - np.nan_to_num(Q3[i][sl], nan=0.0)) ** 2).mean()), 0)  # sem deslocamento
            for s_ in range(-ms, ms + 1):
                j = sl - s_
                ok = (j >= 0) & (j < len(G[i]))
                q = np.full(len(sl), np.nan)
                q[ok] = Q3[i][j[ok]]
                if np.isfinite(q).sum() < 0.6 * len(sl):
                    continue
                e = float(((g - np.nan_to_num(q, nan=0.0)) ** 2).mean())  # buraco conta como 0, como na métrica
                if e < best[0]:
                    best = (e, s_)
            j = sl - best[1]
            ok = (j >= 0) & (j < len(G[i]))
            P3b[i][sl[ok]] = Q3[i][j[ok]]
    out["deriva local"] = _snr(G, P3b)
    Q3 = P3b
    # 4 ganho por derivação
    P4 = Q3.copy()
    for i in range(12):
        m = np.isfinite(G[i]) & np.isfinite(Q3[i])
        if m.sum() < 10:
            continue
        g = G[i][m] - G[i][m].mean()
        q = Q3[i][m] - Q3[i][m].mean()
        k = float(g @ q / max(q @ q, 1e-12))
        P4[i][m] = q * np.clip(k, 0.3, 3.0) + G[i][m].mean()
    out["ganho"] = _snr(G, P4)
    # 5 linha de base: diferença lenta
    P5 = P4.copy()
    w = int(1.4 * fs)
    for i in range(12):
        m = np.isfinite(G[i]) & np.isfinite(P4[i])
        if m.sum() < w:
            continue
        d = np.where(m, G[i] - P4[i], 0.0)
        cnt = uniform_filter1d(m.astype(float), w)
        low = uniform_filter1d(d, w) / np.maximum(cnt, 1e-6)
        P5[i] = np.where(m, P4[i] + low, P4[i])
    out["linha de base"] = _snr(G, P5)
    return out


def main(art: Path):
    res = json.loads((art / "results.json").read_text())["results"]
    meta = {(r["id"], r["type"]): r for r in res}
    rows = []
    for f in sorted((art / "sig").glob("*.npz")):
        rid, typ = f.stem.split("-")
        d = np.load(f)
        G, P, fs = d["gt"].astype(float), d["pred"].astype(float), float(d["fs"])
        dec = decompose(G, P, fs)
        dec.update(id=rid, type=typ, layout=meta.get((rid, typ), {}).get("layout"))
        rows.append(dec)
        dec = {k: (float(v) if isinstance(v, (float, np.floating)) else v) for k, v in dec.items()}
        print(rid, typ, " ".join(f"{k}={v:.2f}" for k, v in dec.items() if isinstance(v, float)), flush=True)
    (art / "decomposition.json").write_text(json.dumps(rows, indent=1, default=float))


if __name__ == "__main__":
    main(Path(sys.argv[1]))
