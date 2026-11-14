"""Modo por segmento: cada derivação é localizada pelo próprio rótulo impresso.

O modelo principal do ecgdig supõe "linhas da página": cada linha tem uma base e se divide
em colunas no tempo. Vários geradores e aparelhos não seguem isso: cada coluna de
derivações tem as suas próprias linhas de base (uma derivação de amplitude grande empurra a
coluna para baixo), e as linhas de colunas diferentes não se alinham.

Aqui a página é lida como um conjunto de segmentos: para cada rótulo legível (I..V6), a base
é o traço logo à direita dele; o segmento vai até o próximo rótulo na mesma faixa (ou até o
fim do traço). O tempo vem da posição x a partir da margem esquerda dos rótulos da primeira
coluna. Cada segmento é rastreado com o mesmo Viterbi do modo principal, numa faixa limitada
pelos segmentos vizinhos que se sobrepõem em x.

A escolha entre este modo e o principal é física (Einthoven/Goldberger nos trechos
simultâneos), não depende de gabarito.
"""
from __future__ import annotations

import numpy as np
import cv2

from . import ocr
from .extract import column_runs, viterbi_track, find_pulses

NAMES = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]


def page_words(B, D, R, min_score=0.5):
    """Rótulos de derivação legíveis na página toda (componentes pequenas e isoladas)."""
    n, lab, st, _ = cv2.connectedComponentsWithStats(B.astype(np.uint8), connectivity=8)
    small = (st[:, cv2.CC_STAT_WIDTH] < 12 * R) & (st[:, cv2.CC_STAT_HEIGHT] < 7 * R)
    small[0] = False
    iso = small[lab]
    out = []
    for (x0, y0, x1, y1) in ocr.find_words(iso, R):
        crop = iso[y0:y1, x0:x1]
        d = ocr.read_label(crop)
        if not d:
            continue
        name = max(d, key=d.get)
        sc = d[name]
        if name not in NAMES or sc < min_score or (name == "I" and list(d) == ["I"] and sc < 0.7):
            continue
        out.append(dict(name=name, score=float(sc), x0=int(x0), y0=int(y0), x1=int(x1), y1=int(y1)))
    return out


def _baseline_near(B, w, R):
    """Base do traço logo à direita do rótulo: mediana da altura da tinta em 25 mm."""
    H, W = B.shape
    xa, xb = int(w["x1"] + 1 * R), min(W, int(w["x1"] + 25 * R))
    yc = 0.5 * (w["y0"] + w["y1"])
    ya, yb = max(0, int(yc - 10 * R)), min(H, int(yc + 10 * R))
    sub = B[ya:yb, xa:xb]
    if sub.size == 0:
        return None
    ys = []
    for j in range(sub.shape[1]):
        col = np.nonzero(sub[:, j])[0]
        if len(col):
            # tinta mais perto da altura do rótulo (o traço vizinho pode entrar na janela)
            k = col[np.argmin(np.abs(col + ya - yc))]
            ys.append(k + ya)
    if len(ys) < 5 * R:
        return None
    return float(np.median(ys))


def segment_mode(B, D, R, speed=25.0, fs=500, gain=None, min_labels=8):
    """Lê a página por segmentos. Devolve dict(leads, signals, per_segment, duration) ou None."""
    H, W = B.shape
    words = page_words(B, D, R)
    # um rótulo por (nome, posição): duplicatas próximas viram uma
    words.sort(key=lambda w: -w["score"])
    keep = []
    for w in words:
        if any(abs(w["x0"] - k["x0"]) < 5 * R and abs(w["y0"] - k["y0"]) < 3 * R for k in keep):
            continue
        keep.append(w)
    words = keep
    if len({w["name"] for w in words}) < min_labels:
        return None
    for w in words:
        w["base"] = _baseline_near(B, w, R)
    words = [w for w in words if w["base"] is not None]
    if len({w["name"] for w in words}) < min_labels:
        return None
    # ganho pelos pulsos (altura modal), senão o padrão
    if gain is None:
        P = find_pulses(B, R)
        hs = [p["foot"] - p["top"] for p in P if 4 * R <= p["foot"] - p["top"] <= 25 * R]
        gain = 10.0
        if len(hs) >= 3:
            h = float(np.median(hs)) / R
            gain = min((5.0, 10.0, 20.0), key=lambda g: abs(g - h)) if min(abs(h - g) for g in (5, 10, 20)) < 1.5 else h
    # fim de cada segmento: próximo rótulo à direita na mesma faixa (±4 mm), ou fim do traço
    x_org = min(w["x0"] for w in words)
    segs = []
    for w in words:
        nxt = [v["x0"] for v in words if v is not w and v["x0"] > w["x1"] + 10 * R and abs(v["base"] - w["base"]) < 4 * R]
        xe = min(nxt) - 1 * R if nxt else None
        if xe is None:
            # fim do traço: última coluna com tinta perto da base
            a, b = max(0, int(w["base"] - 6 * R)), min(H, int(w["base"] + 6 * R))
            has = B[a:b, int(w["x1"]):].any(0)
            idx = np.nonzero(has)[0]
            if not len(idx):
                continue
            # corta em vãos > 8 mm
            gaps = np.nonzero(np.diff(idx) > 8 * R)[0]
            last = idx[gaps[0]] if len(gaps) else idx[-1]
            xe = int(w["x1"]) + int(last)
        xs = int(w["x0"])
        if xe - xs < 15 * R:
            continue
        segs.append(dict(name=w["name"], base=w["base"], xa=xs, xb=int(xe), score=w["score"]))
    if len({s["name"] for s in segs}) < min_labels:
        return None
    dur = max(s["xb"] for s in segs) - x_org
    duration = dur / (speed * R)
    N = int(round(duration * fs))
    if N < fs:
        return None
    sig = {n_: np.full(N, np.nan) for n_ in NAMES}
    per_seg = {n_: [] for n_ in NAMES}
    for s in segs:
        # faixa: metade da distância aos segmentos vizinhos que se sobrepõem em x
        nb = [t["base"] for t in segs if t is not s and min(t["xb"], s["xb"]) - max(t["xa"], s["xa"]) > 5 * R]
        up = min([s["base"] - v for v in nb if v < s["base"]] + [25 * R])
        dn = min([v - s["base"] for v in nb if v > s["base"]] + [25 * R])
        ya, yb = max(0, int(s["base"] - 0.9 * up)), min(H, int(s["base"] + 0.9 * dn))
        xa, xb = max(0, s["xa"]), min(W, s["xb"])
        runs = column_runs(B, D, ya, yb, xa, xb)
        sel = viterbi_track(runs, float(s["base"]), R, others=[float(v) for v in nb])
        cen = np.array([runs[i][k][2] if k is not None else np.nan for i, k in enumerate(sel)], float)
        ok = np.isfinite(cen)
        if ok.sum() < 10 * R:
            continue
        xs_ = xa + np.arange(len(cen))
        base = float(np.nanmedian(cen))
        v = (base - cen) / (gain * R)
        t = (xs_ - x_org) / (speed * R)
        n = np.clip(np.round(t * fs).astype(int), 0, N - 1)
        tt = np.arange(n[ok].min(), n[ok].max() + 1)
        vv = np.interp(tt / fs, t[ok], v[ok])
        m = np.full(N, np.nan); m[tt] = vv
        per_seg[s["name"]].append(m)
        cur = sig[s["name"]]
        sig[s["name"]] = np.where(np.isnan(cur), m, np.where(np.isnan(m), cur, 0.5 * (cur + m)))
    leads = [n_ for n_ in NAMES if np.isfinite(sig[n_]).any()]
    if len(leads) < min_labels:
        return None
    return dict(leads=leads, signals=np.stack([sig[n_] for n_ in leads]).astype(np.float32),
                per_segment={k: v for k, v in per_seg.items() if v}, duration=duration, gain=gain,
                n_words=len(words))


def physics_residual(leads, signals):
    """Resíduo normalizado de Einthoven (I+III=II) e Goldberger (aVR+aVL+aVF=0) nos trechos
    simultâneos. Menor = leitura mais coerente. None se não houver trechos para testar."""
    d = {n: s for n, s in zip(leads, signals)}
    res = []
    for combo, coef in ((("I", "III", "II"), (1, 1, -1)), (("aVR", "aVL", "aVF"), (1, 1, 1))):
        if all(c in d for c in combo):
            V = np.stack([d[c] for c in combo]).astype(float)
            m = np.isfinite(V).all(0)
            if m.sum() < 200:
                continue
            V = V[:, m] - np.median(V[:, m], 1, keepdims=True)
            r = np.tensordot(np.array(coef, float), V, 1)
            res.append(float(np.sqrt(np.mean(r ** 2)) / (np.sqrt(np.mean(V ** 2)) + 1e-9)))
    return float(np.mean(res)) if res else None
