"""Realinhamento por coluna: páginas em que cada coluna de derivações tem as suas próprias
linhas de base.

Alguns geradores (e aparelhos que desenham cada derivação no seu nível DC) não alinham as
linhas entre colunas: numa coluna as três derivações ficam juntas, na outra uma onda S funda
empurra as de baixo. O modelo de "linhas da página" lê isso como linhas extras ou trocadas.

Quando o traço se divide em colunas por vãos verticais comuns (sem tinta em toda a altura do
traçado), em posições que dão colunas da mesma largura, e cada coluna tem o mesmo número de
linhas, cada faixa (coluna, linha) é deslocada só na vertical até a altura de referência da
linha (mediana entre colunas). Deslocar não muda amplitude nem tempo: o resto do programa vê
uma página comum.
"""
from __future__ import annotations

import numpy as np
import cv2
from scipy.signal import find_peaks

ENABLED = [True]


def _gaps(B, R, ya, yb):
    P = B[ya:yb].sum(0).astype(np.float64)
    ink = P > 0
    xs = np.nonzero(ink)[0]
    if len(xs) < 100 * R:
        return None, None
    L, Rr = int(xs[0]), int(xs[-1])
    base = np.median(P[L:Rr][P[L:Rr] > 0]) if (P[L:Rr] > 0).any() else 0
    blank = P[L:Rr] <= max(1.0, 0.03 * base)
    d = np.diff(np.concatenate([[0], blank.astype(np.int8), [0]]))
    s_, e_ = np.nonzero(d == 1)[0], np.nonzero(d == -1)[0]
    gaps = [(L + a, L + b) for a, b in zip(s_, e_) if 2 * R <= b - a <= 15 * R]
    return gaps, (L, Rr)


def _equal_columns(gaps, L, Rr, R):
    if not gaps or len(gaps) > 5:
        return False
    edges = [L] + [x for g in gaps for x in g] + [Rr]
    widths = [edges[2 * i + 1] - edges[2 * i] for i in range(len(gaps) + 1)]
    if min(widths) < 40 * R:
        return False
    if len(gaps) >= 2:
        inner = widths[1:-1]
        w = float(np.median(inner))
        if max(inner) - min(inner) > 0.03 * w:
            return False
        return all(0.97 * w <= x <= w + 30 * R for x in (widths[0], widths[-1]))
    return max(widths) - min(widths) <= 0.03 * np.median(widths) + 30 * R and \
        min(widths) >= 0.9 * max(widths) - 30 * R


def _col_rows(B, R, xa, xb):
    k = int(2 * R) | 1
    Bd = cv2.dilate(B[:, xa:xb].astype(np.uint8), np.ones((k, 1), np.uint8))
    cov = Bd.mean(1).astype(np.float64)
    cov = cv2.GaussianBlur(cov.reshape(-1, 1), (0, 0), R).ravel()
    if cov.max() <= 0:
        return np.array([], int)
    pk, _ = find_peaks(cov, distance=int(8 * R), height=0.35 * cov.max(), prominence=0.15 * cov.max())
    return pk


def col_align(rect, pap, D, B, R):
    """Devolve (rect, pap, D, B, info) realinhados, ou as entradas e None."""
    if not ENABLED[0]:
        return rect, pap, D, B, None
    H, W = B.shape
    prof = B.mean(1)
    ys = np.nonzero(prof > 0.05 * prof.max())[0]
    if len(ys) < 10:
        return rect, pap, D, B, None
    ya, yb = int(ys[0]), int(ys[-1]) + 1
    gaps, lr = _gaps(B, R, ya, yb)
    if gaps and _equal_columns(gaps, lr[0], lr[1], R):
        cuts = [0] + [int(0.5 * (a + b)) for a, b in gaps] + [W]
        cols = [(cuts[i], cuts[i + 1]) for i in range(len(cuts) - 1)]
        inner = cols
    else:
        return rect, pap, D, B, None
    pks = [_col_rows(B, R, a, b) for a, b in inner]
    n = len(pks[0])
    if n < 2 or any(len(p) != n for p in pks):
        return rect, pap, D, B, None
    P = np.array(pks, float)              # (colunas, linhas)
    ref = np.median(P, 0)
    dl = ref[None, :] - P
    sp = float(np.median(np.diff(ref)))
    if np.abs(dl).max() < 3 * R or sp < 8 * R:
        return rect, pap, D, B, None
    if np.abs(dl).max() > 1.5 * sp:       # deslocamento maior que uma linha: casamento duvidoso
        return rect, pap, D, B, None
    rect2 = np.full_like(rect, 255); pap2 = np.zeros_like(pap); D2 = np.zeros_like(D); B2 = np.zeros_like(B)
    for c, (xa, xb) in enumerate(cols):
        p = P[c]
        bounds = [0] + [int(0.5 * (p[i] + p[i + 1])) for i in range(n - 1)] + [H]
        for i in range(n):
            s0, s1 = bounds[i], bounds[i + 1]
            d = int(round(dl[c, i]))
            t0, t1 = s0 + d, s1 + d
            # recorta ao quadro
            o0, o1 = max(0, t0), min(H, t1)
            if o1 <= o0:
                continue
            a0, a1 = o0 - d, o1 - d
            sl_t, sl_s = np.s_[o0:o1, xa:xb], np.s_[a0:a1, xa:xb]
            B2[sl_t] |= B[sl_s]
            D2[sl_t] = np.maximum(D2[sl_t], D[sl_s])
            rect2[sl_t] = np.minimum(rect2[sl_t], rect[sl_s])
            pap2[sl_t] = np.maximum(pap2[sl_t], pap[sl_s])
    return rect2, pap2, D2, B2, dict(cols=cols, peaks=P.tolist(), ref=ref.tolist())
