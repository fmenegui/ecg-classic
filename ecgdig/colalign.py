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
LONG_EXTENT = [True]
ROW_DIST6 = [True]   # linhas de uma coluna a >= 6 mm (antes 8): aVL/aVF apertadas


def _gaps(B, R, ya, yb):
    P = B[ya:yb].sum(0).astype(np.float64)
    ink = P > 0
    xs = np.nonzero(ink)[0]
    if len(xs) < 100 * R:
        return None, None
    L, Rr = int(xs[0]), int(xs[-1])
    if LONG_EXTENT[0]:
        # ilhas estreitas (< 20 mm) nas pontas, separadas do corpo por >= 3 mm quase sem tinta,
        # são marca de canto, tracinho de grade escura ou texto: não esticam as colunas
        for _ in range(4):
            seg = P[L:Rr + 1]
            b0 = np.median(seg[seg > 0]) if (seg > 0).any() else 0
            d_ = np.diff(np.concatenate([[0], (seg <= max(1.0, 0.03 * b0)).astype(np.int8), [0]]))
            runs = [(a, b) for a, b in zip(np.nonzero(d_ == 1)[0], np.nonzero(d_ == -1)[0]) if b - a >= 3 * R]
            if not runs:
                break
            L0, R0 = L, Rr
            a, b = runs[-1]
            if (Rr - L + 1) - b < 20 * R and a >= 100 * R:
                Rr = L + a - 1
            a, b = runs[0]
            if a < 20 * R and Rr - (L + b) >= 100 * R:
                L = L + b
            if (L, Rr) == (L0, R0):
                break
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
    pk, _ = find_peaks(cov, distance=int((6 if ROW_DIST6[0] else 8) * R), height=0.35 * cov.max(), prominence=0.15 * cov.max())
    return pk


EQUAL_SPLIT = [True]
# colunas com linhas em alturas próprias (cada coluna com o seu espaçamento): as faixas
# (coluna, linha) vão para linhas novas de passo comum, em vez da mediana entre colunas
RELAYOUT = [True]


def _rows_are_traces(B, R, cols, pks, lr):
    """Toda linha de toda coluna é traço (tinta perto da altura em >= 85% da largura da
    coluna, dentro da extensão do traçado): cabeçalho ou rodapé de texto numa coluna só não
    passam."""
    H = B.shape[0]
    for (xa, xb), p in zip(cols, pks):
        a, b = max(xa, lr[0]), min(xb, lr[1] + 1)
        if b - a < 20 * R:
            return False
        sub = B[:, a:b]
        a, b = 0, b - a
        for y in p:
            band = sub[max(0, int(y - 3 * R)):min(H, int(y + 3 * R)), a:b]
            if band.any(0).mean() < 0.85:
                return False
    return True


def _equal_split(B, R, ya, yb, lr):
    L, Rr = lr
    P = B[ya:yb].sum(0).astype(np.float64)
    for k in (4, 2):
        w = (Rr - L) / k
        if w < 40 * R:
            continue
        cuts = [0]
        for i in range(1, k):
            x = L + i * w
            a, b = int(x - 4 * R), int(x + 4 * R)
            cuts.append(a + int(np.argmin(P[a:b])))
        cuts.append(B.shape[1])
        cols = [(cuts[i], cuts[i + 1]) for i in range(k)]
        pks = [_col_rows(B, R, a, b) for a, b in cols]
        n = len(pks[0])
        if n >= 2 and all(len(p) == n for p in pks):
            P_ = np.array(pks, float)
            dl = np.median(P_, 0)[None, :] - P_
            sp = float(np.median(np.diff(np.median(P_, 0)))) if n > 1 else 0.0
            # sem vão que confirme as colunas, o deslocamento tem de ser de bloco: alguma
            # coluna inteira fora (mediana >= 3 mm), nenhum casamento ambíguo (> meia linha:
            # cabeçalho numa coluna e rodapé na outra trocam o índice das linhas) e >= 5 mm
            if sp > 0 and np.abs(dl).max() >= 5 * R and np.abs(dl).max() <= 0.5 * sp and \
                    np.median(np.abs(dl), 1).max() >= 3 * R:
                return cols
            if RELAYOUT[0] and sp > 0 and np.abs(dl).max() > 0.5 * sp and _rows_are_traces(B, R, cols, pks, lr):
                return cols
    return None


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
    cols = None
    if gaps and _equal_columns(gaps, lr[0], lr[1], R):
        cuts = [0] + [int(0.5 * (a + b)) for a, b in gaps] + [W]
        cols = [(cuts[i], cuts[i + 1]) for i in range(len(cuts) - 1)]
        pks = [_col_rows(B, R, a, b) for a, b in cols]
        n = len(pks[0])
        if n < 2 or any(len(p) != n for p in pks):
            cols = None
    if cols is None and EQUAL_SPLIT[0] and lr is not None:
        # sem vão em toda a altura (traço que emenda entre colunas, tira de ritmo embaixo):
        # hipóteses de colunas de mesma largura (4 e 2), corte no mínimo de tinta perto da
        # posição nominal; vale se todas as colunas tiverem o mesmo número de linhas
        cols = _equal_split(B, R, ya, yb, lr)
        if cols is None:
            return rect, pap, D, B, None
        pks = [_col_rows(B, R, a, b) for a, b in cols]
        n = len(pks[0])
    if cols is None:
        return rect, pap, D, B, None
    inner = cols
    P = np.array(pks, float)              # (colunas, linhas)
    ref = np.median(P, 0)
    dl = ref[None, :] - P
    sp = float(np.median(np.diff(ref)))
    if np.abs(dl).max() < 3 * R or sp < 8 * R:
        return rect, pap, D, B, None
    relay = False
    if np.abs(dl).max() > 0.5 * sp and RELAYOUT[0] and n >= 2 and _rows_are_traces(B, R, cols, pks, lr):
        # linhas novas: passo = maior espaçamento mediano entre as colunas, a partir da linha
        # mais alta; cada faixa vai inteira para a sua linha nova
        S = max(float(np.median(np.diff(p))) for p in P)
        y0 = float(P[:, 0].min())
        ref = y0 + S * np.arange(n)
        dl = ref[None, :] - P
        relay = True
        Hn = int(ref[-1] + 0.5 * S + 10 * R)
        if Hn > H:  # aumenta a página para caber
            pad = Hn - H
            rect = np.concatenate([rect, np.full((pad,) + rect.shape[1:], 255, rect.dtype)])
            pap = np.concatenate([pap, np.zeros((pad,) + pap.shape[1:], pap.dtype)])
            D = np.concatenate([D, np.zeros((pad, W), D.dtype)])
            B = np.concatenate([B, np.zeros((pad, W), B.dtype)])
            H = Hn
    elif np.abs(dl).max() > 1.5 * sp:       # deslocamento maior que uma linha: casamento duvidoso
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
    return rect2, pap2, D2, B2, dict(cols=cols, peaks=P.tolist(), ref=ref.tolist(), relayout=relay)
