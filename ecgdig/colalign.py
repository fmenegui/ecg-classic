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
STRUCT = [True]     # estrutura por trilhos quando não há vãos em branco entre as colunas
STRUCT_FIRST = [False]   # trilhos antes da hipótese de colunas de mesma largura


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


EQUAL_SPLIT = [True]


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
            if np.abs(dl).max() >= 3 * R:
                return cols
    return None


def _win_peaks(B, R, xa, xb, ya, yb, dist_mm=3.5, with_height=False):
    """Linhas de base numa janela estreita: picos da cobertura horizontal (traço dilatado 2 mm)."""
    k = int(2 * R) | 1
    Bd = cv2.dilate(B[ya:yb, xa:xb].astype(np.uint8), np.ones((k, 1), np.uint8))
    cov = cv2.GaussianBlur(Bd.mean(1).astype(np.float64).reshape(-1, 1), (0, 0), 0.6 * R).ravel()
    if cov.max() <= 0.05:
        return np.array([], int)
    pk, pr = find_peaks(cov, distance=int(dist_mm * R), height=max(0.3 * cov.max(), 0.25), prominence=0.12 * cov.max())
    if with_height:
        return pk + ya, pr["peak_heights"]
    return pk + ya


def col_struct(B, R, win_mm=20.0, step_mm=5.0, link_mm=1.5, min_len_mm=30.0):
    """Estrutura por colunas sem depender de vãos em branco.

    A página é varrida em janelas de 20 mm; em cada uma, os picos da cobertura horizontal são
    as linhas de base locais. Picos que se repetem na mesma altura de janela em janela formam
    um trilho (um trecho de derivação: altura, início, fim). Numa página comum os trilhos
    atravessam a largura toda. Quando cada coluna tem as suas alturas, os trilhos começam e
    terminam nas trocas de coluna: os pontos em que vários trilhos acabam e outros começam
    são as fronteiras, e cada coluna fica com as suas linhas.

    Devolve (cols, P) com cols = [(xa, xb), ...] e P (colunas, linhas) = alturas, ou None
    quando a página é comum (ou a estrutura não é clara)."""
    H, W = B.shape
    prof = B.mean(1)
    ys = np.nonzero(prof > 0.05 * prof.max())[0] if prof.max() > 0 else []
    xsum = B.sum(0)
    xs = np.nonzero(xsum > 0)[0]
    if len(ys) < 10 or len(xs) < 100 * R:
        return None
    # folga de 5 mm: a primeira/última linha de base não pode cair na borda da faixa (pico
    # na borda não é pico para o find_peaks)
    ya, yb = max(0, int(ys[0] - 5 * R)), min(H, int(ys[-1]) + 1 + int(5 * R))
    L, Rr = int(xs[0]), int(xs[-1]) + 1
    win, step, tol = int(win_mm * R), int(step_mm * R), link_mm * R
    x0s = list(range(L, max(L + 1, Rr - win + 1), step))
    if len(x0s) < 8:
        return None
    peaks = [_win_peaks(B, R, x, x + win, ya, yb) for x in x0s]
    # trilhos: encadeia picos de janelas consecutivas (tolera 1 janela sem o pico)
    tracks = []   # dict(ys=[...], j0, j1)
    open_ = []
    for j, pk in enumerate(peaks):
        used = set()
        for t in open_:
            if j - t["j1"] > 2:
                continue
            y_ = float(np.median(t["ys"][-4:]))
            cand = [(abs(float(v) - y_), i) for i, v in enumerate(pk) if i not in used and abs(float(v) - y_) <= tol]
            if cand:
                _, i = min(cand)
                used.add(i); t["ys"].append(float(pk[i])); t["j1"] = j
        open_ = [t for t in open_ if j - t["j1"] <= 2]
        for i, v in enumerate(pk):
            if i not in used:
                t = dict(ys=[float(v)], j0=j, j1=j)
                tracks.append(t); open_.append(t)
    nmin = max(3, int(round((min_len_mm - win_mm) / step_mm)) + 1)
    tr = [dict(y=float(np.median(t["ys"])), xa=x0s[t["j0"]], xb=x0s[t["j1"]] + win, n=t["j1"] - t["j0"] + 1)
          for t in tracks if t["j1"] - t["j0"] + 1 >= nmin]
    # só trilhos de pelo menos 45 mm definem a estrutura (rodapé, QR e platôs de onda S/T
    # dão trilhos curtos)
    tr = [t for t in tr if t["xb"] - t["xa"] >= 45 * R]
    if len(tr) < 4:
        return None
    # extensão do traçado pelos próprios trilhos (tinta solta nas margens não conta)
    L, Rr = min(t["xa"] for t in tr), max(t["xb"] for t in tr)
    span = Rr - L
    full = [t for t in tr if t["xb"] - t["xa"] >= 0.85 * span]
    part = [t for t in tr if t["xb"] - t["xa"] < 0.85 * span]
    if len(part) < 4 or len(part) < len(full):
        return None        # página comum: as linhas atravessam a largura
    # fronteiras: pontos do interior em que trilhos parciais acabam/começam, agrupados
    ev = sorted([(t["xb"] - win / 2, 0, t["y"]) for t in part if t["xb"] < Rr - 0.08 * span] +
                [(t["xa"] + win / 2, 1, t["y"]) for t in part if t["xa"] > L + 0.08 * span])
    if not ev:
        return None
    groups = [[ev[0]]]
    for e in ev[1:]:
        if e[0] - groups[-1][-1][0] <= 0.75 * win:
            groups[-1].append(e)
        else:
            groups.append([e])
    # uma fronteira tem trilho acabando E trilho começando
    def _same(g):
        """Trilhos que acabam na fronteira e recomeçam na MESMA altura (± 2 mm) do outro lado:
        a linha só foi interrompida (pulso, rótulo), não mudou de altura."""
        e0 = [y for _, k, y in g if k == 0]; e1 = [y for _, k, y in g if k == 1]
        return sum(1 for y in e0 if e1 and min(abs(y - z) for z in e1) <= 2 * R), min(len(e0), len(e1))
    cand = [(float(np.median([e[0] for e in g])), len(g), _same(g)) for g in groups
            if len(g) >= 2 and any(e[1] == 0 for e in g) and any(e[1] == 1 for e in g)]
    if not cand or len(cand) > 5:
        return None
    # as colunas de um ECG têm a mesma largura: fica o conjunto de fronteiras (1 ou 3: duas ou
    # quatro colunas) que divide a extensão em partes iguais (±20%) com mais trilhos mudando.
    # Um trilho que muda de altura no meio de uma derivação (ST, onda T larga) dá uma
    # "fronteira" fora do lugar, e cortar ali fatia o traço
    from itertools import combinations
    best = None
    for k in (1, 3):
        for sub in combinations(cand, k):
            ed = [L] + [z[0] for z in sub] + [Rr]
            wd = np.diff(ed)
            if wd.min() < 35 * R or np.abs(wd - wd.mean()).max() > 0.2 * wd.mean():
                continue
            key = (sum(z[1] for z in sub), k)
            if best is None or key > best[0]:
                best = (key, [z[0] for z in sub], max(z[1] for z in sub),
                        sum(z[2][0] for z in sub), sum(z[2][1] for z in sub))
    if best is None:
        return None
    cuts = best[1]
    # página comum com a linha interrompida na troca de coluna: metade ou mais das linhas
    # continua na mesma altura do outro lado -> não há alturas próprias por coluna
    if best[3] >= 0.5 * max(best[4], 1):
        return None
    # linhas que atravessam a página pesam contra: a fronteira mais forte precisa de pelo menos
    # o dobro de eventos (uma página comum com platôs e rodapé dá trilhos parciais soltos)
    if best[2] < max(4, 2 * len(full)):
        return None
    edges = [L] + cuts + [Rr]
    y_lo = max(0, int(min(t["y"] for t in tr) - 3 * R)); y_hi = min(H, int(max(t["y"] for t in tr) + 3 * R))

    def rows_of(ca, cb, frac, n=None):
        m = 0.5 * (1.0 - frac) * (cb - ca)
        pk, hh = _win_peaks(B, R, int(ca + m), int(cb - m), y_lo, y_hi, with_height=True)
        if n is not None and len(pk) > n:
            # platô longo de ST/onda S dá um pico a mais: ficam as n alturas de maior cobertura
            pk = np.sort(pk[np.argsort(-hh)[:n]])
        return pk.astype(float)

    # linhas de cada coluna medidas no miolo dela (longe das fronteiras, ainda imprecisas)
    rows_c = [rows_of(edges[i], edges[i + 1], 0.5) for i in range(len(edges) - 1)]
    cnt = [len(r_) for r_ in rows_c]
    n = int(np.bincount(cnt).argmax())
    if n < 2 or min(cnt) < n or sum(c == n for c in cnt) < 0.5 * len(cnt):
        return None
    # só estruturas que existem: 2 colunas de 3 a 9 linhas (6x2 e variantes com ritmo) ou
    # 4 colunas de 3 a 6 (3x4 e variantes). O resto é traçado confuso, não coluna
    K = len(cnt)
    if not ((K == 2 and 3 <= n <= 9) or (K == 4 and 3 <= n <= 6)):
        return None
    rows_c = [rows_of(edges[i], edges[i + 1], 0.5, n) for i in range(len(edges) - 1)]
    P = np.array(rows_c, float)
    if np.abs(P - np.median(P, 0)[None, :]).max() < 3 * R:
        return None        # mesmas alturas em todas as colunas: nada a realinhar
    cols_c = [(edges[i], edges[i + 1]) for i in range(len(edges) - 1)]
    # corte fino de cada fronteira: a coluna x em que a tinta troca das alturas de uma coluna
    # para as da outra (soma, nas linhas que mudam de altura, de tinta na faixa de cá menos a de lá)
    fine = []
    for c in range(len(cols_c) - 1):
        xc = int(cuts[c]); lo, hi = max(0, int(xc - 1.5 * win)), min(W, int(xc + 1.5 * win))
        sc = np.zeros(hi - lo)
        hb = int(1.0 * R)
        for i in range(n):
            y1, y2 = int(P[c, i]), int(P[c + 1, i])
            if abs(y1 - y2) < 2.5 * R:
                continue
            a_ = B[max(0, y1 - hb):y1 + hb + 1, lo:hi].any(0).astype(float)
            b_ = B[max(0, y2 - hb):y2 + hb + 1, lo:hi].any(0).astype(float)
            sc += a_ - b_
        if not sc.any():
            fine.append(xc); continue
        # corte k maximiza soma(sc[:k]) - soma(sc[k:])
        cs = np.cumsum(sc)
        obj = 2 * np.r_[0.0, cs] - cs[-1]
        fine.append(lo + int(np.argmax(obj)))
    bounds = [0] + fine + [W]
    cols = [(int(bounds[i]), int(bounds[i + 1])) for i in range(len(bounds) - 1)]
    # alturas finais com as fronteiras finas (miolo de 70%); se a contagem mudar, ficam as de antes
    e2 = [L] + fine + [Rr]
    if np.diff(e2).min() >= 35 * R:
        r2 = [rows_of(e2[i], e2[i + 1], 0.7, n) for i in range(len(e2) - 1)]
        if all(len(r_) == n for r_ in r2):
            P = np.array(r2, float)
    return cols, P


def _fits_plain_rows(B, R, P):
    """As linhas de sempre (projeção da página inteira) já ficam a menos de 40% do passo das
    linhas de todas as colunas: não é preciso realinhar."""
    from .extract import find_rows
    pk0 = np.asarray(find_rows(B, R)[0], float)
    if len(pk0) != P.shape[1] or len(pk0) < 2:
        return False
    sp0 = float(np.median(np.diff(pk0)))
    return bool(np.abs(P - pk0[None, :]).max() <= 0.4 * sp0)


def col_align(rect, pap, D, B, R):
    """Devolve (rect, pap, D, B, info) realinhados, ou as entradas e None.

    Três jeitos de achar as colunas, nesta ordem: vãos em branco em toda a altura ("vaos"),
    hipótese de 4 ou 2 colunas de mesma largura ("iguais") e trilhos de linha de base que
    começam e acabam nas trocas de coluna ("trilhos", `col_struct`). STRUCT_FIRST põe os
    trilhos antes das larguras iguais."""
    if not ENABLED[0]:
        return rect, pap, D, B, None
    H, W = B.shape
    prof = B.mean(1)
    ys = np.nonzero(prof > 0.05 * prof.max())[0]
    if len(ys) < 10:
        return rect, pap, D, B, None
    ya, yb = int(ys[0]), int(ys[-1]) + 1
    gaps, lr = _gaps(B, R, ya, yb)

    def _plain():
        """Vãos e larguras iguais: (cols, P, como) ou None."""
        cols, how = None, "vaos"
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
            how = "iguais"
            if cols is None:
                return None
            pks = [_col_rows(B, R, a, b) for a, b in cols]
        if cols is None:
            return None
        P = np.array(pks, float)              # (colunas, linhas)
        ref = np.median(P, 0)
        dl = ref[None, :] - P
        sp = float(np.median(np.diff(ref)))
        if np.abs(dl).max() < 3 * R or sp < 8 * R:
            return None
        if np.abs(dl).max() > 1.5 * sp:       # deslocamento maior que uma linha: casamento duvidoso
            return None
        if how == "iguais" and _fits_plain_rows(B, R, P):
            return None
        return cols, P, how

    def _tracks():
        """Sem vãos em branco (grade escura, rótulo ou QRS no vão): estrutura pelos trilhos."""
        if not STRUCT[0]:
            return None
        st = col_struct(B, R)
        if st is None:
            return None
        cols, P = st
        # diferença pequena entre as colunas (< 40% do passo) e as linhas de sempre já saem
        # certas: a extração dá conta sozinha (faixa larga + linha de base por trecho), e
        # realinhar só arriscaria fatiar um QRS grande na emenda das faixas
        if _fits_plain_rows(B, R, P):
            return None
        return cols, P, "trilhos"

    got = None
    for f in ((_tracks, _plain) if STRUCT_FIRST[0] else (_plain, _tracks)):
        got = f()
        if got is not None:
            break
    if got is None:
        return rect, pap, D, B, None
    cols, P, how = got
    n = P.shape[1]
    if how != "trilhos":
        ref = np.median(P, 0)
        H2 = H
    else:
        # alturas de referência: a primeira linha fica onde está (mediana) e cada intervalo é
        # o MAIOR entre as colunas, para que nenhuma faixa se sobreponha à vizinha depois de
        # deslocada (numa coluna "apertada" as linhas ficam a 4-10 mm; na outra, a 25-35 mm)
        gaps_ = np.max(np.diff(P, axis=1), axis=0)
        ref = np.concatenate([[float(np.median(P[:, 0]))], float(np.median(P[:, 0])) + np.cumsum(gaps_)])
        H2 = int(max(H, ref[-1] + np.max(H - P[:, -1]) + 1))
    dl = ref[None, :] - P
    rect2 = np.full((H2,) + rect.shape[1:], 255, rect.dtype); pap2 = np.zeros((H2, W), pap.dtype)
    D2 = np.zeros((H2, W), D.dtype); B2 = np.zeros((H2, W), B.dtype)
    for c, (xa, xb) in enumerate(cols):
        p = P[c]
        bounds = [0] + [int(0.5 * (p[i] + p[i + 1])) for i in range(n - 1)] + [H]
        for i in range(n):
            s0, s1 = bounds[i], bounds[i + 1]
            d = int(round(dl[c, i]))
            t0, t1 = s0 + d, s1 + d
            # recorta ao quadro
            o0, o1 = max(0, t0), min(H2, t1)
            if o1 <= o0:
                continue
            a0, a1 = o0 - d, o1 - d
            sl_t, sl_s = np.s_[o0:o1, xa:xb], np.s_[a0:a1, xa:xb]
            B2[sl_t] |= B[sl_s]
            D2[sl_t] = np.maximum(D2[sl_t], D[sl_s])
            rect2[sl_t] = np.minimum(rect2[sl_t], rect[sl_s])
            pap2[sl_t] = np.maximum(pap2[sl_t], pap[sl_s])
    return rect2, pap2, D2, B2, dict(cols=cols, peaks=P.tolist(), ref=ref.tolist(), how=how)
