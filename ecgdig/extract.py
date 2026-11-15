"""
Extração do sinal na imagem retificada (R px/mm, linhas grossas em múltiplos de 5R).

1. Mapa de tinta: normalização de iluminação por canal + remoção da grade por
   modelo de transmitância (tinta sobre grade = T_grade * T_tinta). T_grade vem de
   (a) mediana periódica na fase mod 5 mm e (b) resíduo por linha/coluna reta
   (percentil 80), o que cobre grades de qualquer cor e espaçamentos irregulares.
2. Layout: linhas de base pelos picos de cobertura horizontal filtrados por
   "parecer traçado" (componentes longas); pulsos de calibração (ganho e 0 mV).
3. Rastreamento: Viterbi sobre os segmentos de tinta de cada coluna (continuidade +
   prior da linha de base + penalidade de fragmentos + estado de falha).
4. Linha central sub-pixel por inversão morfológica: o traço impresso é a curva
   dilatada por uma pena elíptica (rh, rv). Borda de cima = erosão(y), de baixo =
   dilatação(y); abertura/fechamento recuperam y, exatos em picos/vales. O raio
   horizontal da pena é auto-calibrado (abertura == fechamento em trechos retos).
5. Oclusões: bordas que caem dentro da tinta de um traço vizinho plano viram NaN.
"""
from __future__ import annotations

import numpy as np
import cv2
from scipy.signal import find_peaks
from skimage.filters import apply_hysteresis_threshold, threshold_otsu


# ------------------------------------------------------------------- mapa de tinta

def _bg(ch, R):
    H, W = ch.shape
    f = 2.0 / R  # 2 px/mm
    s = cv2.resize(ch, (max(8, int(W * f)), max(8, int(H * f))), interpolation=cv2.INTER_AREA)
    s = cv2.dilate(s, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))  # ~4.5 mm
    s = cv2.medianBlur(s.astype(np.float32), 5)
    s = cv2.GaussianBlur(s, (0, 0), 3)
    return cv2.resize(s, (W, H), interpolation=cv2.INTER_LINEAR)


TEMPLATE_MM = [10.0]
TEMPLATE_AUTO = [True]
GLEV_RELATIVE = [False]


def _median5_cells(cells: np.ndarray) -> np.ndarray:
    """Mediana 5x5 entre células vizinhas, fase a fase: o mesmo que
    scipy.ndimage.median_filter(cells, size=(5, 5, 1, 1), mode="nearest"), mas com a mediana 5x5
    do OpenCV em cada uma das P*P imagens pequenas (ny, nx) — era ~2,5 s por página."""
    ny, nx, P1, P2 = cells.shape
    a = np.ascontiguousarray(cells.reshape(ny, nx, P1 * P2).transpose(2, 0, 1), dtype=np.float32)
    out = np.empty_like(a)
    for k in range(a.shape[0]):
        out[k] = cv2.medianBlur(a[k], 5)
    return out.transpose(1, 2, 0).reshape(ny, nx, P1, P2)


def ink_map(rect: np.ndarray, paper: np.ndarray, R: float, return_grid_level: bool = False, tmin: float | None = None):
    f = rect.astype(np.float32)
    n = [f[..., c] / (_bg(f[..., c], R) + 1e-3) for c in range(3)]
    t = np.clip(np.maximum(np.maximum(n[0], n[1]), n[2]), 0, 1.2)  # transmitância no canal mais claro
    # molde de 10 mm: cobre a periodicidade de 5 mm e também grades com linha extra a cada
    # 10 mm (as de 5 mm deixavam essa linha como "tinta" e criavam linhas falsas)
    H, W = t.shape

    def _tmpl(P):
        ny, nx = H // P, W // P
        blk = t[: ny * P, : nx * P].reshape(ny, P, nx, P)
        pm = paper[: ny * P, : nx * P].reshape(ny, P, nx, P).min(axis=(1, 3)) > 0
        tiles = blk.transpose(0, 2, 1, 3)[pm]  # (k, P, P)
        tm = np.median(tiles, axis=0) if len(tiles) > 10 else np.ones((P, P), np.float32)
        return ny, nx, blk, pm, tm

    P = int(round(TEMPLATE_MM[0] * R))
    ny, nx, blk, pm, tmpl = _tmpl(P)
    if TEMPLATE_AUTO[0] and TEMPLATE_MM[0] == 10.0:
        # o molde de 10 mm só vale quando a grade tem mesmo estrutura de 10 mm (linha extra);
        # senão o de 5 mm, com o dobro de células e vizinhança local de 25 mm, é melhor
        P5 = int(round(5.0 * R))
        r5 = _tmpl(P5)
        if P == 2 * P5:
            d = np.abs(tmpl - np.tile(r5[4], (2, 2)))
            if float(np.percentile(d, 95)) < 0.08:
                P = P5
                ny, nx, blk, pm, tmpl = r5
    # grade escura (scan P&B, fotocópia): as linhas grossas são tão pretas quanto o traço.
    # O molde precisa descer até elas (senão a linha da grade vira "tinta" contínua que
    # estende as linhas até a borda da folha e cria linhas falsas)
    dark_grid = float(tmpl.min()) < 0.15
    ink_map.tmpl_min = float(tmpl.min())
    ink_map.dark_grid = dark_grid
    if tmin is None:
        tmin = 0.05 if dark_grid else 0.3
        if TMIN_ADAPT[0] and not dark_grid:
            # grade escura mas não preta (foto escurecida): linhas a 0,15-0,3 de transmitância
            # passariam do piso 0,3 e virariam "tinta" na página toda
            tmin = float(np.clip(0.85 * tmpl.min(), 0.05, 0.3))
    tmpl = np.clip(tmpl, tmin, 1.0)
    reps = (int(np.ceil(H / P)), int(np.ceil(W / P)))
    T = np.tile(tmpl, reps)[:H, :W]
    # molde LOCAL: mediana por fase (mod 5 mm) numa vizinhança de 5x5 células (25 mm).
    # Em fotos a nitidez/escuridão da grade varia pela folha; o molde global deixa a
    # grade "vazar" como tinta onde ela é mais escura que a média.
    if ny >= 5 and nx >= 5:
        cells = blk.transpose(0, 2, 1, 3).copy()               # (ny, nx, P, P)
        cells[~pm] = tmpl                                        # fora do papel: global
        loc = _median5_cells(cells)
        loc = np.clip(loc, tmin, 1.0)
        TL = np.ones((reps[0] * P, reps[1] * P), np.float32)
        TL[: ny * P, : nx * P] = loc.transpose(0, 2, 1, 3).reshape(ny * P, nx * P)
        TL = TL[:H, :W]
        # bordas que sobram (fração de célula) ficam com o global
        TL[ny * P:, :] = T[ny * P:, :]; TL[:, nx * P:] = T[:, nx * P:]
        T = TL
    # resíduo por linha/coluna: linhas retas da grade que o molde periódico não explica
    # (grades irregulares, linha extra-escura a cada 25 mm...).
    t1 = t / T
    tt = np.where(paper > 0, np.minimum(t1, 1.0), np.nan)
    with np.errstate(all="ignore"):
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # percentil 80: o traço (mesmo em linha de base longa e plana) cobre bem menos
            # de 80% de uma linha da grade; uma linha da grade de verdade cobre ~100%
            lv = np.nan_to_num(np.nanpercentile(tt, 80, axis=0), nan=1.0)
            lh = np.nan_to_num(np.nanpercentile(tt, 80, axis=1), nan=1.0)
    lv = np.where(lv < 0.92, lv, 1.0); lh = np.where(lh < 0.92, lh, 1.0)
    T = T * np.clip(lh[:, None] * lv[None, :], 0.2, 1.0)
    if dark_grid:
        # grade escura (scan P&B): o molde precisa cobrir o pequeno erro de posição das
        # linhas (±1 px) ou as bordas delas vazam como tinta
        T = cv2.erode(T.astype(np.float32), np.ones((3, 3), np.uint8))
    D = 1.0 - np.clip(t / T, 0, 1)
    pe = cv2.erode(paper.astype(np.uint8), np.ones((int(2 * R), int(2 * R)), np.uint8))
    D[pe == 0] = 0
    D = D.astype(np.float32)
    if return_grid_level:
        # nível residual da grade: tinta aparente nos pixels de linha da grade (o traço
        # ocupa uma fração mínima deles, então o percentil 95 mede o vazamento da grade)
        # "tinta forte" relativa: traço borrado/claro fica em 0,2-0,4 e, se não for excluído,
        # o halo dele vira "vazamento da grade" e o limiar sobe até apagar o próprio traço
        vv = D[D > 0.08]
        s_th = 0.55 if not GLEV_RELATIVE[0] or vv.size < 100 else min(0.55, max(0.15, 0.5 * float(np.percentile(vv, 90))))
        kd = 7 if not GLEV_RELATIVE[0] else max(7, int(1.0 * R) | 1)
        strong = cv2.dilate((D > s_th).astype(np.uint8), np.ones((kd, kd), np.uint8)) > 0
        gp = (T < 0.9) & (pe > 0) & ~strong
        glev = float(np.percentile(D[gp], 95)) if gp.sum() > 1000 else 0.0
        return D, glev
    return D


def binarize(D: np.ndarray, R: float = 10.0, grid_level: float = 0.0, scale: float = 1.0) -> np.ndarray:
    # Otsu só sobre estruturas finas: tarjas, carimbos e blocos de texto preto (> 2.5 mm
    # de espessura) puxariam o limiar para cima e apagariam traçados claros/finos
    kb = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (int(2.5 * R) | 1,) * 2)
    blob = cv2.morphologyEx((D > 0.25).astype(np.uint8), cv2.MORPH_OPEN, kb)
    blob = cv2.dilate(blob, np.ones((int(0.8 * R) | 1,) * 2, np.uint8)) > 0
    v = D[(D > 0.08) & ~blob]
    if v.size < 100:
        B = D > 0.4
    else:
        floor = 0.2
        if OTSU_FLOOR_REL[0]:
            # página toda de baixo contraste (gama alto, toner fraco): o traço mais escuro não
            # passa de ~0,2; o piso fixo de 0,2 apagaria o traço inteiro. O piso passa a ser
            # relativo ao contraste do próprio traço (p99), sempre acima do vazamento da grade.
            floor = min(0.2, max(grid_level + 0.05, 0.35 * float(np.percentile(v, 99))))
        th = float(np.clip(threshold_otsu(v), floor, 0.6))
        if scale != 1.0:
            # traço muito claro (opacidade baixa, toner fraco): o Otsu é decidido por texto e
            # QR pretos; tenta limiares menores, sempre acima do vazamento da grade
            th = max(grid_level + 0.04, 0.05, th * scale)
        # limiar baixo da histerese acima do vazamento da grade (linhas da grade tocam o
        # traço e seriam "puxadas" junto pela conectividade)
        lo = max(0.55 * th, min(grid_level + 0.08, 0.85 * th))
        B = apply_hysteresis_threshold(D, lo, max(th, lo + 0.05))
    # linhas retas da grade que sobraram (grade preta grossa, molde 1 px fora): linhas/colunas
    # de pixel quase todas "tinta" ao longo da página, várias, em passo regular. Um traço
    # não cobre 70% de uma linha de pixel na página inteira.
    if GRID_LINE_CLEAN[0]:
        B = _clean_grid_lines(B, R)
    # remove sal-e-pimenta: componentes minúsculas (< ~0.4 mm²) não são traçado
    n, lab, st, _ = cv2.connectedComponentsWithStats(B.astype(np.uint8), connectivity=8)
    small = st[:, cv2.CC_STAT_AREA] < max(4, int(0.4 * R * R))
    small[0] = False
    if small.any():
        B = B & ~small[lab]
    return B


GRID_LINE_CLEAN = [True]
OTSU_FLOOR_REL = [True]
TMIN_ADAPT = [True]


def _clean_grid_lines(B, R, frac=0.7):
    H, W = B.shape
    ys, xs = np.nonzero(B.any(1))[0], np.nonzero(B.any(0))[0]
    if len(ys) < 10 or len(xs) < 10:
        return B
    out = B.copy()
    for axis in (1, 0):
        span = (xs[-1] - xs[0] + 1) if axis == 1 else (ys[-1] - ys[0] + 1)
        cov = B.sum(axis) / max(span, 1)
        lines = np.nonzero(cov > frac)[0]
        if len(lines) < 3:
            continue
        # agrupa pixels vizinhos numa linha só e exige passo regular (múltiplo de ~5 mm)
        grp = np.split(lines, np.nonzero(np.diff(lines) > 1)[0] + 1)
        cen = np.array([g.mean() for g in grp])
        if len(cen) < 3:
            continue
        d = np.diff(cen) / R
        reg = np.mean(np.abs(d / 5.0 - np.round(d / 5.0)) < 0.15)
        if reg < 0.7:
            continue
        # grade de verdade: a maioria dos vizinhos a 5 mm (ou 1 mm). Linhas de base planas de
        # traçado longo também cobrem a página e ficam em múltiplos de 5 mm, mas a 25-40 mm
        near = np.mean((np.abs(d - 5.0) < 0.75) | (np.abs(d - 1.0) < 0.3))
        if len(cen) < 5 or near < 0.6:
            continue
        for g in grp:
            lo, hi = max(0, g[0] - 1), g[-1] + 2
            if axis == 1:
                out[lo:hi, :] = False
            else:
                out[:, lo:hi] = False
    return out


# ------------------------------------------------------------------------ layout

def find_rows(B: np.ndarray, R: float):
    H, W = B.shape
    k = int(2 * R) | 1
    Bd = cv2.dilate(B.astype(np.uint8), np.ones((k, 1), np.uint8))
    cov = Bd.mean(1).astype(np.float64)
    cov = cv2.GaussianBlur(cov.reshape(-1, 1), (0, 0), R).ravel()
    pk, pr = find_peaks(cov, distance=int(8 * R), height=0.3 * cov.max(), prominence=0.1 * cov.max())
    return pk, cov


def find_pulses(B: np.ndarray, R: float):
    """Todos os pulsos de calibração (retângulos) da página, sem saber onde estão as linhas.
    Bordas verticais de 4..25 mm (abertura vertical), aos pares separados por 1,5..12 mm, com
    topos e pés alinhados e platô de tinta entre elas. Devolve lista de dict(x_start, x_end,
    top, foot)."""
    H, W = B.shape
    Bu = B.astype(np.uint8)
    kc = max(3, int(0.8 * R) | 1)
    Bc = cv2.morphologyEx(Bu, cv2.MORPH_CLOSE, np.ones((kc, 1), np.uint8))
    V = cv2.morphologyEx(Bc, cv2.MORPH_OPEN, np.ones((int(4 * R), 1), np.uint8))
    n, lab, st, _ = cv2.connectedComponentsWithStats(V, connectivity=8)
    edges = []
    for i in range(1, n):
        x, y, w, h, a = st[i]
        if h > 26 * R or w > 2.5 * R:
            continue
        edges.append((x + 0.5 * w, float(y), float(y + h - 1), x, w))
    edges.sort()
    out = []
    for i, (xc1, t1, b1, x1, w1) in enumerate(edges):
        for xc2, t2, b2, x2, w2 in edges[i + 1:]:
            sep = (xc2 - xc1) / R
            if sep > 12:
                break
            if sep < 1.5 or abs(t1 - t2) > 1.0 * R:
                continue
            h = min(b1, b2) - 0.5 * (t1 + t2)
            if not (4 * R <= h <= 25 * R) or abs(b1 - b2) > 10 * R:
                continue
            ty = int(round(0.5 * (t1 + t2)))
            band = B[max(0, ty - int(0.5 * R)):ty + int(0.5 * R) + 1, int(x1 + w1):int(x2)]
            if band.size == 0 or band.any(0).mean() < 0.8:
                continue
            # pé: a base do traço logo à esquerda da subida (ou o fim da borda esquerda)
            out.append(dict(x_start=int(x1), x_end=int(x2 + w2 - 1), top=float(0.5 * (t1 + t2)),
                            foot=float(b1), x_fall=float(xc2)))
            break
    return out


def pulse_rows(B: np.ndarray, R: float):
    """Linhas de base das linhas de traçado pelos pulsos de calibração: o maior grupo de
    pulsos de mesma altura (±12%) na mesma coluna (±5 mm), um por linha. None se < 3."""
    P = find_pulses(B, R)
    H, W = B.shape
    # pulso de verdade emenda no traço: logo depois dele há tinta perto do pé por boa parte
    # dos 25 mm seguintes (bloco de QR code, caixa de texto e cabeçalho não)
    def _cont(p):
        a, b = max(0, int(p["foot"] - 4 * R)), min(H, int(p["foot"] + 4 * R))
        x0, x1 = int(p["x_end"] + 1), min(W, int(p["x_end"] + 25 * R))
        if x1 - x0 < 5 * R:
            return False
        return B[a:b, x0:x1].any(0).mean() >= 0.5
    P = [p for p in P if _cont(p)]
    best = None
    for p in P:
        h = p["foot"] - p["top"]
        grp = [q for q in P if abs(q["x_start"] - p["x_start"]) <= 5 * R and
               abs((q["foot"] - q["top"]) - h) <= 0.12 * h]
        feet = sorted(q["foot"] for q in grp)
        uniq = [f for i, f in enumerate(feet) if i == 0 or f - feet[i - 1] > 3 * R]
        if len(uniq) >= 2 and np.min(np.diff(uniq)) < 8 * R:
            continue  # linhas de ECG ficam a >= ~10 mm umas das outras
        # pontas fora do passo do grupo (retângulo de QR code ou carimbo bem abaixo/acima do
        # bloco): saem
        while len(uniq) >= 4:
            dd = np.diff(uniq); md = float(np.median(dd))
            if dd[-1] > 1.5 * md:
                uniq = uniq[:-1]
            elif dd[0] > 1.5 * md:
                uniq = uniq[1:]
            else:
                break
        key = (len(uniq), -p["x_start"])
        if best is None or key > best[0]:
            best = (key, uniq, grp)
    if best is None or len(best[1]) < 3:
        return None, P
    return np.array(best[1]), best[2]


def pulse_column(B: np.ndarray, R: float, P=None, all_groups: bool = False):
    """Coluna de pulsos de calibração que não emendam no traço (impressos à parte, na linha
    nominal, com o traço deslocado pelo nível DC da derivação). Exige >= 3 pulsos com o
    mesmo x (±2 mm), a mesma altura (±8%), a mesma largura (±1 mm) e passo vertical regular
    (±15%): um bloco de texto ou um QR code não forma isso. Devolve dict(x_start, x_end,
    height, feet) ou None."""
    P = find_pulses(B, R) if P is None else P
    best, groups = None, []
    for p in P:
        h = p["foot"] - p["top"]; w = p["x_end"] - p["x_start"]
        grp = [q for q in P if abs(q["x_start"] - p["x_start"]) <= 2 * R
               and abs((q["foot"] - q["top"]) - h) <= 0.08 * h
               and abs((q["x_end"] - q["x_start"]) - w) <= 1 * R]
        feet = sorted({round(q["foot"]) for q in grp})
        feet = [f for i, f in enumerate(feet) if i == 0 or f - feet[i - 1] > 3 * R]
        if len(feet) < 3:
            continue
        d = np.diff(feet)
        if d.min() < 8 * R or (d.max() - d.min()) > 0.15 * np.median(d):
            continue
        cand = dict(x_start=float(np.median([q["x_start"] for q in grp])),
                    x_end=float(np.median([q["x_end"] for q in grp])),
                    height=float(np.median([q["foot"] - q["top"] for q in grp])), feet=feet)
        if all_groups:
            if not any(abs(g["x_start"] - cand["x_start"]) <= 3 * R for g in groups):
                groups.append(cand)
            continue
        if best is None or len(feet) > len(best["feet"]):
            best = cand
    return sorted(groups, key=lambda g: g["x_start"]) if all_groups else best


def _runs(mask_1d):
    d = np.diff(np.concatenate([[0], mask_1d.astype(np.int8), [0]]))
    return np.nonzero(d == 1)[0], np.nonzero(d == -1)[0] - 1


def row_extent(B, y0, half, R, max_gap_mm=3.0):
    """Maior trecho horizontal contínuo de tinta na faixa da linha."""
    H, W = B.shape
    a, b = max(0, int(y0 - half)), min(H, int(y0 + half))
    has = B[a:b].any(0)
    # fecha lacunas pequenas
    k = int(max_gap_mm * R) | 1
    hc = cv2.morphologyEx(has.astype(np.uint8).reshape(1, -1), cv2.MORPH_CLOSE, np.ones((1, k), np.uint8)).ravel()
    s, e = _runs(hc > 0)
    if len(s) == 0:
        return None
    i = int(np.argmax(e - s))
    lo, hi = int(s[i]), int(e[i])
    if ROW_CHAIN[0]:
        # colunas separadas por um vão (pulso de calibração no começo de cada coluna, margem
        # entre colunas): emenda trechos longos na mesma faixa a até 12 mm de distância
        # (o traço da outra coluna pode vir partido em pedaços; basta que a sequência de
        # pedaços seja densa e chegue a um trecho longo)
        gap, minlen = 12 * R, 15 * R
        for direction in (-1, 1):
            j = i + direction; acc = []
            edge = lo if direction < 0 else hi
            while 0 <= j < len(s):
                g_ = (edge - e[j]) if direction < 0 else (s[j] - edge)
                if g_ > gap:
                    break
                acc.append(j)
                edge = s[j] if direction < 0 else e[j]
                if (e[j] - s[j]) >= minlen:  # chegou a um trecho longo: aceita a cadeia
                    if direction < 0:
                        lo = int(s[j])
                    else:
                        hi = int(e[j])
                    acc = []
                j += direction
    return lo, hi


ROW_CHAIN = [True]


def detect_pulse(B, y0, half, xa, R, up=None, close_mm=0.8):
    """Pulso de calibração (retângulo) no início da linha.
    Exige: duas bordas verticais altas (4..25 mm) separadas por 1.5..12 mm, de alturas,
    topos e pés alinhados (< 1 mm), topo horizontal coberto de tinta entre elas e pé
    perto da linha de base. Isso descarta QRS (ponta, não platô) e texto.
    Retorna dict(top, foot, x_start, x_end) ou None."""
    H, W = B.shape
    up = half if up is None else up
    a, b = max(0, int(y0 - up)), min(H, int(y0 + 0.6 * half))
    x0s = max(0, int(xa - 10 * R))
    # 30 mm: em fotos a extensão da linha pode começar na borda da folha/grade, antes do pulso
    x1 = min(W, int(xa + 30 * R))
    sub = B[a:b, x0s:x1].astype(np.uint8)
    kv = max(3, int(close_mm * R) | 1)  # fecha quebras verticais (resolução baixa / JPEG)
    sub = cv2.morphologyEx(sub, cv2.MORPH_CLOSE, np.ones((kv, 1), np.uint8)) > 0
    n = sub.shape[1]
    tops = np.full(n, np.nan); bots = np.full(n, np.nan)
    for j in range(n):
        s_, e_ = _runs(sub[:, j])
        if len(s_):
            i = int(np.argmax(e_ - s_))
            if e_[i] - s_[i] >= 4 * R:
                tops[j], bots[j] = s_[i] + a, e_[i] + a
    cs, ce = _runs(~np.isnan(tops))
    for i in range(len(cs)):
        for k in range(i + 1, len(cs)):
            sep = (cs[k] - ce[i]) / R
            if sep > 12:
                break
            if sep < 1.5:
                continue
            j1 = np.arange(cs[i], ce[i] + 1); j2 = np.arange(cs[k], ce[k] + 1)
            t1, b1 = np.median(tops[j1]), np.median(bots[j1])
            t2, b2 = np.median(tops[j2]), np.median(bots[j2])
            # a borda de SUBIDA vai de 0 a 1 mV; a de descida termina no valor do ECG em
            # t = 0 (qualquer). Pé = borda esquerda; topo = platô (média das duas).
            # altura pela borda mais curta (a mais longa pode carregar sombra/vinco ou o
            # valor do ECG em t = 0); a linha de base final vem do platô + ganho comum
            h = min(b1, b2) - 0.5 * (t1 + t2)
            if not (4 * R <= h <= 25 * R):
                continue
            if abs(t1 - t2) > 1.0 * R and abs(b1 - b2) < 1.0 * R and min(b1 - t1, b2 - t2) >= 2 * R:
                # pulso em DEGRAU (ganho dividido, ex. 10/5 mm/mV): platô alto à esquerda,
                # degrau vertical, platô baixo à direita. Verifica os dois platôs.
                foot = min(b1, b2)
                if abs(foot - y0) > 4 * R:
                    continue
                xa_, xb_ = x0s + ce[i] + 1, x0s + cs[k]
                if xb_ - xa_ < 2 * R:
                    continue
                hb = int(0.5 * R)
                cov1 = B[max(0, int(t1) - hb):int(t1) + hb + 1, xa_:xb_].any(0)
                cov2 = B[max(0, int(t2) - hb):int(t2) + hb + 1, xa_:xb_].any(0)
                # o platô 1 cobre um prefixo e o 2 um sufixo, juntos cobrem quase tudo
                if (cov1 | cov2).mean() < 0.8 or cov1[: len(cov1) // 4].mean() < 0.6 or \
                        cov2[-(len(cov2) // 4):].mean() < 0.6:
                    continue
                return dict(top=float(t1), top2=float(t2), foot=float(foot), x_start=int(x0s + cs[i]),
                            x_end=int(x0s + ce[k]), x_fall=float(x0s + 0.5 * (cs[k] + ce[k])), step=True)
            # a borda de descida termina no valor do ECG em t = 0 e pode emendar na marca
            # separadora: o pé dela fica bem abaixo do da subida (até ~1 mV)
            if abs(t1 - t2) > 1.0 * R or abs(b1 - b2) > 10.0 * R:
                continue
            top = 0.5 * (t1 + t2); foot = min(b1, b2)
            if abs(foot - y0) > 4 * R:
                continue
            # platô: tinta numa faixa estreita em torno do topo, entre as bordas
            ty = int(round(top))
            band = B[max(0, ty - int(0.5 * R)):ty + int(0.5 * R) + 1, x0s + ce[i] + 1:x0s + cs[k]]
            if band.size == 0 or band.any(0).mean() < 0.8:
                continue
            return dict(top=float(top), foot=float(foot), x_start=int(x0s + cs[i]), x_end=int(x0s + ce[k]),
                        x_fall=float(x0s + 0.5 * (cs[k] + ce[k])))
    # meio pulso (borda esquerda cortada pela margem da imagem): platô >= 1.5 mm
    # terminando numa borda de descida alta, pé perto da linha de base
    for i in range(len(cs)):
        j1 = np.arange(cs[i], ce[i] + 1)
        t1, b1 = np.median(tops[j1]), np.median(bots[j1])
        h = b1 - t1
        if not (4 * R <= h <= 25 * R) or abs(b1 - y0) > 4 * R:
            continue
        ty = int(round(t1))
        xl = x0s + cs[i] - int(1.5 * R)
        if xl < 0:
            continue
        band = B[max(0, ty - int(0.5 * R)):ty + int(0.5 * R) + 1, xl:x0s + cs[i]]
        left = B[max(0, ty + int(1.0 * R)):int(b1 - 1.0 * R), max(0, xl - int(10 * R)):xl]
        if band.size and band.any(0).mean() >= 0.8 and (left.size == 0 or left.any(1).mean() < 0.5):
            return dict(top=float(t1), foot=float(b1), x_start=int(x0s + cs[i]), x_end=int(x0s + ce[i]),
                        x_fall=float(x0s + 0.5 * (cs[i] + ce[i])), half=True)
    return None


# ----------------------------------------------------------------- rastreamento

def column_runs(B, D, ya, yb, xa, xb):
    """Lista por coluna de segmentos (top, bot, centróide, massa).

    Todas as colunas de uma vez (antes era um laço por coluna em Python). Os valores saem
    como int/float do Python: o Viterbi faz milhões de contas escalares com eles, e com
    escalares do NumPy cada uma custa várias vezes mais."""
    sub = B[ya:yb, xa:xb]
    h, w = sub.shape
    if w == 0:
        return []
    if h == 0:
        return [[] for _ in range(w)]
    pad = np.zeros((w, h + 2), np.int8)
    pad[:, 1:-1] = sub.T
    d = np.diff(pad, axis=1)
    cs, ss = np.nonzero(d == 1)          # coluna, primeira linha do segmento (ordem: coluna, linha)
    _, ee = np.nonzero(d == -1)
    ee = ee - 1                           # última linha do segmento
    if len(ss) == 0:
        return [[] for _ in range(w)]
    # funde fragmentos da mesma coluna separados por <= 3 px (halo de antialiasing / JPEG)
    first = np.r_[True, (cs[1:] != cs[:-1]) | ((ss[1:] - ee[:-1] - 1) > 3)]
    last = np.r_[first[1:], True]
    c_, a_, b_ = cs[first], ss[first], ee[last]
    # massa e centróide de cada segmento fundido (inclui os pixels do vão), em float64
    Dt = np.zeros(w * h + 1, np.float64)
    Dt[:-1] = D[ya:yb, xa:xb].T.ravel()
    idx = np.empty(2 * len(a_), np.int64)
    idx[0::2] = c_ * h + a_
    idx[1::2] = c_ * h + b_ + 1
    m = np.add.reduceat(Dt, idx)[0::2] + 1e-9
    yy = np.arange(ya, ya + h, dtype=np.float64)   # yb pode passar do fim da imagem
    Dy = Dt.copy()
    Dy[:-1] *= np.tile(yy, w)
    cen = np.add.reduceat(Dy, idx)[0::2] / m
    top = (a_ + ya).tolist(); bot = (b_ + ya).tolist(); cen = cen.tolist(); m = m.tolist()
    cnt = np.bincount(c_, minlength=w).tolist()
    out = []
    i = 0
    for n in cnt:
        if n == 0:
            out.append([])
            continue
        col = [(top[k], bot[k], cen[k], m[k]) for k in range(i, i + n)]
        i += n
        # descarta fragmentos sem massa quando há alternativa
        if n > 1:
            big = [c for c in col if c[3] >= 0.6]
            col = big or col
        out.append(col)
    return out


def _bridge_table(D, xa, n, R, floor=0.0):
    """Soma acumulada (em y) da tinta máxima em 3 colunas (x-2..x): evidência de um traço
    fraco ligando dois segmentos na vertical (subida/descida rápida do QRS)."""
    from scipy.ndimage import maximum_filter1d
    H, W = D.shape
    a, b = max(0, xa - 2), min(W, xa + n)
    Dm = maximum_filter1d(np.clip(D[:, a:b].astype(np.float32) - floor, 0, None), 3, axis=1, origin=1)
    Dm = Dm[:, xa - a:xa - a + n] if xa - a >= 0 else Dm
    from scipy.ndimage import uniform_filter1d
    return uniform_filter1d(Dm, 3, axis=0)  # média vertical de 3 px: buraco de 1 px no traço fraco não conta


def viterbi_track(runs, base, R, w_base=0.004, c_gap=1.0, others=(), lw=None, w_edge=0.0,
                  D=None, xa=0, bridge_ref=None, rigid=False, bridge_floor=0.0):
    """Escolhe um segmento por coluna (ou falha) minimizando custo de continuidade.

    Com D: um salto vertical entre segmentos que a tinta (mesmo fraca, abaixo do limiar da
    binarização) liga custa pouco. A pena é contínua: a subida e a descida do QRS, finas e
    claras no scan/foto, se partem no binário, e sem isso o rastreio pulava o pico do R ou o
    vale do S (o erro dominante nos picos)."""
    INF = 1e18
    n = len(runs)
    if n == 0:
        return []
    base, R = float(base), float(R)
    others = [float(o) for o in others]
    cs_br = None
    if D is not None and bridge_ref is not None and bridge_ref > 0 and n > 0:
        cs_br = _bridge_table(D, xa, n, R, bridge_floor)
        bridge_ref = float(bridge_ref - bridge_floor)
        Hb = cs_br.shape[0]
    states = []  # por coluna: lista de (top, bot, center)
    backs = []
    flat_h = float(3.0 * (lw if lw else 0.3 * R))
    w_base, c_gap = float(w_base), float(c_gap)
    mx = [max(m for (_, _, _, m) in col) for col in runs if col]
    m_ref = float(np.median(mx)) if mx else 1.0

    def emis(t, b, c, m):
        # distância do intervalo à linha de base (0 se o segmento a cruza)
        dm = max(0.0, t - base, base - b) / R
        # prior saturado: ondas de 2-4 mV são fisiológicas; a penalidade só precisa
        # desempatar entre traços próximos, não empurrar um R alto de volta à base
        e = min(w_base * dm * dm, 0.1)
        # trecho plano em cima da linha de base de OUTRA derivação = traço vizinho.
        # (um R alto cruza a linha vizinha na vertical: segmento alto, sem penalidade)
        if others and (b - t) <= flat_h and dm > 6:
            for ob in others:
                if abs(c - ob) < 4 * R:
                    e += 0.5
                    break
        if (b - t) / R > 30:
            e += 5.0
        if m < 0.35 * m_ref:  # fragmento fraco (halo/ruído)
            e += 0.8
        return e

    _we = float(w_edge)
    _rigid = rigid

    def trans(pt, pb, pc, t, b, c, i=None):
        g = max(0.0, t - pb, pt - b) / R
        g0 = g
        if g > 0.2 and cs_br is not None and i is not None and i < cs_br.shape[1]:
            # intervalo vertical vazio entre os segmentos
            y0, y1 = (int(pb) + 1, int(t)) if t > pb else (int(b) + 1, int(pt))
            y0, y1 = max(0, y0), min(Hb, y1)
            if y1 > y0:
                # quantil baixo ao longo do vão: um traço fraco é contínuo; mancha/pontilhado
                # de scan tem buracos (a média sozinha atravessava manchas escuras)
                ev = float(cs_br[y0:y1, i].min()) / bridge_ref
                if ev > 0.35:
                    g *= max(0.05, 1.0 - (ev - 0.35) / 0.35)
        # continuidade das BORDAS: num traço desenhado por pena, de uma coluna para a
        # outra pelo menos uma das bordas (topo/base) anda pouco; letra colada ao traço
        # (rótulo atravessado por um R) fica no meio e quebra as duas
        # (traço íngreme: o segmento inteiro translada de uma coluna para a outra, as duas
        # bordas andam juntas; só a letra colada muda topo e base de forma desigual)
        ed = min(abs(t - pt), abs(b - pb))
        if _rigid and g0 == 0.0:  # só com sobreposição: o mesmo traço continuando
            ed = min(ed, abs((t - pt) - (b - pb)))
        return 2.0 * g + 0.5 * g * g + 0.02 * abs(c - pc) / R + _we * ed / R

    # contas em escalares do Python (float/int): mesmos valores IEEE, sem o custo dos
    # escalares do NumPy em cada soma
    prev_states = [(base, base, base)]
    prev_cost = [0.0]
    last_cost = prev_cost
    for i in range(n):
        col = runs[i]
        cs = []
        bk = []
        for (t, b, c, m) in col:
            best, arg = INF, -1
            k = 0
            for (pt, pb, pc) in prev_states:
                v = prev_cost[k] + trans(pt, pb, pc, t, b, c, i)
                if v < best:
                    best, arg = v, k
                k += 1
            cs.append(best + emis(t, b, c, m))
            bk.append(arg)
        # estado de falha (último índice)
        pmin = min(prev_cost)
        kmin = prev_cost.index(pmin)        # primeiro mínimo, como o np.argmin
        cs.append(pmin + c_gap)
        bk.append(kmin)
        new_states = [(t, b, c) for (t, b, c, m) in col]
        new_states.append(prev_states[kmin] if kmin < len(prev_states) else prev_states[-1])
        states.append(new_states)
        backs.append(bk)
        prev_states, prev_cost = new_states, cs
        last_cost = cs
    # backtrack
    sel = [None] * n
    k = last_cost.index(min(last_cost))
    for i in range(n - 1, -1, -1):
        is_gap = k == len(states[i]) - 1
        sel[i] = None if is_gap else k
        k = backs[i][k]
    return sel


def selected_envelopes(D, runs, sel, xa):
    """Bordas sub-pixel (nível 50% da tinta cheia) do segmento escolhido em cada coluna."""
    n = len(runs)
    tb = np.full((n, 2), np.nan)
    pk = np.full(n, np.nan)
    for i, k in enumerate(sel):
        if k is None:
            continue
        t, b = runs[i][k][0], runs[i][k][1]
        col = D[:, xa + i]
        tb[i] = t, b
        pk[i] = col[t:b + 1].max()
    return tb, pk


def _edge_up(col, t, b, lev):
    k = t
    while k - 1 >= 0 and col[k - 1] >= lev and k > t - 3:
        k -= 1
    while k < b and col[k] < lev:
        k += 1
    if k == 0:
        return k - 0.5
    d0, d1 = col[k - 1], col[k]
    return (k - 1) + (lev - d0) / (d1 - d0) if d1 > d0 and d1 >= lev else k - 0.5


def _edge_dn(col, t, b, lev):
    k = b
    H = len(col)
    while k + 1 < H and col[k + 1] >= lev and k < b + 3:
        k += 1
    while k > t and col[k] < lev:
        k -= 1
    if k + 1 >= H:
        return k + 0.5
    d0, d1 = col[k], col[k + 1]
    return k + (d0 - lev) / (d0 - d1) if d0 > d1 and d0 >= lev else k + 0.5


def envelope_edges(D, runs, sel, xa, dfull):
    lev = 0.5 * dfull
    n = len(runs)
    top = np.full(n, np.nan); bot = np.full(n, np.nan)
    for i, k in enumerate(sel):
        if k is None:
            continue
        t, b = runs[i][k][0], runs[i][k][1]
        col = D[:, xa + i]
        top[i] = _edge_up(col, t, b, lev)
        bot[i] = _edge_dn(col, t, b, lev)
    return top, bot


def _shift_pad(a, d):
    if d == 0:
        return a
    p = np.pad(a, (abs(d), abs(d)), mode="edge")
    return p[abs(d) - d: abs(d) - d + len(a)]


def _gamma_phi(top, bot, r, rh, U):
    n = len(top)
    xs = np.arange(n, dtype=np.float64)
    okt, okb = ~np.isnan(top), ~np.isnan(bot)
    ok = okt | okb
    if okt.sum() < 2 or okb.sum() < 2:
        okt = okb = okt & okb
    t = np.interp(xs, xs[okt], top[okt]); b = np.interp(xs, xs[okb], bot[okb])
    xf = np.arange(0, n - 1 + 1e-9, 1.0 / U)
    tf = np.interp(xf, xs, t); bf = np.interp(xf, xs, b)
    rU = int(np.floor(rh * U))
    gam = np.full_like(tf, -np.inf); phi = np.full_like(bf, np.inf)
    for d in range(-rU, rU + 1):
        h = r * np.sqrt(max(1.0 - (d / U / rh) ** 2, 0.0))
        gam = np.maximum(gam, _shift_pad(tf, d) + h)
        phi = np.minimum(phi, _shift_pad(bf, d) - h)
    return xf, tf, bf, gam, phi, ok


def calibrate_pen(segments, r, U=4):
    """Auto-calibração da meia-largura horizontal da pena (pena elíptica: impressoras
    térmicas/renderizadores raramente são isotrópicos). No modelo correto, abertura (do
    topo) e fechamento (da base) coincidem nos traços inclinados; escolhe o rh que
    minimiza a discordância mediana, numa faixa plausível (0.7..1.1 x rv)."""
    best, best_rh = np.inf, r
    for f in np.arange(0.7, 1.11, 0.05):  # faixa plausível de anisotropia da pena
        rh = r * f
        errs = []
        for top, bot in segments:
            ok = ~np.isnan(top) & ~np.isnan(bot)
            if ok.sum() < 20:
                continue
            xf, tf, bf, gam, phi, _ = _gamma_phi(top, bot, r, rh, U)
            # colunas de traço inclinado (borda de cima e de baixo bem separadas)
            straight = (bf - tf) > 6 * r
            if straight.sum() > 5:
                errs.append(np.abs(gam - phi)[straight])
        if not errs:
            return r
        e = float(np.median(np.concatenate(errs)))
        if e < best:
            best, best_rh = e, rh
    return best_rh


def centerline_from_envelopes(top, bot, r, U=4, rh=None):
    """Inverte o modelo 'traço = curva (+) pena elíptica (rh, r)':
       topo = erosão(y), base = dilatação(y). Abertura do topo (<= y) é exata em
       picos para cima; fechamento da base (>= y) é exato em vales. Escolhe por curvatura."""
    n = len(top)
    xs = np.arange(n, dtype=np.float64)
    ok = ~np.isnan(top) | ~np.isnan(bot)
    if (~np.isnan(top) & ~np.isnan(bot)).sum() < 3:
        return np.full(n, np.nan), ok
    rh = r if rh is None else rh
    xf, tf, bf, gam, phi, ok = _gamma_phi(top, bot, r, rh, U)
    m = 0.5 * (gam + phi)
    k = max(1, int(np.floor(rh * U)))
    c = _shift_pad(m, k) + _shift_pad(m, -k) - 2 * m
    yf = np.where(c > 0, gam, phi)
    agree = np.abs(gam - phi) < 0.5
    yf = np.where(agree, m, yf)
    y = np.interp(xs, xf, yf)
    return y, ok


def refine_steep(D, xs, yc, rh, min_dy=2.0):
    """Média a posteriori do INSTANTE nos trechos íngremes (QRS).

    Numa coluna, o traço quase vertical ocupa um run alto e o centro dele só diz o x com
    resolução de 1 coluna (0,1 mm = 4 ms a 25 mm/s); num QRS de 50 mV/s isso já é 0,2 mV
    de erro. Aqui o papel de x e y se inverte: em cada linha de pixel que o traço íngreme
    cruza, o centróide horizontal da tinta (pesado pela densidade, sub-pixel) é a estimativa
    de mínimo erro quadrático do instante em que a curva passa por aquela altura.
    Trechos planos ficam com o centro por coluna. Devolve (x, y) ordenados em x."""
    xs = np.asarray(xs, float); yc = np.asarray(yc, float)
    n = len(xs)
    if n < 4:
        return xs, yc
    H, W = D.shape
    dy = np.diff(yc)
    contig = np.diff(xs) == 1
    sgn = np.sign(dy)
    steep = (np.abs(dy) >= min_dy) & contig & np.isfinite(dy)
    # mesmo sentido nos vizinhos: longe da volta do pico (onde subida e descida se fundem)
    same = np.zeros_like(steep)
    same[1:-1] = (sgn[1:-1] == sgn[:-2]) & (sgn[1:-1] == sgn[2:])
    steep &= same
    w = int(np.ceil(rh)) + 2
    px, py = [], []
    for i in np.nonzero(steep)[0]:
        y0, y1 = yc[i], yc[i + 1]
        lo, hi = (y0, y1) if y0 < y1 else (y1, y0)
        rows = np.arange(int(np.floor(lo + 1.0)) + 1, int(np.ceil(hi - 1.0)))
        for Y in rows:
            if Y < 0 or Y >= H:
                continue
            x0 = xs[i] + (Y - y0) / (y1 - y0)
            a, b = max(0, int(np.floor(x0)) - w), min(W, int(np.ceil(x0)) + w + 1)
            prof = D[Y, a:b].astype(float)
            if prof.size < 3 or prof.max() < 0.2:
                continue
            thr = 0.3 * prof.max()
            on = prof > thr
            k = int(np.clip(round(x0) - a, 0, len(prof) - 1))
            if not on[k]:
                cand = np.nonzero(on)[0]
                k = int(cand[np.argmin(np.abs(cand - (x0 - a)))])
            l_, r_ = k, k
            while l_ - 1 >= 0 and on[l_ - 1]:
                l_ -= 1
            while r_ + 1 < len(prof) and on[r_ + 1]:
                r_ += 1
            if l_ == 0 or r_ == len(prof) - 1 or (r_ - l_ + 1) > 2 * rh + 4:
                continue  # encostado na janela ou largo demais: dois traços/borrão
            wt = prof[l_:r_ + 1] - thr
            xc = a + l_ + float((wt * np.arange(r_ - l_ + 1)).sum() / max(wt.sum(), 1e-9))
            if abs(xc - x0) > 1.5:
                continue
            if not (min(xs[i], xs[i + 1]) < xc < max(xs[i], xs[i + 1])):
                continue  # fora do intervalo entre as duas colunas: quebraria a monotonia
            px.append(xc); py.append(float(Y))
    if not px:
        return xs, yc
    # os pontos de coluna ficam (no pico/vale a inversão da pena é exata); os de linha entram
    # entre eles e dão o instante sub-coluna das subidas e descidas
    X = np.concatenate([xs, px]); Y = np.concatenate([yc, py])
    o = np.argsort(X, kind="stable")
    X, Y = X[o], Y[o]
    ok = np.r_[True, np.diff(X) > 1e-3]
    return X[ok], Y[ok]


def mark_occlusions(jobs, lw, margin=1.0):
    """Traços sobrepostos (linhas vizinhas que se cruzam): a borda superior/inferior
    de um traço que cai dentro da tinta de OUTRO traço não é confiável -> NaN.
    O envelope é então interpolado sobre a oclusão."""
    for j in jobs:
        j["top_o"], j["bot_o"] = j["top"].copy(), j["bot"].copy()
    for j in jobs:
        xj = j["xa"] + np.arange(len(j["top"]))
        for o in jobs:
            if o is j or o["row"] == j["row"]:
                continue
            xo0, xo1 = o["xa"], o["xa"] + len(o["top"])
            lo, hi = max(xj[0], xo0), min(xj[-1] + 1, xo1)
            if hi <= lo:
                continue
            sj = slice(lo - j["xa"], hi - j["xa"])
            so = slice(lo - xo0, hi - xo0)
            ot, ob = o["top"][so] - margin, o["bot"][so] + margin
            tj, bj = j["top"][sj], j["bot"][sj]
            with np.errstate(invalid="ignore"):
                # só um traço vizinho localmente plano (não "engoliu" um QRS alheio) oclui
                flat_o = (o["bot"][so] - o["top"][so]) <= 2.5 * lw
                occ_t = flat_o & (tj >= ot) & (tj <= ob)
                occ_b = flat_o & (bj >= ot) & (bj <= ob)
            j["top_o"][sj][occ_t] = np.nan
            j["bot_o"][sj][occ_b] = np.nan
    for j in jobs:
        j["top"], j["bot"] = j.pop("top_o"), j.pop("bot_o")


def trace_likeness(B, peaks, R, min_len_mm=25.0, return_raw=False):
    """Para cada linha candidata: largura total coberta por componentes conexas longas
    (traçados), centradas perto da linha. Texto/cabeçalho = muitos pedaços curtos."""
    k = int(1.5 * R) | 1
    Bc = cv2.dilate(B.astype(np.uint8), np.ones((3, k), np.uint8))  # junta quebras pequenas
    n, lab, st, cen = cv2.connectedComponentsWithStats(Bc, connectivity=8)
    long_ = [i for i in range(1, n) if st[i, cv2.CC_STAT_WIDTH] >= min_len_mm * R]
    scores = []
    for y in peaks:
        cov = 0.0
        for i in long_:
            y0, h = st[i, cv2.CC_STAT_TOP], st[i, cv2.CC_STAT_HEIGHT]
            if y0 <= y <= y0 + h:
                # a componente precisa ter tinta perto da linha ao longo de boa parte da largura
                x0, w = st[i, cv2.CC_STAT_LEFT], st[i, cv2.CC_STAT_WIDTH]
                band = lab[max(0, int(y - 3 * R)):int(y + 3 * R), x0:x0 + w] == i
                cov += band.any(0).sum()
        scores.append(cov / R)  # em mm
    if not return_raw:
        return np.array(scores)
    # cobertura "crua": componentes longas SEM juntar quebras de 1.5 mm. Texto (rodapé
    # "50mm/s 10mm/mV ...") vira uma componente longa depois da dilatação, mas cru são
    # letras de poucos mm; traçado continua longo.
    B8 = B.astype(np.uint8)
    # retira réguas retas (borda da área quadriculada etc.): retas de > 60 mm na horizontal
    lines = cv2.morphologyEx(B8, cv2.MORPH_OPEN, np.ones((1, int(60 * R) | 1), np.uint8))
    lines = cv2.dilate(lines, np.ones((5, 1), np.uint8))
    Br = cv2.morphologyEx(B8 & (1 - lines), cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    n2, lab2, st2, _ = cv2.connectedComponentsWithStats(Br, connectivity=8)
    # linhas retas (borda da área quadriculada, réguas) não são traçado: um traço de
    # > 50 mm sempre tem QRS/ondas com > 1.5 mm de altura
    w2, h2 = st2[:, cv2.CC_STAT_WIDTH], st2[:, cv2.CC_STAT_HEIGHT]
    # palavras: largas mas baixas (< 4 mm); pedaços de traçado têm ondas ou são longos
    long2 = np.nonzero(((w2 >= 6.0 * R) & (h2 >= 4.0 * R)) | (w2 >= 25.0 * R))[0]
    long2 = long2[long2 > 0]
    keep = np.zeros(n2, bool); keep[long2] = True
    raw = []
    for y in peaks:
        band = keep[lab2[max(0, int(y - 3 * R)):int(y + 3 * R)]]
        raw.append(band.any(0).sum() / R)
    return np.array(scores), np.array(raw)


def horizontal_halfwidth(D, jobs, lev, rv):
    """Meia largura horizontal da pena medida diretamente: no meio de traços quase
    verticais (QRS), perfil horizontal da tinta no nível de 50%. Traços de subida e
    descida colados dão larguras dobradas -> usa um percentil baixo."""
    H, W = D.shape
    out = []
    for j in jobs:
        top, bot, xa = j["top"], j["bot"], j["xa"]
        for i in range(2, len(top) - 2):
            if np.isnan(top[i]) or np.isnan(bot[i]) or bot[i] - top[i] < 8 * rv:
                continue
            y = int(round(0.5 * (top[i] + bot[i])))
            x = xa + i
            if x < 12 or x + 12 >= W or not (0 <= y < H):
                continue
            prof = D[y, x - 12:x + 13]
            c = 12 + int(np.argmax(prof[10:15])) - 2
            if prof[c] < lev:
                continue
            l = c
            while l > 0 and prof[l - 1] >= lev:
                l -= 1
            r = c
            while r < 24 and prof[r + 1] >= lev:
                r += 1
            if l == 0 or r == 24:
                continue
            le = (l - 1) + (lev - prof[l - 1]) / max(prof[l] - prof[l - 1], 1e-6)
            re = r + (prof[r] - lev) / max(prof[r] - prof[r + 1], 1e-6)
            out.append(0.5 * (re - le))
    if len(out) < 20:
        return rv
    return float(np.clip(np.percentile(out, 25), 0.7 * rv, 1.0 * rv))
