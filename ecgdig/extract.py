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


def ink_map(rect: np.ndarray, paper: np.ndarray, R: float, return_grid_level: bool = False, tmin: float | None = None):
    f = rect.astype(np.float32)
    n = np.stack([f[..., c] / (_bg(f[..., c], R) + 1e-3) for c in range(3)], -1)
    t = np.clip(n.max(-1), 0, 1.2)  # transmitância no canal mais claro
    P = int(round(5 * R))
    H, W = t.shape
    ny, nx = H // P, W // P
    blk = t[: ny * P, : nx * P].reshape(ny, P, nx, P)
    pm = paper[: ny * P, : nx * P].reshape(ny, P, nx, P).min(axis=(1, 3)) > 0
    tiles = blk.transpose(0, 2, 1, 3)[pm]  # (k, P, P)
    tmpl = np.median(tiles, axis=0) if len(tiles) > 10 else np.ones((P, P), np.float32)
    # grade escura (scan P&B, fotocópia): as linhas grossas são tão pretas quanto o traço.
    # O molde precisa descer até elas (senão a linha da grade vira "tinta" contínua que
    # estende as linhas até a borda da folha e cria linhas falsas)
    dark_grid = float(tmpl.min()) < 0.15
    ink_map.dark_grid = dark_grid
    if tmin is None:
        tmin = 0.05 if dark_grid else 0.3
    tmpl = np.clip(tmpl, tmin, 1.0)
    reps = (int(np.ceil(H / P)), int(np.ceil(W / P)))
    T = np.tile(tmpl, reps)[:H, :W]
    # molde LOCAL: mediana por fase (mod 5 mm) numa vizinhança de 5x5 células (25 mm).
    # Em fotos a nitidez/escuridão da grade varia pela folha; o molde global deixa a
    # grade "vazar" como tinta onde ela é mais escura que a média.
    if ny >= 5 and nx >= 5:
        from scipy.ndimage import median_filter
        cells = blk.transpose(0, 2, 1, 3).copy()               # (ny, nx, P, P)
        cells[~pm] = tmpl                                        # fora do papel: global
        loc = median_filter(cells, size=(5, 5, 1, 1), mode="nearest")
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
        strong = cv2.dilate((D > 0.55).astype(np.uint8), np.ones((7, 7), np.uint8)) > 0
        gp = (T < 0.9) & (pe > 0) & ~strong
        glev = float(np.percentile(D[gp], 95)) if gp.sum() > 1000 else 0.0
        return D, glev
    return D


def binarize(D: np.ndarray, R: float = 10.0, grid_level: float = 0.0) -> np.ndarray:
    # Otsu só sobre estruturas finas: tarjas, carimbos e blocos de texto preto (> 2.5 mm
    # de espessura) puxariam o limiar para cima e apagariam traçados claros/finos
    kb = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (int(2.5 * R) | 1,) * 2)
    blob = cv2.morphologyEx((D > 0.25).astype(np.uint8), cv2.MORPH_OPEN, kb)
    blob = cv2.dilate(blob, np.ones((int(0.8 * R) | 1,) * 2, np.uint8)) > 0
    v = D[(D > 0.08) & ~blob]
    if v.size < 100:
        B = D > 0.4
    else:
        th = float(np.clip(threshold_otsu(v), 0.2, 0.6))
        # limiar baixo da histerese acima do vazamento da grade (linhas da grade tocam o
        # traço e seriam "puxadas" junto pela conectividade)
        lo = max(0.55 * th, min(grid_level + 0.08, 0.85 * th))
        B = apply_hysteresis_threshold(D, lo, max(th, lo + 0.05))
    # remove sal-e-pimenta: componentes minúsculas (< ~0.4 mm²) não são traçado
    n, lab, st, _ = cv2.connectedComponentsWithStats(B.astype(np.uint8), connectivity=8)
    small = st[:, cv2.CC_STAT_AREA] < max(4, int(0.4 * R * R))
    small[0] = False
    if small.any():
        B = B & ~small[lab]
    return B


# ------------------------------------------------------------------------ layout

def find_rows(B: np.ndarray, R: float):
    H, W = B.shape
    k = int(2 * R) | 1
    Bd = cv2.dilate(B.astype(np.uint8), np.ones((k, 1), np.uint8))
    cov = Bd.mean(1).astype(np.float64)
    cov = cv2.GaussianBlur(cov.reshape(-1, 1), (0, 0), R).ravel()
    pk, pr = find_peaks(cov, distance=int(8 * R), height=0.3 * cov.max(), prominence=0.1 * cov.max())
    return pk, cov


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
    return int(s[i]), int(e[i])


def detect_pulse(B, y0, half, xa, R, up=None):
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
    kv = max(3, int(0.8 * R) | 1)  # fecha quebras verticais (resolução baixa / JPEG)
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
    """Lista por coluna de segmentos (top, bot, centróide, massa)."""
    out = []
    sub = B[ya:yb, xa:xb]
    subD = D[ya:yb, xa:xb]
    yy = np.arange(ya, yb, dtype=np.float64)
    for j in range(sub.shape[1]):
        s, e = _runs(sub[:, j])
        # funde fragmentos separados por <= 3 px (halo de antialiasing / JPEG)
        if len(s) > 1:
            keep = np.r_[True, (s[1:] - e[:-1] - 1) > 3]
            grp = np.cumsum(keep) - 1
            s = np.array([s[grp == g].min() for g in range(grp.max() + 1)])
            e = np.array([e[grp == g].max() for g in range(grp.max() + 1)])
        col = []
        for a, b in zip(s, e):
            w = subD[a:b + 1, j].astype(np.float64)
            m = w.sum() + 1e-9
            col.append((a + ya, b + ya, float((w * yy[a:b + 1]).sum() / m), float(m)))
        # descarta fragmentos sem massa quando há alternativa
        if len(col) > 1:
            big = [c for c in col if c[3] >= 0.6]
            col = big or col
        out.append(col)
    return out


def viterbi_track(runs, base, R, w_base=0.004, c_gap=1.0, others=(), lw=None, w_edge=0.0):
    """Escolhe um segmento por coluna (ou falha) minimizando custo de continuidade."""
    INF = 1e18
    n = len(runs)
    states = []  # por coluna: lista de (top, bot, center)
    costs, backs = [], []
    flat_h = 3.0 * (lw if lw else 0.3 * R)
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

    _we = w_edge

    def trans(pt, pb, pc, t, b, c):
        g = max(0.0, t - pb, pt - b) / R
        # continuidade das BORDAS: num traço desenhado por pena, de uma coluna para a
        # outra pelo menos uma das bordas (topo/base) anda pouco; letra colada ao traço
        # (rótulo atravessado por um R) fica no meio e quebra as duas
        return 2.0 * g + 0.5 * g * g + 0.02 * abs(c - pc) / R + _we * min(abs(t - pt), abs(b - pb)) / R

    prev_states = [(base, base, base)]
    prev_cost = np.array([0.0])
    for i in range(n):
        cur = [(t, b, c) for (t, b, c, m) in runs[i]]
        ms = [m for (t, b, c, m) in runs[i]]
        cs = []
        bk = []
        for (t, b, c), m in zip(cur, ms):
            best, arg = INF, -1
            for k, (pt, pb, pc) in enumerate(prev_states):
                v = prev_cost[k] + trans(pt, pb, pc, t, b, c)
                if v < best:
                    best, arg = v, k
            cs.append(best + emis(t, b, c, m))
            bk.append(arg)
        # estado de falha (último índice)
        kmin = int(np.argmin(prev_cost))
        cs.append(prev_cost[kmin] + c_gap)
        bk.append(kmin)
        new_states = cur + [prev_states[kmin] if kmin < len(prev_states) else prev_states[-1]]
        states.append(new_states)
        costs.append(np.array(cs))
        backs.append(bk)
        prev_states, prev_cost = new_states, np.array(cs)
    # backtrack
    sel = [None] * n
    k = int(np.argmin(costs[-1]))
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
