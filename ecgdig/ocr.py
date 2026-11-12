"""OCR clássico dos rótulos das derivações (sem aprendizado).

O vocabulário é minúsculo (~15 palavras), então reconhecimento por casamento de
modelos funciona: cada rótulo é renderizado em várias fontes TrueType; a palavra
candidata na imagem é recortada, normalizada em altura e comparada por correlação
normalizada (após suavização, para tolerar diferenças de fonte), com penalidade de
razão de aspecto. Devolve, por palavra, uma distribuição sobre os rótulos.
"""
from __future__ import annotations

import os
from functools import lru_cache

import numpy as np
import cv2

# rótulo impresso -> nome canônico (layouts.LEAD_NAMES)
VOCAB = {
    "I": "I", "II": "II", "III": "III",
    "aVR": "aVR", "aVL": "aVL", "aVF": "aVF", "AVR": "aVR", "AVL": "aVL", "AVF": "aVF",
    "-aVR": "-aVR", "-AVR": "-aVR",
    "V1": "V1", "V2": "V2", "V3": "V3", "V4": "V4", "V5": "V5", "V6": "V6",
    "v1": "V1", "v2": "V2", "v3": "V3", "v4": "V4", "v5": "V5", "v6": "V6",
    "avr": "aVR", "avl": "aVL", "avf": "aVF",
    # tira de ritmo rotulada entre parênteses: "Rhythm (II) 10 mm/mV"
    "(I)": "I", "(II)": "II", "(III)": "III", "(V1)": "V1", "(V5)": "V5", "(aVF)": "aVF",
}
_FONT_DIRS = ["/usr/share/fonts/truetype/freefont", "/usr/share/fonts/truetype/liberation",
              "/usr/share/fonts/truetype/dejavu"]
_FONTS = ["FreeSerif.ttf", "FreeSerifBold.ttf", "FreeSans.ttf", "FreeSansBold.ttf", "FreeMono.ttf",
          "LiberationSans-Regular.ttf", "LiberationSerif-Regular.ttf", "LiberationSansNarrow-Regular.ttf",
          "DejaVuSans.ttf", "DejaVuSerif.ttf", "DejaVuSansCondensed.ttf", "DejaVuSansMono.ttf"]
TH = 32  # altura normalizada (px)


def _font_paths():
    out = []
    for f in _FONTS:
        for d in _FONT_DIRS:
            p = os.path.join(d, f)
            if os.path.exists(p):
                out.append(p)
                break
    return out


def _norm(mask: np.ndarray, width: int | None = None) -> np.ndarray:
    """Recorte justo -> altura TH (largura proporcional ou forçada), suavizado, média 0 / norma 1."""
    ys, xs = np.nonzero(mask)
    if len(ys) == 0:
        return None
    m = mask[ys.min():ys.max() + 1, xs.min():xs.max() + 1].astype(np.float32)
    h, w = m.shape
    W = width or max(1, int(round(TH * w / h)))
    r = cv2.resize(m, (W, TH), interpolation=cv2.INTER_AREA)
    # margem: um "I" recortado justo é um bloco uniforme (variância zero); com margem a
    # forma (barra fina dentro do vazio) passa a ser comparável
    r = cv2.GaussianBlur(r, (0, 0), 1.6)
    r = r - r.mean()
    n = np.linalg.norm(r)
    return r / n if n > 0 else None


@lru_cache(maxsize=1)
def templates():
    """[(texto, nome_canônico, aspecto, molde_normalizado)]"""
    from PIL import Image, ImageDraw, ImageFont
    out = []
    for fp in _font_paths():
        font = ImageFont.truetype(fp, 64)
        for txt, canon in VOCAB.items():
            for t in ({txt, txt.replace("-", "–")} if "-" in txt else {txt}):
                im = Image.new("L", (64 * (len(t) + 2), 110), 0)
                ImageDraw.Draw(im).text((20, 10), t, fill=255, font=font)
                m = np.array(im) > 127
                ys, xs = np.nonzero(m)
                if len(ys) == 0:
                    continue
                asp = (np.ptp(xs) + 1) / (np.ptp(ys) + 1)
                out.append((t, canon, asp, m))
    return out


_TV = {}


def _tvec(i, W):
    k = (i, W)
    if k not in _TV:
        _TV[k] = _norm(templates()[i][3], W)
    return _TV[k]


def classify_word(mask: np.ndarray) -> dict:
    """Distribuição {nome_canônico: similaridade em [0,1]} para um recorte binário."""
    ys, xs = np.nonzero(mask)
    if len(ys) < 6:
        return {}
    asp_c = (np.ptp(xs) + 1) / (np.ptp(ys) + 1)
    # "I": uma única barra vertical — recorte justo é um bloco uniforme (correlação
    # normalizada indefinida). Decide pela forma: 1 componente, aspecto de barra, cheio.
    n_cc, _, st_cc, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    big = [i for i in range(1, n_cc) if st_cc[i, cv2.CC_STAT_AREA] > 0.1 * len(ys)]
    if len(big) == 1 and asp_c < 0.42:
        fill = len(ys) / float((np.ptp(xs) + 1) * (np.ptp(ys) + 1))
        if fill > 0.5:
            return {"I": float(min(1.0, 0.5 + fill / 2))}
    best = {}
    cache = {}
    for i, (txt, canon, asp, m) in enumerate(templates()):
        W = max(4, int(round(TH * asp)))
        if W not in cache:
            cache[W] = _norm(mask, W)
        a = cache[W]
        b = _tvec(i, W)
        if a is None or b is None:
            continue
        ncc = float((a * b).sum())
        pen = np.exp(-abs(np.log(asp_c / asp)) / 0.35)
        sc = max(0.0, ncc) * pen
        if sc > best.get(canon, 0.0):
            best[canon] = sc
    return best


def find_words(text_mask: np.ndarray, R: float, min_h_mm=1.4, max_h_mm=8.0):
    """Agrupa componentes conexas em palavras (caixas). Retorna lista de (x0,y0,x1,y1)."""
    n, lab, st, _ = cv2.connectedComponentsWithStats(text_mask.astype(np.uint8), connectivity=8)
    boxes = []
    for i in range(1, n):
        x, y, w, h, a = st[i]
        if a < 4 or h > max_h_mm * R or w > 25 * R:
            continue
        boxes.append([x, y, x + w, y + h])
    boxes.sort(key=lambda b: b[0])
    words = []
    used = [False] * len(boxes)
    for i, b in enumerate(boxes):
        if used[i]:
            continue
        cur = list(b); used[i] = True
        changed = True
        while changed:
            changed = False
            for j, c in enumerate(boxes):
                if used[j]:
                    continue
                hh = max(cur[3] - cur[1], c[3] - c[1], 1)
                vo = min(cur[3], c[3]) - max(cur[1], c[1])
                gap = max(c[0] - cur[2], cur[0] - c[2])
                # minúsculas/sinal de menos: sobreposição vertical parcial basta
                if gap < 0.7 * hh and vo > -0.1 * hh and vo > 0.25 * min(c[3] - c[1], cur[3] - cur[1]) - 1:
                    cur = [min(cur[0], c[0]), min(cur[1], c[1]), max(cur[2], c[2]), max(cur[3], c[3])]
                    used[j] = True; changed = True
        h = cur[3] - cur[1]
        if min_h_mm * R <= h <= max_h_mm * R:
            words.append(tuple(cur))
    return words


@lru_cache(maxsize=1)
def char_templates():
    """[(caractere, aspecto, máscara)] para dígitos 1-6 e R/L/F."""
    from PIL import Image, ImageDraw, ImageFont
    out = []
    for fp in _font_paths():
        font = ImageFont.truetype(fp, 64)
        for ch in "123456RLF":
            im = Image.new("L", (120, 110), 0)
            ImageDraw.Draw(im).text((20, 10), ch, fill=255, font=font)
            m = np.array(im) > 127
            ys, xs = np.nonzero(m)
            out.append((ch, (np.ptp(xs) + 1) / (np.ptp(ys) + 1), m))
    return out


def _last_char(mask: np.ndarray):
    """Recorte do último caractere: última componente conexa (ou os 40% da direita)."""
    ys, xs = np.nonzero(mask)
    m = mask[ys.min():ys.max() + 1, xs.min():xs.max() + 1].astype(np.uint8)
    n, lab, st, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    H = m.shape[0]
    comps = [i for i in range(1, n) if st[i, cv2.CC_STAT_HEIGHT] >= 0.45 * H]
    if len(comps) >= 2:
        i = max(comps, key=lambda c: st[c, cv2.CC_STAT_LEFT])
        return lab == i
    w = m.shape[1]
    out = np.zeros_like(m, bool)
    out[:, int(0.6 * w):] = m[:, int(0.6 * w):] > 0
    return out


def classify_char(mask: np.ndarray, allowed: str) -> dict:
    ys, xs = np.nonzero(mask)
    if len(ys) < 4:
        return {}
    asp_c = (np.ptp(xs) + 1) / (np.ptp(ys) + 1)
    best = {}
    for i, (ch, asp, m) in enumerate(char_templates()):
        if ch not in allowed:
            continue
        W = max(4, int(round(TH * asp)))
        a = _norm(mask, W)
        k = ("c", i, W)
        if k not in _TV:
            _TV[k] = _norm(m, W)
        b = _TV[k]
        if a is None or b is None:
            continue
        sc = max(0.0, float((a * b).sum())) * np.exp(-abs(np.log(asp_c / asp)) / 0.5)
        best[ch] = max(best.get(ch, 0.0), sc)
    return best


def _trim_marks(mask: np.ndarray):
    """Remove marcas baixas nas pontas da palavra (seta '↓ V1', ponto, dois-pontos):
    grupos de colunas cuja altura é < 60% da palavra. Retorna None se nada mudou."""
    colp = mask.any(0)
    nz = np.nonzero(colp)[0]
    if len(nz) == 0:
        return None
    runs, a = [], nz[0]
    for i in range(1, len(nz)):
        if nz[i] != nz[i - 1] + 1:
            runs.append((a, nz[i - 1])); a = nz[i]
    runs.append((a, nz[-1]))
    H = np.ptp(np.nonzero(mask.any(1))[0]) + 1

    def tall(r):
        rr = np.nonzero(mask[:, r[0]:r[1] + 1].any(1))[0]
        return np.ptp(rr) + 1 >= 0.6 * H
    i0, i1 = 0, len(runs) - 1
    while i0 < i1 and not tall(runs[i0]):
        i0 += 1
    while i1 > i0 and not tall(runs[i1]):
        i1 -= 1
    if i0 == 0 and i1 == len(runs) - 1:
        return None
    m = mask[:, runs[i0][0]:runs[i1][1] + 1]
    rr = np.nonzero(m.any(1))[0]
    return m[rr[0]:rr[-1] + 1]


def read_label(mask: np.ndarray) -> dict:
    """Palavra inteira + refinamento do último caractere para as famílias V* e aV*.
    Se a leitura for fraca, tenta de novo sem marcas baixas nas pontas (setas, pontos)."""
    d = _read_label(mask)
    if not d or max(d.values()) < 0.5:
        t = _trim_marks(mask)
        if t is not None and t.sum() >= 10:
            d2 = _read_label(t)
            if d2 and (not d or max(d2.values()) > max(d.values())):
                return d2
    return d


def _read_label(mask: np.ndarray) -> dict:
    d = classify_word(mask)
    if not d:
        return d
    top = max(d, key=d.get)
    fam_v = [k for k in d if k.startswith("V")]
    fam_a = [k for k in d if k in ("aVR", "aVL", "aVF")]
    if top.startswith("V") and fam_v:
        c = classify_char(_last_char(mask), "123456")
        if c:
            for k in fam_v:
                d[k] = 0.4 * d[k] + 0.6 * c.get(k[1], 0.0)
    elif top in ("aVR", "aVL", "aVF") and fam_a:
        c = classify_char(_last_char(mask), "RLF")
        if c:
            for k in fam_a:
                d[k] = 0.4 * d[k] + 0.6 * c.get(k[2], 0.0)
    return d


SPEED_WORDS = {"25mm/s": 25.0, "50mm/s": 50.0, "25 mm/s": 25.0, "50 mm/s": 50.0,
               "12.5mm/s": 12.5, "12,5mm/s": 12.5,
               # negativos: palavras parecidas que NÃO são velocidade
               "10mm/mV": 0.0, "5mm/mV": 0.0, "20mm/mV": 0.0, "10 mm/mV": 0.0, "10/5mm/mV": 0.0,
               "150 Hz": 0.0, "40Hz": 0.0, "bpm": 0.0}


@lru_cache(maxsize=1)
def speed_templates():
    from PIL import Image, ImageDraw, ImageFont
    out = []
    for fp in _font_paths():
        font = ImageFont.truetype(fp, 64)
        for t, v in SPEED_WORDS.items():
            im = Image.new("L", (64 * (len(t) + 2), 110), 0)
            ImageDraw.Draw(im).text((20, 10), t, fill=255, font=font)
            m = np.array(im) > 127
            ys, xs = np.nonzero(m)
            out.append((t, v, (np.ptp(xs) + 1) / (np.ptp(ys) + 1), m))
    return out


def read_speed(text_mask: np.ndarray, R: float, min_score=0.45, debug=False):
    """Procura 'NN mm/s' no texto da página. Retorna (velocidade, escore) ou (None, 0).
    A palavra inteira identifica o padrão 'mm/s'; o 1º dígito (2/5/1) decide a velocidade."""
    best = (None, 0.0)
    for w in find_words(text_mask, R, min_h_mm=1.0, max_h_mm=6.0):
        x0, y0, x1, y1 = w
        crop = text_mask[y0:y1, x0:x1]
        ys, xs = np.nonzero(crop)
        if len(ys) < 20:
            continue
        asp_c = (np.ptp(xs) + 1) / (np.ptp(ys) + 1)
        if asp_c < 2.0:
            continue
        sc_best, v_best = 0.0, None
        for i, (t, v, asp, m) in enumerate(speed_templates()):
            W = max(4, int(round(TH * asp)))
            a = _norm(crop, W)
            k = ("s", i, W)
            if k not in _TV:
                _TV[k] = _norm(m, W)
            if a is None or _TV[k] is None:
                continue
            sc = max(0.0, float((a * _TV[k]).sum())) * np.exp(-abs(np.log(asp_c / asp)) / 0.35)
            if sc > sc_best:
                sc_best, v_best = sc, v
        if debug:
            print("   palavra", w, "melhor", v_best, round(sc_best, 3))
        if sc_best < min_score or not v_best:
            continue
        # 1º dígito: componente mais à esquerda
        # pela projeção de colunas (letras quebradas por impressão térmica/limiar viram
        # várias componentes; a 1ª coluna vazia depois da tinta separa o 1º dígito)
        colp = crop.any(0)
        nz = np.nonzero(colp)[0]
        first = None
        if len(nz):
            a = nz[0]
            b = a
            while b + 1 < len(colp) and colp[b + 1]:
                b += 1
            ch = crop[:, a:b + 1]
            rr = np.nonzero(ch.any(1))[0]
            if len(rr) and np.ptp(rr) + 1 >= 0.5 * crop.shape[0]:
                first = ch
        if first is not None:
            d = classify_char(first, "125")
            if d:
                dig = max(d, key=d.get)
                v_best = {"2": 25.0, "5": 50.0, "1": 12.5}[dig]
        if sc_best > best[1]:
            best = (v_best, sc_best)
    return best
