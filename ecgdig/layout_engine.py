"""
Motor de layout por hipótese e verificação.

Entrada: a estrutura já detectada (linhas de traçado, pulsos com 1 ou 2 níveis, eixo do
tempo por linha, traço rápido de cada linha inteira). Saída: o layout mais provável —
nomes impressos de cada segmento de cada linha, ganho de cada segmento — com uma
pontuação decomposta e uma confiança.

Hipóteses: famílias da biblioteca (padrão e Cabrera; 3x4, 6x2, 12x1) no topo da página,
com as linhas restantes como tiras de ritmo de derivação livre, e uma hipótese "livre"
(nº de segmentos por linha pelas evidências de fronteira, nomes pelo OCR).

Pontuação (log-verossimilhança informal, termos independentes):
  OCR       rótulo lido perto do início de cada segmento
  fronteira marcas/rótulos nas posições de troca de derivação (e ausência nas outras)
  membros   instantes com >= 3 derivações de membros simultâneas devem caber no espaço
            2D de Einthoven/Goldberger (resíduo de mínimos quadrados)
  duplicata a mesma derivação em duas tiras no mesmo instante deve coincidir
"""
from __future__ import annotations

import warnings

import numpy as np
import cv2
from scipy.ndimage import median_filter

from .layouts import LIBRARY, LEAD_NAMES, LIMB_COEF, is_limb
from . import ocr

FAMILIES = ["3x4", "3x4-cabrera", "6x2", "6x2-cabrera", "12x1", "12x1-cabrera",
            "6x1-prec", "6x1-limb", "6x1-limb-cabrera"]
STRIPS3 = [True]
if STRIPS3[0]:
    FAMILIES += ["3x1-limb", "3x1-aug", "3x1-v13", "3x1-v46"]
FRACS = {1: [], 2: [0.5], 3: [1 / 3, 2 / 3], 4: [0.25, 0.5, 0.75]}
W_OCR, W_BND, W_LIMB, W_DUP = 4.0, 1.5, 15.0, 3.0
_UNEXPLAINED = [True]
CABRERA_NO_EVIDENCE = [10.0]
LABEL_HI = [15.0]
BND_RELATIVE = [True]
BND_OTHER_ROWS = [True]
BND_COHERENT = [True]
LIMB_LOG = [True]


def _canon(name):
    return "aVR" if name == "-aVR" else name


def _sign(name):
    return -1.0 if name == "-aVR" else 1.0



def _limb_ratio(V, A):
    """Resíduo relativo de Einthoven/Goldberger: V (n, T) medidas, A (n, 2) coeficientes
    em (I, II). Mínimos quadrados por instante, vetorizado por padrão de amostras válidas."""
    ok = ~np.isnan(V)
    num = den = 0.0
    nt = 0
    keys = np.packbits(ok, axis=0)
    cols = {}
    for t in range(V.shape[1]):
        cols.setdefault(keys[:, t].tobytes(), []).append(t)
    for _, ts in cols.items():
        m = ok[:, ts[0]]
        if m.sum() < 3:
            continue
        Am = A[m]
        P = Am @ np.linalg.pinv(Am)
        X = V[m][:, ts]
        Rz = X - P @ X
        num += float((Rz ** 2).sum()); den += float((X ** 2).sum())
        nt += len(ts)
    if den <= 0:
        return np.nan, nt
    return float(np.sqrt(num / den)), nt

class Ctx:
    """Tudo o que o motor precisa, já medido pelo pipeline."""

    def __init__(self, B, D, rows, pulses, R, x0_r, sc_r, speed, duration, qtr, base_r, levels):
        self.B, self.D, self.rows, self.pulses, self.R = B, D, list(rows), pulses, R
        self.x0_r, self.sc_r, self.speed, self.duration = x0_r, sc_r, speed, duration
        self.qtr, self.base_r, self.levels = qtr, base_r, levels  # levels[r] = (g1, g2|None)
        self.n = len(rows)
        self.t = np.arange(0.0, duration, 0.01)  # grade de 100 Hz para pontuar
        self.vals = [self._row_px(r) for r in range(self.n)]
        self.words = self._read_words()
        self.bnd = {r: {q: self._boundary_evidence(r, q) for q in sorted({f for v in FRACS.values() for f in v})}
                    for r in range(self.n)}
        if BND_RELATIVE[0]:
            # evidência relativa ao fundo da própria linha: grade vazando, marcas de tempo ou
            # ruído dão "fronteira" em qualquer x; o que conta é a posição se destacar
            allf = [f for v in FRACS.values() for f in v]
            bgq = [q for q in np.arange(0.08, 0.93, 0.05) if min(abs(q - f) for f in allf) > 0.03]
            for r in range(self.n):
                bg = float(np.median([self._boundary_evidence(r, q) for q in bgq]))
                if bg > 0.05:
                    self.bnd[r] = {q: float(np.clip((e - bg) / (1 - bg + 1e-6), 0, 1))
                                   for q, e in self.bnd[r].items()}
        if BND_COHERENT[0] and self.n >= 3:
            # fronteira de coluna é a mesma em todas as linhas do bloco; um QRS que cai por
            # acaso na posição só aparece numa linha. Evidência vista em menos da metade das
            # linhas vale proporcionalmente menos
            qs = list(self.bnd[0].keys())
            for q in qs:
                c = np.mean([self.bnd[r][q] > 0.5 for r in range(self.n)])
                f = min(1.0, c / 0.5)
                for r in range(self.n):
                    self.bnd[r][q] *= f
        self._hp = {}

    # ------------------------------------------------------------------ sinais
    def x_of(self, r, t):
        return self.x0_r[r] + self.sc_r[r] * np.asarray(t) * self.speed * self.R

    def _row_px(self, r):
        q = self.qtr[r]
        xs = q["x0"] + np.arange(len(q["cen"]))
        ok = ~np.isnan(q["cen"])
        if ok.sum() < 10:
            return np.full(len(self.t), np.nan)
        v = np.interp(self.x_of(r, self.t), xs[ok], q["cen"][ok], left=np.nan, right=np.nan)
        return self.base_r[r] - v  # px acima da linha de base

    def seg_gain(self, r, name):
        g1, g2 = self.levels[r]
        if g2 is not None and not is_limb(name):
            return g2
        return g1

    def seg_signal(self, r, k, c, name):
        """(máscara de tempo, sinal em mV no nome CANÔNICO, passa-alta) do segmento c de k."""
        n = len(self.t)
        i0, i1 = c * n // k, (c + 1) * n // k
        key = (r, k, c, name)
        if key not in self._hp:
            v = np.full(n, np.nan)
            seg = self.vals[r][i0:i1] / (self.seg_gain(r, name) * self.R)
            # margem de 40 ms longe das fronteiras (marcas, troca de derivação)
            m = 4
            seg2 = seg.copy()
            if c > 0:
                seg2[:m] = np.nan
            if c < k - 1:
                seg2[-m:] = np.nan
            v[i0:i1] = seg2 * _sign(name)
            ok = ~np.isnan(v)
            if ok.sum() > 20:
                vi = np.interp(np.arange(n), np.nonzero(ok)[0], v[ok])
                hp = vi - median_filter(vi, size=61, mode="nearest")  # remove deriva (~0.6 s)
                hp[~ok] = np.nan
            else:
                hp = v
            self._hp[key] = hp
        return self._hp[key]

    # --------------------------------------------------------------------- OCR
    def _trace_mask(self):
        H, W = self.B.shape
        m = np.zeros((H, W), np.uint8)
        for q in self.qtr:
            for i, (t, b) in enumerate(zip(q["top"], q["bot"])):
                if not np.isnan(t):
                    x = q["x0"] + i
                    if 0 <= x < W:
                        m[max(0, int(t) - 2):min(H, int(b) + 3), x] = 1
        for p in self.pulses:  # pulsos não são texto
            if p is not None:
                top = min(p["top"], p.get("top2", p["top"]))
                m[max(0, int(top) - 3):int(p["foot"]) + 4, max(0, p["x_start"] - 3):p["x_end"] + 4] = 1
        return cv2.dilate(m, np.ones((3, 3), np.uint8)) > 0

    def _read_words(self):
        R = self.R
        # texto = componentes pequenas e isoladas (não encostam no traço) + tinta fora do
        # caminho do traço (rótulo que encosta num QRS).
        Bt = self.B
        n_, lab, st, _ = cv2.connectedComponentsWithStats(Bt.astype(np.uint8), connectivity=8)
        small = (st[:, cv2.CC_STAT_WIDTH] < 25 * R) & (st[:, cv2.CC_STAT_HEIGHT] < 9 * R)
        small[0] = False
        tm = self._trace_mask()
        pulse_m = np.zeros_like(tm)
        for p in self.pulses:
            if p is not None:
                top = min(p["top"], p.get("top2", p["top"]))
                pulse_m[max(0, int(top) - 3):int(p["foot"]) + 4, max(0, p["x_start"] - 3):p["x_end"] + 4] = True
        iso = small[lab] & ~pulse_m
        txt = (iso | (Bt & ~tm)) & ~pulse_m
        words = []
        for w in ocr.find_words(txt, R):
            x0, y0, x1, y1 = w
            crop = txt[y0:y1, x0:x1]
            d = ocr.read_label(crop)
            if not d:
                continue
            best = max(d.values())
            if best < 0.35:
                continue
            isolated = iso[y0:y1, x0:x1].sum() >= 0.8 * crop.sum()
            single = list(d) == ["I"]
            if single and not isolated:  # barra vertical colada ao traço = fragmento de QRS
                continue
            cy = 0.5 * (y0 + y1)
            r = int(np.argmin([abs(cy - y) for y in self.rows]))
            dk = float(np.percentile(self.D[y0:y1, x0:x1][crop], 90))
            words.append(dict(box=w, row=r, x=x0, dist=d, best=best, h=y1 - y0, single=single, dark=dk))
        # consenso de altura: todos os rótulos de uma página usam a mesma fonte
        conf = [w for w in words if not w["single"] and w["best"] >= 0.55]
        conf_ok = len(conf) >= 3
        # barras soltas ("I") muito maiores ou menores que os rótulos da página são pulso ou QRS
        multi = [w["h"] for w in words if not w["single"] and w["best"] >= 0.45]
        if len(multi) >= 2:
            hm = float(np.median(multi))
            words = [w for w in words if not w["single"] or 0.7 * hm <= w["h"] <= 1.35 * hm]
        hl = None
        if conf_ok:
            hl = float(np.median([w["h"] for w in conf]))
            dl = float(np.median([w["dark"] for w in conf]))
            # mesma fonte (altura) e mesma tinta (intensidade): vazamento de grade e
            # fragmentos de traço são mais claros que o texto impresso da própria página
            words = [w for w in words if 0.7 * hl <= w["h"] <= 1.35 * hl and w["dark"] >= 0.8 * dl]
        # palavra pertence à linha cuja faixa a contém (rótulo acima OU abaixo do traço)
        # 2ª passada local: linha sem rótulo no início -> abertura 3x3 na janela esperada
        # (remove fiapos finos de grade grudados nas letras)
        H, W = txt.shape
        for r in range(self.n):
            xs = float(self.x_of(r, 0.0))
            if any(w["row"] == r and xs - 25 * R <= w["x"] <= xs + 15 * R for w in words):
                continue
            y = self.rows[r]
            up = (y - self.rows[r - 1]) if r > 0 else 30 * R
            dn = (self.rows[r + 1] - y) if r + 1 < self.n else 30 * R
            ya, yb = max(0, int(y - 0.5 * up)), min(H, int(y + 0.5 * dn))
            xa, xb = max(0, int(xs - 25 * R)), min(W, int(xs + 15 * R))
            win = cv2.morphologyEx((txt[ya:yb, xa:xb]).astype(np.uint8), cv2.MORPH_OPEN,
                                   np.ones((3, 3), np.uint8))
            # restos de grade viram pontinhos: remove componentes muito menores que uma letra
            hl_ = hl if conf_ok else 2.5 * R
            n2, l2, st2, _ = cv2.connectedComponentsWithStats(win, connectivity=8)
            keep = st2[:, cv2.CC_STAT_AREA] >= max(25, 0.06 * hl_ * hl_)
            keep[0] = False
            win = keep[l2]
            for w in ocr.find_words(win, R):
                x0, y0, x1, y1 = w
                d = ocr.read_label(win[y0:y1, x0:x1])
                if not d or max(d.values()) < 0.4 or list(d) == ["I"]:
                    continue
                box = (x0 + xa, y0 + ya, x1 + xa, y1 + ya)
                dk = float(np.percentile(self.D[box[1]:box[3], box[0]:box[2]][win[y0:y1, x0:x1]], 90))
                words.append(dict(box=box, row=r, x=box[0], dist=d, best=max(d.values()), h=y1 - y0,
                                  single=False, dark=dk))
        return words

    def label_near(self, r, x_mm_lo, x_mm_hi, x_ref_mm=None):
        """Palavra da linha r cuja borda esquerda cai em [lo, hi] mm; entre várias, a de
        maior escore ponderado pela proximidade de x_ref (início do segmento)."""
        R = self.R
        out, best = None, -1.0
        for w in self.words:
            x0 = w["box"][0]
            if w["row"] != r or not (x_mm_lo * R <= x0 <= x_mm_hi * R):
                continue
            sc = w["best"]
            if w.get("single"):
                # barra solta lida como "I": fragmento de QRS, aresta de pulso ou uma barra
                # de "II" que sobrou; perde para uma leitura de vários caracteres ao lado
                sc *= 0.7
            if x_ref_mm is not None:
                sc *= float(np.exp(-abs(x0 / R - x_ref_mm) / 8.0))
            if sc > best:
                out, best = w, sc
        return out

    # -------------------------------------------------------------- fronteiras
    def _boundary_evidence(self, r, q):
        """[0,1]: marca vertical que não é traço, ou rótulo logo após a posição."""
        R = self.R
        x = float(self.x_of(r, q * self.duration))
        H, W = self.B.shape
        y = self.base_r[r]
        e = 0.0
        w = self.label_near(r, (x - 1.5 * R) / R, (x + 8 * R) / R)
        if w is not None:
            e = max(e, min(1.0, w["best"] / 0.6))
        # marca vertical: tinta fora do traço numa coluna perto de x cruzando a base
        tm = getattr(self, "_tm", None)
        if tm is None:
            tm = self._tm = self._trace_mask_thin()
        if BND_OTHER_ROWS[0]:
            # QRS alto de uma linha vizinha cruza esta linha: é traço, não marca de fronteira
            st = getattr(self, "_steep", None)
            if st is None:
                st = self._steep = [self._trace_mask_steep(k) for k in range(self.n)]
            tm = tm.copy()
            for k in range(self.n):
                if k != r:
                    tm |= st[k]
        a, b = max(0, int(y - 4 * R)), min(H, int(y + 4 * R))
        xa, xb = max(0, int(x - 1.0 * R)), min(W, int(x + 1.0 * R) + 1)
        if xb > xa and b > a:
            sub = self.B[a:b, xa:xb] & ~tm[a:b, xa:xb]
            col = sub.sum(0).max() if sub.size else 0
            e = max(e, min(1.0, col / (4.0 * R)))
        return e

    def _trace_mask_steep(self, k):
        H, W = self.B.shape
        m = np.zeros((H, W), bool)
        q = self.qtr[k]
        for i, (t, b) in enumerate(zip(q["top"], q["bot"])):
            if not np.isnan(t) and b - t >= 3 * self.R:
                x = q["x0"] + i
                if 0 <= x < W:
                    m[max(0, int(t) - 2):min(H, int(b) + 3), max(0, x - 2):x + 3] = True
        return m

    def _trace_mask_thin(self):
        H, W = self.B.shape
        m = np.zeros((H, W), bool)
        for q in self.qtr:
            for i, (t, b) in enumerate(zip(q["top"], q["bot"])):
                if not np.isnan(t) and b - t < 3 * self.R:  # só trechos não verticais
                    x = q["x0"] + i
                    if 0 <= x < W:
                        m[max(0, int(t) - 2):min(H, int(b) + 3), x] = True
        return m

    # --------------------------------------------------------------- pontuação
    def score(self, spec):
        """spec = lista de linhas (cada uma lista de nomes impressos). Retorna (total, partes)."""
        R = self.R
        parts = dict(ocr=0.0, bnd=0.0, limb=0.0, dup=0.0)
        segs = []
        for r, row in enumerate(spec):
            k = len(row)
            for c, name in enumerate(row):
                segs.append((r, k, c, name))
            # fronteiras
            want = set(np.round(FRACS.get(k, []), 4))
            for q, e in self.bnd[r].items():
                if round(q, 4) in want:
                    parts["bnd"] += e - 0.3
                elif e > 0.3:
                    parts["bnd"] -= e - 0.3
        # OCR
        for (r, k, c, name) in segs:
            xs = float(self.x_of(r, c / k * self.duration)) / R
            lo = xs - (25 if c == 0 else 2.0)
            w = self.label_near(r, lo, xs + LABEL_HI[0], x_ref_mm=xs)
            if w is None:
                continue
            d = w["dist"]
            # teto por segmento: uma leitura errada (sinal de menos perdido, dígito trocado)
            # custa no máximo 1 unidade; a física (Goldberger, duplicatas) decide o resto
            top = max(d, key=d.get)
            # I/II/III só diferem pelo número de barras: uma barra colada ao traço ou ao
            # pulso some e "II" vira "I". Confusão entre eles custa metade
            cap = 0.5 if (top in ("I", "II", "III") and _canon(name) in ("I", "II", "III")) else 1.0
            if w.get("single") and top == "I" and _canon(name) in ("II", "III"):
                cap = 0.0  # uma barra só não desmente II/III: as outras podem ter colado no traço
            parts["ocr"] += max(-cap, (d.get(name, 0.0) - d[top]) / 0.15)
        # rótulo confiável sem segmento que o explique: a hipótese ignora uma derivação impressa
        # (p.ex. 6x1 numa página 6x2 com "V1".."V6" no meio da linha)
        if _UNEXPLAINED[0]:
            starts = {}
            for (r, k, c, name) in segs:
                xs = float(self.x_of(r, c / k * self.duration)) / R
                starts.setdefault(r, []).append((xs - (25 if c == 0 else 2.0), xs + LABEL_HI[0]))
            for w in self.words:
                if w.get("single") or w["best"] < 0.55:
                    continue
                x0 = w["box"][0] / R
                if not any(lo <= x0 <= hi for lo, hi in starts.get(w["row"], [])):
                    parts["ocr"] -= 1.0
        # membros
        sig = {}
        has_grid = any(len(row) > 1 for row in spec)
        rhythm = []
        for (r, k, c, name) in segs:
            v = self.seg_signal(r, k, c, name)
            if has_grid and k == 1:
                rhythm.append((v, name))
            else:
                sig.setdefault(_canon(name), []).append((v, name))
        # tira de ritmo só entra na física se for simultânea à cópia da grade: algumas
        # páginas (montagens, impressões de outro trecho) trazem um ritmo de outro instante,
        # e aí nem Goldberger nem a duplicata dizem nada sobre o nome
        for v, name in rhythm:
            cps = sig.get(_canon(name), [])
            if cps:
                with np.errstate(all="ignore"), warnings.catch_warnings():
                    warnings.simplefilter("ignore", RuntimeWarning)
                    a = np.nanmean(np.stack([x for x, _ in cps]), 0)
                m = ~np.isnan(a) & ~np.isnan(v)
                if m.sum() >= 50 and a[m].std() > 0 and v[m].std() > 0:
                    nr = np.linalg.norm(a[m] - v[m]) / max(np.linalg.norm(a[m]), np.linalg.norm(v[m]), 1e-9)
                    # derivação quase plana (< ~80 µV RMS): o NRMSE é só ruído de quantização
                    flat = min(np.std(a[m]), np.std(v[m])) < 0.08
                    if nr > 0.85 and not flat:
                        parts["dup"] -= 0.5
                        continue
            sig.setdefault(_canon(name), []).append((v, name))
        limb_names = [n for n in sig if is_limb(n)]
        if len(limb_names) >= 3:
            T = len(self.t)
            Vs, As = [], []
            for nme in limb_names:
                arr = np.nanmean(np.stack([s for s, _ in sig[nme]]), 0) if len(sig[nme]) > 1 else sig[nme][0][0]
                Vs.append(arr); As.append(LIMB_COEF[nme])
            V = np.stack(Vs); A = np.array(As)
            T = len(self.t)
            ratio, ntest = _limb_ratio(V[:, ::2], A)
            if STRIPS3[0] and not has_grid and len(spec) <= 3 and np.isfinite(ratio):
                # tiras inteiras: a física tem de fechar em cada quarto da página. Numa 3x4 lida
                # como 3 tiras o 1º quarto é mesmo I/II/III e fecha; os outros (aVR, V1, V4...) não
                Vq = V[:, ::2]; qn = Vq.shape[1] // 4
                rq = [_limb_ratio(Vq[:, i * qn:(i + 1) * qn], A) for i in range(4)]
                rq = [x for x, nt in rq if nt > 20 and np.isfinite(x)]
                if len(rq) >= 3:
                    ratio = max(ratio, max(rq))
            if ntest > 20 and np.isfinite(ratio):
                ref = 0.45
                if LIMB_LOG[0]:
                    # verossimilhança: resíduo de Einthoven/Goldberger pequeno é quase
                    # impossível por acaso com os nomes errados; a evidência cresce como log
                    # piso de 0,2: abaixo disso as hipóteses já cumprem a física e a diferença é
                    # ruído (Cabrera e padrão com as mesmas derivações empatam)
                    lr = float(np.clip(np.log(ref / max(ratio, 0.2)), -1.0, 1.5))
                    parts["limb"] = lr * min(1.0, ntest / (0.25 * T))
                else:
                    parts["limb"] = (ref - ratio) * min(1.0, ntest / (0.25 * T))
        # duplicatas
        for nme, lst in sig.items():
            for i in range(len(lst)):
                for j in range(i + 1, len(lst)):
                    a, b = lst[i][0], lst[j][0]
                    m = ~np.isnan(a) & ~np.isnan(b)
                    if m.sum() < 50 or a[m].std() == 0 or b[m].std() == 0:
                        continue
                    if max(a[m].std(), b[m].std()) < 0.08:
                        continue  # duas cópias quase planas: o NRMSE é só ruído, não é evidência
                    # erro RMS normalizado: sensível a forma E amplitude (derivações
                    # vizinhas têm correlação alta, mas amplitudes diferentes)
                    na, nb = np.linalg.norm(a[m]), np.linalg.norm(b[m])
                    nrmse = np.linalg.norm(a[m] - b[m]) / max(na, nb, 1e-9)
                    parts["dup"] += 0.5 - nrmse
        # tiras de ritmo de um mesmo exame não repetem a mesma derivação (nenhum formato
        # de fabricante imprime, p.ex., II duas vezes como ritmo)
        rhythm = [_canon(row[0]) for row in spec if len(row) == 1]
        if any(len(row) > 1 for row in spec) and len(rhythm) > len(set(rhythm)):
            parts["rep"] = -(len(rhythm) - len(set(rhythm)))
        total = W_OCR * parts["ocr"] + W_BND * parts["bnd"] + W_LIMB * parts["limb"] + W_DUP * parts["dup"] \
            + 3.0 * parts.get("rep", 0.0)
        return total, parts

    # ------------------------------------------------------------- hipóteses
    def _best_rhythm_names(self, grid_spec, rest_rows):
        """Nomeia cada tira de ritmo (linha inteira, 1 derivação) pelo melhor escore conjunto."""
        spec = [list(r) for r in grid_spec]
        rest_rows = list(rest_rows)
        # convenção dos fabricantes: 1 tira de ritmo = II (GE, Philips, Mortara, Schiller);
        # 2 = V1 e II; 3 = V1, II, V5. Prior moderado: rótulo lido ou a física podem vencer.
        canon = {1: [{"II"}], 2: [{"V1", "II"}, {"V1", "II"}], 3: [{"V1", "II", "V5"}] * 3}.get(
            len(rest_rows), [set()] * len(rest_rows))
        for j, r in enumerate(rest_rows):
            best, best_n = -np.inf, None
            for nm in LEAD_NAMES:
                sc, parts_ = self.score(spec + [[nm]])
                # o nome da tira vem do rótulo, da comparação direta com a mesma derivação na
                # grade e da convenção; Goldberger (página inteira) não entra: com tira não
                # simultânea ou traço ruim ele puxa para III/aVF/V3 sem motivo
                sc -= W_LIMB * parts_["limb"]
                sc += 1.5 if nm in canon[j] else 0.0
                # o escore inclui a grade; só a diferença entre nomes importa
                if sc > best:
                    best, best_n = sc, nm
            spec.append([best_n])
        return spec

    def candidates(self):
        out = []
        for fam in FAMILIES:
            grid = LIBRARY[fam]
            if len(grid) > self.n:
                continue
            rest = self.n - len(grid)
            if (fam.startswith("12x1") or fam.startswith("6x1") or fam.startswith("3x1")) and rest:
                continue
            if fam.startswith("3x1") and not STRIPS3[0]:
                continue
            if fam in ("3x1-v13", "3x1-v46"):
                # precordiais não têm física entre si (nada fecha como Einthoven): só com rótulo
                # V lido no início de alguma linha; sem isso a hipótese vence só por não ter
                # termo de física a perder
                vs = 0
                for r in range(self.n):
                    xs = float(self.x_of(r, 0.0)) / self.R
                    w = self.label_near(r, xs - 25, xs + LABEL_HI[0])
                    if w and max(w["dist"], key=w["dist"].get).startswith("V"):
                        vs += 1
                if vs < 2:
                    continue
            if rest > 3:
                continue
            spec = self._best_rhythm_names(grid, range(len(grid), self.n)) if rest else [list(r) for r in grid]
            name = fam if not rest else f"{fam.split('-')[0]}+{rest}" + ("-cabrera" if "cabrera" in fam else "")
            out.append((name, spec))
        # hipótese livre: k por linha pelas fronteiras, nomes pelo OCR
        free = []
        for r in range(self.n):
            bestk, bests = 1, -np.inf
            for k, fr in FRACS.items():
                sc = sum((e - 0.3) if any(abs(q - f) < 1e-3 for f in fr) else -(max(e, 0.3) - 0.3)
                         for q, e in self.bnd[r].items())
                if sc > bests:
                    bestk, bests = k, sc
            row = []
            for c in range(bestk):
                xs = float(self.x_of(r, c / bestk * self.duration)) / self.R
                w = self.label_near(r, xs - (25 if c == 0 else 2.0), xs + 15)
                row.append(max(w["dist"], key=w["dist"].get) if w else f"?{r}{c}")
            free.append(row)
        if all(not n.startswith("?") for row in free for n in row):
            out.append(("livre", free))
        return out

    def choose(self):
        cands = self.candidates()
        scored = []
        for name, spec in cands:
            sc, parts = self.score(spec)
            if name != "livre":
                sc += 3.0  # prior para formatos de fabricante (a hipótese livre depende só do OCR)
                if "cabrera" in name:
                    sc -= 1.5  # Cabrera é regional (Escandinávia): precisa de alguma evidência
                    # sem "-aVR" lido em lugar nenhum da página, a física sozinha não basta:
                    # Goldberger fecha "quase" igual para várias permutações das derivações
                    if not any(w["dist"].get("-aVR", 0.0) >= 0.5 for w in self.words):
                        sc -= CABRERA_NO_EVIDENCE[0]
            else:
                # numa grade cada derivação aparece uma vez (tiras de ritmo são linhas de 1)
                grid_names = [nm for row in spec if len(row) > 1 for nm in row]
                sc -= 3.0 * (len(grid_names) - len(set(map(_canon, grid_names))))
            scored.append((sc, name, spec, parts))
        scored.sort(key=lambda z: -z[0])
        if not scored:
            return None
        best = scored[0]
        margin = best[0] - scored[1][0] if len(scored) > 1 else 10.0
        conf = float(1 / (1 + np.exp(-margin / 2.0)))
        warn = None
        n_lab = 0
        for r in range(self.n):  # rótulos legíveis no início de alguma linha
            xs = float(self.x_of(r, 0.0)) / self.R
            w = self.label_near(r, xs - 25, xs + 15)
            if w is not None and max(w["dist"].values()) >= 0.45:
                n_lab += 1
        if n_lab == 0 and len(scored) > 1:
            # sem nenhum rótulo legível, só a física decide; páginas de 6 tiras de
            # precordiais em ordem imitam o leque de Cabrera (fecham Goldberger "quase")
            sets = lambda sp: {_canon(n) for row in sp for n in row}
            shape = lambda sp: [len(row) for row in sp]
            alts = [z for z in scored[1:] if sets(z[2]) != sets(best[2])]
            alts = [z for z in alts if shape(z[2]) == shape(best[2])] or alts
            if alts:
                conf = min(conf, 0.55)
                warn = f"nenhum rótulo lido; {best[1]} decidido só pela fisiologia (alternativa: {alts[0][1]})"
        return dict(name=best[1], spec=best[2], score=best[0], parts=best[3], confidence=conf,
                    warning=warn, ranking=[(round(s, 2), n) for s, n, _, _ in scored[:6]])
