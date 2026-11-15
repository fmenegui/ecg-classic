"""
Geometria da página: segmentação do papel, mapa da grade, deskew, escala (px/mm),
retificação por homografia via fase local da grade + unwrapping guiado por qualidade,
e refinamento não-rígido (deslocamento polinomial) para papel curvado.

Ideia central: a grade milimetrada é um padrão periódico conhecido (1 mm / 5 mm).
Em cada bloco da imagem a fase do coeficiente de Fourier na frequência da grade dá
a posição sub-pixel das linhas. Desenrolando essas fases (como em interferometria /
InSAR) obtém-se correspondências imagem <-> coordenadas em mm, e daí a homografia.
"""
from __future__ import annotations

import heapq
import numpy as np
import cv2


# ----------------------------------------------------------------------------- utils

def _bg_max(ch: np.ndarray, win_frac: float = 0.015, small: int = 900) -> np.ndarray:
    """Fundo (brilho do papel) por filtro de máximo + mediana em baixa resolução."""
    H, W = ch.shape
    f = small / max(H, W)
    s = cv2.resize(ch, (max(8, int(W * f)), max(8, int(H * f))), interpolation=cv2.INTER_AREA)
    k = max(3, int(win_frac * max(H, W) * f) | 1)
    s = cv2.dilate(s, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    s = cv2.medianBlur(s.astype(np.float32), 5)
    s = cv2.GaussianBlur(s, (0, 0), k / 2)
    return cv2.resize(s, (W, H), interpolation=cv2.INTER_LINEAR)


PAPER_OTSU = [True]
RESCUE_CORR = [True]
PAPER_CROP = [True]


def _paper_otsu(lab, border):
    labb = cv2.GaussianBlur(lab, (0, 0), 1.5)
    db = np.linalg.norm(labb - np.median(border, 0), axis=2)
    d8 = np.clip(db * 255 / max(1.0, float(db.max())), 0, 255).astype(np.uint8)
    t, _ = cv2.threshold(d8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    fg = (d8 > t).astype(np.uint8)
    fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n, lab_, st, _ = cv2.connectedComponentsWithStats(fg)
    if n < 2:
        return None
    k = 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))
    comp = (lab_ == k).astype(np.uint8)
    cnts, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    c = max(cnts, key=cv2.contourArea)
    hull = cv2.convexHull(c)
    mm = np.zeros_like(comp)
    cv2.fillPoly(mm, [hull], 1)
    (_, _), (rw, rh), _ = cv2.minAreaRect(c)
    if comp.sum() < 0.95 * mm.sum() or mm.sum() < 0.9 * rw * rh or mm.mean() < 0.03:
        return None
    return mm


def paper_crop(img: np.ndarray, max_frac: float = 0.3, margin: float = 0.12):
    """Recorta a foto à folha quando ela ocupa pouco do quadro (fundo largo em volta, foto de
    longe). A margem deixa o fundo na borda do recorte: a máscara da folha depende disso
    (recorte justo faz a borda parecer papel e a página vira "scan", com fundo e tudo).
    Devolve (img, (x0, y0)) — a origem do recorte; sem recorte, (img, (0, 0))."""
    if not PAPER_CROP[0]:
        return img, (0, 0)
    m = paper_mask(img)
    if m.all():
        return img, (0, 0)
    ys, xs = np.nonzero(m)
    if not len(xs):
        return img, (0, 0)
    H, W = m.shape
    x0, x1, y0, y1 = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1
    if (x1 - x0) * (y1 - y0) > max_frac * H * W:
        return img, (0, 0)
    mx, my = int(margin * (x1 - x0)), int(margin * (y1 - y0))
    x0, y0 = max(0, x0 - mx), max(0, y0 - my)
    x1, y1 = min(W, x1 + mx), min(H, y1 + my)
    return img[y0:y1, x0:x1].copy(), (int(x0), int(y0))


PAPER_QUAD = [True]
# a borda da foto pode ser papel (folha encostada no quadro): pixels da borda com a cor do
# papel não viram semente de fundo do GrabCut (senão a margem branca vira fundo e a máscara
# abraça só a grade, cortando a 1ª linha ou um lado)
PAPER_SEEDFIX = [True]


def _quad_complete(mm, lab, mc, mb):
    """A folha é um quadrilátero. Quando ela encosta na borda da foto, a cor "de fundo" da
    borda inclui papel e o GrabCut corta um canto inteiro (o começo das primeiras linhas some).
    Os 4 lados mais longos da envoltória, prolongados, dão o quadrilátero; a área que ele
    acrescenta entra se tiver a cor do papel (e não a do fundo)."""
    h, w = mm.shape
    cnts, _ = cv2.findContours(mm, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return mm
    hull = cv2.convexHull(max(cnts, key=cv2.contourArea))[:, 0, :].astype(np.float64)
    hull = cv2.approxPolyDP(hull.astype(np.float32), 0.01 * max(h, w), True)[:, 0, :].astype(np.float64)
    n = len(hull)
    if n <= 4:
        return mm
    L = np.linalg.norm(np.roll(hull, -1, 0) - hull, axis=1)
    top = np.sort(np.argsort(L)[-4:])
    if L[top].sum() < 0.6 * L.sum():
        return mm  # contorno arredondado/irregular: não é um quadrilátero cortado
    pts = []
    for a, b in zip(top, np.roll(top, -1)):
        p1, p2 = hull[a], hull[(a + 1) % n]
        q1, q2 = hull[b], hull[(b + 1) % n]
        d1, d2 = p2 - p1, q2 - q1
        den = d1[0] * d2[1] - d1[1] * d2[0]
        if abs(den) < 1e-6 * (np.linalg.norm(d1) * np.linalg.norm(d2) + 1e-9):
            return mm
        t = ((q1[0] - p1[0]) * d2[1] - (q1[1] - p1[1]) * d2[0]) / den
        pts.append(p1 + t * d1)
    quad = np.array(pts)
    if np.abs(quad).max() > 3 * max(h, w):
        return mm
    mq = np.zeros_like(mm)
    cv2.fillPoly(mq, [np.round(quad).astype(np.int32)], 1)
    add = (mq > 0) & (mm == 0)
    if add.sum() < 0.005 * mm.sum() or add.sum() > 0.25 * mm.sum():
        return mm
    # canto cortado é um triângulo grosso; folha curvada dá só lascas finas ao longo dos lados
    # (os lados são arcos e o quadrilátero passa um pouco além deles, sobre a mesa)
    if float(cv2.distanceTransform(add.astype(np.uint8), cv2.DIST_L2, 3).max()) < 0.0175 * max(h, w):
        return mm
    dp = np.linalg.norm(lab[add] - mc, axis=1); db = np.linalg.norm(lab[add] - mb, axis=1)
    if float(np.mean(dp < db)) < 0.7:
        return mm
    return ((mq > 0) | (mm > 0)).astype(np.uint8)


def paper_mask(img: np.ndarray) -> np.ndarray:
    """Máscara da folha. Se a borda da imagem tem a mesma cor do centro, é um scan
    (tudo é papel). Senão: GrabCut (borda = fundo, centro = papel) + envoltória convexa
    da maior componente — robusto a sombras fortes que quebram limiarização global."""
    H, W = img.shape[:2]
    f = 400 / max(H, W)
    small = cv2.resize(img, (max(8, int(W * f)), max(8, int(H * f))), interpolation=cv2.INTER_AREA)
    h, w = small.shape[:2]
    lab = cv2.cvtColor(small, cv2.COLOR_BGR2LAB).astype(np.float32)
    m = max(2, int(0.03 * min(h, w)))
    border = np.concatenate([lab[:m].reshape(-1, 3), lab[-m:].reshape(-1, 3),
                             lab[:, :m].reshape(-1, 3), lab[:, -m:].reshape(-1, 3)])
    center = lab[h // 4:3 * h // 4, w // 4:3 * w // 4].reshape(-1, 3)
    mb, mc = np.median(border, 0), np.median(center, 0)
    dist = np.linalg.norm(mb - mc)
    # scan: borda tão clara quanto o centro (margem branca / grade) -> tudo é papel
    if dist < 20 or mb[0] >= mc[0] - 8:
        return np.ones((H, W), np.uint8)
    gc = np.full((h, w), cv2.GC_PR_FGD, np.uint8)
    gc[:m] = gc[-m:] = cv2.GC_BGD
    gc[:, :m] = gc[:, -m:] = cv2.GC_BGD
    if PAPER_SEEDFIX[0]:
        # cor do papel: os pixels mais claros do centro (fora da grade e do traço)
        cL = center[:, 0]
        pw = np.median(center[cL >= np.percentile(cL, 80)], 0)
        if np.linalg.norm(pw - mb) > 25:
            ring = np.zeros((h, w), bool)
            ring[:m] = ring[-m:] = True
            ring[:, :m] = ring[:, -m:] = True
            near = ring & (np.linalg.norm(lab - pw, axis=2) < 15)
            gc[near] = cv2.GC_PR_FGD
    # pixels da borda parecidos com a mediana da borda -> fundo provável em toda a imagem
    db = np.linalg.norm(lab - np.median(border, 0), axis=2)
    gc[(db < 12) & (gc == cv2.GC_PR_FGD)] = cv2.GC_PR_BGD
    gc[h // 3:2 * h // 3, w // 3:2 * w // 3] = cv2.GC_FGD
    bgm = np.zeros((1, 65), np.float64); fgm = np.zeros((1, 65), np.float64)
    try:
        cv2.setRNGSeed(12345)  # GrabCut usa k-means aleatório: fixa para ser determinístico
        cv2.grabCut(small, gc, None, bgm, fgm, 5, cv2.GC_INIT_WITH_MASK)
    except cv2.error:
        return np.ones((H, W), np.uint8)
    fg = ((gc == cv2.GC_FGD) | (gc == cv2.GC_PR_FGD)).astype(np.uint8)
    n, lab_, stats, _ = cv2.connectedComponentsWithStats(fg)
    if n < 2:
        return np.ones((H, W), np.uint8)
    k = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    comp = (lab_ == k).astype(np.uint8)
    cnts, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    hull = cv2.convexHull(max(cnts, key=cv2.contourArea))
    mm = np.zeros((h, w), np.uint8)
    cv2.fillPoly(mm, [hull], 1)
    solid = comp.sum() / max(1, mm.sum())
    if PAPER_OTSU[0] and (solid < 0.9 or mm.mean() < 0.15):
        # GrabCut vazou para um fundo texturizado (papelão, madeira) ou a folha é pequena na
        # foto: limiar de Otsu na distância de cor ao fundo da borda; aceita só um quadrilátero
        # compacto (folha inteira, sem buracos).
        mo = _paper_otsu(lab, border)
        if mo is not None:
            mm = mo
    if mm.mean() < (0.03 if PAPER_OTSU[0] else 0.15):  # segmentação implausível -> usa tudo
        return np.ones((H, W), np.uint8)
    if PAPER_QUAD[0]:
        mm = _quad_complete(mm, lab, mc, mb)
    mm = cv2.erode(mm, np.ones((3, 3), np.uint8))
    return cv2.resize(mm, (W, H), interpolation=cv2.INTER_NEAREST)


def grid_map(img: np.ndarray, paper: np.ndarray, clamp_ink: bool = True) -> np.ndarray:
    """Mapa de 'estrutura escura' normalizado pela iluminação, com a tinta atenuada."""
    f = img.astype(np.float32)
    gray = f.mean(2)
    mx = f.max(2)
    S = 1.0 - gray / (_bg_max(gray) + 1e-3)
    ink = 1.0 - mx / (_bg_max(mx) + 1e-3)
    G = np.clip(S, 0, 1)
    if clamp_ink:
        G = np.where(ink > 0.5, 0.25, np.minimum(G, 0.5))  # tinta vira constante (não periódica)
    G[paper == 0] = 0
    return G


# ------------------------------------------------------------------- deskew + escala

def estimate_rotation(G: np.ndarray) -> float:
    """Ângulo (graus) que, aplicado com cv2.getRotationMatrix2D, alinha a grade.
    Maximiza a variância das projeções (busca grossa e fina)."""
    H, W = G.shape

    def score_at(Gs, a):
        h, w = Gs.shape
        M = cv2.getRotationMatrix2D((w / 2, h / 2), a, 1.0)
        R = cv2.warpAffine(Gs, M, (w, h), flags=cv2.INTER_LINEAR)
        m = cv2.warpAffine(np.ones_like(Gs), M, (w, h), flags=cv2.INTER_NEAREST)
        rs = R.sum(1) / (m.sum(1) + 1)
        cs = R.sum(0) / (m.sum(0) + 1)
        return np.var(np.diff(rs)) + np.var(np.diff(cs))

    f1 = 700 / max(H, W)
    G1 = cv2.resize(G, (int(W * f1), int(H * f1)), interpolation=cv2.INTER_AREA)
    angs = np.arange(-45, 45, 0.5)
    sc = [score_at(G1, a) for a in angs]
    a0 = angs[int(np.argmax(sc))]
    f2 = min(1.0, 1600 / max(H, W))
    G2 = cv2.resize(G, (int(W * f2), int(H * f2)), interpolation=cv2.INTER_AREA)
    angs = np.arange(a0 - 0.6, a0 + 0.6, 0.04)
    sc = [score_at(G2, a) for a in angs]
    i = int(np.argmax(sc))
    if 0 < i < len(sc) - 1:  # interpolação parabólica
        d = sc[i - 1] - 2 * sc[i] + sc[i + 1]
        off = 0.5 * (sc[i - 1] - sc[i + 1]) / d if d != 0 else 0
        return float(angs[i] + off * 0.04)
    return float(angs[i])


def _profile_period(p: np.ndarray, lo: int = 3, hi: int = 400):
    """Autocorrelação normalizada de um perfil 1D."""
    p = p - cv2.blur(p.reshape(1, -1).astype(np.float32), (101, 1)).ravel()
    p = p - p.mean()
    n = len(p)
    F = np.fft.rfft(p, 2 * n)
    ac = np.fft.irfft(F * np.conj(F))[:n]
    ac /= ac[0] + 1e-9
    return ac[: min(hi, n - 1)]


def estimate_ppmm(G: np.ndarray, angle: float, paper: np.ndarray) -> tuple[float, bool]:
    """Pixels por mm a partir do período da grade. Retorna (ppmm, tem_linhas_grossas)."""
    H, W = G.shape
    M = cv2.getRotationMatrix2D((W / 2, H / 2), angle, 1.0)
    R = cv2.warpAffine(G, M, (W, H))
    pm = cv2.warpAffine(paper.astype(np.float32), M, (W, H))
    # autocorrelação média em blocos locais (robusta a perspectiva/rotação residual)
    ts = max(64, min(H, W) // 4)
    cands = []
    for fy in (0.3, 0.5, 0.7):
        for fx in (0.25, 0.5, 0.75):
            y0, x0 = int(fy * H - ts / 2), int(fx * W - ts / 2)
            if y0 < 0 or x0 < 0 or y0 + ts > H or x0 + ts > W:
                continue
            if pm[y0:y0 + ts, x0:x0 + ts].mean() < 0.95:
                continue
            blk = R[y0:y0 + ts, x0:x0 + ts]
            for axis in (0, 1):
                cands.append(_profile_period(blk.mean(axis).astype(np.float32), hi=ts // 2))
    if not cands:
        for axis in (0, 1):
            prof = R.sum(axis) / (pm.sum(axis) + 1)
            cands.append(_profile_period(prof.astype(np.float32)))
    ac = np.mean([c[: min(len(c) for c in cands)] for c in cands], 0)
    acs = cv2.GaussianBlur(ac.reshape(1, -1), (0, 0), 0.7).ravel()
    n = len(acs)
    z = np.nonzero(acs < 0)[0]
    start = int(z[0]) if len(z) else 3
    pk = [i for i in range(max(3, start), n - 1) if acs[i] > acs[i - 1] and acs[i] >= acs[i + 1]]
    if not pk:
        return float(max(H, W) / 280.0), False

    def refine(i):
        i = int(round(i))
        i = max(1, min(i, n - 2))
        j = i + int(np.argmax(ac[i - 1:i + 2])) - 1
        j = max(1, min(j, n - 2))
        d = ac[j - 1] - 2 * ac[j] + ac[j + 1]
        return j + (0.5 * (ac[j - 1] - ac[j + 1]) / d if d != 0 else 0)

    def prom(x, half):
        """Proeminência de um pico local perto de x em relação aos vales a +-half."""
        x = int(round(x)); h = max(1, int(round(half)))
        if x - h < 0 or x + h >= n:
            return -1.0
        top = acs[max(0, x - 1):x + 2].max()
        return float(top - 0.5 * (acs[x - h:x].min() + acs[x + 1:x + h + 1].min()))

    # pico mais forte = período dominante L (normalmente a linha grossa de 5 mm)
    L = refine(max(pk, key=lambda i: acs[i]))
    # se há pico forte em L/2, L é múltiplo do período
    # se há picos fortes em todos os m*L/k (k = 2, 3), L é múltiplo do período
    changed = True
    while changed:
        changed = False
        pL = prom(L, L / 4)
        for k in (2, 3):
            if L / k < 3:
                continue
            ok = all(prom(m * L / k, L / (2 * k)) > 0.4 * pL and
                     acs[int(round(m * L / k))] > 0.3 * acs[int(round(L))] for m in range(1, k))
            if ok:
                L = refine(L / k)
                changed = True
                break
    # linhas finas: picos em k*L/5 (k = 1..4) com proeminência
    pr5 = np.mean([prom(k * L / 5, L / 10) for k in range(1, 5)]) if L / 5 >= 2.5 else -1
    if pr5 > 0.04:
        return float(L / 5.0), True
    # grade uniforme: prior do tamanho da folha (80..600 mm visíveis)
    width_mm_if_minor = max(H, W) / L
    if 80 <= width_mm_if_minor <= 600:
        return float(L), False
    return float(L / 5.0), False


# ------------------------------------------------------ fase local + unwrapping

def _tile_phases(Gc: np.ndarray, T: float, tile: int, step: int):
    """Para cada bloco: posição das linhas (mod T) em x e y + qualidade."""
    H, W = Gc.shape
    ys = list(range(0, max(1, H - tile + 1), step))
    xs = list(range(0, max(1, W - tile + 1), step))
    j = np.arange(tile)
    e = np.exp(-2j * np.pi * j / T)
    out = np.zeros((len(ys), len(xs), 6), np.float64)  # phx, phy, qx, qy, cx, cy
    for a, y in enumerate(ys):
        for b, x in enumerate(xs):
            blk = Gc[y:y + tile, x:x + tile]
            px = blk.mean(0); py = blk.mean(1)
            px = px - px.mean(); py = py - py.mean()
            cxp = (px * e).sum(); cyp = (py * e).sum()
            nx = np.sqrt((px ** 2).sum() * tile) + 1e-9
            ny = np.sqrt((py ** 2).sum() * tile) + 1e-9
            # posição da linha: onde cos(2*pi*j/T - arg) é máximo -> j = arg*T/2pi
            ox = (-np.angle(cxp) * T / (2 * np.pi)) % T
            oy = (-np.angle(cyp) * T / (2 * np.pi)) % T
            out[a, b] = [x + ox, y + oy, abs(cxp) / nx, abs(cyp) / ny, x + tile / 2, y + tile / 2]
    return out


def _unwrap_quality(ph: np.ndarray, q: np.ndarray, T: float, qmin: float) -> np.ndarray:
    """Unwrapping 2D guiado por qualidade (flood fill priorizado)."""
    h, w = ph.shape
    out = np.full((h, w), np.nan)
    ok = q >= qmin
    if not ok.any():
        return out
    s = np.unravel_index(np.argmax(np.where(ok, q, -1)), q.shape)
    out[s] = ph[s]
    heap = []

    def push(i, j):
        for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            a, b = i + di, j + dj
            if 0 <= a < h and 0 <= b < w and ok[a, b] and np.isnan(out[a, b]):
                heapq.heappush(heap, (-q[a, b], a, b, i, j))

    push(*s)
    while heap:
        _, a, b, i, j = heapq.heappop(heap)
        if not np.isnan(out[a, b]):
            continue
        d = ph[a, b] - out[i, j]
        d -= T * np.round(d / T)
        out[a, b] = out[i, j] + d
        push(a, b)
    return out


def _correspondences(Gc, T_major, T_minor, use_major, tile, step):
    """Correspondências (pontos na imagem Gc) <-> (coordenadas ideais da grade)."""
    T = T_major if use_major else T_minor
    P = _tile_phases(Gc, T, tile, step)
    phx, phy, qx, qy, cx, cy = [P[..., k] for k in range(6)]
    qmin_x = max(0.08, np.percentile(qx, 40) * 0.6)
    qmin_y = max(0.08, np.percentile(qy, 40) * 0.6)
    ux = _unwrap_quality(phx, qx, T, qmin_x)
    uy = _unwrap_quality(phy, qy, T, qmin_y)
    if use_major:  # refina com a fase da linha fina (5x mais precisa)
        Pm = _tile_phases(Gc, T_minor, tile, step)
        fx, fy = Pm[..., 0], Pm[..., 1]
        ux = ux + (((fx - ux) + T_minor / 2) % T_minor - T_minor / 2) * (Pm[..., 2] > 0.15)
        uy = uy + (((fy - uy) + T_minor / 2) % T_minor - T_minor / 2) * (Pm[..., 3] > 0.15)
    good = ~np.isnan(ux) & ~np.isnan(uy)
    src = np.stack([cx[good], cy[good]], 1)              # pontos na imagem
    dst = np.stack([cx[good] - ux[good], cy[good] - uy[good]], 1)  # coords onde a grade cai em k*T
    w = np.minimum(qx[good], qy[good])
    return src, dst, w


def prewarp_quad(img: np.ndarray, min_dev_deg: float = 2.0):
    """Correção de perspectiva grosseira pelo contorno da área quadriculada.

    Em fotos inclinadas, a fase da grade de 5 mm deriva tanto ao longo da folha que o ajuste
    por fases (ambíguo a cada 5 mm) se perde. A área quadriculada é um retângulo no papel:
    acha a região de textura de grade (variância local), aproxima por um quadrilátero e
    leva-o a um retângulo. Só atua se o quadrilátero se afasta do retângulo (> min_dev_deg).
    Devolve (img_corrigida, H) ou (img, None)."""
    H0, W0 = img.shape[:2]
    f = 800.0 / max(H0, W0)
    sm = cv2.resize(img, (int(W0 * f), int(H0 * f)), interpolation=cv2.INTER_AREA)
    paper = paper_mask(sm)
    G = grid_map(sm, paper)
    k = max(5, int(0.02 * max(sm.shape[:2])) | 1)
    m1 = cv2.blur(G, (k, k)); m2 = cv2.blur(G * G, (k, k))
    sd = np.sqrt(np.clip(m2 - m1 * m1, 0, None))
    if sd.max() <= 0:
        return img, None
    from skimage.filters import threshold_otsu
    th = threshold_otsu(sd[paper > 0]) if (paper > 0).sum() > 100 else threshold_otsu(sd)
    mask = (sd > th).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((k, k), np.uint8))
    n, lab, st, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if n < 2:
        return img, None
    i = 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))
    if st[i, cv2.CC_STAT_AREA] < 0.15 * mask.size:
        return img, None
    cnts, _ = cv2.findContours((lab == i).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    hull = cv2.convexHull(max(cnts, key=cv2.contourArea))
    peri = cv2.arcLength(hull, True)
    quad = None
    for eps in np.linspace(0.01, 0.08, 15):
        ap = cv2.approxPolyDP(hull, eps * peri, True)
        if len(ap) == 4:
            quad = ap.reshape(4, 2).astype(np.float64)
            break
    if quad is None:
        return img, None
    # ordena: topo-esq, topo-dir, baixo-dir, baixo-esq
    c = quad.mean(0)
    ang = np.arctan2(quad[:, 1] - c[1], quad[:, 0] - c[0])
    quad = quad[np.argsort(ang)]
    tl = int(np.argmin(quad.sum(1)))
    quad = np.roll(quad, -tl, axis=0)
    def angle(a, b, cc):
        v1, v2 = a - b, cc - b
        return np.degrees(np.arccos(np.clip(v1 @ v2 / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-9), -1, 1)))
    angs = [angle(quad[(j - 1) % 4], quad[j], quad[(j + 1) % 4]) for j in range(4)]
    if max(abs(a - 90) for a in angs) < min_dev_deg:
        return img, None
    q = quad / f
    wt = 0.5 * (np.linalg.norm(q[1] - q[0]) + np.linalg.norm(q[2] - q[3]))
    ht = 0.5 * (np.linalg.norm(q[3] - q[0]) + np.linalg.norm(q[2] - q[1]))
    m = 0.08 * max(wt, ht)  # margem: rótulos, pulsos e a folha em volta continuam visíveis
    dst = np.array([[m, m], [m + wt, m], [m + wt, m + ht], [m, m + ht]], np.float64)
    Hq = cv2.getPerspectiveTransform(q.astype(np.float32), dst.astype(np.float32))
    out = cv2.warpPerspective(img, Hq, (int(wt + 2 * m), int(ht + 2 * m)), flags=cv2.INTER_CUBIC,
                              borderValue=(255, 255, 255))
    return out, Hq



def _refine_long_period(G, angle, ppmm, mm=(10.0, 20.0)):
    """Refina px/mm pelo pico da autocorrelação do perfil da página em 10 e 20 mm: o erro do
    período de 1 mm, multiplicado por 5, vira 5% de escala."""
    H, W = G.shape
    M = cv2.getRotationMatrix2D((W / 2, H / 2), angle, 1.0)
    Rg = cv2.warpAffine(G, M, (W, H))
    est = []
    for axis in (0, 1):
        p = Rg.mean(axis).astype(np.float64)
        p = p - p.mean()
        n = len(p)
        f = np.fft.rfft(p, 2 * n)
        ac = np.fft.irfft(f * np.conj(f))[:n]
        for L in mm:
            c = int(round(L * ppmm))
            h = max(2, int(0.08 * c))
            if c + h + 2 >= n:
                continue
            j = c - h + int(np.argmax(ac[c - h:c + h + 1]))
            d = ac[j - 1] - 2 * ac[j] + ac[j + 1]
            jj = j + (0.5 * (ac[j - 1] - ac[j + 1]) / d if d != 0 else 0)
            est.append(jj / L)
    if not est:
        return None
    v = float(np.median(est))
    return v if abs(v / ppmm - 1) < 0.1 else None

SCALE_X5 = [True]


def _homography_sane(Hm, shape, tol=0.35):
    """Escala local (det. do jacobiano) nos cantos dentro de um fator 3 da do centro, e sem
    ponto no infinito."""
    h, w = shape[:2]
    pts = [(0, 0), (w, 0), (w, h), (0, h), (w / 2, h / 2)]

    def jdet(x, y):
        v = Hm @ np.array([x, y, 1.0])
        if v[2] <= 1e-6:
            return None
        e = 1.0
        a = Hm @ np.array([x + e, y, 1.0]); b = Hm @ np.array([x, y + e, 1.0])
        p0 = v[:2] / v[2]; pa = a[:2] / a[2]; pb = b[:2] / b[2]
        return abs(np.linalg.det(np.stack([pa - p0, pb - p0])))
    d = [jdet(x, y) for x, y in pts]
    if any(x is None or x <= 0 for x in d):
        return False
    c = d[-1]
    # foto com perspectiva forte muda a escala local até ~2x de um canto ao outro; só um
    # fator > 3 (ou ponto no infinito) é degenerado
    return all(1 / 3.0 <= x / c <= 3.0 for x in d[:-1]) and 0.33 <= c <= 3.0


H_SANE = [False]
NONRIGID_CLIP = [True]
CLIP_MARGIN = [0.1]


def rectify(img: np.ndarray, R: float = 10.0, ppmm_hint: float | None = None, nonrigid: str = "cell",
            cell_min_mm: float = 0.06, major_snap: bool = True, fine_snap: bool = True,
            fine_min_mm: float = 0.06):
    """Retorna (img_ret, paper_ret, info). img_ret tem R px/mm e linhas grossas em múltiplos de 5R.

    nonrigid: "poly" (polinômio cúbico global), "cell" (polinômio + campo local por células de
    5 mm, interpolado bilinearmente — acompanha vincos) ou "none"."""
    paper = paper_mask(img)
    G = grid_map(img, paper)
    ang = estimate_rotation(G)
    ppmm, has_major = estimate_ppmm(G, ang, paper)
    rescued = False
    rescued_ok = False
    if ppmm_hint:  # escala conhecida (ex.: DPI do scanner / 25.4)
        ppmm = float(ppmm_hint)
    elif SCALE_X5[0] and max(img.shape[:2]) / ppmm > 600 and max(img.shape[:2]) / (5 * ppmm) >= 100:
        # imagem "maior que 60 cm": o passo de 1 mm foi tomado pelo de 5 mm. Acontece com grade
        # pontilhada preta: os pontos grossos são escuros como tinta e saem do mapa da grade.
        # Refaz o mapa sem atenuar a tinta, liga os pontos em linhas (fechamento de ~1 mm na
        # escala corrigida) e reestima rotação e escala
        p5 = 5.0 * ppmm
        k = int(1.3 * p5) | 1
        G0 = grid_map(img, paper, clamp_ink=False)
        G2 = np.maximum(cv2.morphologyEx(G0, cv2.MORPH_CLOSE, np.ones((1, k), np.uint8)),
                        cv2.morphologyEx(G0, cv2.MORPH_CLOSE, np.ones((k, 1), np.uint8)))
        G2 = np.minimum(G2, 0.5)
        ang2 = estimate_rotation(G2)
        pp2, hm2 = estimate_ppmm(G2, ang2, paper)
        ang, ppmm = ang2, 5.0 * pp2 if max(img.shape[:2]) / pp2 > 600 else pp2
        ppmm = _refine_long_period(G0, ang, ppmm) or ppmm
        rescued = True

    H, W = G.shape
    s = 5.0  # px/mm do quadro grosso
    M = cv2.getRotationMatrix2D((W / 2, H / 2), ang, s / ppmm)
    corners = np.array([[0, 0, 1], [W, 0, 1], [W, H, 1], [0, H, 1]], np.float64)
    pc = corners @ M.T
    tx, ty = -pc[:, 0].min(), -pc[:, 1].min()
    M[:, 2] += [tx, ty]
    Wc, Hc = int(np.ceil(np.ptp(pc[:, 0]))), int(np.ceil(np.ptp(pc[:, 1])))
    Gc = cv2.warpAffine(G, M, (Wc, Hc), flags=cv2.INTER_AREA if s < ppmm else cv2.INTER_LINEAR)

    tile, step = int(25 * s), int(10 * s)
    if not rescued:
        src, dst, w = _correspondences(Gc, 5 * s, s, has_major, tile, step)
    elif RESCUE_CORR[0]:
        # escala resgatada: a estimativa global errou o harmônico, mas na escala corrigida a
        # fase da linha de 1 mm (ou, se ela não aparece, a de 5 mm) ainda dá a perspectiva.
        # Fica a que tiver mais blocos de boa qualidade; a homografia só vale se explicar a
        # maioria deles (senão fica rotação + escala)
        c1 = _correspondences(Gc, 5 * s, s, False, tile, step)
        c5 = _correspondences(Gc, 5 * s, 5 * s, False, tile, step)
        src, dst, w = max((c1, c5), key=lambda c: float(np.sum(c[2])) if len(c[0]) else 0.0)
    else:
        src, dst, w = [], [], []
    A3 = np.vstack([M, [0, 0, 1]])
    if len(src) >= 8:
        Hm, inl = cv2.findHomography(dst, src, cv2.RANSAC, 0.6 * s)
        if Hm is None or (rescued and (inl is None or inl.mean() < 0.6)):
            Hm = np.eye(3)
        elif rescued:
            rescued_ok = True
        if H_SANE[0] and not _homography_sane(Hm, Gc.shape):
            # homografia degenerada (RANSAC num conjunto de blocos mal desembrulhado): a escala
            # local muda > 35% de um canto ao outro. Fica a afim robusta, ou rotação + escala
            A2, _ = cv2.estimateAffine2D(dst, src, method=cv2.RANSAC, ransacReprojThreshold=0.6 * s)
            Hm = np.vstack([A2, [0, 0, 1]]) if A2 is not None else np.eye(3)
            if not _homography_sane(Hm, Gc.shape):
                Hm = np.eye(3)
    else:
        Hm = np.eye(3)
    # quadro retificado: u,v em mm*R. coarse(x,y) = Hm @ (u/R*s, v/R*s)
    Sc = np.diag([s / R, s / R, 1.0])
    T_total = np.linalg.inv(A3) @ Hm @ Sc  # retificado -> original

    # extensão: cantos do papel no quadro retificado
    Tinv = np.linalg.inv(T_total)
    ys, xs = np.nonzero(cv2.resize(paper, (W // 8 + 1, H // 8 + 1), interpolation=cv2.INTER_NEAREST))
    pts = np.stack([xs * 8.0, ys * 8.0, np.ones_like(xs, dtype=float)], 1) @ Tinv.T
    pts = pts[:, :2] / pts[:, 2:3]
    u0, v0 = np.floor(pts.min(0) / (5 * R)) * 5 * R - 5 * R
    u1, v1 = np.ceil(pts.max(0) / (5 * R)) * 5 * R + 5 * R
    Wo, Ho = int(u1 - u0), int(v1 - v0)
    if Wo * Ho > 40e6 or Wo > 12000 or Ho > 12000:
        # folha de mais de ~1 m: a escala (px/mm) foi mal estimada; seguir estouraria a memória
        raise RuntimeError(f"escala implausível: quadro retificado {Wo}x{Ho} px")
    Tsh = np.array([[1, 0, u0], [0, 1, v0], [0, 0, 1.0]])
    T_total = T_total @ Tsh  # pixel de saída -> original (linhas grossas em múltiplos de 5R)

    out = cv2.warpPerspective(img, T_total, (Wo, Ho), flags=cv2.INTER_CUBIC | cv2.WARP_INVERSE_MAP,
                              borderValue=(255, 255, 255))
    pap = cv2.warpPerspective(paper, T_total, (Wo, Ho), flags=cv2.INTER_NEAREST | cv2.WARP_INVERSE_MAP)

    # refinamento não-rígido: resíduo de fase -> polinômio cúbico -> remap
    rms, rms_cell, snap_mm, fine_mm = float("nan"), float("nan"), 0.0, 0.0
    snp = None
    if nonrigid in ("poly", "cell") and (not rescued or rescued_ok):
        out_p, pap_p, rms, mx, my = _refine_nonrigid(out, pap, R, return_map=True)
        cx_ = cy_ = None
        if nonrigid == "cell" and mx is not None:
            loc = _local_field(out_p, pap_p, R)
            # só aplica o campo local quando ele se destaca do papel plano: em scans e
            # sintéticos sem dobra o resíduo local fica em 0,01-0,04 mm (ruído de fase) e
            # aplicá-lo só mexe em decisões marginais; vincos e fotos dão 0,09-0,7 mm
            if loc is not None and loc[2] < cell_min_mm:
                rms_cell = loc[2]
                loc = None
            if loc is not None:
                lx, ly, rms_cell = loc
                H, W = out.shape[:2]
                gx, gy = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32))
                # composição dos mapas (uma única reamostragem da imagem original retificada)
                cx_ = cv2.remap(mx, gx + lx, gy + ly, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
                cy_ = cv2.remap(my, gx + lx, gy + ly, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
                out_p = cv2.remap(out, cx_, cy_, cv2.INTER_CUBIC, borderValue=(255, 255, 255))
                pap_p = cv2.remap(pap, cx_, cy_, cv2.INTER_NEAREST)
        # imagem de baixa resolução (< 6 px/mm): a linha grossa mal se distingue da fina e o
        # ajuste vira ruído
        if mx is not None and major_snap and ppmm >= 6.0:
            if cx_ is None:
                cx_, cy_ = mx, my
            snp = _major_snap(out_p, pap_p, R)
            if snp is not None:
                sx, sy, snap_mm = snp
                H, W = out.shape[:2]
                gx, gy = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32))
                cx2 = cv2.remap(cx_, gx + sx, gy + sy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
                cy2 = cv2.remap(cy_, gx + sx, gy + sy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
                out_p = cv2.remap(out, cx2, cy2, cv2.INTER_CUBIC, borderValue=(255, 255, 255))
                pap_p = cv2.remap(pap, cx2, cy2, cv2.INTER_NEAREST)
        if mx is not None and fine_snap and ppmm >= 6.0:
            if cx_ is None:
                cx_, cy_ = mx, my
            if snp is not None:
                cx_, cy_ = cx2, cy2
            fn = _major_snap(out_p, pap_p, R, band_mm=20.0, step_mm=5.0, search_mm=0.45,
                             min_rms_mm=fine_min_mm, max_rms_mm=0.5, period_mm=1.0, outlier_mm=0.15,
                             smooth=1.0, con_floor=0.006)
            if fn is not None:
                fx, fy, fine_mm = fn
                H, W = out.shape[:2]
                gx, gy = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32))
                cx3 = cv2.remap(cx_, gx + fx, gy + fy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
                cy3 = cv2.remap(cy_, gx + fx, gy + fy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
                out_p = cv2.remap(out, cx3, cy3, cv2.INTER_CUBIC, borderValue=(255, 255, 255))
                pap_p = cv2.remap(pap, cx3, cy3, cv2.INTER_NEAREST)
        out, pap = out_p, pap_p
    info = dict(angle=ang, ppmm=ppmm, has_major=has_major, n_corr=int(len(src)),
                residual_rms_mm=rms, residual_cell_rms_mm=rms_cell, major_snap_mm=snap_mm, fine_snap_mm=fine_mm, T=T_total)
    return out, pap, info


def _major_snap(out, pap, R, band_mm=20.0, step_mm=10.0, search_mm=2.2, min_rms_mm=0.15, max_rms_mm=0.8,
                period_mm=5.0, outlier_mm=0.6, smooth=0.8, con_floor=0.03):
    """Ajuste fino pelas LINHAS GROSSAS (5 mm), localizadas uma a uma.

    Depois do polinômio e do campo por células, a grade ainda pode estar fora do lugar numa
    faixa da folha: scans com o papel escorregando/emendado deslocam a grade em 0,3-1,5 mm
    (às vezes em degrau) e a fase de 1 mm é ambígua nesses valores. A linha grossa é
    inequívoca dentro de ±2,5 mm: em faixas de 20 mm, a mediana das linhas da imagem dá o
    perfil da grade sem o traço; o máximo perto de cada múltiplo de 5 mm é a linha.
    Com period_mm=1 (e busca < 0,5 mm) é o ajuste FINO: cada linha de 1 mm localizada, o
    equivalente clássico da grade indexada por interseção dos primeiros colocados do Kaggle.
    Retorna (dx, dy, rms_mm) densos (amostrar em x+dx, y+dy) ou None."""
    import warnings
    if out.ndim == 3:
        # escuridão no canal em que a grade mais aparece (grade vermelha/rosa: verde e azul);
        # a média dos canais quase apaga a linha fina
        dk = [np.clip(1.0 - out[..., c].astype(np.float32) / (_bg_max(out[..., c].astype(np.float32)) + 1e-3), 0, 1)
              for c in range(3)]
        S = np.max(dk, axis=0)
        # o traço (escuro em TODOS os canais) sai da medida: um QRS alto cobre boa parte da
        # faixa de 20 mm e a mediana o tomaria pela linha da grade, justo onde o tempo mais
        # importa. Em grade preta (P&B) não dá para separar por cor: fica como está
        # traço = escuro e neutro (preto/cinza); linha de grade = colorida (vermelha/rosa:
        # escura em verde/azul, clara em vermelho), mesmo quando a foto a escurece
        mn, mxk = np.min(dk, axis=0), np.max(dk, axis=0)
        ink = (mn > 0.35) & ((mxk - mn) < 0.25) & (pap > 0)
        if ink.sum() < 0.08 * max(1, int((pap > 0).sum())):
            ink = cv2.dilate(ink.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
            S = np.where(ink, np.nan, S)
    else:
        f = out.astype(np.float32)
        S = np.clip(1.0 - f / (_bg_max(f) + 1e-3), 0, 1)
    S[pap == 0] = np.nan
    H, W = S.shape
    T5, bh, st, sr = period_mm * R, int(band_mm * R), int(step_mm * R), int(round(search_mm * R))
    fields = []
    for axis in (0, 1):  # 0: linhas verticais (dx); 1: linhas horizontais (dy)
        A = S if axis == 0 else S.T
        n_par, n_len = A.shape  # bandas ao longo de n_par, perfil ao longo de n_len
        b0s = list(range(0, max(1, n_par - bh + 1), st))
        ks = [k for k in range(1, int(n_len // T5)) if sr + 1 <= k * T5 < n_len - sr - 1]
        off = np.full((len(b0s), len(ks)), np.nan)
        con = np.zeros_like(off)
        for i, b0 in enumerate(b0s):
            band = A[b0:b0 + bh]
            if np.isnan(band).mean() > 0.5:
                continue
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                prof = np.nanmedian(band, axis=0)
            for j, k in enumerate(ks):
                x = int(round(k * T5))
                w = prof[x - sr:x + sr + 1]
                if np.isnan(w).any():
                    continue
                m = int(np.argmax(w))
                if 0 < m < len(w) - 1:
                    a, b, c = w[m - 1], w[m], w[m + 1]
                    den = a - 2 * b + c
                    sub = 0.5 * (a - c) / den if den < -1e-9 else 0.0
                else:
                    sub = 0.0
                off[i, j] = m - sr + float(np.clip(sub, -0.5, 0.5))
                con[i, j] = w[m] - np.median(w)
        cmed = float(np.nanmedian(con[np.isfinite(off)])) if np.isfinite(off).any() else 0.0
        ok = np.isfinite(off) & (con > max(con_floor, _SNAP_CON[0] * cmed))
        if ok.mean() < _SNAP_OKMIN[0]:
            return None
        off[~ok] = np.nan
        # vizinhos (3x3) que discordam muito: linha fina/traço/texto tomado pela grossa
        from scipy.ndimage import generic_filter
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            med = generic_filter(off, np.nanmedian, size=3, mode="nearest")
        off[np.abs(off - med) > outlier_mm * R] = np.nan
        ok = np.isfinite(off)
        if ok.mean() < _SNAP_OKMIN[0]:
            return None
        Wt = ok.astype(np.float32)
        sm = _fill_nan(off, Wt, smooth)
        cov = cv2.GaussianBlur(Wt, (0, 0), smooth)
        sm = np.where(cov > 1e-3, sm, 0.0).astype(np.float32)
        # amostras em (centro da banda, k*5R) -> campo denso por interpolação bilinear
        yc = np.array(b0s, float) + bh / 2
        xc = np.array(ks, float) * T5
        gy = np.clip(np.interp(np.arange(n_par), yc, np.arange(len(yc))), 0, len(yc) - 1)
        gx = np.clip(np.interp(np.arange(n_len), xc, np.arange(len(xc))), 0, len(xc) - 1)
        GX, GY = np.meshgrid(gx.astype(np.float32), gy.astype(np.float32))
        dense = cv2.remap(sm, GX, GY, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        fields.append((dense if axis == 0 else dense.T, off))
    rms = float(np.sqrt(np.nanmean(fields[0][1] ** 2) + np.nanmean(fields[1][1] ** 2)) / R)
    _major_snap.last = [dict(off=f[1], ok_frac=float(np.isfinite(f[1]).mean())) for f in fields]
    # < 0,15 mm: nada a corrigir além do ruído. > 0,8 mm depois do polinômio e das células não
    # é papel deslocado, é a linha fina sendo tomada pela grossa (imagem de baixa resolução,
    # grade sem linhas grossas distintas)
    if not (min_rms_mm <= rms <= max_rms_mm):
        return None
    return fields[0][0].astype(np.float32), fields[1][0].astype(np.float32), rms


import os as _os
_SNAP_CON = [float(_os.environ.get("ECGDIG_SNAP_CON", 0.5))]
_SNAP_OKMIN = [float(_os.environ.get("ECGDIG_SNAP_OKMIN", 0.4))]


def _fill_nan(F: np.ndarray, W: np.ndarray, sigma: float) -> np.ndarray:
    """Preenche/suaviza por convolução normalizada (média ponderada gaussiana dos vizinhos válidos)."""
    ok = ~np.isnan(F)
    num = cv2.GaussianBlur(np.where(ok, F * W, 0).astype(np.float32), (0, 0), sigma)
    den = cv2.GaussianBlur(np.where(ok, W, 0).astype(np.float32), (0, 0), sigma)
    return np.where(den > 1e-6, num / np.maximum(den, 1e-6), 0.0)


def _local_field(out, pap, R, cell_mm: float = 5.0, tile_mm: float = 10.0):
    """Resíduo local da grade de 1 mm numa malha de células de 5 mm, depois do polinômio.

    1. fase da linha de 1 mm em blocos de 10 mm centrados em cada vértice de 5 mm;
    2. desenrolamento 2D guiado por qualidade (período 1 mm) — o campo pode passar de ±0,5 mm
       ao atravessar um vinco, desde que vértices vizinhos difiram menos de 0,5 mm;
    3. limpeza à moda do Gridder: só a maior região conexa de vértices confiáveis, descarte de
       vértices que destoam da mediana 3×3, faltantes preenchidos pelos vizinhos;
    4. interpolação bilinear entre vértices = transformação de 4 cantos por célula.
    Retorna (dx, dy, rms_mm) densos, ou None se a grade local não for confiável."""
    G = grid_map(out, pap)
    H, W = G.shape
    step, tile = int(round(cell_mm * R)), int(round(tile_mm * R))
    P = _tile_phases(G, R, tile, step)
    phx, phy, qx, qy, cx, cy = [P[..., k] for k in range(6)]
    if P.shape[0] < 3 or P.shape[1] < 3:
        return None
    q = np.minimum(qx, qy)
    good = (qx > 0.25) & (qy > 0.25)
    if good.mean() < 0.2:
        return None
    # maior componente conexa de vértices confiáveis
    n, lab = cv2.connectedComponents(good.astype(np.uint8), connectivity=4)
    if n < 2:
        return None
    big = 1 + int(np.argmax(np.bincount(lab.ravel())[1:]))
    good = lab == big
    fields = []
    for ph, qq in ((phx, qx), (phy, qy)):
        d = (ph + R / 2) % R - R / 2                        # resíduo assinado (embrulhado)
        u = _unwrap_quality(d, np.where(good, qq, 0), R, 1e-6)
        u[~good] = np.nan
        u -= R * np.round(np.nanmedian(u) / R)              # o polinômio já zerou o resíduo médio
        # vértices que destoam dos vizinhos (fase saltou um período ou bloco contaminado)
        win = np.lib.stride_tricks.sliding_window_view(np.pad(u, 1, constant_values=np.nan), (3, 3))
        with np.errstate(all="ignore"):
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                med = np.nanmedian(win.reshape(*u.shape, 9), axis=2)
        bad = np.abs(u - med) > 0.25 * R
        u[bad] = np.nan
        fields.append((u, u - med))
    w = np.where(good, q, 0).astype(np.float32)
    rms = float(np.sqrt(np.nanmean(fields[0][0] ** 2 + fields[1][0] ** 2)) / R)
    miss = cv2.GaussianBlur(w, (0, 0), 0.7) < 1e-3
    sm = [np.where(miss, 0.0, _fill_nan(u, w, 0.7)) for u, _ in fields]   # suaviza ruído de fase
    # Encolhimento (Wiener) pelo ruído do próprio campo: em papel plano o resíduo depois do
    # polinômio é < 0,01 mm e o campo medido é quase só ruído — e 0,03 mm de erro horizontal num
    # QRS íngreme já custa ~0,06 mV. Só se aplica o que se destaca do ruído (vincos: 0,3–2 mm).
    rough = np.concatenate([r[np.isfinite(r)] for _, r in fields])
    sigma = 1.4826 * np.median(np.abs(rough)) if rough.size else 0.0
    tau = max(3.0 * sigma, 0.05 * R)
    mag2 = sm[0] ** 2 + sm[1] ** 2
    gain = cv2.GaussianBlur((mag2 / (mag2 + tau ** 2)).astype(np.float32), (0, 0), 0.7)
    dense = []
    for f in sm:
        f = f * gain
        # amostras nos centros dos blocos -> campo denso bilinear
        x0, y0 = cx[0, 0], cy[0, 0]
        Fd = cv2.resize(f.astype(np.float32), (f.shape[1] * step, f.shape[0] * step),
                        interpolation=cv2.INTER_LINEAR)
        # alinha: amostra (i, j) fica no pixel (y0 + i*step, x0 + j*step)
        Mt = np.float32([[1, 0, x0 - (step - 1) / 2], [0, 1, y0 - (step - 1) / 2]])
        dense.append(cv2.warpAffine(Fd, Mt, (W, H), borderMode=cv2.BORDER_REPLICATE))
    return dense[0], dense[1], rms


def _poly_terms(u, v, deg=3):
    return np.stack([u ** i * v ** j for i in range(deg + 1) for j in range(deg + 1 - i)], 1)


def _refine_nonrigid(out, pap, R, return_map=False):
    G = grid_map(out, pap)
    tile, step = int(20 * R), int(8 * R)
    P = _tile_phases(G, R, tile, step)  # período fino = 1 mm
    phx, phy, qx, qy, cx, cy = [P[..., k].ravel() for k in range(6)]
    # resíduo assinado em torno de 0 (as linhas deveriam estar em múltiplos de R)
    dx = (phx + R / 2) % R - R / 2
    dy = (phy + R / 2) % R - R / 2
    good = (qx > 0.2) & (qy > 0.2)
    if good.sum() < 15:
        return (out, pap, float("nan"), None, None) if return_map else (out, pap, float("nan"))
    H, W = G.shape
    un, vn = cx[good] / W - 0.5, cy[good] / H - 0.5
    A = _poly_terms(un, vn)
    wts = np.minimum(qx[good], qy[good])
    sol = []
    for d in (dx[good], dy[good]):
        c = np.linalg.lstsq(A * wts[:, None], d * wts, rcond=None)[0]
        # uma iteração robusta (descarta resíduos grandes)
        r = d - A @ c
        k = np.abs(r) < 3 * (np.median(np.abs(r)) * 1.4826 + 0.05)
        c = np.linalg.lstsq(A[k] * wts[k, None], d[k] * wts[k], rcond=None)[0]
        sol.append(c)
    rms_before = float(np.sqrt(np.mean(dx[good] ** 2 + dy[good] ** 2)) / R)
    yy, xx = np.mgrid[0:H:8, 0:W:8].astype(np.float32)
    At = _poly_terms((xx / W - 0.5).ravel(), (yy / H - 0.5).ravel())
    fx_, fy_ = At @ sol[0], At @ sol[1]
    # o polinômio deforma a folha (escala local fora de 0,7-1,4 em > 2% do papel)?
    def _bad(fx, fy):
        gx_ = (xx.ravel() + fx).reshape(xx.shape); gy_ = (yy.ravel() + fy).reshape(yy.shape)
        dxu, dxv = np.gradient(gx_, 8.0, axis=1), np.gradient(gx_, 8.0, axis=0)
        dyu, dyv = np.gradient(gy_, 8.0, axis=1), np.gradient(gy_, 8.0, axis=0)
        det = dxu * dyv - dxv * dyu
        pm_ = pap[::8, ::8][: det.shape[0], : det.shape[1]] > 0
        if not pm_.any():
            return False
        return float(np.mean((det[pm_] < 0.7) | (det[pm_] > 1.4))) > 0.02
    if NONRIGID_CLIP[0] and _bad(fx_, fy_):
        # o polinômio cúbico extrapola longe dos blocos bons (canto fora de foco, borda): o
        # deslocamento fica no intervalo que ele tem sobre os blocos medidos (o resíduo medido
        # é a fase mod 1 mm; nada fora desse intervalo foi observado)
        for f_, c_ in ((fx_, sol[0]), (fy_, sol[1])):
            v_ = A @ c_
            np.clip(f_, v_.min() - CLIP_MARGIN[0] * R, v_.max() + CLIP_MARGIN[0] * R, out=f_)
    mx = (xx.ravel() + fx_).reshape(xx.shape).astype(np.float32)
    my = (yy.ravel() + fy_).reshape(yy.shape).astype(np.float32)
    mx = cv2.resize(mx, (W, H), interpolation=cv2.INTER_LINEAR) + 0  # aproximação densa
    my = cv2.resize(my, (W, H), interpolation=cv2.INTER_LINEAR)
    # corrige o deslocamento de meio passo do resize
    gx, gy = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32))
    mx = gx + (mx - cv2.resize(xx, (W, H), interpolation=cv2.INTER_LINEAR))
    my = gy + (my - cv2.resize(yy, (W, H), interpolation=cv2.INTER_LINEAR))
    out2 = cv2.remap(out, mx, my, cv2.INTER_CUBIC, borderValue=(255, 255, 255))
    pap2 = cv2.remap(pap, mx, my, cv2.INTER_NEAREST)
    if return_map:
        return out2, pap2, rms_before, mx, my
    return out2, pap2, rms_before
