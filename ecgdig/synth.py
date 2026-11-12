"""
Gerador de ECGs sintéticos com ground truth exato, para validar o digitalizador.

- Sinal: modelo de dipolo cardíaco 3D (ondas P, Q, R, S, T gaussianas com vetores
  próprios) projetado nas 12 derivações -> as relações de Einthoven/Goldberger são
  satisfeitas por construção (II = I + III, aVR + aVL + aVF = 0).
- Imagem: papel milimetrado 25 mm/s, 10 mm/mV, layout 3x4 + ritmo (II), pulsos de
  calibração, rótulos, cabeçalho, e degradações (rotação, perspectiva, sombra,
  ruído, blur, JPEG).
"""
from __future__ import annotations

import numpy as np
import cv2

LEADS = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
FS = 500
DUR = 10.0

# Layout 3x4 padrão (linhas x colunas) + tira de ritmo
LAYOUT_3x4 = [["I", "aVR", "V1", "V4"],
              ["II", "aVL", "V2", "V5"],
              ["III", "aVF", "V3", "V6"]]


def _lead_vectors():
    """Vetores de derivação no espaço (x, y, z) do dipolo."""
    I = np.array([1.0, 0.0, 0.0])
    II = np.array([np.cos(np.pi / 3), np.sin(np.pi / 3), 0.0])
    III = II - I
    aVR = -(I + II) / 2
    aVL = I - II / 2
    aVF = II - I / 2
    vecs = [I, II, III, aVR, aVL, aVF]
    # precordiais no plano horizontal (x, z), z anterior
    for ang in [115, 95, 75, 55, 35, 15]:
        a = np.deg2rad(ang)
        vecs.append(np.array([np.cos(a), 0.15, np.sin(a)]) * 1.4)
    return np.stack(vecs)  # (12, 3)


def synth_signals(rng: np.random.Generator) -> np.ndarray:
    """Retorna array (12, 5000) em mV."""
    n = int(FS * DUR)
    t = np.arange(n) / FS
    hr = rng.uniform(50, 115)
    rr = 60.0 / hr
    # tempos dos batimentos com leve variabilidade
    beats = []
    tb = rng.uniform(0.1, rr)
    while tb < DUR + 0.5:
        beats.append(tb)
        tb += rr * rng.normal(1.0, 0.03)

    # ondas: (offset rel. R [s], largura [s], vetor 3D)
    def rv(scale):
        v = rng.normal(0, 1, 3)
        v /= np.linalg.norm(v) + 1e-9
        return v * scale

    axis = np.deg2rad(rng.uniform(-20, 90))
    rvec = np.array([np.cos(axis), np.sin(axis), rng.uniform(-0.6, 0.9)])
    rvec *= rng.uniform(0.9, 2.0) / np.linalg.norm(rvec)
    waves = [
        (-rng.uniform(0.14, 0.2), rng.uniform(0.018, 0.03), rvec * rng.uniform(0.08, 0.15) + rv(0.03)),  # P
        (-0.025, rng.uniform(0.007, 0.011), -rvec * rng.uniform(0.05, 0.15) + rv(0.1)),                    # Q
        (0.0, rng.uniform(0.008, 0.013), rvec),                                                            # R
        (0.028, rng.uniform(0.008, 0.013), rv(rng.uniform(0.3, 0.9)) - rvec * 0.2),                        # S
        (rng.uniform(0.22, 0.3), rng.uniform(0.04, 0.065), rvec * rng.uniform(0.15, 0.3) + rv(0.08)),      # T
    ]
    dip = np.zeros((n, 3))
    for b in beats:
        for off, w, vec in waves:
            g = np.exp(-0.5 * ((t - (b + off)) / w) ** 2)
            dip += g[:, None] * vec[None, :]
    # deriva de linha de base e ruído no dipolo (mantém Einthoven)
    for k in range(3):
        f = rng.uniform(0.05, 0.4)
        dip[:, k] += rng.uniform(0, 0.06) * np.sin(2 * np.pi * f * t + rng.uniform(0, 6.28))
    dip += rng.normal(0, 0.006, dip.shape)
    sig = _lead_vectors() @ dip.T  # (12, n)
    return sig.astype(np.float32)


GRID_STYLES = [
    # (cor linhas finas BGR, cor linhas grossas BGR, cor papel BGR)
    ((190, 190, 250), (120, 120, 235), (245, 245, 255)),   # rosa/vermelho clássico
    ((200, 205, 255), (90, 100, 230), (250, 250, 255)),
    ((170, 200, 245), (80, 140, 230), (235, 248, 255)),    # laranja
    ((215, 215, 215), (160, 160, 160), (252, 252, 252)),   # cinza (impressão P&B)
    ((235, 215, 195), (200, 160, 120), (255, 252, 248)),   # azul
]


HERSHEY = [cv2.FONT_HERSHEY_SIMPLEX, cv2.FONT_HERSHEY_DUPLEX, cv2.FONT_HERSHEY_COMPLEX,
           cv2.FONT_HERSHEY_TRIPLEX, cv2.FONT_HERSHEY_PLAIN]


def render(sig: np.ndarray, rng: np.random.Generator, ppmm: float | None = None,
           degrade: str = "medium", layout="3x4+1", ticks: bool | None = None,
           speed: float = 25.0, gain_limb: float = 10.0, gain_prec: float = 10.0,
           label_style: str | None = None):
    """Renderiza o ECG em papel. layout: nome em layouts.LIBRARY ou lista de linhas.
    Retorna (img_bgr, shown (12,N) bool, meta) — meta traz o ground truth do layout."""
    from .layouts import LIBRARY, lead_signal, is_limb, to_standard
    if ppmm is None:
        ppmm = float(rng.uniform(5.5, 12.0))
    spec = LIBRARY[layout] if isinstance(layout, str) else layout
    if ticks is None:
        ticks = bool(rng.random() < 0.6)
    if label_style is None:
        label_style = ["above", "below", "start"][rng.integers(3)]
    W_mm, H_mm = 280.0, 216.0
    W, H = int(round(W_mm * ppmm)), int(round(H_mm * ppmm))
    minor_c, major_c, paper_c = GRID_STYLES[rng.integers(len(GRID_STYLES))]
    img = np.full((H, W, 3), paper_c, np.uint8)

    gx0, gy0, gx1, gy1 = 5.0, 25.0, 275.0, 211.0
    lw_minor = max(1, int(round(0.10 * ppmm)))
    lw_major = max(1, int(round(0.22 * ppmm)))
    SH = 4
    S = 1 << SH

    def P(xmm, ymm):
        return (int(round(xmm * ppmm * S)), int(round(ymm * ppmm * S)))

    for k in range(int(gx1 - gx0) + 1):
        if k % 5:
            cv2.line(img, P(gx0 + k, gy0), P(gx0 + k, gy1), minor_c, lw_minor, cv2.LINE_AA, SH)
    for k in range(int(gy1 - gy0) + 1):
        if k % 5:
            cv2.line(img, P(gx0, gy0 + k), P(gx1, gy0 + k), minor_c, lw_minor, cv2.LINE_AA, SH)
    for k in range(0, int(gx1 - gx0) + 1, 5):
        cv2.line(img, P(gx0 + k, gy0), P(gx0 + k, gy1), major_c, lw_major, cv2.LINE_AA, SH)
    for k in range(0, int(gy1 - gy0) + 1, 5):
        cv2.line(img, P(gx0, gy0 + k), P(gx1, gy0 + k), major_c, lw_major, cv2.LINE_AA, SH)

    ink = tuple(int(c) for c in rng.integers(0, 60, 3))
    lw = max(1, int(round(rng.uniform(0.18, 0.35) * ppmm)))
    font = HERSHEY[rng.integers(len(HERSHEY))]
    fs_txt = 0.09 * ppmm * (1.6 if font == cv2.FONT_HERSHEY_PLAIN else 1.0)
    thick = max(1, int(round(0.012 * ppmm * 10 * rng.uniform(0.8, 1.4))))
    gtxt = f"{gain_limb:g}mm/mV" if gain_limb == gain_prec else f"{gain_limb:g}/{gain_prec:g}mm/mV"
    cv2.putText(img, "ID: 000%d   %gmm/s   %s   0.05-150 Hz" % (rng.integers(1000, 9999), speed, gtxt),
                (int(10 * ppmm), int(10 * ppmm)), font, fs_txt, ink, 1, cv2.LINE_AA)
    cv2.putText(img, "HR %d bpm  PR 160 ms  QRS 92 ms  QT/QTc 380/410 ms" % rng.integers(50, 110),
                (int(10 * ppmm), int(18 * ppmm)), font, fs_txt * 0.9, ink, 1, cv2.LINE_AA)

    x_sig0 = 22.5
    n_all = sig.shape[1]
    dur = min(DUR, 250.0 / speed)                 # duração exibida
    n = int(round(dur * FS))
    nr = len(spec)
    top, bottom = 55.0, 200.0
    row_base = list(np.linspace(top, bottom, nr)) if nr > 1 else [130.0]
    sp = (row_base[1] - row_base[0]) if nr > 1 else 40.0
    shown = np.zeros((12, n_all), dtype=bool)
    segments = []

    def draw_trace(name, i0, i1, y0, g):
        v = lead_signal(sig, name)[i0:i1]
        xs = x_sig0 + (np.arange(i0, i1) / FS) * speed
        ys = y0 - v * g
        pts = np.stack([np.round(xs * ppmm * S), np.round(ys * ppmm * S)], 1).astype(np.int32)
        cv2.polylines(img, [pts], False, ink, lw, cv2.LINE_AA, SH)
        std, _ = to_standard(name, v)
        shown[LEADS.index(std), i0:i1] = True

    def cal_pulse(y0, g1, g2):
        xa = 12.0
        if g1 == g2:
            pts = [(xa, y0), (xa + 2, y0), (xa + 2, y0 - g1), (xa + 7, y0 - g1), (xa + 7, y0), (xa + 9, y0)]
        else:  # pulso em degrau: ganho dos membros e depois das precordiais
            pts = [(xa, y0), (xa + 2, y0), (xa + 2, y0 - g1), (xa + 4.5, y0 - g1), (xa + 4.5, y0 - g2),
                   (xa + 7, y0 - g2), (xa + 7, y0), (xa + 9, y0)]
        pts = np.array([P(*p) for p in pts], np.int32)
        cv2.polylines(img, [pts], False, ink, lw, cv2.LINE_AA, SH)

    lab_dy = min(12.0, 0.4 * sp)
    for r, row in enumerate(spec):
        y0 = row_base[r]
        gl = [gain_limb if is_limb(nm) else gain_prec for nm in row]
        has_l = any(is_limb(nm) for nm in row); has_p = any(not is_limb(nm) for nm in row)
        g1 = gain_limb if has_l else gain_prec
        g2 = gain_prec if has_p else gain_limb
        cal_pulse(y0, g1, g2)
        k = len(row)
        for c, name in enumerate(row):
            i0, i1 = c * n // k, (c + 1) * n // k
            draw_trace(name, i0, i1, y0, gl[c])
            segments.append(dict(row=r, lead=name, i0=i0, i1=i1, gain=gl[c]))
            xl = x_sig0 + i0 / FS * speed
            if label_style == "above":
                org = (int((xl + 1.0) * ppmm), int((y0 - lab_dy) * ppmm))
            elif label_style == "below":
                org = (int((xl + 1.0) * ppmm), int((y0 + 0.35 * sp) * ppmm))
            else:  # junto ao início, um pouco acima da linha de base
                org = (int((xl + 0.8) * ppmm), int((y0 - 3.5) * ppmm))
            if c == 0 and label_style != "above":
                org = (int((x_sig0 - 1.0) * ppmm), org[1])
            cv2.putText(img, name, org, font, fs_txt, ink, thick, cv2.LINE_AA)
            if c > 0 and ticks:
                cv2.line(img, P(xl, y0 - 3), P(xl, y0 + 3), ink, lw, cv2.LINE_AA, SH)

    meta = dict(ppmm=ppmm, angle=0.0, layout=layout if isinstance(layout, str) else "custom",
                rows=[list(r) for r in spec], speed=speed, gain_limb=gain_limb, gain_prec=gain_prec,
                n_shown=n, segments=segments, x_sig0_mm=x_sig0, row_base_mm=row_base)
    img = degrade_image(img, rng, degrade, meta)
    return img, shown, meta


def _nonrigid(img, rng, bg):
    """Deformação não rígida de papel fotografado: curvatura cilíndrica suave + 1–2 vincos.

    Vinco = deslocamento contínuo com quebra de derivada ao longo de uma reta (as duas metades
    da folha ficam em planos diferentes), com faixa de sombra. Nenhum polinômio global de grau
    baixo representa essa quebra; é o caso que motiva a retificação local."""
    H, W = img.shape[:2]
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    dx = np.zeros((H, W), np.float32); dy = np.zeros((H, W), np.float32)
    shade = np.ones((H, W), np.float32)
    # curvatura (folha arqueada): compressão perto das bordas + barriga vertical
    a = rng.uniform(0.004, 0.012) * W
    dx += a * np.sin(np.pi * (xx / W - 0.5))
    dy += rng.uniform(-0.008, 0.008) * H * np.cos(np.pi * (xx / W - 0.5))
    for _ in range(int(rng.integers(1, 3))):
        th = rng.uniform(0, np.pi) if rng.random() < 0.4 else rng.choice([0.0, np.pi / 2]) + rng.normal(0, 0.08)
        nx, ny = np.cos(th), np.sin(th)
        c = rng.uniform(0.3, 0.7, 2) * [W, H]
        dist = (xx - c[0]) * nx + (yy - c[1]) * ny
        k = rng.uniform(0.008, 0.025) * rng.choice([-1, 1])     # mudança de inclinação na dobra
        t = rng.uniform(-0.01, 0.01)                             # cisalhamento ao longo do vinco
        r = np.maximum(dist, 0)
        dx += k * r * nx - t * r * ny
        dy += k * r * ny + t * r * nx
        shade *= 1 - rng.uniform(0.08, 0.25) * np.exp(-(dist / (rng.uniform(0.004, 0.015) * W)) ** 2)
    out = cv2.remap(img, xx + dx, yy + dy, cv2.INTER_LINEAR, borderValue=bg)
    return np.clip(out.astype(np.float32) * shade[..., None], 0, 255).astype(np.uint8)


def degrade_image(img, rng, level, meta):
    H, W = img.shape[:2]
    if level == "none":
        return img
    fold = level == "fold"
    lv = {"light": 0.4, "medium": 1.0, "heavy": 1.8, "fold": 1.0}[level]
    # rotação + translação sobre fundo de "mesa"
    ang = rng.uniform(-3, 3) * lv
    meta["angle"] = ang
    pad = int(0.06 * max(H, W))
    bg = tuple(int(c) for c in rng.integers(40, 140, 3))
    canvas = cv2.copyMakeBorder(img, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=bg)
    Hc, Wc = canvas.shape[:2]
    M = cv2.getRotationMatrix2D((Wc / 2, Hc / 2), ang, 1.0)
    canvas = cv2.warpAffine(canvas, M, (Wc, Hc), flags=cv2.INTER_LINEAR, borderValue=bg)
    if fold:
        canvas = _nonrigid(canvas, rng, bg)
    # perspectiva leve
    if lv >= 1.0:
        d = 0.025 * (lv - 0.6) * min(Hc, Wc)
        src = np.float32([[0, 0], [Wc, 0], [Wc, Hc], [0, Hc]])
        dst = src + rng.uniform(-d, d, (4, 2)).astype(np.float32)
        Hm = cv2.getPerspectiveTransform(src, dst)
        canvas = cv2.warpPerspective(canvas, Hm, (Wc, Hc), flags=cv2.INTER_LINEAR, borderValue=bg)
    img = canvas.astype(np.float32)
    # iluminação não uniforme (sombra)
    yy, xx = np.mgrid[0:Hc, 0:Wc].astype(np.float32)
    gx, gy = rng.uniform(-1, 1, 2)
    shade = 1.0 - 0.25 * lv * (0.5 + 0.5 * ((gx * xx / Wc + gy * yy / Hc)))
    img *= shade[..., None]
    # tom de cor (balanço de branco)
    img *= rng.uniform(1 - 0.06 * lv, 1 + 0.06 * lv, 3)[None, None, :]
    # blur + ruído
    k = rng.uniform(0.3, 1.0) * lv
    img = cv2.GaussianBlur(img, (0, 0), k)
    img += rng.normal(0, 4 * lv, img.shape)
    img = np.clip(img, 0, 255).astype(np.uint8)
    q = int(rng.uniform(95 - 40 * lv, 95))
    ok, enc = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, max(q, 25)])
    return cv2.imdecode(enc, cv2.IMREAD_COLOR)


def make_dataset(out_dir, n=20, seed=0, degrade="medium", layout="3x4+1", speed=25.0,
                 gain_limb=10.0, gain_prec=10.0):
    """layout="mix" sorteia layout, velocidade e ganho dividido (conjunto de layout difícil)."""
    import os, json
    from .layouts import LIBRARY
    os.makedirs(out_dir, exist_ok=True)
    rng = np.random.default_rng(seed)
    names = list(LIBRARY)
    for i in range(n):
        sig = synth_signals(rng)
        lay, sp, gl, gp = layout, speed, gain_limb, gain_prec
        if layout == "mix":
            lay = names[rng.integers(len(names))]
            sp = 50.0 if rng.random() < 0.2 else 25.0
            gl, gp = [(10, 10), (10, 10), (10, 5), (5, 5), (20, 10)][rng.integers(5)]
            nrows = len(LIBRARY[lay])
            if nrows > 7 and gl > 5:      # 12x1: linhas a ~12 mm -> ganho baixo (como os aparelhos)
                gl, gp = (5, 5)
            elif nrows > 4 and gl > 10:   # 6x2 com 20 mm/mV não cabe
                gl, gp = (10, 5)
        img, shown, meta = render(sig, rng, degrade=degrade, layout=lay, speed=sp,
                                  gain_limb=float(gl), gain_prec=float(gp))
        cv2.imwrite(f"{out_dir}/ecg_{i:03d}.png", img)
        np.savez_compressed(f"{out_dir}/ecg_{i:03d}.npz", sig=sig, shown=shown,
                            ppmm=meta["ppmm"], angle=meta["angle"])
        with open(f"{out_dir}/ecg_{i:03d}.json", "w") as fh:
            json.dump({k: v for k, v in meta.items() if k not in ("row_base_mm",)}, fh)


if __name__ == "__main__":
    import sys
    make_dataset(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 20,
                 degrade=sys.argv[3] if len(sys.argv) > 3 else "medium",
                 layout=sys.argv[4] if len(sys.argv) > 4 else "3x4+1",
                 seed=int(sys.argv[5]) if len(sys.argv) > 5 else 0)
