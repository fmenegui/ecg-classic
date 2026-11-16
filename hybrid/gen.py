"""Training-crop generator for the trace segmenter (hybrid mode).

Principle: the network only learns WHERE THE ECG PEN INK IS (trace, calibration pulse,
separator mark) on an image already rectified by the grid (10 px/mm). Everything else stays
classical. To avoid memorizing a specific generator (the Kaggle/PhysioNet one), each crop draws
paper, grid, pen, colours, text and degradations independently and broadly; there is no fixed
page, layout or position.

Label: anti-aliased pen coverage (0..1), drawn with supersampling, BEFORE the degradations
(the network learns to undo blur, JPEG, shadow, stippling, moiré).
"""
from __future__ import annotations

import numpy as np
import cv2

R = 10.0  # px/mm of the rectified image


def _ecg_wave(rng, n):
    """Plausible ECG signal (mV) with n samples at 500 Hz: uses the ecgdig dipole generator when
    available and draws lead, window and gain at random."""
    try:
        from ecgdig.synth import synth_signals
        S = synth_signals(rng)
        lead = S[rng.integers(12)]
        if rng.random() < 0.3:          # morfologias extremas (QRS gigante, baixa voltagem)
            lead = lead * rng.choice([0.3, 2.0, 3.0])
        i0 = rng.integers(0, max(1, S.shape[1] - n))
        v = lead[i0:i0 + n]
        if len(v) < n:
            v = np.pad(v, (0, n - len(v)), mode="edge")
        return v
    except Exception:  # noqa: BLE001
        t = np.arange(n) / 500.0
        return 1.2 * np.exp(-((t % 0.8 - 0.3) / 0.012) ** 2) - 0.2 * np.exp(-((t % 0.8 - 0.33) / 0.02) ** 2)


def _rand_color(rng, dark=True):
    if dark:
        base = rng.integers(0, 70, 3)
        if rng.random() < 0.3:  # coloured trace (blue, dark green, purple)
            base[rng.integers(3)] = rng.integers(60, 160)
        return tuple(int(c) for c in base)
    return tuple(int(c) for c in rng.integers(150, 256, 3))


def _blend(img, alpha, color):
    a = np.clip(alpha, 0, 1)[..., None]
    return img * (1 - a) + np.array(color, np.float32)[None, None, :] * a


def _grid_layer(rng, H, W, S):
    """Millimetre grid (or dotted, or none) with period ~R and random phase/rotation."""
    out = []
    if rng.random() < 0.1:
        return out
    per = R * rng.uniform(0.97, 1.03)
    ang = np.deg2rad(rng.normal(0, 0.6))
    ph = rng.uniform(0, 5 * per, 2)
    yy, xx = np.mgrid[0:H * S, 0:W * S].astype(np.float32) / S
    u = xx * np.cos(ang) + yy * np.sin(ang) + ph[0]
    v = -xx * np.sin(ang) + yy * np.cos(ang) + ph[1]
    w_min = rng.uniform(0.6, 1.6)
    w_maj = w_min * rng.uniform(1.0, 2.6)
    dotted = rng.random() < 0.15

    def lines(c, period, w):
        d = np.abs(((c / period) + 0.5) % 1.0 - 0.5) * period
        return (d < w / 2).astype(np.float32)

    minor = np.maximum(lines(u, per, w_min), lines(v, per, w_min))
    if dotted:
        minor = lines(u, per, w_min * 1.5) * lines(v, per, w_min * 1.5)
    major = np.maximum(lines(u, 5 * per, w_maj), lines(v, 5 * per, w_maj))
    # black/grey grid (B&W printer, darkened scan) weighted up: it is where grid and trace share
    # a colour and the network errs most
    kind = int(rng.choice(5, p=[0.3, 0.12, 0.13, 0.33, 0.12]))
    if kind == 0:      # classic red/pink
        cmin = (int(rng.integers(150, 230)), int(rng.integers(150, 225)), int(rng.integers(220, 256)))
        cmaj = (int(rng.integers(60, 170)), int(rng.integers(60, 160)), int(rng.integers(180, 256)))
    elif kind == 1:    # laranja/marrom
        cmin = (int(rng.integers(140, 210)), int(rng.integers(180, 230)), int(rng.integers(220, 256)))
        cmaj = (int(rng.integers(40, 120)), int(rng.integers(90, 170)), int(rng.integers(180, 240)))
    elif kind == 2:    # verde/azul
        g = rng.integers(2)
        cmin = (int(rng.integers(170, 240)),) * 3
        cmin = tuple(int(v_) if i != 1 + g else int(rng.integers(200, 256)) for i, v_ in enumerate(cmin))
        cmaj = tuple(int(c) - 70 for c in cmin)
    elif kind == 3:    # grey/black (B&W printer, darkened scan): the hard case
        gmin = int(rng.integers(120, 220)); gmaj = int(rng.integers(20, max(30, gmin - 30)))
        cmin, cmaj = (gmin,) * 3, (gmaj,) * 3
    else:              # cores livres claras
        cmin = _rand_color(rng, dark=False)
        cmaj = tuple(int(c * rng.uniform(0.4, 0.85)) for c in cmin)
    a_min = rng.uniform(0.4, 1.0)
    out.append((minor * a_min, cmin))
    out.append((major * rng.uniform(0.6, 1.0), cmaj))
    return out


def _trace_layer(rng, H, W, S):
    """ECG traces, pulses and separator marks (the LABEL) with supersampling."""
    lay = np.zeros((H * S, W * S), np.uint8)
    pen = rng.uniform(0.6, 4.0)
    th = max(1, int(round(pen * S)))
    speed = 50.0 if rng.random() < 0.15 else 25.0
    n_tr = int(rng.integers(1, 4))
    ys = np.sort(rng.uniform(0.15 * H, 0.85 * H, n_tr))
    for y0 in ys:
        gain = float(rng.choice([5.0, 10.0, 10.0, 10.0, 20.0])) * rng.uniform(0.95, 1.05)
        x0 = rng.uniform(-0.3 * W, 0.6 * W) if rng.random() < 0.3 else -5.0
        x1 = W + 5.0 if rng.random() < 0.8 else rng.uniform(x0 + 0.3 * W, W + 5)
        n = int((x1 - x0) / R / speed * 500) + 2
        v = _ecg_wave(rng, n) + rng.normal(0, 0.01, n).cumsum() * 0.02
        xs = x0 + np.arange(n) / 500.0 * speed * R
        yv = y0 - v * gain * R / 10.0 * (R / R)
        # gain em mm/mV -> px: v (mV) * gain (mm/mV) * R (px/mm)
        yv = y0 - v * gain * R
        pts = np.stack([xs * S, yv * S], 1).astype(np.int32)
        cv2.polylines(lay, [pts], False, 255, th, cv2.LINE_AA)
        if rng.random() < 0.25:   # calibration pulse
            xa = rng.uniform(0.05, 0.6) * W
            h = gain * R * 1.0
            P = np.array([[xa - 2 * R, y0], [xa, y0], [xa, y0 - h], [xa + 5 * R, y0 - h], [xa + 5 * R, y0],
                          [xa + 7 * R, y0]]) * S
            cv2.polylines(lay, [P.astype(np.int32)], False, 255, th, cv2.LINE_AA)
        if rng.random() < 0.3:    # marca separadora
            xt = rng.uniform(0.1, 0.9) * W
            hh = rng.uniform(2, 8) * R
            cv2.line(lay, (int(xt * S), int((y0 - hh / 2) * S)), (int(xt * S), int((y0 + hh / 2) * S)), 255,
                     max(th, int(rng.uniform(1, 2.5) * th)), cv2.LINE_AA)
    return lay


def _distractors(rng, img, H, W, ink):
    """Dark things that are NOT trace: text, rulers, borders, stamps, QR, stains."""
    font = [cv2.FONT_HERSHEY_SIMPLEX, cv2.FONT_HERSHEY_PLAIN, cv2.FONT_HERSHEY_DUPLEX,
            cv2.FONT_HERSHEY_COMPLEX][rng.integers(4)]
    alph = "IVaRLFV123456 mm/s mV HzQRST ID"
    for _ in range(int(rng.poisson(1.5))):
        s = "".join(alph[i] for i in rng.integers(len(alph), size=int(rng.integers(1, 9))))
        col = ink if rng.random() < 0.6 else _rand_color(rng)
        cv2.putText(img, s, (int(rng.uniform(-20, W)), int(rng.uniform(10, H + 10))), font,
                    rng.uniform(0.3, 1.2), col, int(rng.integers(1, 3)), cv2.LINE_AA)
    if rng.random() < 0.15:  # borda/linha reta longa
        y = int(rng.uniform(0, H)); cv2.line(img, (0, y), (W, y + int(rng.normal(0, 3))), _rand_color(rng), int(rng.integers(1, 4)))
    if rng.random() < 0.08:  # bloco tipo QR
        x, y, s = int(rng.uniform(0, W)), int(rng.uniform(0, H)), int(rng.uniform(15, 60))
        q = (rng.random((8, 8)) < 0.5).astype(np.uint8) * 255
        q = cv2.resize(q, (s, s), interpolation=cv2.INTER_NEAREST)
        sub = img[y:y + s, x:x + s]
        m = q[:sub.shape[0], :sub.shape[1]] > 0
        sub[m] = 20
    return img


def _degrade(rng, img):
    H, W = img.shape[:2]
    f = img.astype(np.float32)
    # low-resolution rasterization before "printing"
    if rng.random() < 0.3:
        k = rng.uniform(1.5, 3.0)
        small = cv2.resize(f, (max(8, int(W / k)), max(8, int(H / k))), interpolation=cv2.INTER_AREA)
        f = cv2.resize(small, (W, H), interpolation=cv2.INTER_LINEAR)
    # blur (printing + capture)
    s = rng.uniform(0, 1.6)
    if s > 0.2:
        f = cv2.GaussianBlur(f, (0, 0), s)
    # illumination: gradient, shadow, colour balance
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    g = 1 - rng.uniform(0, 0.45) * (0.5 + 0.5 * np.sin(rng.uniform(0, 6.28) + (xx * rng.normal() + yy * rng.normal()) / (3 * W)))
    if rng.random() < 0.25:  # sombra dura
        cx, cy = rng.uniform(0, W), rng.uniform(0, H)
        g *= 1 - rng.uniform(0.2, 0.5) * (((xx - cx) * rng.normal() + (yy - cy) * rng.normal()) > 0)
        g = cv2.GaussianBlur(g, (0, 0), rng.uniform(2, 15))
    f = f * g[..., None] * rng.uniform(0.85, 1.12, 3)[None, None, :] + rng.uniform(-25, 15)
    # stains/stippling of a darkened scan
    if rng.random() < 0.15:
        blob = cv2.GaussianBlur((rng.random((H, W)) < 0.002).astype(np.float32), (0, 0), rng.uniform(5, 20))
        blob = blob / (blob.max() + 1e-6)
        speck = (rng.random((H, W)) < blob * rng.uniform(0.2, 0.7)).astype(np.float32)
        f = f * (1 - 0.8 * speck[..., None])
    # screen moiré
    if rng.random() < 0.1:
        a = rng.uniform(0, 3.14); fr = rng.uniform(0.6, 1.4)
        f = f * (1 - rng.uniform(0.05, 0.25) * (0.5 + 0.5 * np.sin(fr * (xx * np.cos(a) + yy * np.sin(a)))))[..., None]
    f = f + rng.normal(0, rng.uniform(0, 10), f.shape)
    f = np.clip(f, 0, 255).astype(np.uint8)
    # stippled B&W (fax/B&W scanner)
    if rng.random() < 0.22:
        gr = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY).astype(np.float32)
        gr = (gr + rng.normal(0, 30, gr.shape) > rng.uniform(110, 170)).astype(np.uint8) * 255
        f = cv2.cvtColor(gr, cv2.COLOR_GRAY2BGR)
    elif rng.random() < 0.2:
        f = cv2.cvtColor(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
    q = int(rng.uniform(30, 96))
    ok, enc = cv2.imencode(".jpg", f, [cv2.IMWRITE_JPEG_QUALITY, q])
    return cv2.imdecode(enc, cv2.IMREAD_COLOR)


def sample(rng, size=192):
    """One crop (BGR uint8, float32 label 0..1)."""
    H = W = size
    S = 4
    paper = np.array(_rand_color(rng, dark=False), np.float32)
    if rng.random() < 0.7:  # papel quase branco/creme na maioria
        paper = np.array([rng.uniform(215, 255)] * 3, np.float32) * rng.uniform(0.94, 1.0, 3)
    img = np.ones((H * S, W * S, 3), np.float32) * paper[None, None, :]
    for a, c in _grid_layer(rng, H, W, S):
        img = _blend(img, a, c)
    lay = _trace_layer(rng, H, W, S)
    if rng.random() < 0.08:  # grid/text-only crop: the network must learn to say "nothing"
        lay[:] = 0
    ink = _rand_color(rng)
    if rng.random() < 0.2:  # trace with the same colour as the major grid (worst case)
        ink = (int(rng.integers(0, 60)),) * 3
    img = _blend(img, lay.astype(np.float32) / 255.0 * rng.uniform(0.75, 1.0), ink)
    img = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
    lab = cv2.resize(lay.astype(np.float32) / 255.0, (W, H), interpolation=cv2.INTER_AREA)
    img = np.clip(img, 0, 255).astype(np.uint8)
    img = _distractors(rng, img, H, W, ink)
    img = _degrade(rng, img)
    return img, lab


if __name__ == "__main__":
    import sys
    rng = np.random.default_rng(int(sys.argv[2]) if len(sys.argv) > 2 else 0)
    tiles = []
    for i in range(16):
        im, lb = sample(rng)
        tiles.append(np.hstack([im, cv2.cvtColor((255 * (1 - lb)).astype(np.uint8), cv2.COLOR_GRAY2BGR)]))
    rows = [np.hstack(tiles[i:i + 4]) for i in range(0, 16, 4)]
    cv2.imwrite(sys.argv[1], np.vstack(rows))
