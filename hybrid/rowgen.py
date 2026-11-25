"""Training strips for the row network: "where is THIS row's signal, column by column".

The classical pipeline finds the rows, their baselines, the time axis and the layout on any page.
What it reads badly is the trace itself inside a row: crossings with the neighbouring rows, faint
or broken ink, grids as dark as the pen, text glued to the trace. The row network takes a strip
of the rectified page (10 px/mm) centred on the row's estimated baseline. The label is SPARSE:
in each column, the height of that row's signal split over the two nearest pixels by its
fractional part (sub-pixel), as the best Kaggle solutions found (a dense ink mask caps the
reconstruction SNR). Neighbours, grid, pulses, separators and text are background.

Each strip draws real 12-lead signals (a PTB-XL bank built by hybrid/ptbxl/ptbxl_bank.py, set by
ECG_BANK; the ecgdig dipole synthesizer when there is none), a random row structure (lead
changes, neighbours at random spacing that overlap the target, calibration pulses, separators,
labels), the paper/grid/degradation generator of hybrid/gen.py and a small residual warp
(imperfect rectification). No page layout, template or test image is used.
"""
from __future__ import annotations

import os

import cv2
import numpy as np

from gen import R, _blend, _degrade, _distractors, _grid_layer, _rand_color

FS = 500
_BANK = None


def _bank():
    global _BANK
    if _BANK is None:
        p = os.environ.get("ECG_BANK", "")
        _BANK = np.load(p, mmap_mode="r") if p and os.path.exists(p) else False
    return _BANK


def _record(rng):
    """(12, 5000) mV at 500 Hz: a real record from the bank, else a synthetic one."""
    b = _bank()
    if b is not False:
        x = np.asarray(b[int(rng.integers(len(b)))], np.float32)
        x = np.nan_to_num(x)
    else:
        from ecgdig.synth import synth_signals
        x = synth_signals(rng).astype(np.float32)
        if x.shape[1] < 5000:
            x = np.pad(x, ((0, 0), (0, 5000 - x.shape[1])), mode="edge")
    if rng.random() < 0.15:   # extreme amplitudes (giant QRS, low voltage)
        x = x * rng.choice([0.4, 1.8, 2.5])
    return x


def _row_signal(rng, n, seg_len):
    """Signal (mV) of one printed row with n samples: consecutive segments of different leads
    (a lead change every seg_len samples, as in 3x4 / 6x2 pages) or one continuous lead."""
    rec = _record(rng)
    leads = rng.permutation(12)
    i0 = int(rng.integers(0, max(1, rec.shape[1] - n)))
    if seg_len <= 0 or seg_len >= n + i0:
        v = rec[leads[0], i0:i0 + n]
        cuts = []
    else:
        v = np.empty(n, np.float32)
        cuts = []
        k, j = 0, 0
        first = int(rng.integers(1, seg_len))
        while j < n:
            e = min(n, j + (first if k == 0 else seg_len))
            v[j:e] = rec[leads[k % 12], i0 + j:i0 + e] if i0 + e <= rec.shape[1] else rec[leads[k % 12], -(e - j):]
            if e < n:
                cuts.append(e)
            j, k = e, k + 1
    if len(v) < n:
        v = np.pad(v, (0, n - len(v)), mode="edge")
    v = v + rng.normal(0, 0.003, n).cumsum() * 0.01   # slow baseline wander
    return v, cuts


def _draw_row(lay, rng, y0, x0, xs_per_s, gain_px, n, seg_len, th, S, pulse, sep):
    v, cuts = _row_signal(rng, n, seg_len)
    xs = x0 + np.arange(n) / FS * xs_per_s
    ys = y0 - v * gain_px
    pts = np.stack([xs * S, ys * S], 1).astype(np.int32)
    # a lead change is a jump: drawn as separate polylines (some printers join them)
    joined = rng.random() < 0.5
    parts = [pts] if joined or not cuts else np.split(pts, cuts)
    for p in parts:
        if len(p) > 1:
            cv2.polylines(lay, [p], False, 255, th, cv2.LINE_AA)
    extra = []   # pulses and separators: ink of the row, but NOT its signal
    if pulse:
        xa = x0 - rng.uniform(0.5, 2.0) * R
        h = gain_px
        P = np.array([[xa - 7 * R, y0], [xa - 5 * R, y0], [xa - 5 * R, y0 - h], [xa, y0 - h], [xa, y0]]) * S
        extra.append(("poly", P.astype(np.int32)))
    if sep:
        for c in cuts:
            xt = xs[min(c, n - 1)]
            hh = rng.uniform(2, 8) * R
            extra.append(("line", (int(xt * S), int((y0 - hh / 2) * S), int(xt * S), int((y0 + hh / 2) * S))))
    return xs, ys, extra


def _warp(rng, H, W):
    """Residual distortion after rectification: slow vertical/horizontal displacement (px)."""
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    a = rng.uniform(0, 0.6) * R * (rng.random() < 0.6)
    lam = rng.uniform(300, 1500)
    dy = a * np.sin(2 * np.pi * xx / lam + rng.uniform(0, 6.3))
    ang = np.deg2rad(rng.normal(0, 0.4))
    dy = dy + (xx - W / 2) * np.tan(ang)
    dx = rng.uniform(0, 0.3) * R * np.sin(2 * np.pi * yy / rng.uniform(300, 1500) + rng.uniform(0, 6.3))
    return (xx + dx).astype(np.float32), (yy + dy).astype(np.float32)


def _sparse_label(xs, ys, mx, my, yc, H, W, S, gap, xup=1):
    """Signal height per output column after the residual warp, split over 2 pixels.

    Coordinates use the pixel-centre convention (row i at y = i). The polyline was drawn at
    supersampled integer positions, i.e. at (v * S) // 1 -> centre (v + 0.5 / S - 0.5).
    xup > 1: the strip is stretched horizontally by xup (cv2.resize convention: output column j
    is at x = (j + 0.5) / xup - 0.5), so the label has W * xup columns, each with the true height
    at its own position (not interpolated between the base columns)."""
    xc = xs + 0.5 / S - 0.5
    yv = ys + 0.5 / S - 0.5
    cols = (np.arange(W * xup, dtype=np.float64) + 0.5) / xup - 0.5
    dy = np.interp(cols, np.arange(W, dtype=np.float64), my[0, :].astype(np.float64))   # my = y + dy(x)
    dx = float(mx[int(np.clip(yc, 0, H - 1)), 0])                     # mx = x + dx(y), at the row
    sc = cols + dx
    W = W * xup
    ok = (sc >= xc[0]) & (sc <= xc[-1])
    if gap is not None:
        ok &= ~((sc >= gap[0]) & (sc <= gap[1]))
    y = np.interp(sc, xc, yv) - dy
    i = np.floor(y).astype(np.int64)
    f = (y - i).astype(np.float32)
    lab = np.zeros((H, W), np.float32)
    c = np.flatnonzero(ok & (i >= 0) & (i + 1 < H))
    lab[i[c], c] = 1.0 - f[c]
    lab[i[c] + 1, c] = f[c]
    return lab


XUP = [int(os.environ.get("ROWGEN_XUP", "1"))]   # horizontal stretch (2 = 20 px/mm across, 10 down)


def sample(rng, size=512, width=None, xup=None):
    """One strip: BGR uint8 (H, W, 3), label float32 (H, W): the centre row's signal height per
    column (sparse, sub-pixel over 2 pixels). xup > 1: the strip is drawn at 10 px/mm, degraded,
    then stretched horizontally by xup exactly as inference stretches the rectified strip, and
    labelled at the stretched columns (W * xup wide)."""
    xup = int(xup or XUP[0])
    H, W = size, width or size
    S = 4
    paper = np.array(_rand_color(rng, dark=False), np.float32)
    if rng.random() < 0.7:
        paper = np.array([rng.uniform(215, 255)] * 3, np.float32) * rng.uniform(0.94, 1.0, 3)
    img = np.ones((H, W, 3), np.float32) * paper[None, None, :]
    for a, c in _grid_layer(rng, H, W, S):
        img = _blend(img, a, c)

    speed = 50.0 if rng.random() < 0.12 else 25.0
    speed *= rng.uniform(0.97, 1.03)
    gain = float(rng.choice([5.0, 10.0, 10.0, 10.0, 10.0, 20.0])) * rng.uniform(0.95, 1.05)
    xs_per_s, gain_px = speed * R, gain * R
    pen = rng.uniform(0.7, 3.5)
    th = max(1, int(round(pen * S)))
    # the strip is centred on the ESTIMATED baseline; the true one is off by a few mm
    yc = H / 2 + float(np.clip(rng.normal(0, 1.5), -4, 4)) * R
    seg_s = float(rng.choice([2.5, 2.5, 5.0, 0.0]))         # lead change every 2.5 s / 5 s / none
    seg_len = int(seg_s * FS)
    x0 = -rng.uniform(0, 3) * xs_per_s if rng.random() < 0.7 else rng.uniform(0.05, 0.5) * W
    x1 = W + 5 if rng.random() < 0.85 else rng.uniform(x0 + 0.3 * W, W)
    n = int(max(10, (x1 - x0) / xs_per_s * FS))

    tgt = np.zeros((H * S, W * S), np.uint8)
    oth = np.zeros((H * S, W * S), np.uint8)
    xs_t, ys_t, extra = _draw_row(tgt, rng, yc, x0, xs_per_s, gain_px, n, seg_len, th, S,
                                  pulse=x0 > 8 * R and rng.random() < 0.5, sep=rng.random() < 0.4)
    # neighbouring rows: same pen, gain and speed, random spacing; large waves reach the target
    for side in (-1, 1):
        y = yc
        for _ in range(int(rng.choice([0, 1, 1, 1, 2]))):
            y = y + side * rng.uniform(12, 45) * R
            if -60 * R < y < H + 60 * R:
                xo = -rng.uniform(0, 3) * xs_per_s if rng.random() < 0.8 else rng.uniform(0, 0.5) * W
                _, _, ex2 = _draw_row(oth, rng, y, xo, xs_per_s, gain_px, int((W + 5 - xo) / xs_per_s * FS),
                                      seg_len, th, S, pulse=xo > 8 * R and rng.random() < 0.5, sep=rng.random() < 0.4)
                extra += ex2
    for kind, g in extra:
        if kind == "poly":
            cv2.polylines(oth, [g], False, 255, th, cv2.LINE_AA)
        else:
            cv2.line(oth, g[:2], g[2:], 255, max(th, int(rng.uniform(1, 2.5) * th)), cv2.LINE_AA)
    gap = None
    if rng.random() < 0.08:   # gap in the target (faded, folded, occluded): no label there either
        xa = rng.uniform(0, W) * S; xb = xa + rng.uniform(5, 60) * R * S
        tgt[:, int(xa):int(xb)] = 0
        gap = (xa / S, xb / S)

    lab = cv2.resize(tgt.astype(np.float32) / 255.0, (W, H), interpolation=cv2.INTER_AREA)
    oth_ = cv2.resize(oth.astype(np.float32) / 255.0, (W, H), interpolation=cv2.INTER_AREA)
    ink = _rand_color(rng)
    if rng.random() < 0.2:
        ink = (int(rng.integers(0, 60)),) * 3
    a_ink = rng.uniform(0.75, 1.0)
    img = _blend(img, np.maximum(lab, oth_) * a_ink, ink)
    img = np.clip(img, 0, 255).astype(np.uint8)
    if rng.random() < 0.5:   # lead names next to the trace (the classic label-glued-to-trace case)
        font = cv2.FONT_HERSHEY_SIMPLEX if rng.random() < 0.6 else cv2.FONT_HERSHEY_DUPLEX
        for _ in range(int(rng.integers(1, 3))):
            s = str(rng.choice(["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]))
            cv2.putText(img, s, (int(rng.uniform(0, W - 40)), int(yc - rng.uniform(-2, 12) * R)), font,
                        rng.uniform(0.6, 1.4), ink if rng.random() < 0.7 else _rand_color(rng),
                        int(rng.integers(1, 3)), cv2.LINE_AA)
    img = _distractors(rng, img, H, W, ink)
    mx, my = _warp(rng, H, W)
    img = cv2.remap(img, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    lab = _sparse_label(xs_t, ys_t, mx, my, yc, H, W, S, gap, xup)
    img = _degrade(rng, img)
    if xup > 1:
        img = cv2.resize(img, (W * xup, H), interpolation=cv2.INTER_CUBIC)
    return img, lab


if __name__ == "__main__":
    import sys
    rng = np.random.default_rng(int(sys.argv[2]) if len(sys.argv) > 2 else 0)
    tiles = []
    for i in range(8):
        im, lb = sample(rng, 384)
        ov = im.copy()
        ov[cv2.dilate((lb > 0.2).astype(np.uint8), np.ones((3, 3), np.uint8)) > 0] = (0, 200, 0)
        tiles.append(np.hstack([im, ov]))
    cv2.imwrite(sys.argv[1], np.vstack([np.hstack(tiles[i:i + 2]) for i in range(0, 8, 2)]))
