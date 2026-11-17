"""Training crops for a generic grid-point network (5 mm intersections), rendered on the GPU.

Where the classical geometry fails (crumpled or curved photos, strong perspective, blur, dark
grids, low resolution), the Fourier phase of the grid is unreliable. A network that marks the 5 mm
intersections in the photo, with sub-pixel precision, gives the geometry another source of image
<-> mm correspondences. It knows nothing about the page: no template, no number of lines, no
position. The integer lattice coordinates still come from the classical unwrapping.

ecg-torch-renderer photographs pages (perspective, curl, folds, shadows, phone optics) and returns
the observed position of every visible 5 mm node; the label is a Gaussian of sigma ~1 px at each.
"""
from __future__ import annotations

import os

import math

import numpy as np
import torch
import torch.nn.functional as F


class GridPoints:
    def __init__(self, device, size=512, page_hw=(1536, 2048), pages=2, sigma=1.0, seed=0):
        from ecg_gpu import ECGSupervisionLayer, GridPointConfig, RenderConfig, SupervisionConfig
        if os.environ.get("RENDER_PAGE"):
            page_hw = tuple(int(v) for v in os.environ["RENDER_PAGE"].split("x"))
        self.dev = torch.device(device)
        self.size, self.pages = size, pages
        self.layer = ECGSupervisionLayer(
            fs=500, config=RenderConfig(height=page_hw[0], width=page_hw[1]),
            supervision=SupervisionConfig(outputs=("grid_points_mask",),
                                          grid_point_config=GridPointConfig(sigma_pixels=sigma)),
        ).to(self.dev)
        self.layer.train()
        p = os.environ.get("ECG_BANK", "")
        self.bank = np.load(p, mmap_mode="r") if p and os.path.exists(p) else None
        self.rng = np.random.default_rng(seed)

    def _signals(self, n):
        if self.bank is not None:
            x = np.stack([np.asarray(self.bank[i], np.float32) for i in self.rng.integers(0, len(self.bank), n)])
        else:
            from ecgdig.synth import synth_signals
            x = np.stack([synth_signals(self.rng)[:, :5000].astype(np.float32) for _ in range(n)])
        return torch.from_numpy(np.nan_to_num(x)).to(self.dev)

    @torch.no_grad()
    def batch(self, n):
        """n crops: images (n, 3, S, S) in [0, 1] and heatmaps (n, 1, S, S) of the 5 mm nodes."""
        b = self.layer(self._signals(self.pages))
        img, hm = b["image"].float().clamp(0, 1), b["grid_points_mask"].float()
        valid = b.get("valid")
        P, _, H, W = img.shape
        S = self.size
        xs, ys = [], []
        for _ in range(n):
            p = int(self.rng.integers(P))
            y0 = int(self.rng.integers(0, max(1, H - S + 1))); x0 = int(self.rng.integers(0, max(1, W - S + 1)))
            xs.append(img[p, :, y0:y0 + S, x0:x0 + S])
            t = hm[p, :, y0:y0 + S, x0:x0 + S]
            if valid is not None:
                t = t * (valid[p, :, y0:y0 + S, x0:x0 + S] > 0.5)
            ys.append(t)
        return torch.stack(xs), torch.stack(ys)


class GridPhase:
    """Continuous grid-phase targets: for every photo pixel, where it lies on the paper's grid.

    u, v = paper position in mm along the two grid axes, measured from a grid line (the renderer's
    grid origin), obtained from the exact inverse field of the photographed page. The targets are
    the 5 mm phase (the main 5 mm squares) and the 1 mm phase (the small squares, for precision):

        ch 0-3: cos, sin(2 pi u / 5), cos, sin(2 pi v / 5)
        ch 4-7: cos, sin(2 pi u),     cos, sin(2 pi v)
        ch 8:   1 where the pixel is grid paper (grid printed and visible), else 0
        ch 9:   weight of the 1 mm channels: 0 when the photo has < 2 px per mm (the 1 mm phase
                is then aliased), 1 from 4 px per mm

    A smooth periodic map is defined everywhere on the paper, also between lines, under traces and
    text, across folds and in perspective, so the network regresses it densely and the geometry
    gets a sub-pixel mm coordinate per pixel (5 mm phase unwrapped, refined by the 1 mm phase).
    Nothing about the page is assumed: no template, number of rows or grid position.
    """

    def __init__(self, device, size=512, page_hw=(1536, 2048), pages=2, seed=0):
        from ecg_gpu import ECGSupervisionLayer, RenderConfig, SupervisionConfig
        if os.environ.get("RENDER_PAGE"):
            page_hw = tuple(int(v) for v in os.environ["RENDER_PAGE"].split("x"))
        self.dev = torch.device(device)
        self.size, self.pages, self.hw = size, pages, page_hw
        self.layer = ECGSupervisionLayer(fs=500, config=RenderConfig(height=page_hw[0], width=page_hw[1]),
                                         supervision=SupervisionConfig(outputs=("trace_mask",))).to(self.dev)
        self.layer.train()
        p = os.environ.get("ECG_BANK", "")
        self.bank = np.load(p, mmap_mode="r") if p and os.path.exists(p) else None
        self.rng = np.random.default_rng(seed)

    _signals = GridPoints._signals

    @torch.no_grad()
    def pages_targets(self):
        """Render self.pages photos -> images (P, 3, H, W) and targets (P, 10, H, W)."""
        H, W = self.hw
        r = self.layer.render_batch(self._signals(self.pages), 500, stage="image", photo=True,
                                    return_field=True)
        img = r.image.float().clamp(0, 1)
        return img, phase_targets(r.inverse_field.float(), r.metadata, H, W)

    @torch.no_grad()
    def batch(self, n):
        img, t = self.pages_targets()
        P, _, H, W = img.shape
        S = self.size
        xs, ys = [], []
        for _ in range(n):
            for _try in range(4):   # prefer crops that contain grid paper
                p = int(self.rng.integers(P))
                y0 = int(self.rng.integers(0, max(1, H - S + 1))); x0 = int(self.rng.integers(0, max(1, W - S + 1)))
                tt = t[p, :, y0:y0 + S, x0:x0 + S]
                if float(tt[8].mean()) > 0.2:
                    break
            xs.append(img[p, :, y0:y0 + S, x0:x0 + S]); ys.append(tt)
        return torch.stack(xs), torch.stack(ys)


def phase_targets(field, meta, H, W):
    """(P, H, W, 2) inverse field (canonical page, normalised, align_corners=False) -> (P, 10, H, W).

    The renderer draws the grid lines at canonical x = (grid_phase_x + k) * ppm (1 mm) and every
    fifth of them as the 5 mm lines; checked on renders: the darkness on the predicted 5 mm lines
    is clearly above the background, and above the other conventions tried."""
    pr = meta["parameters"]
    ppm = meta["ppm"].float().view(-1, 1, 1)
    xc = (field[..., 0] + 1) / 2 * W - 0.5
    yc = (field[..., 1] + 1) / 2 * H - 0.5
    u = xc / ppm - pr["grid_phase_x"].float().view(-1, 1, 1)
    v = yc / ppm - pr["grid_phase_y"].float().view(-1, 1, 1)
    grid_on = ((pr["grid_present"] > 0) & (pr["grid_strength"] > 0)).float().view(-1, 1, 1)
    paper = (field.abs() <= 1).all(-1) & (yc >= meta["top"].float().view(-1, 1, 1) * 0.8)
    conf = paper.float() * grid_on
    # local photo resolution in px per mm from the Jacobian of (u, v) w.r.t. the photo pixels
    du_x = F.pad(u[:, :, 1:] - u[:, :, :-1], (0, 1), mode="replicate")
    dv_x = F.pad(v[:, :, 1:] - v[:, :, :-1], (0, 1), mode="replicate")
    du_y = F.pad((u[:, 1:] - u[:, :-1])[:, None], (0, 0, 0, 1), mode="replicate")[:, 0]
    dv_y = F.pad((v[:, 1:] - v[:, :-1])[:, None], (0, 0, 0, 1), mode="replicate")[:, 0]
    det = (du_x * dv_y - dv_x * du_y).abs().clamp_min(1e-9)
    px_mm = det.rsqrt()
    w1 = ((px_mm - 2.0) / 2.0).clamp(0, 1) * conf
    tau = 2 * math.pi
    ch = [torch.cos(tau * u / 5), torch.sin(tau * u / 5), torch.cos(tau * v / 5), torch.sin(tau * v / 5),
          torch.cos(tau * u), torch.sin(tau * u), torch.cos(tau * v), torch.sin(tau * v), conf, w1]
    out = torch.stack(ch, 1)
    return torch.where(conf[:, None] > 0, out, torch.zeros_like(out)).nan_to_num(0.0)


# the 8 orientations of the square grid: swap of the two line families, reversal of each axis.
# The loss and the metric take the best one per crop, so the network is never asked to guess the
# page orientation where nothing in the crop shows it (a crop of bare grid in a rotated photo).
def _orient(y, swap, fu, fv):
    a = y[:, [2, 3, 0, 1, 6, 7, 4, 5]] if swap else y[:, :8]
    sg = torch.tensor([1, -1 if fu else 1, 1, -1 if fv else 1] * 2, device=y.device, dtype=y.dtype)
    return a * sg.view(1, 8, 1, 1)


ORIENTS = [(s, a, b) for s in (0, 1) for a in (0, 1) for b in (0, 1)]


def phase_loss(out, y):
    """out: (n, 9, S, S) network output (8 phase channels + confidence logit); y: (n, 10, S, S).
    Masked MSE of the phase channels (1 mm channels weighted by the resolution weight), best of
    the 8 grid orientations per crop, plus BCE of the grid-paper confidence."""
    conf, w1 = y[:, 8:9], y[:, 9:10]
    w = torch.cat([conf.expand(-1, 4, -1, -1), w1.expand(-1, 4, -1, -1)], 1)
    den = w.sum((1, 2, 3)).clamp_min(1.0)
    per = torch.stack([(((out[:, :8] - _orient(y, *o)) ** 2) * w).sum((1, 2, 3)) / den for o in ORIENTS], 1)
    reg = per.min(1).values.mean()
    bce = F.binary_cross_entropy_with_logits(out[:, 8:9], conf)
    return reg + 0.2 * bce


def decode(out):
    """Network output -> (u, v) in mm modulo 5 (5 mm phase, refined by the 1 mm phase) and the
    confidence. out: (n, 9, S, S)."""
    tau = 2 * math.pi
    def axis(c5, s5, c1, s1):
        u5 = torch.atan2(s5, c5) / tau * 5
        u1 = torch.atan2(s1, c1) / tau
        return (u5 + torch.remainder(u1 - u5 + 0.5, 1.0) - 0.5) % 5, u5 % 5
    u, u5 = axis(out[:, 0], out[:, 1], out[:, 4], out[:, 5])
    v, v5 = axis(out[:, 2], out[:, 3], out[:, 6], out[:, 7])
    return u, v, u5, v5, torch.sigmoid(out[:, 8])


def phase_error_mm(out, y):
    """Per-pixel position error (mm), best orientation per crop: the refined error where the photo
    has >= 3 px per mm (1 mm phase usable), and the 5 mm-only error on all grid paper."""
    m = y[:, 8] > 0.5
    m1 = m & (y[:, 9] > 0.5)
    best = None
    for o in ORIENTS:
        t = _orient(y, *o)
        u, v, u5, v5, _ = decode(torch.cat([t, y[:, 8:9]], 1))
        pu, pv, pu5, pv5, _ = decode(out)
        cd = lambda a, b: (torch.remainder(a - b + 2.5, 5.0) - 2.5).abs()  # noqa: E731
        e = torch.maximum(cd(pu, u), cd(pv, v)); e5 = torch.maximum(cd(pu5, u5), cd(pv5, v5))
        s = torch.stack([(e * m).sum((1, 2)) / m.sum((1, 2)).clamp_min(1)])[0]
        if best is None:
            best = [s, e, e5]
        else:
            k = s < best[0]
            best[0] = torch.where(k, s, best[0])
            best[1] = torch.where(k[:, None, None], e, best[1]); best[2] = torch.where(k[:, None, None], e5, best[2])
    return best[1][m1], best[2][m]


def peaks(prob, thr=0.3, k=5):
    """Sub-pixel node positions from a heatmap: local maxima above thr, refined by the weighted
    centroid of their 3x3 neighbourhood. prob: (H, W) tensor -> (N, 2) xy in pixel centres."""
    import torch.nn.functional as F
    p = prob[None, None]
    mx = F.max_pool2d(p, k, 1, k // 2)
    yy, xx = torch.nonzero((p == mx)[0, 0] & (prob > thr), as_tuple=True)
    if not len(yy):
        return torch.zeros((0, 2), device=prob.device)
    pad = F.pad(p, (1, 1, 1, 1))[0, 0]
    off = torch.arange(-1, 2, device=prob.device)
    oy, ox = torch.meshgrid(off, off, indexing="ij")
    w = torch.stack([pad[yy + 1 + dy, xx + 1 + dx] for dy, dx in zip(oy.flatten(), ox.flatten())], 1)
    cx = (w * ox.flatten()[None].float()).sum(1) / w.sum(1).clamp_min(1e-6)
    cy = (w * oy.flatten()[None].float()).sum(1) / w.sum(1).clamp_min(1e-6)
    return torch.stack([xx.float() + cx, yy.float() + cy], 1)


if __name__ == "__main__":
    import sys
    import cv2
    g = GridPoints("cuda" if torch.cuda.is_available() else "cpu", size=384, page_hw=(768, 1024), pages=1)
    x, y = g.batch(4)
    tiles = []
    for a, b in zip(x, y):
        im = (a.permute(1, 2, 0).cpu().numpy()[..., ::-1] * 255).astype(np.uint8).copy()
        for px, py in peaks(b[0]).cpu().numpy():
            cv2.circle(im, (int(round(px)), int(round(py))), 4, (0, 200, 0), 1)
        tiles.append(im)
    cv2.imwrite(sys.argv[1], np.vstack([np.hstack(tiles[:2]), np.hstack(tiles[2:])]))
