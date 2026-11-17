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

import numpy as np
import torch


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
