"""Row-network strips rendered on the GPU with ecg-torch-renderer (fmenegui/ecg-torch-renderer).

The CPU generator (hybrid/rowgen.py) limits training speed: the GPU waits for crops. The renderer
draws whole ECG pages on the GPU (layouts, fonts, paper, print, pulses, labels) and returns, for
every printed occurrence, its row, x range, baseline, time window, polarity, pixels per mm, speed
and gain. That is enough to cut row strips like rowgen's and to label them exactly:

- the page is rendered as a rectified document (photo=False) at ~10 px/mm, the resolution of the
  rectified image in ecgdig;
- a strip of size x size pixels is sampled around a row's baseline (+-1.5 mm of estimation error,
  as in rowgen) over a random x window;
- the label is sparse: in each strip column, the height of that row's signal (every occurrence of
  the row, so lead changes are included) split over the two nearest pixels. Other rows, grid,
  pulses and text are background.

Signals come from the same PTB-XL bank as rowgen (ECG_BANK, evaluation records already removed).
Verified on CPU: the metadata places the source signal within ~0.1 px of the drawn trace centre.
"""
from __future__ import annotations

import os

import numpy as np
import torch
import torch.nn.functional as F

R = 10.0   # px/mm of the strips (ecgdig's rectified image)


class TorchStrips:
    def __init__(self, device, size=512, page_hw=(1920, 2712), pages=2, seed=0):
        from ecg_gpu import ECGSupervisionLayer, RenderConfig
        if os.environ.get("RENDER_PAGE"):            # e.g. "1060x1500" for a small CPU test
            page_hw = tuple(int(v) for v in os.environ["RENDER_PAGE"].split("x"))
        self.dev = torch.device(device)
        self.size, self.pages = size, pages
        self.synth = ECGSupervisionLayer(fs=500, config=RenderConfig(height=page_hw[0], width=page_hw[1]),
                                         photo=False).to(self.dev)
        self.synth.train()   # random capture parameters on every call
        p = os.environ.get("ECG_BANK", "")
        self.bank = np.load(p, mmap_mode="r") if p and os.path.exists(p) else None
        self.rng = np.random.default_rng(seed)

    def _signals(self, n):
        if self.bank is not None:
            idx = self.rng.integers(0, len(self.bank), n)
            x = np.stack([np.asarray(self.bank[i], np.float32) for i in idx])
        else:
            from ecgdig.synth import synth_signals
            x = np.stack([synth_signals(self.rng)[:, :5000].astype(np.float32) for _ in range(n)])
        return torch.from_numpy(np.nan_to_num(x)).to(self.dev)

    @torch.no_grad()
    def batch(self, n):
        """n strips: images (n, 3, size, size) in [0, 1] RGB and sparse labels (n, 1, size, size)."""
        S = self.size
        sig = self._signals(self.pages)
        rb = self.synth.render_batch(sig, stage="all", photo=False, return_instances=True)
        img = rb.image.float().clamp(0, 1)                       # (P, 3, H, W)
        ref = rb.reference.float()                               # (P, 12, T) mV, source
        m = {k: (v.float() if torch.is_tensor(v) else v) for k, v in rb.metadata.items()}
        P, _, H, W = img.shape
        fs = float(m["sample_rate_hz"][0]) if "sample_rate_hz" in m else 500.0
        # rows of each page: occurrences grouped by their row index
        rows = []
        for p in range(P):
            act = (m["active"][p] > 0.5).nonzero().flatten().tolist()
            by = {}
            for k in act:
                by.setdefault(int(m["row"][p, k]), []).append(k)
            rows += [(p, ks) for ks in by.values()]
        if not rows:
            return None
        xs_out, ys_out = [], []
        j = torch.arange(S, device=self.dev, dtype=torch.float32)
        for _ in range(n):
            p, ks = rows[int(self.rng.integers(len(rows)))]
            ppm = float(m["ppm"][p]); sp = float(m["speed"][p]); g = float(m["gain"][p])
            s = ppm / R                                          # page px per strip px
            base = float(np.mean([float(m["baseline"][p, k]) for k in ks]))
            yc = base + float(np.clip(self.rng.normal(0, 1.5), -4, 4)) * ppm
            x_lo = min(float(m["x0"][p, k]) for k in ks); x_hi = max(float(m["x1"][p, k]) for k in ks)
            span = S * s
            xs = float(self.rng.uniform(x_lo - 0.3 * span, max(x_lo - 0.3 * span + 1, x_hi - 0.7 * span)))
            # page coordinates (pixel-centre convention) of the strip pixel centres
            px = xs + j * s
            py = yc + (j - S / 2) * s
            gx = (px + 0.5) / W * 2 - 1
            gy = (py + 0.5) / H * 2 - 1
            grid = torch.stack(torch.meshgrid(gy, gx, indexing="ij")[::-1], -1)[None]   # (1, S, S, 2) as (x, y)
            xs_out.append(F.grid_sample(img[p:p + 1], grid, mode="bilinear", padding_mode="border",
                                        align_corners=False)[0])
            # label: the row's signal height in every strip column
            yl = torch.full((S,), float("nan"), device=self.dev)
            for k in ks:
                x0 = float(m["x0"][p, k]); x1 = float(m["x1"][p, k]); s0 = float(m["start_s"][p, k])
                sgn = float(m["amplitude_sign"][p, k]); lead = int(m["lead_id"][p, k])
                inside = (px >= x0) & (px <= x1)
                if not inside.any():
                    continue
                t = s0 + (px + 0.5 - x0) / (ppm * sp)
                ti = (t * fs).clamp(0, ref.shape[-1] - 1.001)
                i0 = ti.floor().long(); fr = ti - i0
                v = ref[p, lead, i0] * (1 - fr) + ref[p, lead, (i0 + 1).clamp(max=ref.shape[-1] - 1)] * fr
                yp = float(m["baseline"][p, k]) - sgn * v * g * ppm
                yl = torch.where(inside, (yp - yc) / s + S / 2, yl)
            lab = torch.zeros((S, S), device=self.dev)
            ok = torch.isfinite(yl)
            i = torch.floor(torch.nan_to_num(yl, nan=-10.0)).long()
            f = torch.nan_to_num(yl, nan=0.0) - i
            c = torch.nonzero(ok & (i >= 0) & (i + 1 < S)).flatten()
            lab[i[c], c] = 1 - f[c]
            lab[i[c] + 1, c] = f[c]
            ys_out.append(lab[None])
        return torch.stack(xs_out), torch.stack(ys_out)


if __name__ == "__main__":
    import sys
    import cv2
    ts = TorchStrips("cuda" if torch.cuda.is_available() else "cpu", size=384, page_hw=(1060, 1500), pages=1)
    x, y = ts.batch(6)
    tiles = []
    for a, b in zip(x, y):
        im = (a.permute(1, 2, 0).cpu().numpy()[..., ::-1] * 255).astype(np.uint8).copy()
        ov = im.copy()
        ov[cv2.dilate((b[0].cpu().numpy() > 0.2).astype(np.uint8), np.ones((3, 3), np.uint8)) > 0] = (0, 200, 0)
        tiles.append(np.hstack([im, ov]))
    cv2.imwrite(sys.argv[1], np.vstack(tiles))
