"""Whole-page training samples for the page-structure network, rendered on the GPU.

The classical reader loses most PM-ECG-ID ECGs before tracing: on ~19% of single-page images it
counts the trace rows wrong (blur, dark or washed-out photos, faint digital traces, QRS that cross
into the next row), and on ~6% more it splits the rows into the wrong columns. The rectified page
itself is usually fine. This network looks at the rectified page and marks its structure:

- channel 0, baseline: a line of Gaussian profile (sigma 0.8 mm) along every printed lead
  occurrence, from its first to its last sample, at its baseline height (rhythm strips included);
- channel 1, lead change: a vertical band (sigma 1 mm) where one lead ends and the next begins in
  the same row, +-5 mm around the baseline;
- channels 2..14, lead name: 0 = no trace band, 1..12 = I..V6 (-aVR counts as aVR), within
  +-3.5 mm of the nearest baseline and inside the occurrence.

Pages come from ecg-torch-renderer (every layout it knows: 3x4, 6x2, 12x1, 2x6, 4x3, +1..3 rhythm
strips, Cabrera, long strips), with PTB-XL signals from ECG_BANK (evaluation records removed). A
share of the pages is photographed and brought back onto the flat page with the exact inverse
field (hybrid/rendergen.rectify), like rendergen. The page is then placed on a canvas at
S px/mm (scale +-12%, rotation sigma 0.4 deg, shift), the way ecgdig's rectified image shows a page
(margins, background beyond the paper), and degraded (blur, low resolution, tone curve,
brightness, faded content, uneven light, noise). No test image is used. The labels are computed
from the renderer's metadata in page coordinates, so they stay exact under the warp.
"""
from __future__ import annotations

import math
import os

import numpy as np
import torch
import torch.nn.functional as F

S_MM = 4.0          # px/mm of the network input (ecgdig's rectified image is resized to this)
N_CLS = 13          # background + 12 leads
CH = 2 + N_CLS


def _blur(x, s):
    k = int(2 * round(3 * s) + 1)
    ax = torch.arange(k, device=x.device, dtype=x.dtype) - k // 2
    w = torch.exp(-ax ** 2 / (2 * s * s)); w = w / w.sum()
    C = x.shape[1]
    x = F.conv2d(F.pad(x, (k // 2,) * 4, mode="replicate"), w.view(1, 1, 1, k).repeat(C, 1, 1, 1), groups=C)
    return F.conv2d(x, w.view(1, 1, k, 1).repeat(C, 1, 1, 1), groups=C)


def degrade(x, g, heavy=0.5):
    """Capture degradations of the canvas (n, 3, H, W) in [0, 1]."""
    n, _, H, W = x.shape
    dev = x.device
    r = lambda: float(torch.rand(1, generator=g))
    out = []
    for i in range(n):
        xi = x[i:i + 1]
        if r() < 0.5:
            xi = xi[:, torch.randperm(3, generator=g).to(dev)]
        if r() < 0.15:
            xi = xi.mean(1, keepdim=True).repeat(1, 3, 1, 1)
        hv = r() < heavy
        if r() < (0.6 if hv else 0.3):    # faded content: everything pulled toward the paper colour
            paper = xi.flatten(2).median(2).values[..., None, None]
            a = 0.12 + 0.88 * r() if hv else 0.5 + 0.5 * r()
            xi = paper + a * (xi - paper)
        if r() < (0.5 if hv else 0.15):   # low resolution
            k = 1.5 + (3.0 if hv else 1.0) * r()
            xi = F.interpolate(F.interpolate(xi, size=(max(8, int(H / k)), max(8, int(W / k))), mode="area"),
                               size=(H, W), mode="bilinear", align_corners=False)
        if r() < (0.6 if hv else 0.3):    # out of focus
            xi = _blur(xi, 0.4 + (2.6 if hv else 0.8) * r())
        if r() < 0.5:                     # uneven light: smooth multiplicative field
            f = torch.rand(1, 1, 4, 6, generator=g).to(dev)
            f = F.interpolate(f, size=(H, W), mode="bicubic", align_corners=False)
            xi = xi * (1 - (0.6 if hv else 0.3) * r() * f)
        if r() < (0.6 if hv else 0.4):    # tone curve
            lim = math.log(8.0 if hv else 2.0)
            xi = xi.clamp(1e-4, 1) ** float(math.exp((2 * r() - 1) * lim))
        if r() < (0.5 if hv else 0.3):    # brightness / contrast, saturating
            c = 0.5 + r() if hv else 0.8 + 0.4 * r()
            xi = (xi - 0.5) * c + 0.5 + (2 * r() - 1) * (0.5 if hv else 0.12)
        xi = xi + torch.randn_like(xi) * (0.06 if hv else 0.025) * r()
        out.append(xi.clamp(0, 1))
    return torch.cat(out)


class PageStructure:
    def __init__(self, device, canvas=(896, 1280), page_hw=(1400, 1980), pages=4, photo_frac=0.4,
                 heavy=0.5, seed=0, reuse=1, buf_size=600):
        from ecg_gpu import ECGSupervisionLayer, RenderConfig
        if os.environ.get("RENDER_PAGE"):            # e.g. "1060x1500" for a small CPU test
            page_hw = tuple(int(v) for v in os.environ["RENDER_PAGE"].split("x"))
        self.dev = torch.device(device)
        self.canvas, self.pages, self.photo_frac, self.heavy = canvas, pages, photo_frac, heavy
        self.reuse, self.buf_size, self._buf = reuse, buf_size, []
        self.synth = ECGSupervisionLayer(fs=500, config=RenderConfig(height=page_hw[0], width=page_hw[1]),
                                         photo=False).to(self.dev)
        self.synth.train()
        p = os.environ.get("ECG_BANK", "")
        self.bank = np.load(p, mmap_mode="r") if p and os.path.exists(p) else None
        self.rng = np.random.default_rng(seed)
        self.g = torch.Generator().manual_seed(seed)

    def _signals(self, n):
        if self.bank is not None:
            x = np.stack([np.asarray(self.bank[i], np.float32) for i in self.rng.integers(0, len(self.bank), n)])
        else:
            from ecgdig.synth import synth_signals
            x = np.stack([synth_signals(self.rng)[:, :5000].astype(np.float32) for _ in range(n)])
        return torch.from_numpy(np.nan_to_num(x)).to(self.dev)

    def _pages(self):
        """Pages for one batch: freshly rendered, or (reuse > 1) drawn from a CPU buffer of the
        last renders, each page reused about `reuse` times under new placement and degradation.
        Rendering is ~55% of a training step on the GPU (profile_gen.py)."""
        if self.reuse <= 1:
            return self._render()
        if not self._buf or self.rng.random() < 1.0 / self.reuse:
            img, seen, m = self._render()
            for p in range(img.shape[0]):
                mp = {}
                for k, v in m.items():
                    if torch.is_tensor(v) and v.dim() > 0 and len(v) == img.shape[0]:
                        v = v[p].cpu()
                        if v.dim() == 1 and k != "ppm":   # per-occurrence: pad to a fixed count
                            v = F.pad(v, (0, max(0, 96 - len(v))))[:96]
                    mp[k] = v

                self._buf.append(((img[p] * 255).round().byte().cpu(), seen[p].bool().cpu(), mp))
            del self._buf[:-self.buf_size]
            return img, seen, m
        idx = self.rng.integers(0, len(self._buf), self.pages)
        img = torch.stack([self._buf[i][0] for i in idx]).to(self.dev).float() / 255
        seen = torch.stack([self._buf[i][1] for i in idx]).to(self.dev).float()
        m = {}
        for k in set().union(*(self._buf[i][2].keys() for i in idx)):
            # flat and photo renders carry different metadata (some None in one of them): keep a
            # key only when every page has it as a tensor of one shape; batch() reads only those
            vals = [self._buf[i][2].get(k) for i in idx]
            if all(torch.is_tensor(v) for v in vals) and len({tuple(v.shape) for v in vals}) == 1:
                m[k] = torch.stack(vals).to(self.dev)
        return img, seen, m

    def _render(self):
        sig = self._signals(self.pages)
        if self.rng.random() < self.photo_frac:
            from rendergen import rectify
            rb = self.synth.render_batch(sig, stage="image", photo=True, return_field=True)
            got = [rectify(rb.image[q].float(), rb.inverse_field[q].float()) for q in range(len(sig))]
            img = torch.stack([g_[0] for g_ in got]).clamp(0, 1)
            seen = torch.stack([g_[1] for g_ in got]).float()[:, None]
        else:
            rb = self.synth.render_batch(sig, stage="all", photo=False, return_instances=True)
            img = rb.image.float().clamp(0, 1)
            seen = torch.ones_like(img[:, :1])
        m = {k: (v.float() if torch.is_tensor(v) else v) for k, v in rb.metadata.items()}
        return img, seen, m

    def _distractors(self, im, free, xs, ys, inside):
        """Things on real pages that are not leads, drawn where no lead is (the labels stay 0):
        frame and separator lines, text lines, a QR-like block, stamps. The renderer's pages
        have few of them, and the first network marked grid frames and text lines as rows."""
        r = self.rng
        dev = im.device
        Hc, Wc = im.shape[2:]
        m = torch.zeros(Hc, Wc, device=dev)
        px = S_MM
        for _ in range(int(r.integers(0, 6))):            # long straight lines
            th = float(r.uniform(0.5, 2.5))
            if r.random() < 0.7:
                y0 = float(r.uniform(0, Hc)); xa = float(r.uniform(0, 0.5 * Wc)); xb = float(r.uniform(0.5 * Wc, Wc))
                m = torch.maximum(m, ((ys - y0).abs() <= th / 2).float() * ((xs >= xa) & (xs <= xb)).float())
            else:
                x0 = float(r.uniform(0, Wc)); ya = float(r.uniform(0, 0.5 * Hc)); yb = float(r.uniform(0.5 * Hc, Hc))
                m = torch.maximum(m, ((xs - x0).abs() <= th / 2).float() * ((ys >= ya) & (ys <= yb)).float())
        for _ in range(int(r.integers(0, 8))):            # text lines: runs of character blobs
            y0 = float(r.uniform(0, Hc)); x0 = float(r.uniform(0, Wc)); n = int(r.integers(5, 40))
            ch = float(r.uniform(1.5, 4.0)) * px; cw = ch * float(r.uniform(0.5, 0.8))
            for k in range(n):
                xa = x0 + k * cw * 1.2
                if xa > Wc or r.random() < 0.15:
                    continue
                hh = ch * float(r.uniform(0.6, 1.0))
                m = torch.maximum(m, ((xs >= xa) & (xs <= xa + cw * 0.8) & (ys >= y0 + ch - hh) & (ys <= y0 + ch)).float()
                                  * float(r.uniform(0.5, 1.0)))
        if r.random() < 0.3:                              # QR-like block (slicing, no gather)
            n = int(r.integers(15, 30)); cell = max(1, int(round(float(r.uniform(0.6, 1.2)) * px)))
            size = n * cell
            if size < min(Hc, Wc):
                x0 = int(r.integers(0, Wc - size)); y0 = int(r.integers(0, Hc - size))
                blk = torch.from_numpy((r.random((n, n)) < 0.5).astype(np.float32)).to(dev)
                blk = blk.repeat_interleave(cell, 0).repeat_interleave(cell, 1)
                m[y0:y0 + size, x0:x0 + size] = torch.maximum(m[y0:y0 + size, x0:x0 + size], blk)
        if not bool(m.any()):
            return im
        m = (m * free.float() * inside[0, 0])[None, None]
        col = torch.rand(1, 3, 1, 1, generator=self.g).to(dev) * 0.35
        a = float(r.uniform(0.5, 1.0))
        return im * (1 - a * m) + col * a * m

    @torch.no_grad()
    def batch(self, n=None):
        """Canvases (P, 3, Hc, Wc) in [0, 1], targets (P, 3, Hc, Wc): baseline, lead change, class
        (as float), and weights (P, 1, Hc, Wc): 0 where the page was not seen."""
        img, seen, m = self._pages()
        P, _, H, W = img.shape
        Hc, Wc = self.canvas
        dev = self.dev
        ys, xs = torch.meshgrid(torch.arange(Hc, device=dev, dtype=torch.float32),
                                torch.arange(Wc, device=dev, dtype=torch.float32), indexing="ij")
        X, T, Wt = [], [], []
        for p in range(P):
            ppm = float(m["ppm"][p])
            s = S_MM * float(self.rng.uniform(0.88, 1.12)) / ppm      # canvas px per page px
            th = math.radians(float(self.rng.normal(0, 0.4)))
            # page centre -> canvas centre + shift (the page may be cut at the canvas border)
            cx = Wc / 2 + float(self.rng.uniform(-0.12, 0.12)) * Wc
            cy = Hc / 2 + float(self.rng.uniform(-0.12, 0.12)) * Hc
            c_, s_ = math.cos(th), math.sin(th)
            u, v = (xs - cx) / s, (ys - cy) / s                        # canvas -> page (inverse map)
            px = c_ * u + s_ * v + W / 2
            py = -s_ * u + c_ * v + H / 2
            grid = torch.stack([(px + 0.5) / W * 2 - 1, (py + 0.5) / H * 2 - 1], -1)[None]
            im = F.grid_sample(img[p:p + 1], grid, mode="bilinear", padding_mode="zeros", align_corners=False)
            inside = ((px >= 0) & (px <= W - 1) & (py >= 0) & (py <= H - 1)).float()[None, None]
            vis = F.grid_sample(seen[p:p + 1], grid, mode="nearest", padding_mode="zeros", align_corners=False) * inside
            # background beyond the paper: a flat colour with low-frequency texture, sometimes
            # with corrugated-cardboard stripes and fine grain
            bg = torch.rand(1, 3, 1, 1, generator=self.g).to(dev) * 0.8 + 0.1
            tex = F.interpolate(torch.rand(1, 1, 12, 16, generator=self.g).to(dev), size=(Hc, Wc), mode="bicubic",
                                align_corners=False)
            bg = bg + 0.15 * (tex - 0.5)
            if self.rng.random() < 0.5:
                per = float(self.rng.uniform(6, 30)); ang = float(self.rng.uniform(0, math.pi))
                ph = (xs * math.cos(ang) + ys * math.sin(ang)) * 2 * math.pi / per
                bg = bg + float(self.rng.uniform(0.03, 0.15)) * torch.sin(ph)[None, None]
                bg = bg + 0.05 * torch.randn(1, 1, Hc, Wc, device=dev)
            bg = bg.clamp(0, 1)
            im = im * inside + bg * (1 - inside)
            # ---- targets from the metadata, in page pixels
            act = [k for k in (m["active"][p] > 0.5).nonzero().flatten().tolist()]
            base_t = torch.zeros(Hc, Wc, device=dev)
            bnd_t = torch.zeros(Hc, Wc, device=dev)
            best_d = torch.full((Hc, Wc), 1e9, device=dev)
            cls_t = torch.zeros(Hc, Wc, device=dev)
            rows = {}
            for k in act:
                rows.setdefault(int(m["row"][p, k]), []).append(k)
            for r_, ks in rows.items():
                ks = sorted(ks, key=lambda k: float(m["x0"][p, k]))
                for j, k in enumerate(ks):
                    x0, x1, b = float(m["x0"][p, k]), float(m["x1"][p, k]), float(m["baseline"][p, k])
                    dx = torch.clamp(torch.maximum(x0 - px, px - x1), min=0) / ppm
                    dy = (py - b) / ppm
                    base_t = torch.maximum(base_t, torch.exp(-(dx ** 2 + dy ** 2) / (2 * 0.8 ** 2)))
                    if j > 0:   # lead change between ks[j-1] and ks[j]
                        xb = 0.5 * (x0 + float(m["x1"][p, ks[j - 1]]))
                        bt = torch.exp(-((px - xb) / ppm) ** 2 / (2 * 1.0 ** 2)) * (dy.abs() <= 5.0)
                        bnd_t = torch.maximum(bnd_t, bt)
                    lead = int(m["lead_id"][p, k])
                    ins = (px >= x0) & (px <= x1) & (dy.abs() <= 3.5) & (dy.abs() < best_d)
                    cls_t = torch.where(ins, torch.full_like(cls_t, lead + 1), cls_t)
                    best_d = torch.where(ins, dy.abs(), best_d)
            tg = torch.stack([base_t, bnd_t, cls_t])[None] * inside
            k_ = int(2 * 6 * S_MM) | 1    # nothing drawn within ~6 mm of a lead band
            near = F.max_pool2d((cls_t > 0).float()[None, None], (k_, 9), 1, (k_ // 2, 4))[0, 0] > 0
            im = self._distractors(im, ~near & (base_t < 0.05) & (bnd_t < 0.05), xs, ys, inside)
            X.append(im); T.append(tg); Wt.append(vis)
        X = degrade(torch.cat(X), self.g, self.heavy)
        return X, torch.cat(T), torch.cat(Wt)


if __name__ == "__main__":
    import sys
    import cv2
    os.environ.setdefault("RENDER_PAGE", "1060x1500")
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    ps = PageStructure("cuda" if torch.cuda.is_available() else "cpu", canvas=(896, 1280), pages=2,
                       photo_frac=float(sys.argv[2]) if len(sys.argv) > 2 else 0.0)
    x, t, w = ps.batch()
    tiles = []
    for a, b, c in zip(x, t, w):
        im = (a.permute(1, 2, 0).cpu().numpy()[..., ::-1] * 255).astype(np.uint8).copy()
        ov = im.copy()
        ov[b[0].cpu().numpy() > 0.5] = (0, 200, 0)
        ov[b[1].cpu().numpy() > 0.5] = (0, 0, 255)
        cl = b[2].cpu().numpy().astype(int)
        lut = (np.random.default_rng(1).random((13, 3)) * 255).astype(np.uint8); lut[0] = 0
        cm = lut[cl]
        tiles.append(np.hstack([im, ov, np.where(cl[..., None] > 0, cm, im // 3)]))
    cv2.imwrite(sys.argv[1], np.vstack(tiles))
