"""Dense page-flattening network: photo -> flat-page coordinate of every photo pixel.

    ECG_BANK=bank.npy python hybrid/train_dewarp.py --minutes 240 --out /workspace/out

The classical geometry fits the grid with a homography plus smooth corrections; a bent or
crumpled sheet, a strong perspective or a page that fills a small part of the photo defeats it
(PM-ECG-ID photos_bents / crumbles ~0 dB, perspective ~8 dB; the Kaggle phone photos lose 30%
of their images below 10 dB). ecg-torch-renderer photographs its pages with perspective, lens
distortion, curl and creases and returns the exact field (photo pixel -> normalised flat-page
coordinate). A U-Net learns that field from the photo (plus a page-visibility mask); ecgdig
inverts it by Newton iterations (ecgdig/dewarp.py, the same scheme as hybrid/rendergen.rectify)
and reads the flattened page.

Every batch is rendered on the GPU from PTB-XL signals (ECG_BANK, evaluation records removed);
no test image is used. Crops with random aspect and position make the network independent of
the photo's framing; the degradations of the page network (blur, low resolution, tone curve,
faded content, noise) are applied on top.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pagegen import degrade  # noqa: E402

IN_HW = (512, 704)


class Photos:
    def __init__(self, device, page_hw=(1400, 1980), pages=4, heavy=0.4, seed=0, reuse=1, buf_size=500):
        from ecg_gpu import ECGSupervisionLayer, RenderConfig
        if os.environ.get("RENDER_PAGE"):
            page_hw = tuple(int(v) for v in os.environ["RENDER_PAGE"].split("x"))
        self.dev = torch.device(device)
        self.pages, self.heavy = pages, heavy
        self.reuse, self.buf_size, self._buf = reuse, buf_size, []
        self.synth = ECGSupervisionLayer(fs=500, config=RenderConfig(height=page_hw[0], width=page_hw[1]),
                                         photo=True).to(self.dev)
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

    def _render(self):
        rb = self.synth.render_batch(self._signals(self.pages), stage="image", photo=True, return_field=True)
        return rb.image.float().clamp(0, 1), rb.inverse_field.float()          # (P, 3, H, W), (P, H, W, 2)

    def _photos(self):
        """Fresh renders, or (reuse > 1) photos from a CPU buffer of the last renders, each used
        about `reuse` times under a new crop and new degradations (rendering dominates the step)."""
        if self.reuse <= 1:
            return self._render()
        if not self._buf or self.rng.random() < 1.0 / self.reuse:
            img, fld = self._render()
            for p in range(len(img)):
                self._buf.append(((img[p] * 255).round().byte().cpu(), fld[p].half().cpu()))
            del self._buf[:-self.buf_size]
            return img, fld
        idx = self.rng.integers(0, len(self._buf), self.pages)
        img = torch.stack([self._buf[i][0] for i in idx]).to(self.dev).float() / 255
        fld = torch.stack([self._buf[i][1] for i in idx]).to(self.dev).float()
        return img, fld

    @torch.no_grad()
    def batch(self):
        """Photos (P, 3, h, w) in [0, 1]; targets (P, 3, h, w): flat-page x, y (normalised, -1..1
        on the page) and page visibility (1 where the photo pixel shows the page)."""
        img, fld = self._photos()
        P, _, H, W = img.shape
        X, T = [], []
        for p in range(P):
            # random crop: 70-100% of each side, independent aspect
            ch = int(H * self.rng.uniform(0.7, 1.0)); cw = int(W * self.rng.uniform(0.7, 1.0))
            y0 = int(self.rng.integers(0, H - ch + 1)); x0 = int(self.rng.integers(0, W - cw + 1))
            if self.rng.random() < 0.3:   # portrait framing of a landscape page, or the reverse
                ch, cw = min(H, int(cw * 1.3)), cw
                y0 = int(self.rng.integers(0, H - ch + 1))
            im = img[p:p + 1, :, y0:y0 + ch, x0:x0 + cw]
            f = fld[p, y0:y0 + ch, x0:x0 + cw].permute(2, 0, 1)[None]
            im = F.interpolate(im, size=IN_HW, mode="area")
            vis = ((f.abs() <= 1.0).all(1, keepdim=True)).float()
            f = F.interpolate(f, size=IN_HW, mode="bilinear", align_corners=False)
            vis = (F.interpolate(vis, size=IN_HW, mode="bilinear", align_corners=False) > 0.5).float()
            X.append(im); T.append(torch.cat([f.clamp(-1.5, 1.5), vis], 1))
        X = degrade(torch.cat(X), self.g, self.heavy)
        return X, torch.cat(T)


def build(weights="imagenet"):
    import segmentation_models_pytorch as smp
    return smp.Unet(encoder_name="resnet34", encoder_weights=None if weights in ("", "none") else weights,
                    in_channels=3, classes=3)


def loss_fn(out, t):
    vis = t[:, 2:3]
    coord = out[:, :2]
    d = vis.sum().clamp(min=1)
    l1 = ((coord - t[:, :2]).abs() * vis).sum() / (2 * d)
    # first differences: a sheet is smooth even where a crease bends it
    gx = (coord[..., 1:] - coord[..., :-1]) - (t[:, :2, :, 1:] - t[:, :2, :, :-1])
    gy = (coord[..., 1:, :] - coord[..., :-1, :]) - (t[:, :2, 1:] - t[:, :2, :-1])
    lg = (gx.abs() * vis[..., 1:]).sum() / (2 * d) + (gy.abs() * vis[:, :, 1:]).sum() / (2 * d)
    lm = F.binary_cross_entropy_with_logits(out[:, 2:3], vis)
    return l1, lg, lm


def evaluate(net, vx, vt, dev):
    """Median page-coordinate error in mm (A4 landscape, 297 x 210 mm) on visible pixels."""
    net.eval(); errs = []; acc = []
    with torch.no_grad():
        for i in range(0, len(vx), 4):
            with torch.autocast(dev, enabled=dev == "cuda"):
                o = net(vx[i:i + 4].to(dev)).float().cpu()
            t = vt[i:i + 4]
            v = t[:, 2] > 0.5
            ex = (o[:, 0] - t[:, 0]) * 297 / 2; ey = (o[:, 1] - t[:, 1]) * 210 / 2
            errs.append(torch.sqrt(ex ** 2 + ey ** 2)[v])
            acc.append(((o[:, 2] > 0) == v).float().mean())
    net.train()
    e = torch.cat(errs)
    return dict(err_mm_median=float(e.median()), err_mm_p90=float(e.quantile(0.9)) if len(e) < 16_000_000 else None,
                mask_acc=float(torch.stack(acc).mean()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--minutes", type=float, default=60)
    ap.add_argument("--steps", type=int, default=10 ** 9)
    ap.add_argument("--pages", type=int, default=6)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--heavy", type=float, default=0.4)
    ap.add_argument("--reuse", type=float, default=1.0, help="each rendered photo is used ~this many times")
    ap.add_argument("--val", type=int, default=48)
    ap.add_argument("--weights", default="imagenet")
    ap.add_argument("--init", type=Path, default=None, help="state_dict to start from (an earlier run)")
    ap.add_argument("--out", type=Path, default=Path("hybrid/out_dewarp"))
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    if dev == "cpu" and os.environ.get("RUNPOD_POD_ID"):
        sys.exit("CUDA is not available on this GPU pod; refusing to train on the CPU")
    torch.backends.cudnn.benchmark = True
    net = build(a.weights)
    if (a.out / "dewarp.pt").exists():
        print("resuming", net.load_state_dict(torch.load(a.out / "dewarp.pt", map_location="cpu")), flush=True)
    elif a.init and a.init.exists():
        print("init from", a.init, net.load_state_dict(torch.load(a.init, map_location="cpu")), flush=True)
    net = net.to(dev)
    gen = Photos(dev, pages=a.pages, heavy=a.heavy, seed=int(time.time()) % 100000, reuse=a.reuse)
    gv = Photos(dev, pages=4, heavy=a.heavy, seed=999)
    vb = [gv.batch() for _ in range(max(1, a.val // 4))]
    vx = torch.cat([b[0] for b in vb]).cpu(); vt = torch.cat([b[1] for b in vb]).cpu()
    del gv, vb
    opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=1e-4)
    scaler = torch.amp.GradScaler(enabled=dev == "cuda")
    t0 = time.time(); budget = a.minutes * 60
    step, best, hist = 0, 1e9, []
    acc = np.zeros(3); n_acc = 0; tl = time.time()

    def export(tag=""):
        net.eval(); m = net.float().cpu()
        torch.onnx.export(m, torch.zeros(1, 3, *IN_HW), str(a.out / f"dewarp{tag}.onnx"), input_names=["x"],
                          output_names=["y"], dynamic_axes={"x": {0: "n"}, "y": {0: "n"}}, opset_version=17, dynamo=False)
        net.to(dev); net.train()

    while step < a.steps and time.time() - t0 < budget:
        frac = (time.time() - t0) / budget
        lr = a.lr * min(1.0, frac / 0.02) if frac < 0.02 else a.lr * 0.5 * (1 + math.cos(math.pi * (frac - 0.02) / 0.98)) + 1e-6
        for g_ in opt.param_groups:
            g_["lr"] = lr
        x, t = gen.batch()
        with torch.autocast(dev, enabled=dev == "cuda"):
            out = net(x)
        l1, lg, lm = loss_fn(out.float(), t)
        loss = l1 + 0.5 * lg + 0.1 * lm
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
        acc += [l1.item(), lg.item(), lm.item()]; n_acc += 1; step += 1
        if step % 50 == 0:
            el = time.time() - tl; tl = time.time()
            print(f"step {step} lr {lr:.2e} l1 {acc[0]/n_acc:.4f} grad {acc[1]/n_acc:.4f} mask {acc[2]/n_acc:.4f} "
                  f"{50/el:.2f} it/s elapsed {(time.time()-t0)/60:.1f} min", flush=True)
            acc[:] = 0; n_acc = 0
        if step % 1000 == 0:
            ev = evaluate(net, vx, vt, dev); ev["step"] = step; hist.append(ev)
            print("val", json.dumps(ev), flush=True)
            torch.save(net.state_dict(), a.out / "dewarp.pt")
            if ev["err_mm_median"] < best:
                best = ev["err_mm_median"]
                torch.save(net.state_dict(), a.out / "dewarp_best.pt")
                export("_best")
                print("best", round(best, 3), flush=True)
            (a.out / "val.json").write_text(json.dumps(hist))
    ev = evaluate(net, vx, vt, dev); ev["step"] = step; hist.append(ev)
    print("final val", json.dumps(ev), flush=True)
    (a.out / "val.json").write_text(json.dumps(hist))
    export()
    (a.out / "dewarp.pt").unlink(missing_ok=True)
    print("done", step, "steps", flush=True)


if __name__ == "__main__":
    main()
