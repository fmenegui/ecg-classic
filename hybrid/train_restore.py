"""Appearance restorer: degraded ECG page -> the same page, clean (Config.restore = its ONNX).

Pairs come from ecg-torch-renderer: a flat page as rendered (the target) and the same page
through pagegen.degrade (blur, low resolution, faded ink, tone curves, uneven light, noise), so
the geometry is identical and the label is exact. 20% of the crops are left undegraded, so the
network learns to give a clean page back unchanged. Regression only (L1 on the image and on its
gradients), no adversary: it sharpens and cleans what is there and has no reason to invent a
trace. No PM-ECG-ID or Kaggle image is used.

  ECG_BANK=bank.npy python hybrid/train_restore.py --minutes 220 --out OUT
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
import pagegen as PG  # noqa: E402


def build(weights="imagenet", encoder="resnet34"):
    import segmentation_models_pytorch as smp
    return smp.Unet(encoder, encoder_weights=None if weights in ("", "none") else weights, in_channels=3, classes=3)


def grads(x):
    return x[..., 1:, :] - x[..., :-1, :], x[..., :, 1:] - x[..., :, :-1]


def loss_fn(pred, clean):
    l1 = (pred - clean).abs().mean()
    py, px = grads(pred); cy, cx = grads(clean)
    return l1 + 0.5 * ((py - cy).abs().mean() + (px - cx).abs().mean())


class Pairs:
    def __init__(self, dev, pages=4, crop=512, crops=8, seed=0, ident=0.2, heavy=0.6, reuse=4.0):
        self.ps = PG.PageStructure(dev, pages=pages, photo_frac=0.0, heavy=heavy, seed=seed, reuse=1)
        self.dev, self.crop, self.crops, self.ident, self.heavy, self.reuse = dev, crop, crops, ident, heavy, reuse
        self.g = torch.Generator().manual_seed(seed + 1)
        self.rng = np.random.default_rng(seed + 2)
        self.buf = None

    def batch(self):
        if self.buf is None or self.rng.random() < 1.0 / self.reuse:
            img, _, _ = self.ps._render()           # (n, 3, H, W) clean pages in [0, 1]
            self.buf = img.detach()
        img = self.buf
        n, _, H, W = img.shape
        c = self.crop
        X, Y = [], []
        for _ in range(self.crops):
            i = int(self.rng.integers(0, n)); y0 = int(self.rng.integers(0, H - c)); x0 = int(self.rng.integers(0, W - c))
            X.append(img[i:i + 1, :, y0:y0 + c, x0:x0 + c])
        clean = torch.cat(X)
        deg = PG.degrade(clean, self.g, self.heavy)
        keep = torch.rand(len(clean), generator=self.g) < self.ident
        deg[keep] = clean[keep]
        return deg, clean


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--minutes", type=float, default=60)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--crops", type=int, default=8)
    ap.add_argument("--crop", type=int, default=512)
    ap.add_argument("--heavy", type=float, default=0.6)
    ap.add_argument("--heavy-down-max", type=float, default=16.0, help="at the render scale (~7 px/mm)")
    ap.add_argument("--heavy-blur-max", type=float, default=9.0, help="sigma, render-scale px")
    ap.add_argument("--reuse", type=float, default=4.0)
    ap.add_argument("--init", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=Path("hybrid/out_restore"))
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    if dev == "cpu" and os.environ.get("RUNPOD_POD_ID"):
        sys.exit("CUDA is not available on this GPU pod; refusing to train on the CPU")
    PG.HEAVY_DOWN_MAX[0], PG.HEAVY_BLUR_MAX[0] = a.heavy_down_max, a.heavy_blur_max
    torch.backends.cudnn.benchmark = True
    net = build()
    if a.init and a.init.exists():
        print("init from", a.init, net.load_state_dict(torch.load(a.init, map_location="cpu"), strict=False), flush=True)
    net = net.to(dev)
    gen = Pairs(dev, crops=a.crops, crop=a.crop, seed=int(time.time()) % 100000, heavy=a.heavy, reuse=a.reuse)
    gv = Pairs(dev, crops=8, crop=a.crop, seed=999, heavy=a.heavy, reuse=1)
    val = [gv.batch() for _ in range(6)]
    vx = torch.cat([v[0] for v in val]).cpu(); vy = torch.cat([v[1] for v in val]).cpu()
    del gv, val
    opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=1e-4)
    scaler = torch.amp.GradScaler(enabled=dev == "cuda")
    t0 = time.time(); budget = a.minutes * 60; step = 0; best = 1e9; hist = []

    def evaluate():
        net.eval(); e_in = e_out = 0.0
        with torch.no_grad():
            for i in range(0, len(vx), 8):
                x = vx[i:i + 8].to(dev); y = vy[i:i + 8].to(dev)
                p = (x + net(x)).clamp(0, 1)
                e_in += (x - y).abs().mean().item(); e_out += (p - y).abs().mean().item()
        net.train(); k = math.ceil(len(vx) / 8)
        return dict(l1_in=round(e_in / k, 5), l1_out=round(e_out / k, 5))

    def export(tag=""):
        net.eval(); m = net.float().cpu()
        torch.onnx.export(m, torch.zeros(1, 3, 256, 256), str(a.out / f"restore{tag}.onnx"), input_names=["x"],
                          output_names=["y"], dynamic_axes={"x": {0: "n", 2: "h", 3: "w"}, "y": {0: "n", 2: "h", 3: "w"}},
                          opset_version=17, dynamo=False)
        net.to(dev); net.train()

    tl = time.time(); acc = 0.0; nacc = 0
    while time.time() - t0 < budget:
        frac = (time.time() - t0) / budget
        lr = a.lr * min(1.0, frac / 0.02) if frac < 0.02 else a.lr * 0.5 * (1 + math.cos(math.pi * (frac - 0.02) / 0.98)) + 1e-6
        for g_ in opt.param_groups:
            g_["lr"] = lr
        x, y = gen.batch()
        with torch.autocast(dev, enabled=dev == "cuda"):
            r = net(x)
        pred = (x + r.float()).clamp(0, 1)          # residual: the network learns the correction
        loss = loss_fn(pred, y)
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
        acc += loss.item(); nacc += 1; step += 1
        if step % 100 == 0:
            el = time.time() - tl; tl = time.time()
            print(f"step {step} lr {lr:.2e} loss {acc / nacc:.4f} {100 / el:.2f} it/s elapsed {(time.time() - t0) / 60:.1f} min", flush=True)
            acc = 0.0; nacc = 0
        if step % 1000 == 0:
            ev = evaluate(); ev["step"] = step; hist.append(ev); print("val", json.dumps(ev), flush=True)
            if ev["l1_out"] < best:
                best = ev["l1_out"]; torch.save(net.state_dict(), a.out / "restore_best.pt")
            (a.out / "val.json").write_text(json.dumps(hist))
    ev = evaluate(); ev["step"] = step; hist.append(ev); print("final val", json.dumps(ev), flush=True)
    (a.out / "val.json").write_text(json.dumps(hist))
    torch.save(net.state_dict(), a.out / "restore.pt")
    export()
    if (a.out / "restore_best.pt").exists():
        net.load_state_dict(torch.load(a.out / "restore_best.pt", map_location=dev)); export("_best")
    (a.out / "meta.json").write_text(json.dumps(dict(kind="restore", residual=True, render_px_per_mm=7.0,
                                                    page_long_side=1980, crop=a.crop)))
    print("done", step, "steps", flush=True)


if __name__ == "__main__":
    main()
