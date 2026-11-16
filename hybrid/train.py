"""Train the small trace-segmenter U-Net on CPU and export ONNX.

    python hybrid/train.py --steps 6000 --out hybrid/out

Data: hybrid/gen.py, generated on the fly (no PhysioNet/Kaggle image).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from gen import sample  # noqa: E402


class Block(nn.Module):
    def __init__(self, a, b):
        super().__init__()
        self.c = nn.Sequential(nn.Conv2d(a, b, 3, padding=1, bias=False), nn.BatchNorm2d(b), nn.ReLU(inplace=True),
                               nn.Conv2d(b, b, 3, padding=1, bias=False), nn.BatchNorm2d(b), nn.ReLU(inplace=True))

    def forward(self, x):
        return self.c(x)


class UNet(nn.Module):
    """4 levels, small channels (~0.5 M parameters): receptive field ~7 mm at 10 px/mm."""

    def __init__(self, ch=(16, 32, 64, 96)):
        super().__init__()
        self.e = nn.ModuleList()
        a = 3
        for c in ch:
            self.e.append(Block(a, c)); a = c
        self.d = nn.ModuleList()
        for i in range(len(ch) - 1, 0, -1):
            self.d.append(Block(ch[i] + ch[i - 1], ch[i - 1]))
        self.out = nn.Conv2d(ch[0], 1, 1)

    def forward(self, x):
        hs = []
        for i, b in enumerate(self.e):
            x = b(x)
            if i < len(self.e) - 1:
                hs.append(x)
                x = F.max_pool2d(x, 2)
        for b in self.d:
            h = hs.pop()
            x = F.interpolate(x, size=h.shape[-2:], mode="bilinear", align_corners=False)
            x = b(torch.cat([x, h], 1))
        return self.out(x)


class Stream(torch.utils.data.IterableDataset):
    def __init__(self, size, seed):
        self.size, self.seed = size, seed

    def __iter__(self):
        wi = torch.utils.data.get_worker_info()
        rng = np.random.default_rng(self.seed + (wi.id if wi else 0) * 7919 + int(time.time()) % 1000)
        while True:
            im, lb = sample(rng, self.size)
            x = torch.from_numpy(im[..., ::-1].copy()).permute(2, 0, 1).float() / 255.0  # RGB
            yield x, torch.from_numpy(lb)[None]


def dice_loss(logit, y):
    p = torch.sigmoid(logit)
    num = 2 * (p * y).sum((1, 2, 3)) + 1
    den = (p + y).sum((1, 2, 3)) + 1
    return (1 - num / den).mean()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--size", type=int, default=192)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--minutes", type=float, default=0, help="stop earlier, by time (0 = no limit)")
    ap.add_argument("--out", type=Path, default=Path("hybrid/out"))
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(max(1, (os.cpu_count() or 2) - a.workers + 1))
    net = UNet()
    print("parameters", sum(p.numel() for p in net.parameters()), flush=True)
    dl = torch.utils.data.DataLoader(Stream(a.size, 1234), batch_size=a.batch, num_workers=a.workers,
                                     persistent_workers=a.workers > 0)
    opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=1e-4)
    import math

    def lr_at(frac):  # short warm-up + cosine; the fraction is the larger of steps and time
        if frac < 0.03:
            return a.lr * frac / 0.03
        return a.lr * 0.5 * (1 + math.cos(math.pi * min(1.0, (frac - 0.03) / 0.97))) + 1e-5
    pos_w = torch.tensor(3.0)
    t0 = time.time(); log = []
    it = iter(dl)
    for step in range(1, a.steps + 1):
        frac = step / a.steps
        if a.minutes:
            frac = max(frac, (time.time() - t0) / (a.minutes * 60))
        for g in opt.param_groups:
            g["lr"] = lr_at(frac)
        x, y = next(it)
        logit = net(x)
        loss = F.binary_cross_entropy_with_logits(logit, y, pos_weight=pos_w) + dice_loss(logit, y)
        opt.zero_grad(); loss.backward(); opt.step()
        if step % 50 == 0:
            el = time.time() - t0
            log.append(dict(step=step, loss=float(loss), sec=round(el, 1)))
            print(f"passo {step} perda {float(loss):.4f} {el / step:.2f} s/passo", flush=True)
        if step % 1000 == 0 or step == a.steps:
            torch.save(net.state_dict(), a.out / "unet.pt")
        if frac >= 1.0:
            print("time limit", flush=True)
            break
    torch.save(net.state_dict(), a.out / "unet.pt")
    net.eval()
    dummy = torch.zeros(1, 3, 256, 256)
    torch.onnx.export(net, dummy, a.out / "trace_seg.onnx", input_names=["x"], output_names=["logit"],
                      dynamic_axes={"x": {0: "n", 2: "h", 3: "w"}, "logit": {0: "n", 2: "h", 3: "w"}},
                      opset_version=17, dynamo=False)
    (a.out / "train_log.json").write_text(json.dumps(log))
    print("ok", a.out / "trace_seg.onnx", flush=True)


if __name__ == "__main__":
    main()
