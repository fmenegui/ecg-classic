"""Where a training step spends its time on the GPU pod: rendering (page, dewarp, strip
generators) against the network's forward/backward. Decides whether a render cache pays."""
import os
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "hybrid"))


def t(f, n=6):
    f(); torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(n):
        f()
    torch.cuda.synchronize()
    return (time.time() - t0) / n


dev = "cuda"
print("GPU", torch.cuda.get_device_name(0), flush=True)
from pagegen import PageStructure  # noqa: E402
from train_page import build as build_page, loss_fn  # noqa: E402
g = PageStructure(dev, pages=4, photo_frac=0.4, seed=1)
x, tg, w = g.batch()
print(f"page: render+targets+degrade {t(g.batch):.3f} s per 4 pages", flush=True)
g0 = PageStructure(dev, pages=4, photo_frac=0.0, seed=2)
print(f"page: same without photos {t(g0.batch):.3f} s", flush=True)
net = build_page(weights="none").to(dev); opt = torch.optim.AdamW(net.parameters()); sc = torch.amp.GradScaler()


def step():
    with torch.autocast("cuda"):
        out = net(x)
    lb, lc, ln = loss_fn(out.float(), tg, w)
    opt.zero_grad(); sc.scale(lb + lc + ln).backward(); sc.step(opt); sc.update()


print(f"page: network step {t(step):.3f} s per 4 pages", flush=True)
from train_dewarp import Photos, build as build_dw  # noqa: E402
p = Photos(dev, pages=6, seed=3)
print(f"dewarp: render {t(p.batch):.3f} s per 6 photos", flush=True)
from rendergen import TorchStrips  # noqa: E402
ts = TorchStrips(dev, size=512, pages=1, photo_frac=0.7)
print(f"rows: rendered strips {t(lambda: ts.batch(13)):.3f} s per 13 strips", flush=True)
dn = build_dw("none").to(dev); od = torch.optim.AdamW(dn.parameters()); xd, td = p.batch()
from train_dewarp import loss_fn as dloss  # noqa: E402


def dstep():
    with torch.autocast("cuda"):
        o = dn(xd)
    l1, lg, lm = dloss(o.float(), td)
    od.zero_grad(); (l1 + lg + lm).backward(); od.step()


print(f"dewarp: network step {t(dstep):.3f} s per 6 photos", flush=True)
