"""Row-start network: the sub-pixel position of a row's first sample (the time origin).

The time origin comes from the trace's start edge (edge + pen radius, digitize._vote_x0). How a
line starts depends on the printer: matplotlib's caps (butt, projecting, round), a calibration
pulse joined to the trace or drawn apart, the pen width and its anti-aliasing. On ecg-image-kit
pages our origin sits 0.1 px (at 10 px/mm, 0.4 ms) late with a joined pulse and 0.1 px early
without one; on our own cv2 pages it is unbiased. Kaggle and PMcardio-style metrics align the
signals only by whole samples, so that fraction of a sample costs 1.2-1.4 dB on clean pages.

This network looks at a crop of the rectified page (10 px/mm) around the estimated start and
returns the offset of the true first sample from the crop centre, in pixels. Training crops come
from pages whose geometry is exact: ecg-image-kit (matplotlib; caps, pulse on/off, resolution,
colours varied) and ecgdig.synth (cv2), both scaled exactly to 10 px/mm with a small affine.

    python hybrid/startnet.py build --eik ecg-image-kit --out crops.npz --pages 2000
    python hybrid/startnet.py train --data crops.npz --out start.onnx
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent))

from ecgdig.startmodel import CH, CW, XUP, crop  # noqa: E402

R = 10.0


def _eik_page(gen_dir, rng, seed, tmp):
    import cv2
    import matplotlib
    import eikstrips_build as EB
    from ecgdig.synth import synth_signals
    matplotlib.rcParams["lines.solid_capstyle"] = str(rng.choice(["butt", "projecting", "round"]))
    o = EB.page_options(rng)
    o["pulse"] = bool(rng.random() < 0.6)
    o["full_mode"] = "None" if rng.random() < 0.3 else "II"
    sig = synth_signals(rng).astype(np.float32)
    work = Path(tmp) / f"s{seed}"
    page, meta = EB.render(Path(gen_dir), sig, work, seed, o)
    shutil.rmtree(work, ignore_errors=True)
    k = R / (o["resolution"] / 25.4)
    starts = []
    for L in meta["leads"]:
        pp = np.asarray(L.get("plotted_pixels") or [], float)
        if len(pp) and L.get("start_sample", 1) == 0:
            starts.append(((pp[0, 1] - 0.5 + 0.5) * k - 0.5, (pp[0, 0] - 0.5 + 0.5) * k - 0.5))
    return page, k, starts, "eik"


def _synth_page(rng):
    from ecgdig.synth import render, synth_signals
    lay = str(rng.choice(["3x4+1", "3x4", "6x2", "6x2+1", "12x1"]))
    ppmm = float(rng.uniform(6, 14))
    img, shown, meta = render(synth_signals(rng), rng, ppmm=ppmm, degrade="none", layout=lay)
    k = R / ppmm
    starts = [((meta["x_sig0_mm"] * ppmm + 0.5) * k - 0.5, (y * ppmm + 0.5) * k - 0.5) for y in meta["row_base_mm"]]
    return img, k, starts, "cv2"


def _work(args):
    gen_dir, seed, n_per, tmp = args
    import cv2
    rng = np.random.default_rng(seed)
    try:
        page, k, starts, src = _eik_page(gen_dir, rng, seed, tmp) if rng.random() < 0.6 else _synth_page(rng)
        H, W = page.shape[:2]
        Hr, Wr = int(round(H * k)), int(round(W * k))
        rect = cv2.resize(page, (Wr, Hr), interpolation=cv2.INTER_AREA if k < 1 else cv2.INTER_CUBIC)
        ang, sc = rng.uniform(-0.3, 0.3), rng.uniform(0.99, 1.01)
        M = cv2.getRotationMatrix2D((Wr / 2, Hr / 2), ang, sc); M[:, 2] += rng.uniform(-3, 3, 2)
        rect = cv2.warpAffine(rect, M, (Wr, Hr), flags=cv2.INTER_CUBIC, borderValue=(255, 255, 255))
        if rng.random() < 0.4:      # mild capture: blur, noise, JPEG
            rect = cv2.GaussianBlur(rect, (0, 0), rng.uniform(0.3, 1.2))
            rect = np.clip(rect + rng.normal(0, rng.uniform(0, 6), rect.shape), 0, 255).astype(np.uint8)
            rect = cv2.imdecode(cv2.imencode(".jpg", rect, [cv2.IMWRITE_JPEG_QUALITY, int(rng.uniform(40, 95))])[1], 1)
        X, Y = [], []
        for x, y in starts:
            xr, yr = M[0, 0] * x + M[0, 1] * y + M[0, 2], M[1, 0] * x + M[1, 1] * y + M[1, 2]
            for _ in range(n_per):
                u = rng.uniform(-3, 3) if rng.random() < 0.8 else rng.uniform(-8, 8)
                v = rng.uniform(-12, 12)
                X.append(crop(rect, xr + u, yr + v)); Y.append(-u)
        return X, Y, src
    except Exception as e:  # noqa: BLE001
        return [], [], f"error {type(e).__name__}: {e}"[:200]


def build(a):
    gen_dir = str((a.eik / "codes" / "ecg-image-generator").resolve())
    tmp = tempfile.mkdtemp(prefix="startnet")
    X, Y, src = [], [], {}
    with ProcessPoolExecutor(a.workers) as ex:
        for x, y, s in ex.map(_work, [(gen_dir, a.seed * 1_000_003 + p, a.per_row, tmp) for p in range(a.pages)], chunksize=2):
            X += x; Y += y; src[s] = src.get(s, 0) + 1
    shutil.rmtree(tmp, ignore_errors=True)
    np.savez_compressed(a.out, x=np.stack(X), y=np.asarray(Y, np.float32))
    print(len(X), "crops from", a.pages, "pages:", src, "->", a.out, flush=True)


class Net:
    @staticmethod
    def make():
        import torch.nn as nn
        def blk(i, o, s):
            return nn.Sequential(nn.Conv2d(i, o, 3, s, 1, bias=False), nn.BatchNorm2d(o), nn.ReLU(inplace=True),
                                 nn.Conv2d(o, o, 3, 1, 1, bias=False), nn.BatchNorm2d(o), nn.ReLU(inplace=True))
        # keep the horizontal resolution longer than the vertical one: the target is an x position
        return nn.Sequential(blk(3, 32, 1), nn.MaxPool2d((2, 1)), blk(32, 48, 1), nn.MaxPool2d((2, 2)),
                             blk(48, 64, 1), nn.MaxPool2d((2, 2)), blk(64, 96, 1), nn.MaxPool2d((2, 2)),
                             blk(96, 128, 1), nn.AdaptiveAvgPool2d((1, 4)), nn.Flatten(), nn.Linear(512, 64),
                             nn.ReLU(inplace=True), nn.Linear(64, 1))


def train(a):
    import torch
    z = np.load(a.data)
    X = torch.from_numpy(np.ascontiguousarray(z["x"][..., ::-1])).permute(0, 3, 1, 2)   # RGB uint8, as inference
    Y = torch.from_numpy(z["y"]).float()[:, None]
    n = len(X); idx = torch.randperm(n, generator=torch.Generator().manual_seed(0))
    nv = max(256, n // 20)
    va, tr = idx[:nv], idx[nv:]
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    net = Net.make().to(dev)
    opt = torch.optim.AdamW(net.parameters(), 2e-3, weight_decay=1e-4)
    steps = a.epochs * (len(tr) // a.batch)
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, 2e-3, total_steps=max(steps, 1))
    torch.manual_seed(0)
    st = 0
    for ep in range(a.epochs):
        net.train()
        perm = tr[torch.randperm(len(tr))]
        for i in range(0, len(perm) - a.batch + 1, a.batch):
            b = perm[i:i + a.batch]
            x, y = X[b].to(dev).float() / 255.0, Y[b].to(dev)
            if torch.rand(1).item() < 0.5:   # photometric jitter
                x = (x * torch.empty(len(x), 1, 1, 1, device=dev).uniform_(0.8, 1.15)
                     + torch.empty(len(x), 3, 1, 1, device=dev).uniform_(-0.08, 0.08)).clamp(0, 1)
            loss = (net(x) - y).abs().mean()
            opt.zero_grad(); loss.backward(); opt.step(); sch.step(); st += 1
        net.eval()
        with torch.no_grad():
            pv = torch.cat([net(X[va[i:i + 512]].to(dev).float() / 255.0).cpu() for i in range(0, len(va), 512)])
        e = (pv - Y[va]).abs()[:, 0]
        near = Y[va][:, 0].abs() <= 3
        print(f"epoch {ep + 1}: train L1 {float(loss):.3f} px | validation |error| median {float(e.median()):.3f} px, "
              f"p90 {float(e.quantile(0.9)):.3f}, within 3 px of the start: median {float(e[near].median()):.3f}, "
              f"mean signed {float((pv - Y[va])[near.nonzero()[:, 0]].mean()):+.3f}", flush=True)
    net.cpu().eval()
    torch.onnx.export(net, torch.zeros(1, 3, CH, CW * XUP), str(a.out), input_names=["x"], output_names=["dx"],
                      dynamic_axes={"x": {0: "n"}, "dx": {0: "n"}}, opset_version=17, dynamo=False)
    Path(str(a.out) + ".json").write_text(json.dumps(dict(ch=CH, cw=CW, xup=XUP, R=R, n=n,
                                                          val_median=float(e.median()))))
    print("->", a.out)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--eik", type=Path, required=True)
    b.add_argument("--out", type=Path, required=True)
    b.add_argument("--pages", type=int, default=2000)
    b.add_argument("--per-row", type=int, default=4)
    b.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    b.add_argument("--seed", type=int, default=0)
    t = sub.add_parser("train")
    t.add_argument("--data", type=Path, required=True)
    t.add_argument("--out", type=Path, required=True)
    t.add_argument("--epochs", type=int, default=12)
    t.add_argument("--batch", type=int, default=128)
    a = ap.parse_args()
    build(a) if a.cmd == "build" else train(a)


if __name__ == "__main__":
    main()
