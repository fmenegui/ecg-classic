"""Train the trace segmenter on a GPU (RunPod pod, see hybrid/gpu/pod.sh) and export ONNX.

    python hybrid/train_gpu.py --minutes 120 --encoder resnet34 --out /workspace/out

Same task and data as hybrid/train.py (generic synthetic crops from hybrid/gen.py, label =
pen coverage before the degradations), with a larger U-Net (pretrained encoder from
segmentation_models_pytorch), mixed precision and many data workers. The ONNX output is a
drop-in replacement for Config(ink_model=...).
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

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import gen  # noqa: E402
import rowgen  # noqa: E402
GENS = {"gen": gen.sample, "rowgen": rowgen.sample}
sample = gen.sample


def _cpus():
    """CPUs this container may really use: the cgroup quota (a RunPod pod sees all the host's
    cores in os.cpu_count(), e.g. 94, but gets ~16), else the affinity mask."""
    try:
        q, p = open("/sys/fs/cgroup/cpu.max").read().split()[:2]
        if q != "max":
            return max(1, int(float(q) / float(p)))
    except Exception:  # noqa: BLE001
        pass
    try:
        q = int(open("/sys/fs/cgroup/cpu/cpu.cfs_quota_us").read()); p = int(open("/sys/fs/cgroup/cpu/cpu.cfs_period_us").read())
        if q > 0:
            return max(1, q // p)
    except Exception:  # noqa: BLE001
        pass
    return len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else (os.cpu_count() or 4)


def _worker_init(_):
    # one thread per data worker: torch and OpenCV would each start one thread per HOST core
    # (94 on a RunPod pod) inside every worker and stall the machine
    import cv2
    torch.set_num_threads(1)
    cv2.setNumThreads(1)


class Stream(torch.utils.data.IterableDataset):
    def __init__(self, size, seed, gen_name="gen"):
        self.size, self.seed, self.gen_name = size, seed, gen_name

    def __iter__(self):
        wi = torch.utils.data.get_worker_info()
        rng = np.random.default_rng(self.seed + (wi.id if wi else 0) * 7919 + int(time.time() * 1000) % 100000)
        while True:
            im, lb = GENS[self.gen_name](rng, self.size)
            x = torch.from_numpy(im[..., ::-1].copy()).permute(2, 0, 1).float() / 255.0  # RGB
            yield x, torch.from_numpy(lb)[None]


def gpu_aug(x, y, g, rot=True):
    """Cheap augmentations on the GPU, so that one generated batch can feed several steps (the CPU crop
    generator is the bottleneck). The label moves with the image; colour changes keep it."""
    if torch.rand(1, generator=g).item() < 0.5:
        x, y = x.flip(3), y.flip(3)
    if torch.rand(1, generator=g).item() < 0.5:
        x, y = x.flip(2), y.flip(2)
    if rot:   # not for row strips: the row runs horizontally through the centre
        k = int(torch.randint(0, 4, (1,), generator=g))
        x, y = torch.rot90(x, k, (2, 3)), torch.rot90(y, k, (2, 3))
    n = x.shape[0]
    dev = x.device
    x = x[:, torch.randperm(3, generator=g).to(dev)]                         # ink / grid / paper colours
    gam = torch.exp(torch.empty(n, 1, 1, 1, device=dev).uniform_(-0.4, 0.4))  # tone curve
    x = x.clamp(1e-4, 1) ** gam
    c = torch.empty(n, 1, 1, 1, device=dev).uniform_(0.7, 1.3)               # contrast and brightness
    b = torch.empty(n, 1, 1, 1, device=dev).uniform_(-0.1, 0.1)
    x = (x - 0.5) * c + 0.5 + b
    x = x + torch.randn_like(x) * torch.empty(n, 1, 1, 1, device=dev).uniform_(0, 0.04)
    return x.clamp(0, 1), y


def render_aug(x, g):
    """Capture degradations for the clean rendered documents: blur, colour/contrast, noise."""
    n = x.shape[0]
    dev = x.device
    out = []
    for i in range(n):
        xi = x[i:i + 1]
        s = float(torch.rand(1, generator=g)) * 1.6
        if s > 0.3:
            k = int(2 * round(3 * s) + 1)
            ax = torch.arange(k, device=dev, dtype=xi.dtype) - k // 2
            w = torch.exp(-ax ** 2 / (2 * s * s)); w = w / w.sum()
            xi = F.conv2d(F.pad(xi, (k // 2,) * 4, mode="replicate"), w.view(1, 1, 1, k).repeat(3, 1, 1, 1), groups=3)
            xi = F.conv2d(xi, w.view(1, 1, k, 1).repeat(3, 1, 1, 1), groups=3)
        out.append(xi)
    x = torch.cat(out)
    x = x * torch.empty(n, 3, 1, 1, device=dev).uniform_(0.8, 1.1) + torch.empty(n, 1, 1, 1, device=dev).uniform_(-0.1, 0.05)
    x = x + torch.randn_like(x) * torch.empty(n, 1, 1, 1, device=dev).uniform_(0, 0.05)
    return x.clamp(0, 1)


def dice_loss(logit, y):
    p = torch.sigmoid(logit)
    num = 2 * (p * y).sum((1, 2, 3)) + 1
    den = (p + y).sum((1, 2, 3)) + 1
    return (1 - num / den).mean()


def build(encoder, weights):
    import segmentation_models_pytorch as smp
    return smp.Unet(encoder_name=encoder, encoder_weights=None if weights in ("", "none") else weights,
                    in_channels=3, classes=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--minutes", type=float, default=60)
    ap.add_argument("--steps", type=int, default=10 ** 9)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--workers", type=int, default=0, help="data workers (0 = the container's CPUs - 1, at most 32)")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--encoder", default="resnet34")
    ap.add_argument("--weights", default="imagenet")
    ap.add_argument("--gen", default="gen", choices=sorted(GENS),
                    help="gen: page ink map; rowgen: the centre row's trace in a strip (row network)")
    ap.add_argument("--init", type=Path, default=None, help="state_dict to start from (e.g. the ink model's unet.pt)")
    ap.add_argument("--render-frac", type=float, default=0.0,
                    help="rowgen only: share of each batch rendered on the GPU by ecg-torch-renderer (hybrid/rendergen.py)")
    ap.add_argument("--reuse", type=int, default=1,
                    help="steps per generated batch (> 1: the extra steps see GPU-augmented copies)")
    ap.add_argument("--val", type=int, default=256, help="fixed validation crops")
    ap.add_argument("--out", type=Path, default=Path("hybrid/out_gpu"))
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    if a.workers <= 0:
        a.workers = int(min(32, max(2, _cpus() - 1)))
    torch.set_num_threads(min(8, _cpus()))
    import cv2
    cv2.setNumThreads(1)   # the parent's OpenCV pool must not be inherited half-locked
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    if dev == "cpu" and os.environ.get("RUNPOD_POD_ID"):
        sys.exit("CUDA is not available on this GPU pod; refusing to train on the CPU")
    torch.backends.cudnn.benchmark = True
    net = build(a.encoder, a.weights)
    if not a.init and (a.out / "unet.pt").exists():   # pod restarted: resume from the last checkpoint
        a.init = a.out / "unet.pt"
        print("resuming after a restart", flush=True)
    if a.init and a.init.exists():
        print("init from", a.init, net.load_state_dict(torch.load(a.init, map_location="cpu"), strict=False), flush=True)
    net = net.to(dev)
    n_par = sum(p.numel() for p in net.parameters())
    print(f"device {dev} encoder {a.encoder} parameters {n_par} workers {a.workers}", flush=True)
    dl = torch.utils.data.DataLoader(Stream(a.size, 1234, a.gen), batch_size=a.batch, num_workers=a.workers,
                                     persistent_workers=True, pin_memory=dev == "cuda", prefetch_factor=4,
                                     worker_init_fn=_worker_init, multiprocessing_context="spawn")
    # fixed validation set (same crops every time)
    vr = np.random.default_rng(999)
    val = [GENS[a.gen](vr, a.size) for _ in range(a.val)]
    vx = torch.stack([torch.from_numpy(im[..., ::-1].copy()).permute(2, 0, 1).float() / 255.0 for im, _ in val])
    vy = torch.stack([torch.from_numpy(lb)[None] for _, lb in val])
    opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=1e-4)
    scaler = torch.amp.GradScaler(enabled=dev == "cuda")
    # row network: sparse signal label (<= 2 px per column) -> heavy positive weight, no Dice
    pos_w = torch.tensor(20.0 if a.gen == "rowgen" else 3.0, device=dev)

    def lr_at(frac):
        if frac < 0.02:
            return a.lr * frac / 0.02
        return a.lr * 0.5 * (1 + math.cos(math.pi * min(1.0, (frac - 0.02) / 0.98))) + 1e-6

    def validate():
        net.eval(); tot = 0.0; inter = 0.0; uni = 0.0; cerr = []
        with torch.no_grad(), torch.autocast(dev, enabled=dev == "cuda"):
            for i in range(0, len(vx), 16):
                x, y = vx[i:i + 16].to(dev), vy[i:i + 16].to(dev)
                lg = net(x).float()
                tot += float(F.binary_cross_entropy_with_logits(lg, y, pos_weight=pos_w)) * len(x)
                p = (torch.sigmoid(lg) > 0.5).float(); yb = (y > 0.5).float()
                inter += float((p * yb).sum()); uni += float(((p + yb) > 0).float().sum())
                if a.gen == "rowgen":   # height error per column (px, 10 px = 1 mm = 0.1 mV at 10 mm/mV)
                    pr = torch.sigmoid(lg)[:, 0]; yy = y[:, 0]
                    hh = torch.arange(pr.shape[1], device=pr.device, dtype=pr.dtype)[None, :, None]
                    # decode as at inference: centroid within +-3 px of each column's peak
                    k = pr.argmax(1, keepdim=True)
                    win = ((hh - k).abs() <= 3).to(pr.dtype)
                    cp = (pr * win * hh).sum(1) / (pr * win).sum(1).clamp_min(1e-6)
                    cy = (yy * hh).sum(1) / yy.sum(1).clamp_min(1e-6)
                    ok = (yy.sum(1) > 0.5) & (pr.amax(1) >= 0.3)
                    cerr.append((cp - cy).abs()[ok].float().cpu())
        net.train()
        validate.col_err = float(torch.cat(cerr).median()) if cerr else float("nan")
        validate.col_err90 = float(torch.cat(cerr).quantile(0.9)) if cerr else float("nan")
        return tot / len(vx), inter / max(uni, 1.0)

    def export(tag=""):
        net.eval()
        m = net.float().cpu()
        dummy = torch.zeros(1, 3, 256, 256)
        torch.onnx.export(m, dummy, a.out / f"trace_seg{tag}.onnx", input_names=["x"], output_names=["logit"],
                          dynamic_axes={"x": {0: "n", 2: "h", 3: "w"}, "logit": {0: "n", 2: "h", 3: "w"}},
                          opset_version=17, dynamo=False)
        net.to(dev).train()

    rs = None
    if a.render_frac > 0 and a.gen == "rowgen":
        try:
            from rendergen import TorchStrips
            rs = TorchStrips(dev, size=a.size, pages=2)
            print("GPU-rendered strips:", a.render_frac, "of each batch", flush=True)
        except Exception as e:  # noqa: BLE001
            print("ecg-torch-renderer not available, CPU strips only:", e, flush=True)
    t_render = 0.0
    t0 = time.time(); log = []; it = iter(dl); best = None; t_wait = 0.0
    g_aug = torch.Generator().manual_seed(7)
    for step in range(1, a.steps + 1):
        frac = max(step / a.steps, (time.time() - t0) / (a.minutes * 60))
        for g in opt.param_groups:
            g["lr"] = lr_at(frac)
        if (step - 1) % a.reuse == 0:
            tw = time.time(); xb, yb = next(it); t_wait += time.time() - tw
            xb, yb = xb.to(dev, non_blocking=True), yb.to(dev, non_blocking=True)
            if rs is not None:   # replace part of the batch with GPU-rendered pages' strips
                tr = time.time()
                n_r = int(round(a.render_frac * len(xb)))
                got = rs.batch(n_r) if n_r else None
                if got is not None:
                    xr, yr = got
                    xb = xb.clone(); yb = yb.clone()
                    xb[:n_r] = render_aug(xr, g_aug); yb[:n_r] = yr
                t_render += time.time() - tr
            x, y = xb, yb
        else:
            x, y = gpu_aug(xb, yb, g_aug, rot=a.gen != "rowgen")
        with torch.autocast(dev, enabled=dev == "cuda"):
            logit = net(x)
        logit = logit.float()
        loss = F.binary_cross_entropy_with_logits(logit, y, pos_weight=pos_w)
        if a.gen != "rowgen":
            loss = loss + dice_loss(logit, y)
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
        if step % 100 == 0:
            el = time.time() - t0
            mem = torch.cuda.max_memory_allocated() / 2 ** 30 if dev == "cuda" else 0.0
            # data wait share: near 100% means the GPU is starved by the CPU crop generator
            print(f"step {step} loss {float(loss):.4f} lr {lr_at(frac):.2e} {el / step:.3f} s/step "
                  f"data wait {100 * t_wait / el:.0f}% render {100 * t_render / el:.0f}% gpu mem {mem:.1f} GB "
                  f"cpus {_cpus()}", flush=True)
        if step % 2000 == 0 or frac >= 1.0:
            vl, iou = validate()
            log.append(dict(step=step, loss=float(loss), val_loss=vl, val_iou=iou, col_err=validate.col_err,
                            col_err90=validate.col_err90, sec=round(time.time() - t0, 1)))
            print(f"VAL step {step} val_loss {vl:.4f} IoU {iou:.4f} col_err median {validate.col_err:.2f} px "
                  f"p90 {validate.col_err90:.2f} px", flush=True)
            (a.out / "train_log.json").write_text(json.dumps(log))
            torch.save(net.state_dict(), a.out / "unet.pt")
            if best is None or iou > best:
                best = iou
                torch.save(net.state_dict(), a.out / "unet_best.pt")
        if frac >= 1.0:
            print("time limit", flush=True)
            break
    torch.save(net.state_dict(), a.out / "unet.pt")
    export()
    (a.out / "meta.json").write_text(json.dumps(dict(encoder=a.encoder, parameters=n_par, steps=step,
                                                     best_val_iou=best, size=a.size, batch=a.batch,
                                                     reuse=a.reuse, gen=a.gen)))
    print("ok", a.out / "trace_seg.onnx", flush=True)


if __name__ == "__main__":
    main()
