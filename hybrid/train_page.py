"""Train the page-structure network (hybrid/pagegen.py) on a GPU; export ONNX for ecgdig.pagemodel.

    ECG_BANK=bank.npy python hybrid/train_page.py --minutes 240 --out /workspace/out

U-Net (resnet34, ImageNet start) with 15 output channels: baseline and lead-change heatmaps
(BCE) and a 13-class lead map (cross entropy, background down-weighted). Every batch is rendered
on the GPU; there is no CPU data path. Validation: fixed rendered pages, decoded with the same
code ecgdig uses (rows found exactly, column count, name accuracy).
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
from pagegen import PageStructure, N_CLS, CH  # noqa: E402
from ecgdig.pagemodel import decode  # noqa: E402


def build(encoder="resnet34", weights="imagenet"):
    import segmentation_models_pytorch as smp
    return smp.Unet(encoder_name=encoder, encoder_weights=None if weights in ("", "none") else weights,
                    in_channels=3, classes=CH)


def loss_fn(out, tg, w):
    vis = w[:, 0]
    pb = torch.tensor(5.0, device=out.device)
    l_base = F.binary_cross_entropy_with_logits(out[:, 0], tg[:, 0], pos_weight=pb, reduction="none")
    l_bnd = F.binary_cross_entropy_with_logits(out[:, 1], tg[:, 1], pos_weight=2 * pb, reduction="none")
    cw = torch.ones(N_CLS, device=out.device); cw[0] = 0.1
    l_cls = F.cross_entropy(out[:, 2:].float(), tg[:, 2].long(), weight=cw, reduction="none")
    d = vis.sum().clamp(min=1)
    return (l_base * vis).sum() / d, (l_bnd * vis).sum() / d, (l_cls * vis).sum() / d


def _sig(z):
    return 1 / (1 + np.exp(-z))


def evaluate(net, vx, vt, dev):
    """Row-count accuracy, baseline error (mm), column-count accuracy and name accuracy."""
    net.eval()
    n_ok = n_col = n_tot = 0; dy = []; nm_ok = nm_tot = 0
    with torch.no_grad():
        for i in range(len(vx)):
            with torch.autocast(dev, enabled=dev == "cuda"):
                o = net(vx[i:i + 1].to(dev)).float()[0].cpu().numpy()
            t = vt[i].numpy()
            oh = np.eye(N_CLS, dtype=np.float32)[t[2].astype(int)].transpose(2, 0, 1)
            ref = decode(t[0], t[1], oh)
            e = np.exp(o[2:] - o[2:].max(0, keepdims=True)); pc = e / e.sum(0, keepdims=True)
            got = decode(_sig(o[0]), _sig(o[1]), pc)
            n_tot += 1
            n_ok += len(ref) == len(got)
            if len(ref) == len(got):
                dy += [abs(a["y"] - b["y"]) / 4.0 for a, b in zip(ref, got)]
                n_col += all(len(a["changes"]) == len(b["changes"]) for a, b in zip(ref, got))
                for a, b in zip(ref, got):
                    if len(a["names"]) == len(b["names"]):
                        for na, nb in zip(a["names"], b["names"]):
                            nm_tot += 1; nm_ok += int(np.argmax(na[1:]) == np.argmax(nb[1:]))
    net.train()
    return dict(rows=n_ok / max(n_tot, 1), cols=n_col / max(n_tot, 1),
                dy_mm=float(np.median(dy)) if dy else float("nan"), names=nm_ok / max(nm_tot, 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--minutes", type=float, default=60)
    ap.add_argument("--steps", type=int, default=10 ** 9)
    ap.add_argument("--pages", type=int, default=4, help="pages rendered per step")
    ap.add_argument("--canvas", default="896x1280")
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--photo", type=float, default=0.4)
    ap.add_argument("--heavy", type=float, default=0.5)
    ap.add_argument("--reuse", type=float, default=1.0, help="each rendered page is used ~this many times")
    ap.add_argument("--val", type=int, default=48)
    ap.add_argument("--init", type=Path, default=None)
    ap.add_argument("--weights", default="imagenet")
    ap.add_argument("--out", type=Path, default=Path("hybrid/out_page"))
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    if dev == "cpu" and os.environ.get("RUNPOD_POD_ID"):
        sys.exit("CUDA is not available on this GPU pod; refusing to train on the CPU")
    torch.backends.cudnn.benchmark = True
    canvas = tuple(int(v) for v in a.canvas.split("x"))
    net = build(weights=a.weights)
    if not a.init and (a.out / "page.pt").exists():
        a.init = a.out / "page.pt"; print("resuming after a restart", flush=True)
    if a.init and a.init.exists():
        print("init from", a.init, net.load_state_dict(torch.load(a.init, map_location="cpu"), strict=False), flush=True)
    net = net.to(dev)
    gen = PageStructure(dev, canvas=canvas, pages=a.pages, photo_frac=a.photo, heavy=a.heavy, seed=int(time.time()) % 100000,
                        reuse=a.reuse)
    gv = PageStructure(dev, canvas=canvas, pages=4, photo_frac=a.photo, heavy=a.heavy, seed=999)
    vb = [gv.batch() for _ in range(max(1, a.val // 4))]
    vx = torch.cat([b[0] for b in vb]).cpu(); vt = torch.cat([b[1] for b in vb]).cpu()
    del gv, vb
    opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=1e-4)
    scaler = torch.amp.GradScaler(enabled=dev == "cuda")
    t0 = time.time(); budget = a.minutes * 60
    step, best = 0, -1.0
    hist = []

    def export(tag=""):
        net.eval()
        m = net.float().cpu()
        torch.onnx.export(m, torch.zeros(1, 3, 256, 256), str(a.out / f"page{tag}.onnx"), input_names=["x"],
                          output_names=["y"], dynamic_axes={"x": {0: "n", 2: "h", 3: "w"}, "y": {0: "n", 2: "h", 3: "w"}},
                          opset_version=17, dynamo=False)
        net.to(dev); net.train()

    tl = time.time(); acc = np.zeros(3); n_acc = 0
    while step < a.steps and time.time() - t0 < budget:
        frac = (time.time() - t0) / budget
        lr = a.lr * min(1.0, frac / 0.02) if frac < 0.02 else a.lr * 0.5 * (1 + math.cos(math.pi * (frac - 0.02) / 0.98)) + 1e-6
        for g_ in opt.param_groups:
            g_["lr"] = lr
        x, tg, w = gen.batch()
        with torch.autocast(dev, enabled=dev == "cuda"):
            out = net(x)
        lb, lc, ln = loss_fn(out.float(), tg, w)
        loss = lb + lc + 0.5 * ln
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(opt); scaler.update()
        acc += [lb.item(), lc.item(), ln.item()]; n_acc += 1
        step += 1
        if step % 50 == 0:
            el = time.time() - tl; tl = time.time()
            print(f"step {step} lr {lr:.2e} loss base {acc[0]/n_acc:.4f} change {acc[1]/n_acc:.4f} names {acc[2]/n_acc:.4f} "
                  f"{50/el:.2f} it/s elapsed {(time.time()-t0)/60:.1f} min", flush=True)
            acc[:] = 0; n_acc = 0
        if step % 1000 == 0:
            ev = evaluate(net, vx, vt, dev)
            ev["step"] = step; hist.append(ev)
            print("val", json.dumps(ev), flush=True)
            torch.save(net.state_dict(), a.out / "page.pt")
            sc = ev["rows"] + 0.5 * ev["cols"] + 0.25 * ev["names"]
            if sc > best:
                best = sc
                torch.save(net.state_dict(), a.out / "page_best.pt")
                print("best", round(sc, 4), flush=True)
            (a.out / "val.json").write_text(json.dumps(hist))
    torch.save(net.state_dict(), a.out / "page.pt")
    ev = evaluate(net, vx, vt, dev); ev["step"] = step; hist.append(ev)
    print("final val", json.dumps(ev), flush=True)
    (a.out / "val.json").write_text(json.dumps(hist))
    export()
    if (a.out / "page_best.pt").exists():
        net.load_state_dict(torch.load(a.out / "page_best.pt", map_location=dev))
        export("_best")
    (a.out / "page.pt").unlink(missing_ok=True)   # resume-only checkpoint: keeps the artifact small
    print("done", step, "steps", flush=True)


if __name__ == "__main__":
    main()
