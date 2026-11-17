"""Stage oracle: where does the classical reader lose dB?

Each page is rendered by ecg-torch-renderer from a PTB-XL record (strat fold 10, never used in
training), so every stage has an exact truth. The same page is read six times, replacing one
stage at a time by its truth:

  photo            the photographed / scanned page, everything automatic (what we really do)
  photo+cal        same image, true layout, speed and gain given to the reader
  rectified        the photo resampled onto the flat page with the exact geometry (perfect
                   rectification, the capture's light, blur, noise and colour kept)
  rectified+cal    perfect geometry + true layout / speed / gain / px per mm
  document         the flat document before capture (paper, grid, ink, stains; no camera)
  document+cal     the flat document with layout, speed, gain and px per mm known

Differences between readings attribute the loss: photo -> rectified = geometry (perspective,
curl, folds, crop); rectified -> document = capture appearance (light, blur, noise, moire,
screen); +cal = layout / calibration; what remains in document+cal is tracing on a clean flat
page (pen width, overlaps, resolution). Both metrics are reported: official Kaggle and
PMcardio style (benchmarks/metrics.py). The truth is the source signal; the score against the
printed signal (after the simulated device filters) is kept too.

    python benchmarks/oracle/run.py --bank fold10.npy --part 0 --nparts 20 --pages 10 --out out/
"""
from __future__ import annotations

import argparse
import json
import signal
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "benchmarks"))
sys.path.insert(0, str(ROOT))
import metrics as MT  # noqa: E402

LEADS = MT.LEADS
FS = 500
READINGS = ["photo", "photo+cal", "rectified", "rectified+cal", "document", "document+cal"]


def _bgr(t):
    return (t.clamp(0, 1).permute(1, 2, 0).cpu().numpy()[..., ::-1] * 255).round().astype(np.uint8).copy()


def rectify(photo, field):
    """Photo (3, H, W) and inverse field (H, W, 2: photo pixel -> page, normalised,
    align_corners=False) -> the photo resampled on the flat page grid (3, Hc, Wc) and the mask of
    page pixels that are seen. Newton iterations on the field, from an affine start."""
    _, H, W = photo.shape
    Hc, Wc = H, W
    f = field.float()
    ok = (f.abs() <= 1.2).all(-1)
    ys, xs = torch.meshgrid(torch.arange(H, dtype=torch.float32), torch.arange(W, dtype=torch.float32), indexing="ij")
    px = torch.stack([(xs + 0.5) / W * 2 - 1, (ys + 0.5) / H * 2 - 1], -1)   # photo, normalised
    # affine start: page = A @ photo + b on the seen pixels, inverted
    sel = ok.flatten().nonzero()[:, 0]
    sel = sel[torch.randperm(len(sel))[:20000]]
    P = torch.cat([px.reshape(-1, 2)[sel], torch.ones(len(sel), 1)], 1)
    C = f.reshape(-1, 2)[sel]
    M = torch.linalg.lstsq(P, C).solution            # (3, 2)
    A, b = M[:2].T, M[2]
    yc, xc = torch.meshgrid(torch.arange(Hc, dtype=torch.float32), torch.arange(Wc, dtype=torch.float32), indexing="ij")
    c = torch.stack([(xc + 0.5) / Wc * 2 - 1, (yc + 0.5) / Hc * 2 - 1], -1)
    p = (c - b) @ torch.linalg.inv(A).T
    # Jacobian of the field in normalised photo units
    fx = torch.zeros_like(f); fy = torch.zeros_like(f)
    fx[:, 1:-1] = (f[:, 2:] - f[:, :-2]) / (2 * 2 / W); fx[:, 0] = fx[:, 1]; fx[:, -1] = fx[:, -2]
    fy[1:-1] = (f[2:] - f[:-2]) / (2 * 2 / H); fy[0] = fy[1]; fy[-1] = fy[-2]
    stack = torch.cat([f, fx, fy], -1).permute(2, 0, 1)[None]        # (1, 6, H, W)
    for _ in range(20):
        s = F.grid_sample(stack, p[None], align_corners=False, padding_mode="border")[0].permute(1, 2, 0)
        r = c - s[..., :2]
        j00, j10, j01, j11 = s[..., 2], s[..., 3], s[..., 4], s[..., 5]
        det = j00 * j11 - j01 * j10
        det = torch.where(det.abs() < 1e-6, torch.full_like(det, 1e-6), det)
        dx = (j11 * r[..., 0] - j01 * r[..., 1]) / det
        dy = (-j10 * r[..., 0] + j00 * r[..., 1]) / det
        p = p + torch.stack([dx, dy], -1).clamp(-0.2, 0.2)
    s = F.grid_sample(stack, p[None], align_corners=False, padding_mode="border")[0].permute(1, 2, 0)
    err_px = ((c - s[..., :2]) * torch.tensor([Wc / 2, Hc / 2])).norm(dim=-1)
    seen = (err_px < 0.5) & (p.abs() <= 1).all(-1)
    out = F.grid_sample(photo[None].float(), p[None], align_corners=False, padding_mode="border")[0]
    paper = out[:, seen].median(1).values if seen.any() else torch.ones(3)
    out = torch.where(seen[None], out, paper[:, None, None])
    return out, seen, err_px


def layout_rows(meta):
    """Rows of printed names, left to right, from the renderer's slots."""
    rows = {}
    for k in range(meta["lead_id"].shape[1]):
        if not bool(meta["active"][0, k]):
            continue
        name = LEADS[int(meta["lead_id"][0, k])]
        if int(meta["amplitude_sign"][0, k]) < 0:
            name = "-" + name
        rows.setdefault(int(meta["row"][0, k]), []).append((int(meta["column"][0, k]), name))
    return [[n for _, n in sorted(v)] for _, v in sorted(rows.items())]


def ground_truth(meta, sig):
    """(12, n) truth on the samples shown on the page, NaN elsewhere (the Kaggle convention)."""
    n = sig.shape[1]
    G = np.full((12, n), np.nan)
    for k in range(meta["lead_id"].shape[1]):
        if not bool(meta["active"][0, k]):
            continue
        li = int(meta["lead_id"][0, k])
        a = max(0, int(round(float(meta["start_s"][0, k]) * FS)))
        b = min(n, int(round(float(meta["end_s"][0, k]) * FS)) + 1)
        G[li, a:b] = sig[li, a:b]
    return G


def _to_grid(r, n):
    S = np.full((12, r["signals"].shape[1]), np.nan)
    for k, nm in enumerate(r["leads"]):
        if nm in LEADS:
            S[LEADS.index(nm)] = r["signals"][k]
    out = np.full((12, n), np.nan)
    m = min(n, S.shape[1])
    out[:, :m] = S[:, :m]
    return out


class _Timeout(Exception):
    pass


def _alarm(*_):
    raise _Timeout()


def read(img, cfg_kw, n, limit=300):
    from ecgdig import Config, digitize
    signal.signal(signal.SIGALRM, _alarm)
    signal.alarm(limit)
    t0 = time.time()
    try:
        r = digitize(img, Config(**cfg_kw))
        P = _to_grid(r, n)
        info = dict(ok=True, layout=r.get("layout"), speed=r.get("speed_mm_per_s"),
                    gains=sorted(set(float(g) for g in r["gains"].values())) if r.get("gains") else None)
    except _Timeout:
        P = np.full((12, n), np.nan); info = dict(ok=False, error="timeout")
    except Exception as e:  # noqa: BLE001
        P = np.full((12, n), np.nan); info = dict(ok=False, error=f"{type(e).__name__}: {e}"[:300])
    finally:
        signal.alarm(0)
    info["sec"] = round(time.time() - t0, 1)
    return P, info


def score(G, P):
    klin, kper = MT.kaggle_image_snr(G, P, FS)
    pm = MT.pm_image_metrics(G, P, FS)
    return dict(kaggle_lin=float(klin), kaggle_db=float(10 * np.log10(max(klin, 1e-12))), kaggle_per_lead=kper, pm=pm)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bank", type=Path, required=True)
    ap.add_argument("--part", type=int, default=0)
    ap.add_argument("--nparts", type=int, default=1)
    ap.add_argument("--pages", type=int, default=10, help="pages of this shard")
    ap.add_argument("--page", default="1700x2200", help="HxW of the rendered photo")
    ap.add_argument("--readings", default=",".join(READINGS))
    ap.add_argument("--config", default="", help='extra Config fields for every reading, e.g. "R=15"')
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    import ast
    extra = {}
    for item in a.config.split():
        k, v = item.split("=", 1)
        try:
            extra[k] = ast.literal_eval(v)
        except (ValueError, SyntaxError):
            extra[k] = v
    from ecg_gpu import ECGSupervisionLayer, RenderConfig, SupervisionConfig
    H, W = (int(v) for v in a.page.split("x"))
    layer = ECGSupervisionLayer(fs=FS, config=RenderConfig(height=H, width=W),
                                supervision=SupervisionConfig(outputs=("trace_mask",)))
    layer.train()
    bank = np.load(a.bank, mmap_mode="r")
    want = [x for x in a.readings.split(",") if x]
    (a.out / "img").mkdir(parents=True, exist_ok=True)
    results = []
    for i in range(a.pages):
        page = a.part * a.pages + i
        rid = page % len(bank)
        rec = dict(page=page, record=int(rid))
        try:
            torch.manual_seed(1000 + page)
            sig = np.nan_to_num(np.asarray(bank[rid], np.float32))[:, :5000]
            r = layer.render_batch(torch.from_numpy(sig)[None], FS, stage="image", photo=True,
                                   return_field=True, return_canonical=True)
            m = r.metadata
            ref = r.reference[0].cpu().numpy() if torch.is_tensor(r.reference) else sig
            G = ground_truth(m, ref)
            Gp = ground_truth(m, m["printed_signals"][0].cpu().numpy()) if "printed_signals" in m else None
            rect, seen, err = rectify(r.image[0], r.inverse_field[0])
            imgs = {"photo": _bgr(r.image[0]), "rectified": _bgr(rect), "document": _bgr(r.canonical_image[0])}
            gains = m["gain"].flatten().tolist(); speed = float(m["speed"].flatten()[0])
            cal = dict(layout=layout_rows(m), speed=speed, gain=float(np.median(gains)))
            ppm = float(m["ppm"][0])
            rec.update(capture=str(m["capture_medium"][0] if isinstance(m["capture_medium"], (list, tuple)) else m["capture_medium"]),
                       layout_class=int(m["layout_class"][0]), speed=speed, gain=cal["gain"], ppm=ppm,
                       rows=cal["layout"], rect_seen=float(seen.float().mean()),
                       rect_err_px=float(err[seen].median()) if seen.any() else None)
            import cv2
            for nm, im in imgs.items():
                sc = min(1.0, 900 / im.shape[1])
                cv2.imwrite(str(a.out / "img" / f"{page:05d}-{nm}.jpg"),
                            cv2.resize(im, None, fx=sc, fy=sc, interpolation=cv2.INTER_AREA), [cv2.IMWRITE_JPEG_QUALITY, 70])
            for nm in want:
                base = nm.split("+")[0]
                kw = dict(extra)
                if nm.endswith("+cal"):
                    kw.update(cal)
                    if base != "photo":
                        kw["ppmm"] = ppm
                P, info = read(imgs[base], kw, G.shape[1])
                s = score(G, P)
                if Gp is not None:
                    s["kaggle_db_printed"] = float(10 * np.log10(max(MT.kaggle_image_snr(Gp, P, FS)[0], 1e-12)))
                rec[nm] = dict(info, **s)
                print(f"page {page} {nm:14s} kaggle {s['kaggle_db']:6.2f} dB  {info.get('error', '')}", flush=True)
        except Exception as e:  # noqa: BLE001
            rec["error"] = f"{type(e).__name__}: {e}"
            traceback.print_exc()
        results.append(rec)
        (a.out / "results.json").write_text(json.dumps(results))
    print("done", len(results), "pages", flush=True)


if __name__ == "__main__":
    main()
