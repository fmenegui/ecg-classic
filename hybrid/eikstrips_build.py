"""Row strips with EXACT labels from pages drawn by ecg-image-kit (the generator of the Kaggle
images), for the row network alongside the synthetic and the real Kaggle strips.

The real Kaggle strips (realstrips_build.py) are labelled through our own reader's geometry and
kept only where that reading is already right, so the network learns our geometry errors and
seldom sees the pages we fail on. ecg-image-kit stores the pixel of every sample it plots
(`plotted_pixels`, store_configs=2), so here the label is the drawn signal itself, whatever our
reader would do. The signals come from the PTB-XL bank (hybrid/ptbxl/ptbxl_bank.py: strat folds
1-9, every PM-ECG-ID ECG and evaluation record removed); no Kaggle image or PM-ECG-ID record is
used. Each page varies what ecg-image-kit can vary (resolution, grid colours, font, lead names,
calibration pulse, 3x4 or 6x2, rhythm strip), is scaled exactly to 10 px/mm, gets a small random
affine (an imperfect rectification) and is cut into strips by realstrips.strips, in the same
chunk format, so train_gpu.py reads them with --eik-dir.

    python hybrid/eikstrips_build.py --eik ecg-image-kit --bank /root/bank.npy --out /root/eik \
        --pages 20000 --workers 30 --xup 2 [--cache $CACHE]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import tempfile
import time
import types
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent))

LEADS = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
R = 10.0                         # rectified px/mm, as the reader's frame
FONTS = ["Times_New_Roman.ttf", "Arial.ttf", "Arial_Bold.ttf", "Courier_New.ttf", "Verdana.ttf",
         "Verdana_Bold.ttf"]
# matplotlib places a point at display coordinate u on the edge-based pixel grid; in the
# pixel-centre convention of the reader (cv2) that is u - MPL_OFF (checked by tests on the ink)
MPL_OFF = [0.5]
_GEN = {}


def _generator(gen_dir: Path):
    """ecg-image-kit's get_paper_ecg (imgaug stubbed: its augmentations are not used here)."""
    if "f" in _GEN:
        return _GEN["f"], _GEN["cfg"]
    sys.modules.setdefault("imgaug", types.ModuleType("imgaug"))
    aug = types.ModuleType("imgaug.augmenters"); sys.modules["imgaug.augmenters"] = aug
    sys.modules["imgaug"].augmenters = aug
    if str(gen_dir) not in sys.path:
        sys.path.insert(0, str(gen_dir))
    import extract_leads
    from helper_functions import read_config_file
    _GEN["f"], _GEN["cfg"] = extract_leads.get_paper_ecg, read_config_file(str(gen_dir / "config.yaml"))
    return _GEN["f"], _GEN["cfg"]


def page_options(rng):
    return dict(resolution=int(rng.choice([150, 200, 200, 250, 300])),
                colour=int(rng.integers(1, 6)),
                names=bool(rng.random() < 0.8),
                pulse=bool(rng.random() < 0.8),
                columns=-1 if rng.random() < 0.75 else 2,
                full_mode=str(rng.choice(["II", "II", "II", "V1", "None"])),
                font=str(rng.choice(FONTS)))


def render(gen_dir: Path, sig: np.ndarray, work: Path, seed: int, o: dict):
    """Draw one record (12 x 5000 mV at 500 Hz). Returns the page (BGR) and its JSON metadata."""
    import cv2
    import wfdb
    from scipy.stats import bernoulli
    work.mkdir(parents=True, exist_ok=True)
    rec = f"r{seed}"
    wfdb.wrsamp(rec, fs=500, units=["mV"] * 12, sig_name=LEADS, p_signal=sig.T.astype(np.float64),
                fmt=["16"] * 12, adc_gain=[1000.0] * 12, baseline=[0] * 12, write_dir=str(work))
    get_paper_ecg, configs = _generator(gen_dir)
    cwd = os.getcwd()
    os.chdir(gen_dir)
    try:
        res = get_paper_ecg(input_file=str(work / f"{rec}.dat"), header_file=str(work / f"{rec}.hea"),
                            configs=configs, mask_unplotted_samples=False, start_index=0, store_configs=2,
                            store_text_bbox=False, output_directory=str(work), resolution=o["resolution"],
                            papersize="", add_lead_names=o["names"], add_dc_pulse=bernoulli(1 if o["pulse"] else 0),
                            add_bw=bernoulli(0), show_grid=bernoulli(1), add_print=bernoulli(0), pad_inches=0,
                            font_type=os.path.join("Fonts", o["font"]), standard_colours=o["colour"],
                            full_mode=o["full_mode"], bbox=False, columns=o["columns"], seed=seed)
    finally:
        os.chdir(cwd)
    png = Path(res[0])
    return cv2.imread(str(png)), json.loads(png.with_suffix(".json").read_text())


def page_segments(meta: dict, k: float, M: np.ndarray):
    """Every plotted lead as a segment in rectified coordinates: page pixels (matplotlib) -> pixel
    centres -> scaled by k -> the 2 x 3 affine M. Rows are the leads' common baselines."""
    segs = []
    for L in meta["leads"]:
        pp = np.asarray(L.get("plotted_pixels") or [], float)
        if len(pp) < 20:
            continue
        y = (pp[:, 0] - MPL_OFF[0] + 0.5) * k - 0.5
        x = (pp[:, 1] - MPL_OFF[0] + 0.5) * k - 0.5
        xr = M[0, 0] * x + M[0, 1] * y + M[0, 2]
        yr = M[1, 0] * x + M[1, 1] * y + M[1, 2]
        segs.append(dict(lead=L.get("lead_name", "II"), base=float(np.median(yr)), x=xr, y=yr))
    if not segs:
        return []
    # rows: leads whose baselines lie within 12 mm of each other share a row
    order = np.argsort([s["base"] for s in segs])
    row, last = -1, -1e9
    for i in order:
        if segs[i]["base"] - last > 12 * R:
            row += 1
        segs[i]["row"] = row
        last = segs[i]["base"]
    return segs


def _work(args):
    gen_dir, bank_path, idx, seed, xup, tmp = args
    import cv2
    import realstrips as RS
    rng = np.random.default_rng(seed)
    try:
        bank = np.load(bank_path, mmap_mode="r")
        sig = np.nan_to_num(np.asarray(bank[idx], np.float32))
        o = page_options(rng)
        work = Path(tmp) / f"p{seed}"
        page, meta = render(Path(gen_dir), sig, work, seed, o)
        shutil.rmtree(work, ignore_errors=True)
        k = R / (o["resolution"] / 25.4)
        H, W = page.shape[:2]
        Hr, Wr = int(round(H * k)), int(round(W * k))
        rect = cv2.resize(page, (Wr, Hr), interpolation=cv2.INTER_AREA if k < 1 else cv2.INTER_CUBIC)
        ang, sc = rng.uniform(-0.3, 0.3), rng.uniform(0.99, 1.01)
        M = cv2.getRotationMatrix2D((Wr / 2, Hr / 2), ang, sc)
        M[:, 2] += rng.uniform(-3, 3, 2)
        rect = cv2.warpAffine(rect, M, (Wr, Hr), flags=cv2.INTER_CUBIC, borderValue=(255, 255, 255))
        segs = page_segments(meta, k, M)
        X, Y = RS.strips(rect, segs, rng, per_row=2, xup=xup)
        if not len(X):
            return None, o
        jp = [cv2.imencode(".jpg", x, [cv2.IMWRITE_JPEG_QUALITY, 92])[1].tobytes() for x in X]
        rows = np.arange(Y.shape[1], dtype=np.float32)[:, None]
        w = Y.sum(1)
        h = np.where(w > 0.5, (Y * rows[None]).sum(1) / np.maximum(w, 1e-6), np.nan).astype(np.float32)
        return (jp, h), o
    except Exception as e:  # noqa: BLE001
        return None, dict(error=f"{type(e).__name__}: {e}"[:200])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eik", type=Path, required=True, help="clone of alphanumericslab/ecg-image-kit")
    ap.add_argument("--bank", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--pages", type=int, default=20000)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    ap.add_argument("--chunk", type=int, default=400)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--xup", type=int, default=2)
    ap.add_argument("--cache", type=Path, default=None, help="persistent folder: sets kept by content key")
    a = ap.parse_args()
    gen_dir = (a.eik / "codes" / "ecg-image-generator").resolve()
    out = a.out
    if a.cache:
        h = hashlib.sha1()
        for f in [Path(__file__).resolve(), HERE / "realstrips.py"]:
            h.update(f.read_bytes())
        h.update(f"{a.pages} {a.chunk} {a.seed} {a.xup} {a.bank.stat().st_size}".encode())
        cdir = a.cache / f"eik-{h.hexdigest()[:12]}"
        cdir.mkdir(parents=True, exist_ok=True)
        if out.is_symlink():
            out.unlink()
        if not out.exists():
            out.parent.mkdir(parents=True, exist_ok=True)
            out.symlink_to(cdir)
        if (cdir / "DONE").exists():
            print(f"ecg-image-kit strips from the cache: {cdir}", flush=True)
            return
        for f in cdir.glob("chunk_*.npz"):
            f.unlink()
        out = cdir
    out.mkdir(parents=True, exist_ok=True)
    n_bank = np.load(a.bank, mmap_mode="r").shape[0]
    rng = np.random.default_rng(a.seed)
    idx = rng.integers(0, n_bank, a.pages)
    tmp = tempfile.mkdtemp(prefix="eik")
    jobs = [(str(gen_dir), str(a.bank), int(i), a.seed * 1_000_003 + p, a.xup, tmp) for p, i in enumerate(idx)]
    buf_x, buf_h, k, done, kept, errs = [], [], 0, 0, 0, {}
    t0 = time.time()

    def flush():
        nonlocal buf_x, buf_h, k
        if not buf_x:
            return
        t = out / f".chunk_{k:04d}.npz"
        np.savez(t, jpg=np.array(buf_x, dtype=object), h=np.stack(buf_h))
        os.replace(t, out / f"chunk_{k:04d}.npz")
        print(f"eik strips: chunk {k} ({len(buf_x)} strips), pages {done}/{len(jobs)}, kept {kept}, "
              f"{time.time() - t0:.0f} s", flush=True)
        buf_x, buf_h, k = [], [], k + 1

    with ProcessPoolExecutor(a.workers) as ex:
        for r, o in ex.map(_work, jobs, chunksize=4):
            done += 1
            if r is None:
                e = o.get("error", "no strip")
                errs[e] = errs.get(e, 0) + 1
                continue
            kept += 1
            jp, h = r
            buf_x += jp; buf_h += list(h)
            if len(buf_x) >= a.chunk:
                flush()
    flush()
    (out / "DONE").write_text(json.dumps(dict(pages=done, kept=kept, errors=errs)))
    print(f"eik strips: {kept} of {done} pages, {time.time() - t0:.0f} s; errors {errs}", flush=True)
    shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
