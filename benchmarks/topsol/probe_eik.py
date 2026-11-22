"""Probe pages drawn by ecg-image-kit, the generator of the Kaggle images, then changed one thing at a time.

    python benchmarks/topsol/probe_eik.py --eik ecg-image-kit --records DIR --out /root/probe [--n 20]

benchmarks/topsol/probe_pages.py draws the Kaggle template GEOMETRY in our own style, and the
hengck23-based winners did not read even that page (stage 0 found no lead-name keypoints): they
depend on the Kaggle images' drawing style too. These pages are drawn by the same generator as
the competition's images (alphanumericslab/ecg-image-kit, its default 3x4 + II sheet at 200 dpi,
calibration pulse, red grid; no augmentation), with real PTB-XL records the winners never saw:

  E0  the generator's default sheet
  E1  no lead names
  E2  blue grid (the generator's colour 3)
  E3  the whole sheet 8% smaller in the same image
  E4  rotated 3 degrees, JPEG quality 70 (a mild scan)
  E5  6x2 (two columns of 5 s)
  E6  rows 30% closer together
  E7  the top 12% of the sheet cut off (a photo that does not show the whole page)

The records are the PTB-XL 500 Hz files in --records (*.hea/*.dat, as probe_pages.py downloads
them). Appends to OUT/test.csv and writes OUT/test/<key>.png, OUT/gt/<key>.npz like probe_pages.py.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import types
from pathlib import Path

import numpy as np

LEADS = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
VARIANTS = ["E0", "E1", "E2", "E3", "E4", "E5", "E6", "E7"]


def _load_generator(gen_dir: Path, row_factor: float):
    """extract_leads.get_paper_ecg with ecg_plot's row height scaled (imgaug stubbed: not used here)."""
    for k in [k for k in sys.modules if k in ("ecg_plot", "extract_leads", "helper_functions")]:
        del sys.modules[k]
    sys.modules.setdefault("imgaug", types.ModuleType("imgaug"))
    aug = types.ModuleType("imgaug.augmenters"); sys.modules["imgaug.augmenters"] = aug
    sys.modules["imgaug"].augmenters = aug
    if str(gen_dir) not in sys.path:
        sys.path.insert(0, str(gen_dir))
    src = (gen_dir / "ecg_plot.py").read_text()
    old = "row_height = (height * y_grid_size/y_grid)/(rows+2)"
    assert old in src, "ecg-image-kit changed: row height line not found"
    src = src.replace(old, old + f" * {row_factor}")
    mod = types.ModuleType("ecg_plot"); mod.__file__ = str(gen_dir / "ecg_plot.py")
    exec(compile(src, mod.__file__, "exec"), mod.__dict__)
    sys.modules["ecg_plot"] = mod
    import extract_leads
    from helper_functions import read_config_file
    return extract_leads.get_paper_ecg, read_config_file(str(gen_dir / "config.yaml"))


def render(gen_dir, rec_stem: Path, out_dir: Path, seed: int, names=True, colour=5, columns=-1, row_factor=1.0):
    from scipy.stats import bernoulli
    cwd = os.getcwd()
    os.chdir(gen_dir)
    try:
        get_paper_ecg, configs = _load_generator(gen_dir, row_factor)
        out_dir.mkdir(parents=True, exist_ok=True)
        res = get_paper_ecg(input_file=str(rec_stem) + ".dat", header_file=str(rec_stem) + ".hea", configs=configs,
                            mask_unplotted_samples=False, start_index=0, store_configs=2, store_text_bbox=False,
                            output_directory=str(out_dir), resolution=200, papersize="", add_lead_names=names,
                            add_dc_pulse=bernoulli(1), add_bw=bernoulli(0), show_grid=bernoulli(1),
                            add_print=bernoulli(0), pad_inches=0, font_type=os.path.join("Fonts", "Times_New_Roman.ttf"),
                            standard_colours=colour, full_mode="II", bbox=False, columns=columns, seed=seed)
    finally:
        os.chdir(cwd)
    png = Path(res[0])
    meta = json.loads(png.with_suffix(".json").read_text())
    return png, meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eik", type=Path, required=True, help="clone of alphanumericslab/ecg-image-kit")
    ap.add_argument("--records", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--variants", default=",".join(VARIANTS))
    a = ap.parse_args()
    import cv2
    import wfdb
    gen = (a.eik / "codes" / "ecg-image-generator").resolve()
    (a.out / "test").mkdir(parents=True, exist_ok=True); (a.out / "gt").mkdir(parents=True, exist_ok=True)
    stems = sorted({p.with_suffix("") for p in a.records.glob("*.hea")})
    k0 = sorted((a.out / "gt").glob("K0_*.npz"))
    if k0:   # the same records as probe_pages.py's pages (its cache also holds records it rejected)
        ids = [int(np.load(f)["ecg_id"]) for f in k0]
        by_id = {int(s.name.split("_")[0]): s for s in stems}
        stems = [by_id[i] for i in ids if i in by_id]
    stems = stems[: a.n]
    tmp = a.out / ".eik"
    rows = []
    for v in a.variants.split(","):
        for p, stem in enumerate(stems):
            key = f"{v}_{p:03d}"
            kw = dict(names=v != "E1", colour=3 if v == "E2" else 5, columns=2 if v == "E5" else -1,
                      row_factor=0.7 if v == "E6" else 1.0)
            png, meta = render(gen, stem.resolve(), tmp / key, seed=p, **kw)
            img = cv2.imread(str(png))
            H, W = img.shape[:2]
            if v == "E3":
                r = cv2.resize(img, (int(W * 0.92), int(H * 0.92)), interpolation=cv2.INTER_AREA)
                img = np.full_like(img, 255); y0, x0 = (H - r.shape[0]) // 2, (W - r.shape[1]) // 2
                img[y0:y0 + r.shape[0], x0:x0 + r.shape[1]] = r
            elif v == "E4":
                M = cv2.getRotationMatrix2D((W / 2, H / 2), 3.0, 1.0)
                img = cv2.warpAffine(img, M, (W, H), borderValue=(255, 255, 255))
                img = cv2.imdecode(cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 70])[1], cv2.IMREAD_COLOR)
            elif v == "E7":
                c = int(0.12 * H)
                img = cv2.resize(img[c:], (W, H - c))
            cv2.imwrite(str(a.out / "test" / f"{key}.png"), img)
            sig, f = wfdb.rdsamp(str(stem))
            names = [n.upper() for n in f["sig_name"]]
            S = np.stack([sig[:5000, names.index(n.upper())] for n in LEADS]).astype(np.float32)
            segs = [dict(lead=L.get("lead_name", "II"), rhythm=(L["end_sample"] - L["start_sample"]) >= 4000,
                         t0=L["start_sample"] / 500.0, t1=L["end_sample"] / 500.0) for L in meta["leads"]]
            ecg_id = int(stem.name.split("_")[0])
            np.savez_compressed(a.out / "gt" / f"{key}.npz", signals=np.nan_to_num(S), leads=np.array(LEADS),
                                segments=json.dumps(segs), ecg_id=ecg_id, variant=v)
            rows += [(key, L, 500, 5000 if L == "II" else 1250) for L in LEADS]
        print(v, "done", flush=True)
    tc = a.out / "test.csv"
    old = list(csv.reader(open(tc)))[1:] if tc.exists() else []
    with open(tc, "w", newline="") as f_:
        w = csv.writer(f_); w.writerow(["id", "lead", "fs", "number_of_rows"]); w.writerows(old + rows)
    print(len(rows) // 12, "ecg-image-kit pages added to", a.out)


if __name__ == "__main__":
    main()
