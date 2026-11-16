"""Command line:

    python -m ecgdig image.jpg -o out               # writes out.csv, out.json, out_debug.jpg
    python -m ecgdig image.jpg --layout 6x2 --fs 500 --duration 10
"""
import argparse
import json
import sys

import cv2
import numpy as np

from .digitize import digitize, Config, LAYOUTS


def main(argv=None):
    ap = argparse.ArgumentParser(prog="ecgdig", description="Classical ECG digitizer (no deep learning)")
    ap.add_argument("image")
    ap.add_argument("-o", "--out", default=None, help="prefix of the output files")
    ap.add_argument("--layout", default="auto", choices=["auto"] + list(LAYOUTS))
    ap.add_argument("--fs", type=int, default=500)
    ap.add_argument("--duration", type=float, default=10.0)
    ap.add_argument("--speed", type=float, default=25.0, help="mm/s")
    ap.add_argument("--gain", type=float, default=None, help="mm/mV (default: measured on the calibration pulse)")
    ap.add_argument("--ppmm", type=float, default=None, help="image px/mm, if known")
    ap.add_argument("--fill-limb", action="store_true", help="rebuild missing limb leads via Einthoven/Goldberger")
    ap.add_argument("--no-einthoven", action="store_true")
    a = ap.parse_args(argv)
    cfg = Config(fs=a.fs, duration=a.duration, speed=a.speed, gain=a.gain, layout=a.layout,
                 einthoven=not a.no_einthoven, fill_limb=a.fill_limb, ppmm=a.ppmm)
    res = digitize(a.image, cfg, return_debug=True)
    out = a.out or a.image.rsplit(".", 1)[0]
    sig = res["signals"]
    t = np.arange(sig.shape[1]) / res["fs"]
    hdr = "t," + ",".join(res["leads"])
    np.savetxt(out + ".csv", np.column_stack([t, sig.T]), delimiter=",", header=hdr, comments="", fmt="%.5f")
    meta = {k: v for k, v in res.items() if k not in ("signals", "debug_image", "rectified", "ink")}
    meta = {k: (float(v) if isinstance(v, (np.floating, float)) else v) for k, v in meta.items()}
    with open(out + ".json", "w") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    cv2.imwrite(out + "_debug.jpg", res["debug_image"])
    print(f"ok: {out}.csv  ({sig.shape[0]} leads x {sig.shape[1]} samples, layout {res['layout']}, "
          f"gain {res['gain_mm_per_mv']} mm/mV)")


if __name__ == "__main__":
    sys.exit(main())
