"""Linha de comando:

    python -m ecgdig imagem.jpg -o saida            # gera saida.csv, saida.json, saida_debug.jpg
    python -m ecgdig imagem.jpg --layout 6x2 --fs 500 --duration 10
"""
import argparse
import json
import sys

import cv2
import numpy as np

from .digitize import digitize, Config, LAYOUTS


def main(argv=None):
    ap = argparse.ArgumentParser(prog="ecgdig", description="Digitalizador clássico de ECG (sem deep learning)")
    ap.add_argument("image")
    ap.add_argument("-o", "--out", default=None, help="prefixo dos arquivos de saída")
    ap.add_argument("--layout", default="auto", choices=["auto"] + list(LAYOUTS))
    ap.add_argument("--fs", type=int, default=500)
    ap.add_argument("--duration", type=float, default=10.0)
    ap.add_argument("--speed", type=float, default=25.0, help="mm/s")
    ap.add_argument("--gain", type=float, default=None, help="mm/mV (padrão: mede no pulso)")
    ap.add_argument("--ppmm", type=float, default=None, help="px/mm da imagem, se conhecido")
    ap.add_argument("--fill-limb", action="store_true", help="reconstrói derivações dos membros via Einthoven/Goldberger")
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
    print(f"ok: {out}.csv  ({sig.shape[0]} derivações x {sig.shape[1]} amostras, layout {res['layout']}, "
          f"ganho {res['gain_mm_per_mv']} mm/mV)")


if __name__ == "__main__":
    sys.exit(main())
