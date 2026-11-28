#!/bin/bash
# ECG-Digitiser (Krones et al., PhysioNet Challenge 2024 winner, BSD-2; github.com/felixkrones/ECG-Digitiser)
# on the LITFL pages (TS_LITFL=1, the only mode so far): nnU-Net segmentation of the 12 lead labels,
# Hough rotation, column-wise vectorisation; its weights expect the Challenge page (3x4 + one 10 s
# strip, ecg-image-kit). Output: one npz per page (signals 12 x 5000 mV, each 2.5 s lead at the
# start of the row, as the method writes it) in $OUT_DIR/litfl_ecgdigitiser_npz.tgz.
set -u
W=/root/ecgd; mkdir -p "$W"
git clone -q --depth 1 https://github.com/felixkrones/ECG-Digitiser "$W/code" || { echo "clone failed"; exit 1; }
(apt-get -qq update && apt-get -qq install -y git-lfs) > /dev/null 2>&1
( cd "$W/code" && git lfs install --local > /dev/null && git lfs pull --include "models/M3/**" ) || echo "git lfs pull failed"
find "$W/code/models/M3" -name "*.pth" -exec ls -la {} \;
pip install -q wfdb==4.1.2 opencv-python-headless scikit-image tqdm matplotlib "pandas==2.2.2" 2>&1 | tail -1   # its pins: pandas 3 breaks its code
( cd "$W/code/nnUNet" && pip install -q . ) 2>&1 | tail -1
pip install -q "pandas==2.2.2" 2>&1 | tail -1   # nnU-Net may pull a newer one
python -c "import pandas, numpy, torch; print('pandas', pandas.__version__, 'numpy', numpy.__version__, 'torch', torch.__version__)"
mkdir -p "$W/in"
python - "$W/in" <<'PY'
import cv2, sys, pathlib
for p in sorted(pathlib.Path("benchmarks/litfl/images").glob("img_*")):
    img = cv2.imread(str(p))
    if img is not None:
        cv2.imwrite(f"{sys.argv[1]}/{p.stem.replace('img_', '')}.png", img)
PY
echo "$(ls $W/in | wc -l) pages"
t0=$(date +%s)
( cd "$W/code" && python -m src.run.digitize -d "$W/in" -o "$W/out" -m models/M3/ -f > "$W/run.log" 2>&1 )
grep -E "Error|Traceback" "$W/run.log" | sort | uniq -c | sort -rn | head -8; grep -B8 -m1 "Error" "$W/run.log" | head -20; tail -3 "$W/run.log"
echo "ECG-Digitiser ran in $(( $(date +%s) - t0 )) s; $(ls $W/out/*.hea 2>/dev/null | wc -l) outputs"
python - "$W/out" "$W/ext" <<'PY'
import sys, pathlib, numpy as np, wfdb
LEADS = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
src, out = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2]); out.mkdir(parents=True, exist_ok=True)
n = 0
for h in sorted(src.glob("*.hea")):
    r = wfdb.rdrecord(str(h.with_suffix("")))
    sig = np.full((12, 5000), np.nan, np.float32)
    for j, nm in enumerate(r.sig_name):
        if nm in LEADS:
            y = r.p_signal[:5000, j].astype(np.float32)
            # it writes NaN as 0 after the lead's own samples: a short lead keeps its 2.5 s only
            if nm != "II":
                y[1250:] = np.nan
            sig[LEADS.index(nm), :len(y)] = y
    np.savez(out / f"{h.stem}.npz", leads=np.array(LEADS), signals=sig, layout=np.array("ecgdigitiser"))
    n += 1
print(f"{n} pages converted")
PY
mkdir -p "$W/ext"; tar czf "$OUT_DIR/litfl_ecgdigitiser_npz.tgz" -C "$W/ext" . && echo "LITFL readings packed"
cp "$W/run.log" "$OUT_DIR/ecgdigitiser.log"
echo done
