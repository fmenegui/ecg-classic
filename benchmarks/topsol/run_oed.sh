#!/bin/bash
# Open-ECG-Digitizer (Ahus-AIM, Akershus University Hospital; Stenhede et al., npj Digit Med 2026;
# CC BY-SA 4.0) on our development sets, on a GPU pod:
#   bash benchmarks/topsol/run_oed.sh          (gpu-train.yml kaggle=true for the Kaggle part)
# Input size: TS_OED_SIZE=WxH fits every image inside WxH (aspect kept, white padding); the default,
# none, keeps the images as they are (it resamples to 3000 px inside). Nov 20, 60 PM-ECG-ID ECGs:
# 2200x1700 3.91 dB, own size 5.09, 3300x2550 5.12 (runs 44834969780, 44834966288, 44834973129). Generic layouts (lead_layouts_all.yml: 12x1, 6x2, 3x4 with 0-3 rhythm strips, ...).
# Options as run_hengck.sh: TS_PM, TS_KAGGLE, TS_LIMIT, TS_PM_FORMATS, TS_PM_PER, TS_PROBE (probe pages K0-K6, E0-E7).
set -u
W=/root/oed; mkdir -p "$W"
git clone -q --depth 1 https://github.com/Ahus-AIM/Open-ECG-Digitizer "$W/code" || { echo "clone failed"; exit 1; }
for f in unet_weights_07072025.pt lead_name_unet_weights_07072025.pt; do   # git-lfs files, by their media URL
  curl -sSL -o "$W/code/weights/$f" "https://media.githubusercontent.com/media/Ahus-AIM/Open-ECG-Digitizer/main/weights/$f"
  echo "weights $f $(du -m "$W/code/weights/$f" | cut -f1) MB"
done
pip install -q yacs torch-tps "ray[tune]" scikit-learn pyyaml scikit-image opencv-python-headless networkx pillow scipy tqdm matplotlib \
  kaggle==2.2.5 kagglesdk==0.1.38 six==1.17.0 remotezip pandas 2>&1 | tail -2

PREP=""
if [ "${TS_PM:-1}" = "1" ]; then
  F=""; [ -n "${TS_PM_FORMATS:-}" ] && F="--formats $TS_PM_FORMATS --pages 1"
  python benchmarks/pmcardio/fetch.py --out /root/pm --per-scenario "${TS_PM_PER:-5}" --threads 8 --seed 0 $F \
    --sel-out "$W/pm_sel.csv" | tail -2
  if [ -n "${TS_LIMIT:-}" ]; then head -n $(( TS_LIMIT + 1 )) "$W/pm_sel.csv" > "$W/s.csv"; mv "$W/s.csv" "$W/pm_sel.csv"; fi
  PREP="$PREP --pm-root /root/pm/visual_data --pm-list $W/pm_sel.csv"
fi
if [ "${TS_KAGGLE:-1}" = "1" ]; then
  python benchmarks/physionet/kaggle_zip.py shard --part -1 --nparts 20 --stride 20 --out /root/kaggle --threads 8 \
    --ids-file "$W/kaggle_ids.txt" | tail -2
  IDS=$(cat "$W/kaggle_ids.txt")
  if [ -n "${TS_LIMIT:-}" ]; then IDS=$(echo "$IDS" | tr ',' '\n' | head -n $(( (TS_LIMIT + 8) / 9 )) | paste -sd,); echo "$IDS" > "$W/kaggle_ids.txt"; fi
  PREP="$PREP --kaggle-dir /root/kaggle --kaggle-ids $IDS"
fi
if [ "${TS_PROBE:-0}" = "1" ]; then
  # probe pages: the Kaggle template geometry, then one change at a time (probe_pages.py, probe_eik.py)
  pip install -q wfdb 2>&1 | tail -1
  python benchmarks/topsol/probe_pages.py --out /root/probe --n "${TS_PROBE_N:-20}" ${TS_PROBE_K:+--variants $TS_PROBE_K} | tail -3
  git clone -q --depth 1 https://github.com/alphanumericslab/ecg-image-kit.git /root/eik && \
    python benchmarks/topsol/probe_eik.py --eik /root/eik --records /root/probe/.records --out /root/probe --n "${TS_PROBE_N:-20}" | tail -3
  PREP="$PREP --probe-dir /root/probe"
fi
if [ "${TS_LITFL:-0}" = "1" ]; then PREP="$PREP --litfl-dir benchmarks/litfl/images"; fi   # LITFL: no truth, signals kept
RS="${TS_OED_SIZE:-none}"; [ "$RS" = none ] && RS=""      # none: the images as they are
python benchmarks/topsol/prepare.py --out "$W/data" ${RS:+--resize $RS} $PREP

cat > "$W/code/src/config/ours.yml" <<YML
MODEL:
  class_path: 'src.model.inference_wrapper.InferenceWrapper'
  KWARGS:
    config:
      SIGNAL_EXTRACTOR: {class_path: 'src.model.signal_extractor.SignalExtractor', KWARGS: {}}
      PERSPECTIVE_DETECTOR: {class_path: 'src.model.perspective_detector.PerspectiveDetector', KWARGS: {num_thetas: 250}}
      DEWARPER: {class_path: 'src.model.dewarper.Dewarper', KWARGS: {abs_peak_threshold: 0.1}}
      SEGMENTATION_MODEL:
        class_path: 'src.model.unet.UNet'
        weight_path: './weights/unet_weights_07072025.pt'
        KWARGS: {num_in_channels: 3, num_out_channels: 4, dims: [32, 64, 128, 256, 320, 320, 320, 320], depth: 2}
      CROPPER: {class_path: 'src.model.cropper.Cropper', KWARGS: {granularity: 80, percentiles: [0.02, 0.98], alpha: 0.85}}
      PIXEL_SIZE_FINDER:
        class_path: 'src.model.pixel_size_finder.PixelSizeFinder'
        KWARGS: {min_number_of_grid_lines: 30, max_number_of_grid_lines: 70, lower_grid_line_factor: 0.3}
      LAYOUT_IDENTIFIER:
        class_path: 'src.model.lead_identifier.LeadIdentifier'
        config_path: 'src/config/lead_layouts_all.yml'
        unet_config_path: 'src/config/lead_name_unet.yml'
        unet_weight_path: './weights/lead_name_unet_weights_07072025.pt'
        KWARGS: {debug: false, device: 'cuda', possibly_flipped: true}
    device: 'cuda'
    resample_size: 3000
    rotate_on_resample: true
    enable_timing: false
    apply_dewarping: true
DATA:
  images_path: '$W/data/test'
  image_extensions: ['.png']
  output_path: '$W/out'
  save_mode: 'timeseries_only'
  layout_should_include_substring: false
YML
t0=$(date +%s)
( cd "$W/code" && python -m src.digitize --config src/config/ours.yml > "$W/oed.log" 2>&1 )
grep -E "Error|Traceback|error" "$W/oed.log" | sort | uniq -c | sort -rn | head -20
echo "Open-ECG-Digitizer ran in $(( $(date +%s) - t0 )) s; $(ls $W/out/*_timeseries_canonical.csv 2>/dev/null | wc -l) outputs"
python benchmarks/topsol/to_npz_oed.py "$W/out" "$W/data/test.csv" "$W/ext"

# nproc shows the host's cores (128 on some pods), not the pod's share: 128 evaluator processes
# with OpenBLAS threads each hit the thread limit and crashed the Kaggle scoring (Nov 20, 44801832861)
if [ "${TS_LITFL:-0}" = "1" ]; then   # public LITFL pages: the readings themselves leave the pod (no Kaggle data)
  tar czf "$OUT_DIR/litfl_oed_npz.tgz" -C "$W/ext" . && echo "LITFL readings: $(ls $W/ext | wc -l) npz"
fi
JOBS=${TS_JOBS:-$(python -c "import os; print(min(16, len(os.sched_getaffinity(0))))")}
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
if [ "${TS_PROBE:-0}" = "1" ]; then
  python benchmarks/topsol/score_probe.py --probe /root/probe --npz "$W/ext" --out "$OUT_DIR/probe_results.json" | tee "$OUT_DIR/probe_report.md"
fi
if [ "${TS_PM:-1}" = "1" ]; then
  ECGDIG_EXTERNAL="$W/ext" python benchmarks/pmcardio/eval_pm.py --root /root/pm/visual_data --gt /root/pm/final_data/data \
    --meta /root/pm/final_data/metadata.csv --list "$W/pm_sel.csv" --out "$W/pm" --jobs "$JOBS" | tail -15
  cp "$W/pm/results.json" "$OUT_DIR/pm_results.json"; cp "$W/pm/report.md" "$OUT_DIR/pm_report.md" 2>/dev/null
fi
if [ "${TS_KAGGLE:-1}" = "1" ]; then
  mkdir -p "$W/kev"; ln -sfn /root/kaggle "$W/kev/data"
  ECGDIG_EXTERNAL="$W/ext" python benchmarks/physionet/run.py eval --n 100000 --jobs "$JOBS" --keep-raw 0 \
    --ids "$(cat "$W/kaggle_ids.txt")" --out "$W/kev" | tail -15
  cp "$W/kev/results.json" "$OUT_DIR/kaggle_results.json"; cp "$W/kev/report.md" "$OUT_DIR/kaggle_report.md"
  python - "$W/kev/results.json" "$W/ext" <<'PY' || FAILED=1
import json, sys, pathlib
n = len(json.load(open(sys.argv[1]))["results"]); m = len(list(pathlib.Path(sys.argv[2]).glob("*-*.npz")))
print(f"Kaggle scoring coverage: {n} of {m} read images scored")
sys.exit(0 if n >= m else "Kaggle scoring incomplete")
PY
  rm -rf "$W/kev/sig"
fi
[ "${FAILED:-0}" = 1 ] && { echo "scoring incomplete"; exit 1; }
echo "done"
