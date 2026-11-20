#!/bin/bash
# A hengck23-based Kaggle top solution (2nd or 3rd place) on our development sets, on a GPU pod:
#   SOL=takashisomeya/physionet-2nd-place-submission  bash benchmarks/topsol/run_hengck.sh
# Needs the Kaggle token (gpu-train.yml kaggle=true): the notebook, its public datasets (code and
# weights) and the evaluation data are downloaded on the pod. Writes only pm_results.json,
# kaggle_results.json and their reports to $OUT_DIR.
#   TS_PM=1 / TS_KAGGLE=1   which sets (both by default)
#   TS_LIMIT=N              at most N images per set (smoke test)
#   TS_PM_FORMATS, TS_PM_PER PM-ECG-ID selection (see the fetch line)
set -u
SOL=${SOL:-takashisomeya/physionet-2nd-place-submission}
W=/root/ts; mkdir -p "$W/input" "$W/work" "$W/data" "$W/nb"
# the solutions need timm >= 1.0.12 (pretrained tags like ra4_e3600_r224_in1k) but < the versions that
# require an hf-hub: prefix for "tu-timm/..." encoders (Nov 20: ValueError with the newest timm)
pip install -q -U kaggle==2.2.5 kagglesdk==0.1.38 six==1.17.0 remotezip pandas connected-components-3d "timm==1.0.15" "segmentation-models-pytorch==0.5.0" 2>&1 | tail -2
kaggle kernels pull "$SOL" -p "$W/nb" -m > /dev/null || { echo "could not pull $SOL"; exit 1; }
NB=$(ls "$W"/nb/*.ipynb | head -1)
python - "$W/nb/kernel-metadata.json" > "$W/datasets.txt" <<'EOF'
import json, sys
for d in json.load(open(sys.argv[1])).get("dataset_sources") or []:
    print(d)
EOF
while read ds; do
  slug=${ds#*/}
  echo "dataset $ds"
  kaggle datasets download "$ds" -p "$W/input/$slug" --unzip -q || echo "could not download $ds"
done < "$W/datasets.txt"
du -sh "$W"/input/* 2>/dev/null
python benchmarks/topsol/make_run.py "$NB" "$W/run.py"

PREP=""
if [ "${TS_PM:-1}" = "1" ]; then
  # TS_PM_FORMATS=3x4+1R: only ECGs printed like the Kaggle sheet (one page); TS_PM_PER: ECGs per
  # scenario (0 = every one of those formats)
  F=""; [ -n "${TS_PM_FORMATS:-}" ] && F="--formats $TS_PM_FORMATS --pages 1"
  python benchmarks/pmcardio/fetch.py --out /root/pm --per-scenario "${TS_PM_PER:-5}" --threads 8 --seed 0 $F \
    --sel-out "$W/pm_sel.csv" | tail -2
  if [ -n "${TS_LIMIT:-}" ]; then head -n $(( TS_LIMIT + 1 )) "$W/pm_sel.csv" > "$W/pm_sel_l.csv"; mv "$W/pm_sel_l.csv" "$W/pm_sel.csv"; fi
  PREP="$PREP --pm-root /root/pm/visual_data --pm-list $W/pm_sel.csv"
fi
if [ "${TS_KAGGLE:-1}" = "1" ]; then
  python benchmarks/physionet/kaggle_zip.py shard --part -1 --nparts 20 --stride 20 --out /root/kaggle --threads 8 \
    --ids-file "$W/kaggle_ids.txt" | tail -2
  IDS=$(cat "$W/kaggle_ids.txt")
  if [ -n "${TS_LIMIT:-}" ]; then IDS=$(echo "$IDS" | tr ',' '\n' | head -n $(( (TS_LIMIT + 8) / 9 )) | paste -sd,); echo "$IDS" > "$W/kaggle_ids.txt"; fi
  PREP="$PREP --kaggle-dir /root/kaggle --kaggle-ids $IDS"
fi
python benchmarks/topsol/prepare.py --out "$W/data" $PREP
t0=$(date +%s)
( cd "$W/work" && TS_DATA="$W/data" TS_INPUT="$W/input" TS_WORK="$W/work" python "$W/run.py" > "$W/solution.log" 2>&1 )
# progress lines (\r) dropped; every error kept with its context
tr '\r' '\n' < "$W/solution.log" | grep -vE "^\s+(\[GPU[0-9]\])?\s*[0-9]+(/[0-9]+)? |^\s*$" | grep -vE "WARN.*imread" | head -150
echo "solution ran in $(( $(date +%s) - t0 )) s on $(ls $W/work/outputs/digitalised 2>/dev/null | wc -l) images"
python benchmarks/topsol/to_npz.py "$W/work/outputs/digitalised" "$W/data/test.csv" "$W/ext"

# nproc shows the host's cores (128 on some pods), not the pod's share: 128 evaluator processes
# with OpenBLAS threads each hit the thread limit and crashed the Kaggle scoring (Nov 20, 44801832861)
JOBS=${TS_JOBS:-$(python -c "import os; print(min(16, len(os.sched_getaffinity(0))))")}
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
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
