#!/bin/bash
# 4th place Kaggle solution (tomoon33: grid multitask + grid line index rectification, an ensemble of
# 4 digitisation models x 5 folds) on our development sets, on a GPU pod:
#   bash benchmarks/topsol/run_4th.sh          (gpu-train.yml kaggle=true disk_gb=150)
# Its notebook is shell cells driven by environment variables (RAW_IMAGE_DIR, TEST_CSV); it is
# converted to a script (nb2sh.py) and pointed at our images (prepare.py). Its submission.csv is
# converted to our npz (to_npz_sub.py) and scored by our evaluators.
# Options as run_hengck.sh: TS_PM, TS_KAGGLE, TS_LIMIT, TS_PM_FORMATS, TS_PM_PER.
set -u
SOL=tomoon33/physionet-submission-4th-place
W=/root/t4; export TS_INPUT=$W/input TS_WORK=$W/work; mkdir -p "$TS_INPUT" "$TS_WORK" "$W/nb"
pip install -q -U kaggle==2.2.5 kagglesdk==0.1.38 six==1.17.0 remotezip pandas uv 2>&1 | tail -1
kaggle kernels pull "$SOL" -p "$W/nb" -m > /dev/null || { echo "could not pull $SOL"; exit 1; }
NB=$(ls "$W"/nb/*.ipynb | head -1)
python - "$W/nb/kernel-metadata.json" > "$W/sources.txt" <<'PY'
import json, sys
m = json.load(open(sys.argv[1]))
for d in m.get("dataset_sources") or []:
    print("dataset", d)
for k in m.get("kernel_sources") or []:
    print("kernel", k)
PY
df -h /root | tail -1     # the weights take ~70 GB unpacked: gpu-train.yml disk_gb=150
while read kind ref; do
  slug=${ref#*/}; echo "$kind $ref"
  if [ "$kind" = dataset ]; then kaggle datasets download "$ref" -p "$TS_INPUT/$slug" --unzip -q || echo "could not download $ref"
  else kaggle kernels output "$ref" -p "$TS_INPUT/$slug" -q > /dev/null || echo "could not download $ref"; fi
done < "$W/sources.txt"
du -sh "$TS_INPUT"/* 2>/dev/null
# an image the solution cannot read aborts its whole submission: NaN for it instead (patch_4th.py)
find "$TS_INPUT" -name run_ecg_digitization_inference.py -exec python benchmarks/topsol/patch_4th.py {} +

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
  # probe pages: the Kaggle template geometry, then one change at a time (probe_pages.py)
  pip install -q wfdb matplotlib pyyaml 2>&1 | tail -1
  python benchmarks/topsol/probe_pages.py --out /root/probe --n "${TS_PROBE_N:-20}" | tail -3
  # the same records drawn by ecg-image-kit, the generator of the Kaggle images (probe_eik.py)
  git clone -q --depth 1 https://github.com/alphanumericslab/ecg-image-kit.git /root/eik && \
    python benchmarks/topsol/probe_eik.py --eik /root/eik --records /root/probe/.records --out /root/probe --n "${TS_PROBE_N:-20}" | tail -3
  PREP="$PREP --probe-dir /root/probe"
fi
python benchmarks/topsol/prepare.py --out "$W/data" $PREP

python benchmarks/topsol/nb2sh.py "$NB" "$W/run.sh"
# the solution reads every image of RAW_IMAGE_DIR listed in TEST_CSV; one GPU here
sed -i "s|--world-size 2|--world-size 1|; s|uv pip install --system|uv pip install --system --python $(which python)|" "$W/run.sh"
echo "export RAW_IMAGE_DIR=$W/data/test TEST_CSV=$W/data/test.csv" >> "$W/env.sh"
python - "$W/run.sh" "$W/env.sh" <<'PY'
import sys
s = open(sys.argv[1]).read(); env = open(sys.argv[2]).read()
# our images and request list instead of the competition's test set
s = s.replace("export RAW_IMAGE_DIR=", "# export RAW_IMAGE_DIR=").replace("export TEST_CSV=", "# export TEST_CSV=")
s = s.replace("set -x\n", "set -x\n" + env, 1)
open(sys.argv[1], "w").write(s)
PY
t0=$(date +%s)
( cd "$TS_WORK" && bash "$W/run.sh" > "$W/solution.log" 2>&1 )
tr '\r' '\n' < "$W/solution.log" | grep -vE "^\s*$|it/s\]|s/it\]" | tail -80
echo "solution ran in $(( $(date +%s) - t0 )) s"
ls -la "$TS_WORK/submission.csv" 2>/dev/null
python benchmarks/topsol/to_npz_sub.py "$TS_WORK/submission.csv" "$W/data/test.csv" "$W/ext"

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
  rm -rf "$W/kev/sig"
fi
echo "done"
