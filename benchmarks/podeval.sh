#!/bin/bash
# Evaluation job of a CPU pod (.github/workflows/pod-eval.yml), run from the repository by
# hybrid/gpu/pod.sh with /workspace on the network volume:
# - the datasets live on the volume (/workspace/data) and are downloaded only when missing;
# - models arrive in $IN_DIR (sent by the workflow), READY marks the end of the upload;
# - work files stay in the run folder; only results.json / report.md go to $OUT_DIR, the one
#   folder the pod serves (never images, ground truth or read signals).
# Environment: WHAT (pm, kaggle or both), PER_SCENARIO (PM ECGs per scenario, 0 = all),
# KAGGLE_PART/KAGGLE_NPARTS (subset of the Kaggle evaluation records), EVAL_CONFIG, JOBS.
set -u
RUN=$(dirname "$OUT_DIR"); D=/workspace/data; IN=${IN_DIR:-$RUN/in}
mkdir -p "$D" "$RUN"
JOBS=${JOBS:-$(nproc)}
echo "waiting for the models"
for i in $(seq 1 180); do [ -f "$IN/READY" ] && break; sleep 10; done
ls -la "$IN" 2>/dev/null
ARGS=""
for m in ink grid row page dewarp; do
  [ -f "$IN/$m.onnx" ] && ARGS="$ARGS --config ${m}_model=$IN/$m.onnx"
done
for kv in ${EVAL_CONFIG:-}; do ARGS="$ARGS --config $kv"; done
echo "config:$ARGS  jobs: $JOBS"
case "${WHAT:-both}" in pm|both)
  t0=$(date +%s)
  SH=""; [ "${NSHARDS:-1}" -gt 1 ] && SH="--shard ${SHARD:-0} --nshards ${NSHARDS}"
  python benchmarks/pmcardio/fetch.py --out "$D/pm" --per-scenario "${PER_SCENARIO:-0}" --threads 8 --seed 0 $SH \
    --sel-out "$RUN/pm_sel.csv" | tail -3
  echo "PM data ready in $(( $(date +%s) - t0 )) s"
  if [ "${DATA_ONLY:-}" = "1" ]; then echo "data only"; else
  python benchmarks/pmcardio/eval_pm.py --root "$D/pm/visual_data" --gt "$D/pm/final_data/data" \
    --meta "$D/pm/final_data/metadata.csv" --list "$RUN/pm_sel.csv" --out "$RUN/pm" --jobs "$JOBS" $ARGS | tail -40
  cp "$RUN/pm/results.json" "$OUT_DIR/pm_results.json"; cp "$RUN/pm/report.md" "$OUT_DIR/pm_report.md" 2>/dev/null
  echo "PM done in $(( $(date +%s) - t0 )) s"
  fi
esac
case "${WHAT:-both}" in kaggle|both)
  t0=$(date +%s)
  if [ -z "${KAGGLE_API_TOKEN:-}" ]; then echo "no Kaggle token: Kaggle part skipped"; else
    KP=${KAGGLE_PART:-0}; KN=${KAGGLE_NPARTS:-1}
    if [ "${NSHARDS:-1}" -gt 1 ]; then KP=$(( KP * NSHARDS + ${SHARD:-0} )); KN=$(( KN * NSHARDS )); fi
    python benchmarks/physionet/kaggle_zip.py shard --part "$KP" --nparts "$KN" --out "$D/kaggle" --threads 8 \
      --ids-file "$RUN/kaggle_ids.txt" | grep -v "^token\|attempt"
    echo "Kaggle data ready in $(( $(date +%s) - t0 )) s"
    if [ "${DATA_ONLY:-}" != "1" ]; then
    mkdir -p "$RUN/kaggle"; ln -sfn "$D/kaggle" "$RUN/kaggle/data"
    python benchmarks/physionet/run.py eval --n 100000 --jobs "$JOBS" --keep-raw 0 --ids "$(cat "$RUN/kaggle_ids.txt")" \
      --out "$RUN/kaggle" $ARGS | tail -30
    cp "$RUN/kaggle/results.json" "$OUT_DIR/kaggle_results.json"; cp "$RUN/kaggle/report.md" "$OUT_DIR/kaggle_report.md"
    rm -rf "$RUN/kaggle/sig"   # signals read next to the ground truth: never kept
    echo "Kaggle done in $(( $(date +%s) - t0 )) s"
    fi
  fi
esac
