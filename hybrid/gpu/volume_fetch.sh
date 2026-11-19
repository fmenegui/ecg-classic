#!/bin/bash
# Job of .github/workflows/volume-fetch.yml on a CPU pod with the data volume at /workspace:
# copies the checkpoints of training run $1 (/workspace/runs/$1/out) into $OUT_DIR, the one
# folder the pod serves. Only names on the list leave the volume; never data.
set -u
RID=$1
case "$RID" in (*[!0-9]*|'') echo "bad run id"; exit 1;; esac
SRC=/workspace/runs/$RID/out
[ -d "$SRC" ] || { echo "no folder for run $RID on the volume"; ls /workspace/runs | tail -20; exit 1; }
ls -la "$SRC"
n=0
for f in "$SRC"/*; do
  b=$(basename "$f")
  if [[ "$b" =~ ^[A-Za-z0-9]+(_[A-Za-z0-9]+)*_best\.(pt|onnx)$ || "$b" == val.json ]]; then
    cp "$f" "$OUT_DIR/$b" && n=$((n + 1)) && echo "copied $b ($(du -m "$f" | cut -f1) MB)"
  fi
done
[ -f "$SRC/log.txt" ] && cp "$SRC/log.txt" "$OUT_DIR/source_log.txt"
echo "$n checkpoint files"
[ "$n" -gt 0 ]
