#!/bin/bash
# The live log of the pod of a running pod-eval / pod-job / volume-fetch run:
#   bash hybrid/gpu/podlog.sh RUN_ID [lines]
# Reads the pod id from the run's podid-* artifact, then the pod's log through the RunPod proxy.
set -eu
RUN=$1; N=${2:-40}
d=$(mktemp -d)
gh run download "$RUN" -p 'podid-*' -D "$d" > /dev/null 2>&1 || { echo "no pod id yet for run $RUN"; exit 1; }
for f in $(find "$d" -name pod_id.txt); do
  p=$(cat "$f"); echo "== pod $p ($(basename "$(dirname "$f")"))"
  curl -sS -m 30 "https://$p-8000.proxy.runpod.net/log.txt" | tail -n "$N" || echo "pod not answering"
done
rm -rf "$d"
