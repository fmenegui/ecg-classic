#!/bin/bash
# Runs inside a RunPod GPU pod (started by .github/workflows/gpu-train.yml).
# Clones the repository at $REF, runs the training command for at most $MAX_HOURS and serves
# /workspace/out over HTTP (port 8000) so the workflow can follow the log and fetch the result.
# The pod receives no secret: the repository is public and the results leave through the HTTP port.
set -u
# OUT_DIR: per-run folder when /workspace is a shared network volume (evaluation pods)
OUT=${OUT_DIR:-/workspace/out}
export OUT_DIR=$OUT
# the code: on the container disk when /workspace is a volume shared by pods of other runs
REPO=${REPO_DIR:-/workspace/repo}
mkdir -p "$OUT"
cd /workspace
# GET of $OUT, and PUT /in/<name> with the run's one-time upload token (models for evaluations)
curl -fsSL "https://raw.githubusercontent.com/fmenegui/ecg-digitalization-classic/${REF}/hybrid/gpu/podserve.py" -o /podserve.py \
  && (python /podserve.py > /dev/null 2>&1 &) || (python -m http.server 8000 --directory "$OUT" > /dev/null 2>&1 &)
# a RunPod container restart (host hiccup, out of memory) runs this script again: keep the
# code and the checkpoints, and let train_gpu.py resume from $OUT/unet.pt
if [ -f "$OUT/status.txt" ]; then echo "POD RESTART detected $(date -u) (was: $(cat "$OUT/status.txt"))" >> "$OUT/log.txt"; fi
echo "setup" > "$OUT/status.txt"
{
  echo "pod start $(date -u)"
  [ -n "${CPU_ONLY:-}" ] && echo "CPU pod: $(nproc) vCPU, $(free -g | awk '/Mem/ {print $2}') GB" || nvidia-smi || true
  # a pod whose DNS or network to GitHub is broken cannot get the code: retry, then let the
  # workflow replace the pod (same path as a broken GPU)
  for i in 1 2 3 4 5 6; do
    [ -d "$REPO" ] || git clone --depth 1 -b "$REF" https://github.com/fmenegui/ecg-digitalization-classic "$REPO" && break
    sleep 20
  done
  if [ ! -d "$REPO" ]; then echo "nogpu" > "$OUT/status.txt"; echo "cannot clone the repository on this pod"; sleep infinity; fi
  cd "$REPO"
  git log --oneline -1
  if [ -n "${CPU_ONLY:-}" ]; then
    # evaluation pod: the package and onnxruntime only, no GPU, no renderer
    pip install -q -e . onnxruntime pandas remotezip requests kaggle==2.2.5 kagglesdk==0.1.38 six==1.17.0
    echo "running" > "$OUT/status.txt"
    echo "command: $JOB"
    timeout "$(python -c "print(int(float('$MAX_HOURS') * 3600))")" bash -c "$JOB"
    echo "exit code $?"
  else
  pip install -q -e . onnx onnxruntime segmentation-models-pytorch==0.3.4 timm wfdb
  # GPU strip renderer (hybrid/rendergen.py): public install, or a read-only token for the private repo
  pip install -q "git+https://${RENDERER_TOKEN:+x-access-token:${RENDERER_TOKEN}@}github.com/fmenegui/ecg-torch-renderer" \
    && echo "ecg-torch-renderer installed" || echo "ecg-torch-renderer not installed (CPU strips only)"
  # the renderer needs torch >= 2.6 and may upgrade it: torchvision (used by timm) must follow
  pip install -q -U torchvision && python -c "import torch, torchvision; print('torch', torch.__version__, 'torchvision', torchvision.__version__)"
  # some hosts expose the GPU to nvidia-smi but fail CUDA initialization ("CUDA unknown error");
  # training would then silently run on the CPU, so stop here and let the workflow try another pod
  if ! python -c "import torch; assert torch.cuda.is_available(); print('cuda ok', torch.cuda.get_device_name(0), (torch.ones(1, device='cuda') * 2).item())"; then
    echo "nogpu" > "$OUT/status.txt"; echo "CUDA not usable on this pod"; sleep infinity
  fi
  echo "running" > "$OUT/status.txt"
  echo "command: $JOB"
  # GPU use every 2 minutes in the log (utilization %, memory): shows whether the GPU is starved
  (while sleep 120; do echo "GPU $(nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader)"; done &)
  timeout "$(python -c "print(int(float('$MAX_HOURS') * 3600))")" bash -c "$JOB"
  echo "exit code $?"
  fi
} >> "$OUT/log.txt" 2>&1
cd /workspace
tar czf "$OUT.result.tar.gz" -C "$OUT" --exclude=result.tar.gz . && mv "$OUT.result.tar.gz" "$OUT/result.tar.gz"
echo "done" > "$OUT/status.txt"
sleep infinity
