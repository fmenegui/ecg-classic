# Compute infrastructure

Two places run the work: GitHub Actions (free; CI, small jobs, the original 20-shard evaluations) and RunPod.

## Training: `gpu-train.yml` → one RunPod GPU pod

The workflow starts a GPU pod (`hybrid/gpu/runpod_ctl.py`). The pod is a cheap card (A5000, 3090, A4500, 4000 Ada, A4000, L4) at no more than 0.5 USD/h.

The pod runs `hybrid/gpu/pod.sh`, which clones the code at the ref, installs, checks CUDA and runs the command. The workflow follows the log and brings the result back as an artifact. It always terminates the pod at the end, even on error or cancellation.

Safeguards added after losses:

| Safeguard | Why |
|---|---|
| Refuses to start a pod the RunPod balance cannot pay for (hours × 0.5 USD/h) | A zero balance stopped two runs after 3-4 h. |
| Copies the best checkpoints (`*_best.onnx/.pt`, `val.json`) to the runner every 30 min | |
| Gives up on a pod silent for 20 min and keeps the last checkpoints | |
| Retries the clone; replaces a pod whose CUDA or network to GitHub is broken | |
| Never lists pods | A pod listing carries the pod's environment, which holds the Kaggle token of runs with `kaggle=true`. |

GitHub stops a job after 6 h. A training run should end within about 5 h 30 min of training, including setup.

### Longer runs: `init_run`

A training longer than one job, or one whose pod was lost, continues in a new run:

- set `init_run=<earlier run>`;
- put `--init /workspace/in/<checkpoint>` (or `$IN_DIR/<checkpoint>`) in the job.

The runner downloads the earlier run's `*_best.pt` and sends it with a one-time upload token. The pod waits for it before starting the job.

### Data volume and caches: `use_volume`, on by default

The pod is first sought in the data center of the data volume, among cheap secure cards there. If none is free, it runs anywhere as before.

On the volume, the job finds `$CACHE`:

- `ptbxl_bank.py --cache $CACHE` reuses a bank built with the same builder, exclusions and folds. Building one took up to 1 h.
- `realstrips_build.py --cache $CACHE` reuses a finished set of real Kaggle strips from the same code and settings. These strips are kept only on the volume.

Jobs write to `$OUT_DIR`, the per-run folder the pod serves, not to `/workspace/out`.

## Evaluation: `pod-eval.yml` → RunPod CPU pods on a network volume

The datasets live on the network volume `ecgdig-data`: 60 GB in EUR-IS-1, about 4 USD/month.

- `/workspace/data/pm` holds PM-ECG-ID from Zenodo.
- `/workspace/data/kaggle` holds the Kaggle training records.
- Each is downloaded once, only what is missing.
- Neither ever leaves the volume.

Each run:

1. Fetches the models from their training runs: `models: "row=RUN page=RUN dewarp=RUN ink=RUN"`. A checkpoint-only row or ink network is converted to ONNX.
2. Starts one CPU pod per shard (`pods`, 1-4). Each pod is a cpu3c/cpu5c with 32 vCPU, about 1 USD/h (cap 1.2), mounting the volume.
3. Sends the models with a one-time upload token. The pod gets no GitHub token.
4. Runs `benchmarks/podeval.sh`:
   - PM: `eval_pm.py` on `PER_SCENARIO` ECGs per scenario.
   - Kaggle: `run.py eval` on part 0 of `kaggle_nparts` parts of the training records.
   - The work files stay in the run's folder on the volume. The read signals stored next to the ground truth are deleted.
5. Returns only `pm_results.json`, `pm_report.md`, `kaggle_results.json`, `kaggle_report.md` and the log. It always terminates the pod.

Shapes of run:

| Use | Inputs | Size | Cost |
|---|---|---|---|
| Development benchmark | `per_scenario=5`, `kaggle_nparts=20` | 300 PM ECGs, ~49 Kaggle records (~440 images) | ~15-20 min, ~0.3 USD |
| Whole databases | `per_scenario=0`, `kaggle_nparts=1`, `pods=3` | 6,000 PM ECGs, 977 Kaggle records | ~1.5-2 h per pod |
| Prefill | `data_only=true` | Downloads what is missing, evaluates nothing | |

Smoke test (run 44532635331): 60 PM ECGs plus 10 Kaggle records, end to end in about 8 minutes on cpu3c × 32 (0.96 USD/h).

The free 20-shard workflows (`pmcardio.yml`, `kaggle-full.yml`) stay available. They are slower: queue, 6 h per job, a download per shard.

The paired comparison scripts read the pod results like the shard results: `pmpair.py` takes the PM `results.json` and `kpair.py` takes the Kaggle results.
