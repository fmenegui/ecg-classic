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
| Asks for at least 46 GB of RAM (`MIN_RAM_GB`) | The row network was killed for memory twice (exit 137) while building real strips on 8-vCPU hosts. |
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

### Faster steps: `--reuse` (page and dewarp networks)

Measured on an RTX 4000 Ada (`hybrid/gpu/profile_gen.py`, run 44543053782):

| Network | Rendering | Network step |
|---|---|---|
| Page | 1.14 s per 4 pages | 0.27 s |
| Dewarp | 1.28 s per 6 photos | 0.13 s |
| Row (strips rendered on the GPU) | 0.51 s per 13 strips | |

So the GPU spends 80-90% of a step rendering. With `--reuse N`, the generator keeps the last renders in a CPU buffer and draws each page about N times, each time with a new placement, crop and degradation. `--reuse 4` should make the page network about 2.5 times faster, and `--reuse 6` the dewarp network about 3.5 times faster.

The row network has `--reuse` already. Its bottleneck is the CPU strip generator: the GPU sits idle about half the time. A larger `--render-frac`, which renders strips on the GPU, shifts that work to the GPU.

### Automatic development benchmark: `eval_models`

After a training run, `eval_models="page=THIS row=<run>"` starts `pod-eval` with the development set (300 PM ECGs, about 440 Kaggle images). `THIS` stands for the run itself. `eval_base=<pod-eval run>` pairs the result with a baseline. The paired table goes into that run's summary.

### Not done, on purpose

- **No image of our own for the pods.** The setup takes about 3 min on most hosts; the slow ones were slow at pulling any image. The image would hold the private renderer, so it would need a private registry and an access token on RunPod.
- **No render cache shared between runs.** Cheap cards are often unavailable in the volume's data center, and training must not wait for them. Reuse inside a run takes most of the gain.

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
