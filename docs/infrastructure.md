# Compute infrastructure

How the experiments run, what they cost, and what protects them. Every workflow named here is in
`.github/workflows/`; every script in `hybrid/gpu/` or `benchmarks/`.

## 1. The rule: pay only for GPUs

| Work | Where | Cost |
|---|---|---|
| Training (GPU) | RunPod, one pod per run (`gpu-train.yml`) | 0.2-0.3 USD/h (cap 0.5) |
| Development benchmark (300 PM ECGs, ~440 Kaggle images) | GitHub Actions, 20 free runners per database | free |
| Whole-database evaluation (6,000 PM ECGs; 8,793 Kaggle images) | GitHub Actions, 20 free runners per database | free |
| Fast evaluation, when waiting is costly | RunPod CPU pods (`pod-eval.yml`) | ~1 USD/h per 32-vCPU pod |
| Recovering checkpoints from the data volume | RunPod, 2-vCPU pod (`volume-fetch.yml`) | ~0.07 USD/h, minutes |
| Data volume `ecgdig-data` (60 GB, EUR-IS-1) | RunPod | ~4 USD/month |

The repository is public, so GitHub Actions runners cost nothing. RunPod is used where a GPU is
needed, or where the free runners are too slow for the question at hand.

Typical spend: a 5 h training on an RTX 4000 Ada or 3090 is 1.1-1.5 USD; its development benchmark
is free.

## 2. Workflows

### `gpu-train.yml`: a training run on one GPU pod

| Input | Meaning |
|---|---|
| `job` | The command run on the pod (see section 3 for examples). |
| `max_hours` | Hard limit of the job on the pod; also sizes the balance check. Keep it at 5.8 or below. |
| `gpu` | Optional card name. Empty = the cheapest card free among A5000, 3090, A4500, 4000 Ada, A4000, L4, at no more than 0.5 USD/h. |
| `init_run` | Continue from the `*_best.pt` of an earlier run (sent to `$IN_DIR` before the job starts). |
| `eval_models` | After a successful training, start the free development benchmark with these models, e.g. `page=THIS row=44206699093` (`THIS` = this run). |
| `kaggle` | Give the pod the Kaggle token (only for jobs that read real Kaggle pages). |
| `use_volume` | Look for the pod in the data volume's data center first. **Off by default** (section 6, incident of Nov 19). |
| `tag` | Name of the artifact `gpu-<tag>-<run>`. |

What a run does:

1. `runpod_ctl.py` checks that the RunPod balance covers `max_hours` × 0.5 USD/h, then starts the cheapest free card with at least 46 GB RAM (`MIN_RAM_GB`).
2. The pod runs `pod.sh`: clone at the ref, install, check CUDA, start the HTTP server (`podserve.py`), wait for the init checkpoints if any, run `job` under `timeout`.
3. The runner follows the log, copies the best checkpoints every 30 min, fetches the result archive at the end, uploads it as an artifact and terminates the pod (always, even on failure or cancellation).
4. With `eval_models`, it starts `pmcardio.yml` (5 ECGs per scenario) and `kaggle-full.yml` (`stride=20`) with tag `dev-<tag>`.

The pod receives no GitHub token and no RunPod key. Results leave only through its HTTP port, which
serves its output folder (`$OUT_DIR`); uploads to it need the run's one-time token.

### `pmcardio.yml` and `kaggle-full.yml`: evaluation on free runners

Both shard the database over 20 runners and merge the shards.

| Workflow | Subset input | Development benchmark | Whole database |
|---|---|---|---|
| `pmcardio.yml` | `per_scenario` (ECGs per scenario, 0 = all) | `per_scenario=5`: 300 ECGs | `per_scenario=0`: 6,000 ECGs |
| `kaggle-full.yml` | `stride` (every k-th record of each shard) | `stride=20`: ~49 records, ~440 images | `stride=1`: 977 records, 8,793 images |

Models come from training runs: `row_model_run`, `page_model_run`, `dewarp_model_run`, `ink_model_run`
(and `grid_model_run` for PM). A checkpoint-only network is converted to ONNX in the workflow.

Compare two evaluations with both metrics and the coverage:

```
python benchmarks/compare.py pm     BASE_RESULTS NEW_RESULTS
python benchmarks/compare.py kaggle BASE_RESULTS NEW_RESULTS
```

(`BASE`/`NEW`: `results.json` files, folders or globs of downloaded artifacts.) The PM table includes
the 60-scenario mean next to PMcardio's 14.48 dB; the Kaggle table splits the holdout
(id % 5 == 0 plus the 15 local records, never trained on) from the rest.

### `experiment.yml` and the development leaderboard (the Kaggle-style loop)

Every candidate is scored on the same two development sets, under one tag starting with `dev-`:

```
gh workflow run experiment.yml -f tag=dev-row3e-page3 -f models="row=44653806160 page=44560948929" [-f config="k=v ..."]
```

- PM-ECG-ID: 300 ECGs (5 per scenario, seed 0) on a RunPod CPU pod, ~40 min (~0.7 USD); longer with the page network (`pm_hours`).
- Kaggle: 60 records, 540 images (`kaggle-full.yml stride=20`), free, ~25 min, with the error decomposition per type.

`python benchmarks/leaderboard.py` collects every `dev*` result, compares them on the images all of them have and writes `docs/leaderboard.md` (PMcardio-style SNR, failed leads, 60-scenario mean; official Kaggle metric on all images, the holdout and per type). One line per tag in `docs/experiments.csv` says what the experiment is. A candidate that wins on the development sets is then confirmed on the whole databases (`pmcardio.yml per_scenario=0`, `kaggle-full.yml stride=1`) before it becomes the default.

### `pod-eval.yml`: evaluation on RunPod CPU pods

Same evaluations on 32-vCPU pods (`pods` = 1-4 in parallel). Use it only when a result is needed
faster than the free runners give it.

| Input | Meaning |
|---|---|
| `models` | `"row=RUN page=RUN dewarp=RUN ink=RUN"` |
| `what` | `pm`, `kaggle` or `both` |
| `per_scenario`, `kaggle_nparts` | Subsets, as above (`kaggle_nparts=20` = part 0 of 20) |
| `volume` | Use the data volume. Off by default: the data is downloaded on the pod. |
| `data_only` | Only download to the volume (4 vCPU) |
| `base_run` | A previous `pod-eval` run to pair with |

The pod returns only `pm_results.json`, `pm_report.md`, `kaggle_results.json`, `kaggle_report.md`
and its log.

### `pod-job.yml`: a command next to the data

Runs `job` on one CPU pod with the data volume at `/workspace` (`$CACHE` = `/workspace/cache`); only the log comes back. Used to build the real Kaggle strips once for all training runs:

```
gh workflow run pod-job.yml -f kaggle=true -f vcpu=32 -f job='OMP_NUM_THREADS=1 python hybrid/realstrips_build.py --out /workspace/realstrips --cache $CACHE --max-pages 7000 --workers 30 --chunk 400'
```

A training run that passes the same `--max-pages --chunk --seed` with `--cache $CACHE` reuses that set (the cache key also covers the code of the builder).

### `dc-probe.yml`: which data centers work

One 2-vCPU pod per data center: does it answer through the RunPod proxy, does the Kaggle archive answer from there. On Nov 20: CPU pods exist in EU-RO-1, EU-CZ-1, US-KS-2 and EUR-IS-1; all four answered. The Kaggle archive answers 404 to pods on the `python:3.11-bookworm` image everywhere and 302 to pods on the RunPod PyTorch image, so every pod uses the PyTorch image.

### `volume-fetch.yml`: checkpoints left on the data volume

A training pod on the volume writes to `/workspace/runs/<run>/out`. If its job ended without an
artifact, `run_id=<run>` copies the checkpoints back with a 2-vCPU pod into an artifact
`gpu-<tag>-<fetch run>`; `init_run=<fetch run>` then continues from it. Only `*_best.pt`,
`*_best.onnx`, `val.json` and the log leave the volume. It needs the volume's data center to answer
(section 6).

## 3. Training jobs

Each job builds the PTB-XL signal bank first (PM-ECG-ID source ECGs and Kaggle evaluation records
excluded; `ptbxl_bank.py` refuses to run without the exclusion list), then trains.

```
# page-structure network, continued from an earlier run
pip install -q pandas && python hybrid/ptbxl/ptbxl_bank.py --out /root/bank.npy --work /root/ptbxl --cache $CACHE \
  && ECG_BANK=/root/bank.npy python hybrid/train_page.py --minutes 285 --pages 4 --reuse 4 --lr 3e-4 \
     --init $IN_DIR/page_best.pt --out $OUT_DIR

# dewarp network
... python hybrid/train_dewarp.py --minutes 280 --pages 6 --reuse 6 --lr 3e-4 --init $IN_DIR/dewarp_best.pt --out $OUT_DIR

# row network with real Kaggle strips (kaggle=true)
pip install -q remotezip pandas \
  && (OMP_NUM_THREADS=1 nohup python hybrid/realstrips_build.py --out /root/real --cache $CACHE --max-pages 4000 \
      --workers 3 > $OUT_DIR/realstrips.log 2>&1 &) \
  && python hybrid/ptbxl/ptbxl_bank.py --out /root/bank.npy --work /root/ptbxl --cache $CACHE \
  && ECG_BANK=/root/bank.npy python hybrid/train_gpu.py --gen rowgen --render-frac 0.5 --render-photo 0.7 \
     --real-dir /root/real --real-frac 0.35 --heavy 0.3 --size 512 --batch 32 --reuse 3 --val 256 \
     --minutes 270 --init $IN_DIR/unet_best.pt --out $OUT_DIR
```

Rules for jobs:

- **Write to `$OUT_DIR`, read init files from `$IN_DIR`, cache to `$CACHE`.** Keep big temporary files (the bank) on the container disk (`/root`): `/workspace` may be a volume shared with other runs.
- **Budget the time.** Setup (bank, install) takes 5-20 min; GitHub ends the job at 355 min and the follower stops at 340. So `--minutes` ≤ 285 for a 5.8 h run.
- **Use `--reuse`** for page and dewarp. Measured on an RTX 4000 Ada (`profile_gen.py`): rendering takes 80-90% of a step (page 1.14 s per 4 pages against a 0.27 s network step; dewarp 1.28 s per 6 photos against 0.13 s). Page v3 with `--reuse 4` ran 38,190 steps in 290 min, against 18,593 for page v2 without it: twice the training for the same cost.
- **Continue instead of restarting.** `init_run` + `--init $IN_DIR/<name>_best.pt` and a lower `--lr` (3e-4).
- **Validation decides the checkpoint kept**, so it must measure what ecgdig uses: page rows go through ecgdig's length and edge filters (`rows`; `rows_raw` before them); dewarp keeps the checkpoint with the lowest local error after the best affine map per photo (`err_local_median`), since ecgdig's own rectification removes global shift, scale and shear.

## 4. Data and secrets

- **RunPod key**: only in the GitHub secret `RUNPOD_API_KEY`, used by the runner. Pods never get it.
- **Kaggle token**: in the secret `KAGGLE_API_TOKEN`; reaches a pod only with `kaggle=true` (training on real pages) or a Kaggle evaluation.
- **Never print a pod listing or a pod's environment.** A listing returns each pod's environment, which holds the Kaggle token (a probe once printed it to a public log; the run was deleted at once). The tools print only ids, prices and status lines; `runpod_v2.py stop` looks up its run's pods by name in memory and prints nothing else.
- **Kaggle competition data is never published**: not in the repository, not in artifacts. Evaluation artifacts hold only `results.json` and `report.md`; the read signals written next to the ground truth (`sig/`) are deleted on the pod. Weights trained on Kaggle pages stay in run artifacts, never in releases.
- **PM-ECG-ID and LITFL are never used for training.** The PM source ECGs are excluded from the PTB-XL bank (by id, and by lead-II match against `eval_pm_lead2.npz`).
- **Kaggle split by record**: the holdout (id % 5 == 0, plus the 15 local records) never enters training.

## 5. Safeguards, and the loss that made each one

| Safeguard | Why |
|---|---|
| Refuse a GPU pod the balance cannot pay for (`max_hours` × 0.5 USD/h) | At 0 USD RunPod stops every pod: two runs were lost at 3-4 h. |
| Refuse a CPU pod unless the balance covers its hours plus a 2 USD reserve | Same, and other pods may be running. |
| Copy the best checkpoints to the runner every 30 min, and once more before giving up | A pod lost late in a run no longer takes the run with it. |
| Hard limit on every whole transfer from the pod | A response trickling through the proxy hung the follower for 5 h. |
| Give up on a pod unreachable for 20 min, or whose log has not grown for 25 min | `pod.sh` logs GPU use (training) or load and free space (CPU pods) every 2 min, so a silent log means a dead job. |
| Stop following at 340 min of the job | GitHub cancels at 355 min, and a cancelled job uploaded nothing (row network, run 44546313205). |
| Fail the run when the job on the pod exits non-zero | A crashed training was reported as a success and started its benchmark. |
| Upload to pods in 16 MB chunks | 98 MB uploads failed with HTTP 502. |
| Retry the clone; replace a pod whose CUDA or network is broken | Some hosts had broken DNS or GPUs. |
| At least 46 GB RAM on training pods | The row network was killed for memory (exit 137) on small hosts. |
| Small Python image for CPU pods, container disk within the per-vCPU cap | CPU pods do not need the 8 GB CUDA image; RunPod refuses 40 GB on 2 vCPU. |
| A prefill pod (only downloads) gets 4 vCPU | A 32-vCPU pod downloaded for 1.5 h at 1.13 USD/h. |
| Pod id as a run annotation | A running pod's log can be read from outside (`https://<pod>-8000.proxy.runpod.net/log.txt`). |
| Atomic, per-writer temporary files for downloads; truncated PNGs fetched again | Cancelled downloads and pods sharing the volume. |

## 6. Incidents

**Nov 19, from ~15:20 UTC: the data volume's data center (EUR-IS-1) stopped answering through the
RunPod HTTP proxy.** Every pod there gave 404, 502 or 520 for hours: the row network's log stopped at
15:19 (its 5 h of training are on the volume), two evaluation pods and a 2-vCPU fetch pod never
answered, and dewarp v2 could not receive its init checkpoint. A pod elsewhere (page v3) worked
throughout. Since then `use_volume` (training) and `volume` (evaluation) are off by default: pods run
in any other data center and fetch their data themselves. About 3-4 USD went to pods that never
answered. Turn the volume back on once a `volume-fetch` run succeeds.

Lessons kept in the tools: read the run annotations for the pod id and look at its log early; a pod
that does not answer within 15-20 min will not, and should be cancelled.

## 7. Costs to expect

| Item | Cost |
|---|---|
| Training, 5 h, RTX 4000 Ada / 3090 / A5000 | 1.1-1.5 USD |
| Development benchmark (free runners) | 0 USD; expected ~20-40 min (queue plus setup dominate) |
| Whole-database evaluation (free runners) | 0 USD; Kaggle took 4 h 40 min (run 44526678562) |
| Same on RunPod (`pod-eval`, 3 pods × 32 vCPU) | ~5 USD, ~2 h |
| Data volume | ~4 USD/month |

Before a batch of runs: the balance must cover the GPU runs at 0.5 USD/h × `max_hours` each, plus
2 USD of reserve.

## 8. Not done, on purpose

- **No pod image of our own.** Setup takes about 3 min on most hosts. The image would hold the private renderer, so it would need a private registry and a token on RunPod.
- **No render cache shared between runs.** `--reuse` inside a run takes most of the gain.
- **The whole Kaggle set is not on the volume.** A single download stream did not fetch the 977 records in 4.5 h; `kaggle_zip.py --threads` now reads in parallel, but the free 20-runner workflow already evaluates the whole set at no cost.
