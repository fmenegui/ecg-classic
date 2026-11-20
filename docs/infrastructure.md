# Compute infrastructure: how it is built, run, rebuilt and kept cheap

Everything that runs outside a laptop: GitHub Actions (free, the orchestrator and most evaluations)
and RunPod (GPUs for training; CPU pods for fast evaluations and jobs next to the data). Every
workflow named here is in `.github/workflows/`; every script in `hybrid/gpu/`, `benchmarks/` or
`hybrid/`.

Related documents:

- [`runpod-troubleshooting.md`](runpod-troubleshooting.md): every problem met on RunPod, how to
  recognise it, what it cost, how it is prevented.
- [`experiments-log.md`](experiments-log.md): every training and evaluation run, with card, hours
  and cost.
- [`lessons.md`](lessons.md): what went right and wrong (method, evaluation, infrastructure).
- [`roadmap.md`](roadmap.md): what comes next.

---

## 1. Principles

These rules are enforced three ways:

- **`CLAUDE.md`** is read by every Claude Code session in this repository.
- **`.claude/hooks/guard.py`** is a PreToolUse hook that blocks the costly commands before they
  run:
  - an expensive GPU;
  - `max_hours` > 5.8 or `--minutes` > 290;
  - more than two 32-vCPU pods;
  - calling the RunPod API from the session;
  - printing secrets;
  - competition data in `git add`;
  - `gpu-stop-all`.

  An explicit `ALLOW_<RULE>` in the command passes one rule when the owner decides it.
- **`tests/test_infra_rules.py`** runs on every push (`tests.yml`). It checks that:
  - the default card list is cheap;
  - the balance guards and follower limits exist;
  - CPU pods use the PyTorch image;
  - every workflow that starts a pod terminates it in an always-run step and bounds its follower;
  - no artifact path can carry data;
  - no tool prints a pod listing.

1. **Pay for GPUs, not for waiting.** Training runs on RunPod GPUs (0.2-0.3 USD/h cards). The
   free GitHub runners evaluate. A paid CPU pod is used only when its speed is worth the money.
2. **Nothing runs blind.** Every pod writes a log that can be read while it runs
   (`hybrid/gpu/podlog.sh RUN_ID`), logs a heartbeat every 2 minutes, and is given up when it stops
   answering.
3. **Every pod is terminated by the run that started it**, on success, failure or cancellation.
4. **Money is checked before it is spent.** No pod starts if the RunPod balance cannot pay for it.
5. **Secrets stay on GitHub.** Pods get the Kaggle token only when the job reads Kaggle data, and
   nothing else.
6. **Competition data never leaves the pods and runners.** Artifacts hold results and weights only.
7. **Every experiment is scored the same way** (section 6), so results are comparable.

---

## 2. Architecture

```
                 GitHub Actions (free)                                   RunPod (paid)
 ┌──────────────────────────────────────────────┐        ┌──────────────────────────────────────────┐
 │ gpu-train.yml ── runpod_ctl.py (GraphQL) ────┼──────► │ GPU pod  runpod/pytorch image             │
 │   follows log, copies checkpoints, uploads   │ HTTP   │   pod.sh: clone, install, CUDA check,     │
 │   artifact gpu-<tag>-<run>, terminates pod   │ proxy  │   podserve.py :8000, job under timeout     │
 │                                              │◄───────┤   GET /log.txt /status.txt /result.tar.gz  │
 │ pod-eval.yml / pod-job.yml / volume-fetch    │        │   PUT /in/<name> (one-time token)          │
 │   runpod_v2.py (REST v2) ───────────────────►│        │ CPU pod (cpu3c/cpu5c, 2-32 vCPU)           │
 │                                              │        │   same pod.sh (CPU_ONLY), same server      │
 │ pmcardio.yml / kaggle-full.yml: 20 free      │        │ Network volume ecgdig-data2 (EU-RO-1)      │
 │   runners each, merge, error decomposition   │        │   /workspace/data, /workspace/cache, runs  │
 │ experiment.yml: one tag, both dev sets       │        └──────────────────────────────────────────┘
 └──────────────────────────────────────────────┘
```

- **GPU pods** are started with RunPod's GraphQL API (`podFindAndDeployOnDemand`) by
  `hybrid/gpu/runpod_ctl.py`.
- **CPU pods and network volumes** use the REST API v2 (`https://api.runpod.io/v2`: `/pods`,
  `/network-volumes`, `/catalog/...`) through `hybrid/gpu/runpod_v2.py`.
- **Every pod runs `hybrid/gpu/pod.sh`**, fetched from the repository at the run's ref. It clones the
  code, installs it and starts `hybrid/gpu/podserve.py` on port 8000. On a GPU pod it then checks
  CUDA; it waits for uploaded inputs if asked and runs the job with a time limit. Finally it packs
  the output folder into `result.tar.gz` and writes `status.txt` = `done`.
- **The runner talks to the pod only through RunPod's HTTP proxy**,
  `https://<pod_id>-8000.proxy.runpod.net`. The pod receives no GitHub token. Files go to it with a
  per-run upload token, in 16 MB chunks.

---

## 3. Rebuilding everything from zero

Follow this order. Each step says how to check it worked.

### 3.1 Accounts and money

1. **RunPod account** with a positive balance. Before a batch of runs, the balance must cover
   every GPU run at 0.5 USD/h × `max_hours`, every CPU pod at its hourly price × hours, plus a
   2 USD reserve. At 0 USD RunPod stops every pod at once (section 7 of the troubleshooting guide).
2. **RunPod API key** with read and write access (it creates and deletes pods and volumes).
3. **Kaggle account** that has accepted the rules of the "PhysioNet - Digitization of ECG Images"
   competition, and an API token (new style, `KGAT_...`). Without the accepted rules the archive
   answers 403/404.

### 3.2 GitHub secrets and variables (repository settings)

| Name | Kind | Used by | Notes |
|---|---|---|---|
| `RUNPOD_API_KEY` | secret | every RunPod workflow | Never reaches a pod. |
| `KAGGLE_API_TOKEN` | secret | Kaggle evaluations; training with `kaggle=true` | Reaches a pod only when the job reads Kaggle data. |
| `KAGGLE_USERNAME` | variable | same | Not secret. |
| `RENDERER_TOKEN` | secret, optional | `gpu-train.yml` | Read-only token for ecg-torch-renderer. Not needed while that repository is public (it is, Nov 20). Leave it unset: when set it would also reach the pods. |

Check: run `kaggle-zip-probe.yml`. The log must show `attempt bearer -> HTTP 302` and
`range request: 206`.

### 3.3 Data center and volume

1. **Probe data centers.** Run `dc-probe.yml` (default list EU-RO-1, EU-CZ-1, US-KS-2, EUR-IS-1).
   Each 2-vCPU pod reports whether it answered through the proxy and whether the Kaggle archive
   answered from there. Cost: a few cents.
2. **Pick the volume's data center** among those that pass. Prefer one where cheap GPUs are also
   often free. The default list in `runpod_v2.py` (`DATA_CENTERS`) is `EU-RO-1,EU-CZ-1,US-KS-2`.
3. **The volume is created on first use.** Name `ecgdig-data2` (`VOLUME_NAME`), 60 GB
   (`VOLUME_GB`), about 4 USD/month. The first workflow that runs `runpod_v2.py volume` creates it.
   Check: that step prints `created volume <id> (60 GB) in <dc>`, then `volume <id> 60 GB in <dc>`
   on later runs.
4. **Fill the volume** (optional; evaluations without the volume download their data themselves):
   - PM-ECG-ID: `pod-eval.yml data_only=true what=pm` (4 vCPU; 1-2 h, limited by Zenodo's 429 answers).
   - Real Kaggle strips for training: `pod-job.yml` (section 5.4).

### 3.4 First runs

1. `pod-eval.yml what=pm per_scenario=1` (60 ECGs). This checks a CPU pod end to end; it takes
   about 10 minutes.
2. `gpu-train.yml` with a short job (`job='python hybrid/gpu/profile_gen.py'`). This checks a GPU
   pod, the renderer install and CUDA; it takes about 15 minutes and costs under 0.10 USD.
3. `experiment.yml tag=dev-base models="row=<a row network run>"`. The PM part takes about 40 min
   on a pod; the Kaggle part about 25 min on free runners. Then run
   `python benchmarks/leaderboard.py`.

---

## 4. Components

### 4.1 `hybrid/gpu/runpod_ctl.py` (GPU pods; GraphQL)

| Command | What it does |
|---|---|
| `run --out DIR` | balance check, start, follow, fetch result, exit with the job's exit code |
| `follow --out DIR` | follow an already started pod (`POD_ID`) |
| `stop` | terminate this run's pods (name `ecgdig-train-<run id>`) |

Behaviour and settings:

- **Card choice:** cheapest free card among A5000, 3090, A4500, 4000 Ada, A4000, L4, 2000 Ada,
  3080, at most `MAX_USD_H` = 0.5 USD/h. It tries the volume's data center first when
  `use_volume`, then any. A card outside the list is used only when named (`gpu` input).
- **Minimum RAM:** `MIN_RAM_GB` 46.
- **Waiting for a card:** retries every 3 min for `WAIT_MIN` = 45 min.
- **Broken pods:** a pod whose clone fails, or whose CUDA or GitHub network is broken, reports
  `nogpu` and is replaced.
- **Follower:**
  - polls `status.txt` and `log.txt` every 60 s;
  - every transfer has a hard limit (4 × its timeout);
  - copies `CKPT_FILES` (`*_best.pt`, `*_best.onnx`, `val.json`) every `CKPT_MIN` = 30 min;
  - gives up, after a last checkpoint copy, when:
    - the pod never answered in `BOOT_MIN` = 35 min;
    - it stopped answering for `LOST_MIN` = 20 min;
    - its log has not grown for `STALL_MIN` = 25 min;
    - `FOLLOW_UNTIL` is reached (the workflows set job start + 340 min).
- **Exit code:** the job's own exit code (read from the log) decides success. A failed job fails
  the run and starts no benchmark.

### 4.2 `hybrid/gpu/runpod_v2.py` (CPU pods and volumes; REST v2)

| Command | What it does |
|---|---|
| `volume` | find or create `VOLUME_NAME` in the first of `DATA_CENTERS` that accepts it; outputs `volume_id`, `data_center` |
| `cpu-start` | balance guard, then a CPU pod |
| `put FILE...` | upload to the pod's `/in/` (16 MB chunks, `PUT_TRIES` 120 × 30 s; every 10th failure printed; the pod log printed if it never succeeds) |
| `stop` | terminate this run's CPU pods (name `ecgdig-eval-<run>-<shard>`), printing nothing about them |

`cpu-start` in more detail:

- **Machine:** cpu5c/cpu3c (cpu5g/cpu3g as fallbacks) × `VCPU` (32 by default; 4 for a prefill;
  2 for fetches and probes). Container disk is min(40, 10 × vCPU) GB, because RunPod caps it per
  vCPU.
- **Image:** `CPU_IMAGE`, the RunPod PyTorch image. Not `python:3.11-bookworm`: the Kaggle archive
  answers 404 to pods on that image.
- **Placement:** with a volume, in its data center. Without one, in `CPU_DCS` minus `EXCLUDE_DCS`.
- **Price cap:** `MAX_USD_H` = 1.2; a dearer pod is deleted at once.
- **Waiting:** retries every 2 min for `WAIT_MIN` = 30 min. When the account is at its 64-vCPU
  limit, it waits in line up to `QUEUE_MIN` = 150 min.
- **Balance guard:** `MAX_HOURS` × price + `BALANCE_RESERVE` (2 USD) must be in the balance.

### 4.3 `hybrid/gpu/pod.sh` and `hybrid/gpu/podserve.py` (on the pod)

- **Folders:** `OUT_DIR`, the only folder served; `IN_DIR`, uploads; `CACHE`, persistent on the
  volume; `REPO_DIR`. With a volume they live under `/workspace/runs/<run>-<shard>/`, and the job
  should keep big temporary files on the container disk (`/root`).
- **GPU branch:** installs the package, onnx, onnxruntime, segmentation-models-pytorch 0.3.4,
  timm, wfdb and the renderer. It checks CUDA and logs GPU use every 2 min.
- **CPU branch (`CPU_ONLY`):** installs the package, onnxruntime, pandas, remotezip, requests and
  the kaggle CLI. It logs CPU load and free space on `/workspace` every 2 min.
- **Inputs:** `WAIT_IN=1` makes the job wait until a `READY` file is uploaded.
- **`podserve.py` endpoints:** GET serves `OUT_DIR`. PUT `/in/<name>` stores a file, or one chunk
  `<name>.cKKKofNNN`, joined atomically into `<name>` once all chunks are in; it needs the run's
  `X-Upload-Token`.

### 4.4 Workflows

| Workflow | What | Where | Typical cost |
|---|---|---|---|
| `gpu-train.yml` | one training job on a GPU pod; optional chaining (`init_run`) and automatic development benchmark (`eval_models`) | GPU pod | 1.1-1.5 USD per 5 h |
| `experiment.yml` | one experiment, both development sets, one tag | starts `pod-eval` (PM) and `kaggle-full` (Kaggle) | ~0.7 USD (PM pod), Kaggle free |
| `pmcardio.yml` | PM-ECG-ID on 20 free runners (`per_scenario`: 5 = dev set, 0 = all) | GitHub | free |
| `kaggle-full.yml` | Kaggle training records on 20 free runners (`stride`: 20 = dev set, 1 = all) + error decomposition | GitHub | free |
| `pod-eval.yml` | the same evaluations on 1-4 CPU pods (`what`, `per_scenario`, `kaggle_nparts`, `models`, `config`, `volume`, `data_only`, `max_hours`) | CPU pods | ~1 USD/h per pod |
| `pod-job.yml` | any command on a CPU pod next to the volume; only the log returns | CPU pod | 0.035 USD/h per vCPU |
| `volume-fetch.yml` | checkpoints of a run left on the volume → artifact usable as `init_run` | 2-vCPU pod | cents |
| `dc-probe.yml` | which data centers answer, and reach Kaggle | 2-vCPU pods | cents |
| `topsol-survey.yml` | public notebooks and code datasets of the top Kaggle solutions | GitHub | free |
| `gpu-stop-all.yml` | terminate every training pod of the account (emergency only: also kills parallel runs) | GitHub | — |
| `kaggle-zip-probe.yml` | Kaggle archive reachable from a runner | GitHub | free |

### 4.5 Watching a run

```
gh run view RUN_ID                      # steps; logs only when the job has ended
bash hybrid/gpu/podlog.sh RUN_ID 60     # live pod log (pod-eval, pod-job, volume-fetch, dc-probe)
```

`podlog.sh` reads the pod id from the run's `podid-*` artifact, which is uploaded right after the pod
starts. Run annotations would show it only when the job ends. `gpu-train.yml` starts and follows in
one step, so it has no live pod id yet; its log arrives when the run ends. Check a training's first
minutes with a short smoke run first.

---

## 5. Recipes

### 5.1 A training run

```
gh workflow run gpu-train.yml -f tag=page4 -f max_hours=5.6 -f init_run=44560948929 \
  -f eval_models="page=THIS row=44653806160" \
  -f job='pip install -q pandas && python hybrid/ptbxl/ptbxl_bank.py --out /root/bank.npy --work /root/ptbxl --cache $CACHE && ECG_BANK=/root/bank.npy python hybrid/train_page.py --minutes 285 --pages 4 --reuse 4 --lr 3e-4 --init $IN_DIR/page_best.pt --out $OUT_DIR'
```

Rules:

- **Time budget.** Setup (install, bank) takes 5-20 min. GitHub ends the job at 355 min and the
  follower stops at 340. So `--minutes` ≤ 285 for `max_hours` 5.6-5.8.
- **Write to `$OUT_DIR`.** Read init files from `$IN_DIR`. Cache to `$CACHE`. Big temporaries go
  on `/root`.
- **`--reuse`** for the page (4) and dewarp (6) networks: rendering is 80-90% of a step.
- **Continue instead of restarting.** Use `init_run` + `--init`, with a lower `--lr`.
- **Smoke first.** Run 10-20 minutes with the same job and the same inputs before a 5 h run, when
  anything in the job changed.

### 5.2 An experiment (scored on the development sets)

```
gh workflow run experiment.yml -f tag=dev3-row3e-page3 -f models="row=44653806160 page=44560948929" [-f config="R=20.0"] [-f pm_hours=3]
echo 'dev3-row3e-page3,rowheavy3 + page v3' >> docs/experiments.csv
python benchmarks/leaderboard.py           # rewrites docs/leaderboard.md
```

### 5.3 Whole-database confirmation (before a model becomes the default)

```
gh workflow run pmcardio.yml -f per_scenario=0 -f tag=full-<name> -f row_model_run=... [-f page_model_run=...]
gh workflow run kaggle-full.yml -f stride=1 -f tag=full-<name> -f row_model_run=...
python benchmarks/compare.py pm   BASE NEW
python benchmarks/compare.py kaggle BASE NEW
```

### 5.4 Real Kaggle strips for the row network (built once, kept on the volume)

```
gh workflow run pod-job.yml -f kaggle=true -f vcpu=32 -f max_hours=3 -f tag=realstrips-full \
  -f job='OMP_NUM_THREADS=1 python hybrid/realstrips_build.py --out /workspace/realstrips --cache $CACHE --max-pages 7000 --workers 30 --readers 12 --chunk 400'
```

A training run reuses that set when it passes the same `--max-pages --chunk --seed` with
`--cache $CACHE`, and runs on the volume (`use_volume=true`). The cache key also covers the
builder's code, so a change to `realstrips.py` or `realstrips_build.py` builds a new set.

### 5.5 A top Kaggle solution on our images

```
gh workflow run gpu-train.yml -f tag=topsol2 -f max_hours=3 -f kaggle=true \
  -f job='SOL=takashisomeya/physionet-2nd-place-submission bash benchmarks/topsol/run_hengck.sh'
```

Options:

- `TS_LIMIT=18`: smoke test.
- `TS_PM_FORMATS=3x4+1R TS_PM_PER=0`: every PM-ECG-ID image printed like the Kaggle sheet.

The solution's readings are scored by our evaluators (`ECGDIG_EXTERNAL`), with both metrics.

### 5.6 Live log, stuck runs, emergencies

- **Live log:** `bash hybrid/gpu/podlog.sh RUN_ID`.
- **Cancelling:** `gh run cancel RUN_ID`. The run's always-run step terminates its pod. Check the
  log for `terminated <pod>` or a 404, which means the pod was already gone.
- **Emergency:** `gpu-stop-all.yml` kills every training pod of the account, including parallel
  runs.
- **Never** list pods or print pod environments to diagnose (section 8).

---

## 6. The development leaderboard (Kaggle-style loop)

Each candidate is scored on two fixed development sets:

| Set | Content | Runs on | Time |
|---|---|---|---|
| PM-ECG-ID dev | 300 ECGs: 5 per scenario, seed 0 (5% of 6,000) | RunPod CPU pod, 32 vCPU | ~40 min (longer with page or dewarp networks, which read pages twice) |
| Kaggle dev | every 20th training record of each of the 20 shards: 60 records, 540 images (6.1% of 8,793) | free runners | ~25 min |

`benchmarks/leaderboard.py` writes `docs/leaderboard.md`.

- **Rows:** it compares all experiments on the images they all have and reports the coverage.
- **PM columns:** PMcardio-style SNR, failed leads, 60-scenario mean.
- **Kaggle columns:** official metric on all images, the holdout and each image type.
- **References:** PMcardio's 14.48 dB (60-subset mean of its paper) and the Kaggle leaders'
  22.6-23.3 dB (private leaderboard: other images).

A winner is confirmed on the whole databases before it becomes the default (section 5.3).

The Kaggle dev set includes holdout records (id % 5 == 0 and 15 local ones) that no network was
trained on. Report the holdout column whenever a network trained on Kaggle pages is involved.

---

## 7. Costs

### 7.1 Prices seen (Nov 16-20, 2026)

| Machine | USD/h | Notes |
|---|---|---|
| RTX A4000 | 0.17-0.25 | slowest of the cheap cards |
| RTX A4500 | 0.19-0.25 | |
| RTX 3090 (community) | 0.22 | the price jumps to 0.5 on secure cloud |
| RTX A5000 | 0.16-0.27 | best cost per step in the Nov 16 comparison |
| RTX 4000 Ada | 0.28 | |
| RTX 4090 | 0.34-0.89 | not worth it: the GPU waited for CPU-generated data 23-37% of the time |
| L40S | 1.09 | never needed |
| CPU cpu3c × 32 | 0.96 | |
| CPU cpu5c × 32 | 1.12 | |
| CPU × 2 / × 4 | 0.06-0.07 / 0.14 | |
| Network volume, 60 GB | ~4 per month | |

### 7.2 What things cost

| Item | Cost |
|---|---|
| Training, 5 h on a cheap card | 1.1-1.5 USD (2.4 when only a secure-cloud 3090 at 0.5 USD/h is free) |
| Development experiment | ~0.7 USD (PM pod) + 0 (Kaggle) |
| Whole PM-ECG-ID + whole Kaggle on free runners | 0 USD; Kaggle ~4 h 40 min |
| Same on pods (3 × 32 vCPU) | ~5 USD, ~2 h |
| Real strips, 7,000 pages, 32 vCPU | ~1-2 USD |

### 7.3 Spend so far (estimated from run durations × hourly price; Nov 16-20)

| | Runs | Estimated USD | Of which lost to failures |
|---|---|---|---|
| GPU training and tests | 41 | ~33 | ~14 (balance at 0, cancelled 6 h runs, 4090/L40S trials, the EUR-IS-1 outage, runs lost at the 6 h limit) |
| CPU pods (evaluations, jobs, probes) | 24 | ~20 | ~14 (pods that never answered, evaluations killed by their time limit, an oversized prefill) |
| **Total** | 65 | **~53** | **~28** |

About half the money went to runs that produced nothing. Every one of those causes now has a
safeguard; the troubleshooting guide lists them one by one.

---

## 8. Data and secrets

- **RunPod key:** only in `RUNPOD_API_KEY`, used by the runner. Pods never get it.
- **Kaggle token:** in `KAGGLE_API_TOKEN`. It reaches a pod only with `kaggle=true` or a Kaggle
  evaluation, and is passed in the pod's environment.
- **Never print a pod listing or a pod's environment.** A listing returns every pod's environment,
  Kaggle token included. On Nov 19 a probe printed it to a public log; the run was deleted at once.
  The tools print ids, prices and status lines only.
- **One-time upload token per run.** It is masked in the workflow logs.
- **Kaggle competition data is never published.**
  - Not in the repository and not in artifacts.
  - Evaluation artifacts hold `results.json`, `report.md`, `decomposition.json` (numbers only) and
    logs.
  - The read signals saved next to the ground truth (`sig/`) are deleted on the pod or runner.
  - Real strips stay on the volume.
  - Weights trained on Kaggle pages stay in run artifacts (30-60 days), never in the public models
    repository. `fmenegui/ecgdig-models`' release workflow refuses them.
- **PM-ECG-ID and LITFL are never used for training.** The PM source ECGs are excluded from the
  PTB-XL bank by id and by lead-II match (`eval_pm_lead2.npz`); `ptbxl_bank.py` refuses to run
  without that file.
- **Kaggle split by record.**
  - The holdout (id % 5 == 0, plus the 15 local records) never enters training.
  - The three copies of the local list (`hybrid/realstrips_build.py`, `benchmarks/compare.py`,
    `benchmarks/physionet/merge_shards.py`) are checked equal by `tests/test_holdout_lists.py`.
  - Until Nov 20 the builder's copy stopped after 8 ids, so 5 local records could become real
    strips. None of them is in the development set.

---

## 9. Model storage

- **`fmenegui/ecgdig-models`** (public): one release per model, with the ONNX file, the PyTorch
  checkpoint and `val.json`.
  - Its workflow `release-model.yml` takes a training run id, downloads the artifact and refuses a
    run whose log shows Kaggle pages (`realstrips`, `--real-dir`, `kaggle_zip`).
  - Released so far: page v1, v2, v3; dewarp v1, v2.
- **Row networks trained on real Kaggle strips** live only in run artifacts, which expire after
  30-60 days. The current best, rowheavy3 (44653806160), expires around Dec 20. They need a private
  home (a private repository or bucket): an open decision.

---

## 10. Maintenance checklist

| When | What |
|---|---|
| Before a batch of runs | Balance ≥ sum of (hours × 0.5 USD/h per GPU run, hours × ~1.1 per 32-vCPU pod) + 2 USD |
| Weekly | `dc-probe.yml` if anything looked odd; `kaggle-zip-probe.yml` if Kaggle evaluations fail |
| After any change to `pod.sh`, `runpod_ctl.py`, `runpod_v2.py`, `podserve.py` | a 10-minute `pod-eval what=pm per_scenario=1` and a short `gpu-train` smoke run |
| After any change to the real-strips builder | a new strips set will be built: plan the 1-2 USD, or pin the old code |
| Monthly | Delete volumes no longer used (the old `ecgdig-data` in EUR-IS-1 holds the rowheavy3d checkpoint and the first PM copy; delete it once that checkpoint is recovered or given up) |
| Before artifacts expire (30-60 days) | Release non-Kaggle models to `ecgdig-models`; keep Kaggle-trained ones privately |
