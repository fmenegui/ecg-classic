# Rules for working in this repository

These rules come from real losses (about 28 of 53 USD spent on RunPod in Nov 16-20, 2026 went to
runs that produced nothing; one secret leaked into a public log). The reasons and the full
procedures are in `docs/infrastructure.md`, `docs/runpod-troubleshooting.md` and `docs/lessons.md`.
Read them before changing anything under `hybrid/gpu/`, `.github/workflows/` or `benchmarks/`.
`.claude/hooks/guard.py` blocks the most expensive mistakes; `tests/test_infra_rules.py` checks the
code on every push.

## Communication

- Talk to the owner in concise Portuguese; write code, commits and docs in English.
- For every training run state the card, hours and cost. For every result state the coverage
  (n of N, %) and both metrics: the official Kaggle metric and the PMcardio-style SNR.
- Never compare our local Kaggle numbers with the leaders' private-leaderboard score as if they
  were on the same images.

## Before a batch of experiments

1. `gh workflow run env-check.yml` (~10 min, cents). Every line of its summary must be OK: tests,
   Kaggle archive from GitHub, RunPod balance and volume, a pod on the volume answering through the
   proxy, the volume writable, and the Kaggle archive reachable from the pod.
2. Smoke-test each new or changed job for 10-20 minutes (same job, small inputs: `TS_LIMIT`,
   `per_scenario=1`, `--minutes 10`) and read its log before the long run.

## Money (RunPod)

- GPUs: cheap cards only (≤ 0.5 USD/h: A5000, 3090, A4500, 4000 Ada, A4000, L4). Never 4090, L40S,
  A100 or H100 unless the owner names the card.
- Before launching anything, add up the worst case of everything running and planned
  (GPU: max_hours × 0.5 USD/h; 32-vCPU pod: hours × 1.12 USD/h) plus 2 USD. If the balance does
  not cover it, ask for funds; RunPod stops every pod at 0 USD.
- Smoke-test any new job or data path for 10-20 minutes before a long run.
- Watch every new kind of pod job in its first 10 minutes: `bash hybrid/gpu/podlog.sh RUN_ID`.
  CPU load far below the vCPU count, a silent log or no proxy answer after ~30 min = cancel.
- Training: `max_hours` ≤ 5.8 and `--minutes` ≤ 285 (GitHub kills jobs at 355 min).
- Evaluations with the page or dewarp network read every page twice: give them `pm_hours` 3.
- Downloads go on small pods (4 vCPU) or free GitHub runners, never on 32 vCPU.
- At most two 32-vCPU pods at a time (account limit 64 vCPU).
- Diagnose one variable at a time with a cheap probe (`dc-probe.yml`, cents) before changing the
  infrastructure.
- Prefer free GitHub runners for evaluation (`kaggle-full.yml`, `pmcardio.yml`); pay for pods only
  when the time matters.

## Secrets and data

- The RunPod key lives only in the GitHub secret `RUNPOD_API_KEY`. Never paste keys in chat or code.
- Pods get no secret except the Kaggle token, and only when the job reads Kaggle data
  (`kaggle=true`).
- Never list pods and never print a pod's environment (it holds the Kaggle token). Do not call the
  RunPod API from this machine; the workflows do.
- Kaggle competition data is never published: not in the repository, not in artifacts. Only
  `results.json`, `report.md`, `decomposition.json` and logs leave a pod or runner. The read signals
  next to the ground truth (`sig/`) are deleted. Real strips stay on the volume.
- Weights trained on Kaggle pages never go to the public `fmenegui/ecgdig-models`.
- PM-ECG-ID and LITFL are never used for training. Kaggle is split by record: the holdout
  (id % 5 == 0 plus the 15 local records) never enters training.
- Only terminate pods this session started.

## Experiments

- Every candidate goes through `experiment.yml` (same development sets) and
  `python benchmarks/leaderboard.py`; one line per tag in `docs/experiments.csv`.
- A candidate becomes the default only after a whole-database paired run (`pmcardio.yml
  per_scenario=0`, `kaggle-full.yml stride=1`) with `benchmarks/compare.py`.
- A result of exactly 0.00 dB change is a bug until the per-image fields show the component ran.
- Every adapter, converter or duplicated list gets a test.
- Calibrate any oracle on a case with no error of its kind before trusting its gain.
- After each result: update `docs/experiments-log.md` (and the leaderboard) the same day.

## Git

- Work on branch `r4`; keep `main` equal: `git push origin r4 && git push origin r4:main`.
- Commit trailer:
  ```
  Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
  ```
