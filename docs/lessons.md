# Lessons: what worked, what did not, and the mistakes not to repeat

Collected from Nov 12-20, 2026. Each item names the evidence; details are in
[`experiments-log.md`](experiments-log.md), [`evaluation-and-roadmap.md`](evaluation-and-roadmap.md),
[`runpod-troubleshooting.md`](runpod-troubleshooting.md) and [`results/`](results/).

## 1. Method: what worked

| What | Evidence |
|---|---|
| Classical geometry from the grid (Fourier phase, unwrapping, homography, 5 mm cell field, line-by-line snapping) | sub-millimetre rectification; 0001 at 25 dB; scans 0.02-0.05 mm |
| Layout by hypothesis and verification (OCR, separator marks, Einthoven/Goldberger residual, duplicates) | format right on 92-95% of Kaggle images; 74-75% of PM single-page ECGs |
| Fixing the measurements under the format, not the format decision | layout oracle (Nov 16): forcing the printed format did not raise the mean (7.91 → 7.91); the wrong pages fail in rows, pulses, gain and origin |
| A learned row network inside the classical pipeline (fuse mode) | PM 8.52 → 9.41 (rowphoto) → 9.46 (rowreal) → 9.71 (rowheavy), whole database |
| Training on real Kaggle strips, split by record | holdout gain equal to train-record gain (no memorisation); rowheavy3e +0.53 dB on the Kaggle dev holdout |
| Rendering photos with their exact geometry, then rectifying with the exact inverse field | first network that helped on real photos (rowphoto) |
| `--reuse` of rendered pages | 2× (page) and 6× (dewarp) more steps per dollar |
| Page network as a second reading with selection | PM +0.5 dB (9.45 → 9.94), once the rhythm-strip margin stops it costing Kaggle |
| Error decomposition by oracles | showed shape, not layout or time, is the remaining Kaggle error |
| Paired whole-database runs for every decision | small samples misled twice (dev20 → dev21: +0.5 on 120 ECGs, 0 on the database; 18 images promised +2.1 dB, the database gave +0.10) |

## 2. Method: what did not work, or not yet

| What | Evidence | Status |
|---|---|---|
| Ink network as a full replacement of the classical ink map | Kaggle 135: 16.76 vs 18.70 dB | used only as a second reading |
| A patient beat template as a global prior | −0.1 to −0.9 dB | dropped |
| Cubic/Akima interpolation of the read samples | 25.0 → 17.8 / 24.6 dB | dropped: amplifies sample noise |
| Grid phase network v3 | PM 8.53 → 8.63 on 85% | off |
| Pulse-column gain | PM 9.56 → 9.55 | neutral, kept |
| Page network alone (no selection) | PM 9.45 → 8.94 | only with selection |
| Page selection without a rhythm margin | Kaggle −0.31 dB: 384 clean pages lost the rhythm strip | margin 0.5 added |
| Dewarp network (v1, v2) | local error 2.9 mm, needs < 1 mm; zero effect on Kaggle | keeps training; judged on PM photos |
| Per-lead oracle of local time drift as a guide | inflated (+1.2-1.5 dB per type) by fitting noise; the geometric part is +0.3-0.45 | use the shared oracle |

## 3. Evaluation: mistakes and the rule each one left

1. **Too small samples.** The 120-ECG and 18-image samples disagreed with the database by
   0.5-2 dB. *Rule:* decide on whole-database paired runs, or on the fixed development sets of the
   leaderboard, confirmed on the whole database before adoption.
2. **A component that never ran was measured as "no effect".** Dewarp v2 needed torch, which the
   evaluation hosts lack; every reading failed silently. *Rule:* a change of exactly 0.00 dB is a bug
   until the per-image fields (`dewarp`, `select`, errors) show it ran.
3. **The validation metric picked the wrong checkpoints.**
   - Page: the row count was measured before ecgdig's filters, so it scored fragments.
   - Dewarp: the raw coordinate error included global shifts that ecgdig removes.
   - Row: the best checkpoint was chosen by mask IoU, not by the trace error.

   *Rule:* the validation must measure what the digitizer uses (fixed Nov 19-20).
4. **An oracle that also fits noise overstated a lever.** *Rule:* calibrate every oracle on a case
   with no error of its kind (noise only) before reading its gain.
5. **The external converter for the 2nd-place solution misplaced 10 of 12 leads**, found by an
   outside review. *Rule:* every adapter gets a synthetic round-trip test (now
   `tests/test_topsol_to_npz.py`).
6. **One holdout list was truncated** (8 of 15 local records in the real-strips builder). *Rule:*
   one test checks that all copies agree (`tests/test_holdout_lists.py`).
7. **Comparing our local CV to the leaders' private score.** Different images, and the leaders saw
   ~4 dB CV-to-private drops. *Rule:* the only direct comparison is the same images (our dev sets
   with their code) or a submission to the hidden test.

## 4. Infrastructure: what worked

- GitHub Actions as the free orchestrator: 20-shard evaluations and the merge cost nothing.
- One pod per run, terminated by the run in every case.
- The balance guards. After they were added, no run was lost to a zero balance.
- Checkpoint copies every 30 min; `init_run` chaining across the 6 h limit.
- Live pod logs (`podlog.sh`) and heartbeats. They found the single-stream bottleneck of the strips
  builder within minutes.
- `dc-probe.yml`: cents to settle "is it the data center, the volume or the image".
- The development leaderboard: one command per experiment, one table.

## 5. Infrastructure: mistakes (about 28 of ~53 USD lost, Nov 16-20)

| Mistake | ~USD lost | Rule |
|---|---|---|
| Expensive GPUs (4090, L40S) for a CPU-bound generator | 5 | cost per step, cheap cards only |
| Balance reached 0 with several pods running | 7 | guard per pod + check the sum before a batch |
| Pods that never answered (EUR-IS-1 volume) kept for an hour or more | 6 | give up after 35 min; probe the data center |
| Job lost at GitHub's 6 h limit | 1.7 | follower stops at 340 min and uploads |
| Evaluations killed by `max_hours` (page/dewarp read twice) | 7 | estimate time from a previous run; `pm_hours` |
| 32 vCPU to download; single-stream downloads | 2.4 | 4 vCPU prefills; parallel readers |
| Diagnosing by changing the infrastructure (image swap) instead of probing | Kaggle part of several benchmarks | one variable at a time, cheap probe first |
| A pod listing printed to a public log | 0 (security) | never list pods or print environments |

## 6. Process

- **Write the result down the day it is measured,** with coverage and both metrics. The docs had
  fallen behind the code by Nov 20 and an outside review quoted stale numbers.
- **Every adapter, converter and list copy gets a test.**
- **State GPU, hours and cost for every training** and keep the total visible.
- **Smoke-test any new data path for 10-20 minutes before a 5 h run.**
- **Read the first heartbeat lines of every new kind of pod job.**
