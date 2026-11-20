# Experiment log

Every training and evaluation run that informed a decision, in order, with what it showed.
Coverage is stated for every number. "PM" = PMcardio-style SNR on PM-ECG-ID (a failed lead counts
0 dB); "Kaggle" = the official competition metric (mean of linear image SNRs, then dB). GPU costs
are run duration × the hourly price in the log (they include setup minutes).

Earlier history (classical pipeline only, Nov 12-16): [`evaluation-and-roadmap.md`](evaluation-and-roadmap.md).
Per-scenario tables: [`results/`](results/). Current development ranking: [`leaderboard.md`](leaderboard.md).

## 1. Where we stand (Nov 20, 2026)

| Benchmark | Best of ours | Coverage | Reference |
|---|---|---|---|
| PM-ECG-ID whole database | 9.71 dB (rowheavy, run 44356949676, Nov 19) | 6,000 / 6,000 (100%) | PMcardio 14.48 dB (60-subset mean of its paper); 14.05 dB / PCC 0.885 / 6.62% failed in its supplementary table |
| PM-ECG-ID dev (300 ECGs) | 9.34 dB (rowheavy3, 44653806160) | 300 / 6,000 (5%) | |
| Kaggle training records, whole | 18.12 dB all, 18.04 holdout (rowheavy) | 8,793 / 8,793 (100%) | leaders 22.6-23.3 dB on the hidden test (private LB), not the same images |
| Kaggle dev (540 images) | 17.95 dB all, 17.67 holdout (rowheavy3) | 540 / 8,793 (6.1%) | |
| Kaggle hidden test (official) | 13.03 dB private, classical only, late submission (Nov 17) | ~1,000 hidden images | 1st place 23.33 dB |

The hybrid (row network) has not been submitted to the hidden test yet. Our local numbers (~18 dB)
cannot be subtracted from the leaders' private score: the images differ, and the leaders themselves
saw CV-to-private drops of ~4 dB.

## 2. GPU trainings

| Date | Run | Tag / network | Card, USD/h | Hours | ~USD | Result |
|---|---|---|---|---|---|---|
| 16/11 | 43937639778 | smoke | 4090, 0.89 | 0.4 | 0.36 | pipeline works |
| 16/11 | 43952046757, 43961592285, 43961901902 | trials | 4090 / L40S | 1.0 | 0.34 | cancelled; L40S at 1.09 USD/h never needed |
| 16/11 | 43962031311 | ink network r34-512 | 4090, 0.89 | 4.0 | 3.55 | step 20,000, val IoU 0.806; Kaggle 135 local: network only 16.76 vs classical 18.70 dB; best of two per image (oracle) 19.16 |
| 17/11 | 43992337940, 43997598960, 44001122850 | trials | 4090 / A4500 | 1.8 | 1.2 | cancelled (data wait 23-37% on the 4090) |
| 17/11 | 44003064714, 44003801795 | render bench | 3090, 0.5 | 0.4 | 0.18 | GPU renderer speed |
| 17/11 | 44006004260 | row-512-mix | 3090, 0.22 | 4.5 | 0.98 | first row network |
| 17/11 | 44048958066 | grid phase v3 | A5000, 0.27 | 2.9 | 0.79 | neutral on PM (8.53 → 8.63 dB on 5,100 ECGs, 85%); left off |
| 17/11 | 44069775455 | row v2 "rowphoto" (rendered photos rectified with their exact field) | 3090, 0.5 | 4.4 | 2.18 | PM 6,000/6,000: 8.52 → 9.41 dB, PCC 0.687 → 0.705, failed 13.4 → 12.9%; first network that helps on real photos |
| 18/11 | 44206699093 | rowreal (35% real Kaggle strips) | 4000 Ada, 0.28 | 4.2 | 1.17 | PM 5,990/6,000: 8.52 → 9.46; Kaggle holdout 1,854 images: 17.42 → 17.88 dB; train records same gain (no memorisation) |
| 18/11 | 44329786685 | rowheavy (short) | 3090, 0.22 | 0.6 | 0.14 | test |
| 18/11 | 44356949676 | rowheavy | A5000, 0.27 | 5.6 | 1.52 | PM 6,000/6,000: 9.55 → 9.71; Kaggle whole: 18.12 all, 18.04 holdout |
| 18/11 | 44356969568 | page v1 | A4000, 0.25 | 4.0 | 1.00 | PM alone 8.94 (worse); with select 9.94 dB (5,700 ECGs, 95%); Kaggle with select 17.77 vs 18.08 (−0.31; 384 images lost the rhythm strip) |
| 18-19/11 | 44406646797, 44420706651, 44420718166, 44427796105 | (row and page runs) | 3090, 0.5 | 14.7 | 7.35 | lost: balance at 0, cancelled, OOM; nothing kept |
| 19/11 | 44500818386 / 44500830779 | page2c / rowheavy3b | 3090, 0.22 | 2.3 | 0.51 | rowheavy3b: first of the rowheavy3 chain |
| 19/11 | 44507014540 | page v2 (distractors) | A5000, 0.27 | 5.1 | 1.38 | 18,593 steps; names 65 → 90%; row validation misleading (unfiltered fragments) |
| 19/11 | 44523858950 | rowheavy3c (continues 3b) | 3090, 0.22 | 2.2 | 0.49 | Kaggle dev 17.41 all / 16.71 holdout; PM dev 8.95 |
| 19/11 | 44526296478 | dewarp v1 | 4000 Ada, 0.28 | 4.7 | 1.31 | 12,041 steps; page-coordinate error 21 → 8.7 mm median; ~5 mm local |
| 19/11 | 44541716846, 44543053782 | profile | 3090 / 4000 Ada | 0.4 | 0.14 | rendering = 80-90% of a step (page 1.14 s per 4 pages vs 0.27 s; dewarp 1.28 s per 6 vs 0.13 s) |
| 19/11 | 44546313205 | rowheavy3d | 4000 Ada, 0.28 | 5.9 | 1.66 | **lost**: EUR-IS-1 proxy outage + the 6 h limit; checkpoint stuck on the old volume |
| 19/11 | 44558780904 | page v3 (first try) | A5000, 0.27 | 0.1 | 0.04 | crashed at the first reused batch (`--reuse` metadata bug) |
| 19/11 | 44560948929 | page v3 (`--reuse 4`) | A5000, 0.27 | 5.1 | 1.36 | 38,190 steps (2× v2); names 97.8%; Kaggle dev with the rhythm margin: +0.01 dB (neutral) |
| 19-20/11 | 44635378101, 44646005538 | dewarp v2 (tries) | 4000 Ada / 3090 | 2.1 | 0.58 | could not receive the init checkpoint (EUR-IS-1 outage) |
| 20/11 | 44653802675 | dewarp v2 (`--reuse 6`) | 3090, 0.5 | 4.8 | 2.38 | 75,827 steps (6× v1); local error 5 → 2.9 mm median; Kaggle dev 0.00 dB (flat images) |
| 20/11 | 44653806160 | rowheavy3e (continues 3c) | 3090, 0.5 | 4.6 | 2.31 | **current best row network**: Kaggle dev 17.95 / holdout 17.67 (+0.54 / +0.95 over 3c; +0.24 / +0.53 over rowreal); PM dev 9.34 (+0.21 over rowreal); few real strips (built during training) |
| 20/11 | 44778865703, 44786097024, 44789063535 | 2nd place solution, smoke | A4000-3090 | ~1 | ~0.3 | pipeline fixes (paths, timm version, layout); see §5 |
| 20/11 | 44796609191 | 2nd place solution, smoke (36 images) | 3090, 0.5 | 1.2 | 0.60 | runs end to end; Kaggle dev 30.07 dB (in-sample: memorised); PM-ECG-ID 3x4+1R 0 of 18 read; see §5 |
| 20/11 | 44797759750 | Open-ECG-Digitizer, smoke (18 images) | A4000, 0.25 | 0.4 | 0.10 | runs end to end, ~19 s per image; see §5 |
| 20/11 | 44801827747 | Open-ECG-Digitizer, PM dev | 4000 Ada, 0.28 | 2.5 | 0.70 | 3.54 dB vs ours 9.34 (300 ECGs); see §5 |
| 20/11 | 44801832861 | Open-ECG-Digitizer, Kaggle dev | A4000, 0.25 | 2.7 | 0.68 | read 540 of 540, but only 45 scored (5 records): the scoring crashed (128 processes on a pod that shows 128 host cores; OpenBLAS thread limit). On those 45: 10.00 vs ours 15.48 dB. Fixed; rerun needed |

Released weights (no Kaggle data): page v1-v3, dewarp v1-v2 in `fmenegui/ecgdig-models`.

### CPU pods and environment checks

| Date | Run | What | Pod | Hours | ~USD | Result |
|---|---|---|---|---|---|---|
| 20/11 | 44789056728 | real Kaggle strips, whole set (`realstrips-full`) | 32 vCPU, 1.12 | 0.8 | 0.9 | 6,447 of 6,894 pages kept (93.5%), 5.2 GB on `ecgdig-data2`; holdout excluded |
| 20/11 | 44789252756 | PM dev: rowheavy3e + dewarp v2 | 32 vCPU, 1.12 | 2.4 | 2.7 | 9.26 dB (-0.07); not adopted (§5) |
| 20/11 | 44787684688 | PM dev: rowheavy3e + page v3 | 32 vCPU, 1.12 | 3.0 | 3.4 | **lost**: hit the 3 h limit with nothing kept; evaluator now keeps partial results |
| 20/11 | 44796740396 | env-check | 2 vCPU | 1.0 (queued) | 0 | cancelled: the two runs above held the account's 64 vCPUs; now fails in 5 min with that reason |

## 3. Whole-database evaluations (paired)

| Date | Change | PM-ECG-ID | Kaggle training records |
|---|---|---|---|
| 12-16/11 | classical pipeline rounds | 6.99 → 8.51 dB (6,000) | 135 local: 3.3 → 14.70 dB |
| 17/11 | row v2 (rowphoto) | 8.52 → 9.41 (6,000) | — |
| 17/11 | `edge_stroke` | — | whole set (8,793): small gain; 0001 decomposition: sub-sample lag up to +1-2.6 dB on some images |
| 17/11 | classical official late submission | — | hidden test: 12.67 public, 13.03 private |
| 18/11 | rowreal | 8.52 → 9.46 (5,990) | holdout 17.42 → 17.88 (1,854) |
| 18/11 | photo invariance fixes (5 mm rescue, lone pulse) | 9.46 → 9.56 (5,997) | 18.08 → 18.08 (8,793) |
| 18/11 | pulse-column gain | 9.56 → 9.55 (5,999): neutral | — |
| 19/11 | page v1 + select (margin 0) | 9.45 → 9.94 (5,700) | 18.08 → 17.77 |
| 19/11 | rowheavy | 9.55 → 9.71 (6,000) | 18.12 all / 18.04 holdout (8,793) |
| 19/11 | page v1 + select, same-layout margin 0.1 | — | 18.13 → 17.82 (8,352, 95%): 384 images 3x4+1 → 3x4 |
| 19/11 | rhythm-drop margin 0.5 (simulated from the two runs above) | −0.03 dB (5,700) | +0.31 dB back to 18.13 |

## 4. Development-set experiments (Nov 20; `docs/leaderboard.md`)

| Experiment | PM dev (300) | Kaggle dev all (540) | Kaggle holdout (117) |
|---|---|---|---|
| base: rowreal 44206699093 | 9.12 | 17.71 | 17.13 |
| + dewarp v2 (torch missing: never ran) | 9.12 | 17.71 | 17.13 |
| + page v3 + rhythm margin | (timed out: 44787684688) | 17.72 | 17.14 |
| rowheavy3c 44523858950 | 8.95 | 17.41 | 16.71 |
| **rowheavy3e 44653806160** | **9.34** | **17.95** | **17.67** |
| rowheavy3e + dewarp v2 (numpy, really running) | 9.26 (-0.07) | 17.95 | 17.67 |
| classical only (no network) | — | 17.22 | 17.15 |
| classical at 20 px/mm (rectified frame; same rules) | — | 17.51 (+0.29) | 17.79 (+0.64) |

Classical at 20 vs 10 px/mm per type: 0001 24.38 → 25.20 (+0.82, as the simulated ceiling
predicts), 0010 +0.13; every other degraded type loses 0.2-1.5 dB (0004 −1.48, 0012 −1.23): the
classical thresholds were tuned at 10 px/mm. The overall gain comes from the clean renders, which
dominate the linear mean. Resolution pays where the trace is clean; a row network trained at
20 px/mm is the test that matters.

### Other methods on the same development sets

| Method | PM dev SNR (ECGs) | PM failed leads | PM Kaggle metric | Kaggle dev (images) | Note |
|---|---|---|---|---|---|
| ours, rowheavy3e | 9.34 (300) | 15.2% | 15.48 | 17.95 (540) | |
| Open-ECG-Digitizer | 3.54 (300) | 24.5% | 5.37 | 10.00 vs ours 15.48 (45 of 540 paired, 8%) | images fitted inside 2200x1700; size sensitivity not checked yet |
| 2nd place Kaggle | 0.00 (18, 3x4+1R only) | 100% | 0.42 | 30.07 (18) | Kaggle records are in its training set; fails the page template on PM |

## 5. Diagnostics

- **Oracle stages (Nov 17, 200 synthetic pages, run 44062555355).** Replacing a stage by its truth:
  capture appearance +4.5 dB (Kaggle metric), geometry +0.8, layout and calibration on the flat
  document +0.7. Everything together: +6.0 Kaggle, +8.6 PM SNR. Details:
  [`results/oracle-stages-200.md`](results/oracle-stages-200.md).
- **Clean-render ceiling (Nov 17).** A column reading at 10 px/mm caps at 26.6 dB; at 20 px/mm at
  32.9 dB (simulated on the ground truth of 15 records). The classical reader is at 25.2 dB on 0001.
  Details: [`results/edge-stroke-0001.md`](results/edge-stroke-0001.md).
- **Error decomposition per Kaggle type (Nov 20, 540 images, runs 44754243797, 44776638008).**
  - 0001 is at 25.2 dB with almost nothing left to the oracles.
  - Degraded types (0003-0012) sit at 12-16 dB.
  - Time drift and local drift look like +2.1-2.4 dB per type, but the per-lead oracles also fit
    noise (+0.34 dB on noise alone). The geometric part, one shift shared by the leads of a page
    column, is only +0.3-0.45 dB.
  - Layout, names, coverage and origin cost ≤ 0.3 dB.
  - The main remaining error is the shape of the trace on degraded images. Details:
    [`results/error-decomposition-2026-11-20.md`](results/error-decomposition-2026-11-20.md).
- **Top Kaggle solutions (Nov 20).**
  - Public notebooks and weights exist for 2nd (takashisomeya), 3rd (hirotetsu), 4th (tomoon33) and
    5th (dangnh0611); 1st is not public.
  - 2nd and 3rd build on hengck23's baseline. Their code hard-wires the Kaggle sheet: 3x4 plus a
    lead II rhythm strip, fixed zero-mV rows at 703.5 / 987.5 / 1271.5 / 1531.5 px, 79 px/mV,
    homography to a reference page.
  - Running them on PM-ECG-ID (out of their distribution) and on our Kaggle dev set (in-sample for
    them: they trained on all training records) is in progress (`benchmarks/topsol/`).
- **Page v3 with rowheavy3e (Nov 20, pod-eval 44787684688, 32 vCPU, 3 h, ~3.4 USD): no result.**
  The PM dev part hit the 3 h limit (exit code 124) with nothing kept. The evaluator now prints
  its progress and stops before the limit with partial results (runpod-troubleshooting section 10).
- **Dewarp v2 with rowheavy3e (Nov 20, pod-eval 44789252756, 32 vCPU, 1.7 h, ~1.9 USD).** PM dev
  300 of 6,000 ECGs (5%): 9.34 -> 9.26 dB, failed leads 15.2% -> 15.9%; Kaggle dev (540 images,
  6.1%) unchanged (17.95). It changed only blur, brightness and contrast images (-0.13 to -0.36 dB)
  and photos_bents (+0.09 dB SNR but failed leads 0% -> 18.3%); crumbled photos, where it should
  help, did not change. Not adopted: both readings are made and the one with the better internal
  quality score is kept (`dewarp_margin` 0), and that score prefers the flattened reading on images
  where it is worse. Next: a margin (`dewarp_margin` > 0) or a score that tracks the error, checked
  on the per-image `dewarp_select` fields.
  - Their methods are summarised in [`kaggle-insights.md`](kaggle-insights.md).
  - 2nd place smoke test (run 44796609191, 24 GB card, 185 s for 36 images,
    `TS_LIMIT=18 TS_PM_FORMATS=3x4+1R`):
    - Kaggle dev, 18 images (2 records): Kaggle score 30.07 dB, PCC 0.999, every type including
      the 0006 photo at 27.9 dB. These records are in its training set (the solution trained on
      every training record), so this measures memorisation, not reading; its private-leaderboard
      score on unseen pages is ~22.6 dB. Our Kaggle dev numbers for the top solutions are
      therefore not comparable to ours; only the hidden test is.
    - PM-ECG-ID, 18 images printed like the Kaggle sheet (3x4 plus a rhythm strip, one page):
      0 of 18 read (100% failed leads, 0.00 dB). Stage 0 (homography to the Kaggle reference
      page) failed on 17 and stage 1 on the last one. Same layout, different printer: the
      solution depends on the Kaggle page template, not only on the lead layout.
- **Open-ECG-Digitizer (Nov 20).** Ahus-AIM (Stenhede et al., CC BY-SA 4.0) added as a
  comparison: a generic pipeline (U-Net segmentation, perspective, dewarp, grid size, lead-name
  U-Net, 13 layouts). Its weights were trained on full pages, so the runner fits every image inside
  2200x1700 px (aspect kept, white padding; `TS_OED_SIZE`) before it reads them
  (`benchmarks/topsol/run_oed.sh`, adapter `to_npz_oed.py`, `tests/test_topsol_oed.py`).
  - Smoke test (run 44797759750, RTX A4000, ~25 min including setup, < 0.15 USD): 18 of 18
    images read in 349 s (~19 s per image). 9 PM-ECG-ID images: SNR 6.27 dB, Kaggle score
    8.59 dB; 9 Kaggle images (one record): Kaggle score 10.95 dB, with 0006 (a photo) matched to
    the 12x1 layout and scored 0 dB. Too few images to compare; full runs 44801827747 (PM dev)
    and 44801832861 (Kaggle dev) started.
  - PM dev (run 44801827747, A4000-class card, 8,072 s for 331 images, ~24 s each; 300 of 6,000
    ECGs, 5%; images fitted inside 2200x1700): **3.54 dB** PMcardio-style SNR (ours, rowheavy3e,
    on the same ECGs: 9.34), PCC 0.476 (0.687), failed leads 24.5% (15.2%), Kaggle metric 5.37 dB
    (15.48). It is better only on bent and crumpled photos (+2.5 / +2.8 dB; both methods below
    2.3 dB there).
    - By its layout match: on the 156 ECGs where it named the right layout it scores 5.92 dB
      (ours 12.19); 77 got no layout (1.03 dB) and 67 a wrong layout or were multi-page (0.88 dB).
    - Rotation (0.54 dB, 53% failed leads) and padding (0.16 dB, 55%) break it; padding plus our
      fit-inside resize shrinks the grid, so the input size may matter more than the page size.
      A sensitivity check (no resize / 2200x1700 / larger) on ~30 images is the next step before
      calling this its final number.

## 6. Running now (Nov 20, 23:12 UTC; credit added)

| Run | What | Limit | Worst case |
|---|---|---|---|
| 44834948746 | env-check | 1 h | cents |
| 44834954765 | rowheavy4 smoke: full real strips from the volume, init rowheavy3e, 10 min | 0.7 h | 0.35 USD |
| 44834958768 | PM dev, rowheavy3e + page v3, on free GitHub runners (pmcardio.yml, 20 shards) | 5 h | free |
| 44834962498 | Open-ECG-Digitizer, Kaggle dev again (scoring fixed) | 4 h | 2.0 USD |
| 44834966288 / 44834969780 / 44834973129 | Open-ECG-Digitizer input size: none / 2200x1700 / 3300x2550, 1 PM-ECG-ID ECG per scenario | 1 h each | 1.5 USD |
| 44834976689 | 2nd place, PM-ECG-ID 3x4+1R pages, 5 per scenario | 1.5 h | 0.75 USD |

After the smoke: rowheavy4, 285 min on the volume (`use_volume=true`).
