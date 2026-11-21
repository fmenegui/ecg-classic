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
| PM-ECG-ID whole database | **10.43 dB** (rowheavy3e + page v3 + photo_norm, Nov 21) | 5,918 / 6,000 (98.6%) | PMcardio 14.48 dB (60-subset mean of its paper); 14.05 dB / PCC 0.885 / 6.62% failed in its supplementary table |
| PM-ECG-ID dev (300 ECGs) | 9.34 dB (rowheavy3, 44653806160) | 300 / 6,000 (5%) | |
| Kaggle training records, whole | **18.30 dB** all, 18.21 holdout (rowheavy3e + page v3 + photo_norm, Nov 21) | 8,352 / 8,793 (95%) | leaders 22.6-23.3 dB on the hidden test (private LB), not the same images |
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
| 20-21/11 | 44834962498 | Open-ECG-Digitizer, Kaggle dev again (scoring fixed; input fitted to 2200x1700) | 3090, 0.5 | 2.55 | 1.28 | 540 of 540 scored: **10.35 dB** official metric vs ours 17.95 (holdout 10.01 vs 17.67); every type lower, 0001 12.73 vs 25.05, photos 0006 9.86 vs 12.32 |
| 20/11 | 44834969780 | Open-ECG-Digitizer input size 2200x1700, 1 ECG per scenario (60 ECGs, 66 images) | GPU | 0.5 | ~0.15 | 3.91 dB PMcardio-style, failed leads 23.8% |
| 20/11 | 44834966288 | Open-ECG-Digitizer, images at their own size (same 60 ECGs) | GPU | 0.8 | ~0.2 | **5.09 dB (+1.19 over 2200x1700)**, failed 22.8%; photos gain most (iphone +10, bents +6.6, samsung +6.1); 2148 s vs 1222 s |
| 20/11 | 44834973129 | Open-ECG-Digitizer input size 3300x2550 (same 60 ECGs) | GPU | 0.8 | ~0.2 | 5.12 dB, same as its own size (+0.03); its own size is kept (it resamples to 3000 px inside) |
| 21/11 | 44840397161 | Open-ECG-Digitizer, PM dev at the images' own size | 3090, 0.5 | 4.3 | 2.15 | 300 of 6,000 ECGs (5%): **4.41 dB** (3.54 fitted to 2200x1700; ours rowheavy3e 9.34, with page v3 + photo_norm 10.35); PCC 0.513, failed leads 24.7%; 44 s per image |
| 20/11 | 44834976689 | 2nd place solution, PM-ECG-ID 3x4+1R pages (5 per scenario) | 3090, 0.5 | 0.3 | 0.16 | **1 of 300 pages read** (stage 0, the homography to the Kaggle reference page, failed on 292; stage 1 on 7); PM SNR 0.00 dB. On the 21 of these ECGs also in our PM dev: ours 11.08 dB |
| 20/11 | 44834954765 | rowheavy4 smoke (10 min) | A4500, 0.25 (EU-RO-1, on the volume) | 0.3 | 0.08 | real strips from the cache, bank cached, 0.64 s/step, data wait 1%; found that only 20,000 of ~50,000 real strips were loaded (`--real-cap` added) |
| 20-21/11 | 44837522288 | rowheavy4: init rowheavy3e, full real strips (`--real-cap 60000`), 285 min | A4500, 0.25 (EU-RO-1, volume) | 4.85 | 1.21 | 29,530 steps; 49,762 real strips in memory; validation col error median 0.29 → 0.25 px, p90 1.60 → 1.40 px (real strips 0.30 → 0.29, p90 1.37 → 1.32); dev benchmarks 44865885556 (PM) and 44865887535 (Kaggle), free |
| 21/11 | 44930386428 | 4th place solution (tomoon33, grid-based rectification, 4 models x 5 folds), smoke: 18 PM-ECG-ID 3x4+1R + 18 Kaggle images (`benchmarks/topsol/run_4th.sh`) | (running) | | | |
| 21/11 | 44930219937 | 3rd place solution (hirotetsu, hengck23 base), smoke: 18 PM-ECG-ID 3x4+1R + 18 Kaggle images | (running) | | | |
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
| 21/11 | rowheavy3e + page v3 + photo_norm (vs rowheavy) | 9.69 → **10.43** (5,918, 98.6%) | 18.14 → **18.30** all, 18.07 → 18.21 holdout (8,352, 95%; 44902767154 vs 44420729713); 0001 +0.28, 0006 −0.09, others ±0.09 |

## 4. Development-set experiments (Nov 20; `docs/leaderboard.md`)

| Experiment | PM dev (300) | Kaggle dev all (540) | Kaggle holdout (117) |
|---|---|---|---|
| base: rowreal 44206699093 | 9.12 | 17.71 | 17.13 |
| + dewarp v2 (torch missing: never ran) | 9.12 | 17.71 | 17.13 |
| + page v3 + rhythm margin | (timed out: 44787684688) | 17.72 | 17.14 |
| rowheavy3e + page v3 (free runners, 44834958768; 19 of 20 shards) | 9.16 → 9.59 (+0.43) on 280 (4.7%); failed leads 15.4 → 11.7%; Kaggle metric on PM 15.62 → 14.22 | (17.72 with rowreal) | |
| rowheavy3c 44523858950 | 8.95 | 17.41 | 16.71 |
| **rowheavy3e 44653806160** | **9.34** | **17.95** | **17.67** |
| rowheavy4 44837522288 (full real strips) | 9.29 (-0.04) | 17.93 (-0.02) | 17.58 (-0.09) |
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
| Open-ECG-Digitizer | 4.41 (300, own size; 3.54 fitted to 2200x1700) | 24.7% | 5.86 | 10.35 (540) | images fitted inside 2200x1700; size sensitivity not checked yet |
| 2nd place Kaggle | 0.00 (300, 3x4+1R only: 1 of 300 pages read) | 100% | 0.46 | 30.07 (18) | Kaggle records are in its training set; fails the page template on PM |

**rowheavy4 (Nov 21): neutral, not adopted.** 2.5× the real Kaggle strips (49,762 vs ~20,000)
and 29,530 more steps from rowheavy3e: PM dev 9.34 → 9.29 dB (300 ECGs, 5%), Kaggle dev 17.95 →
17.93 (540 images, 6.1%), holdout 17.67 → 17.58; validation column error improved (0.29 → 0.25 px
median) without moving the benchmarks. More of the same real strips does not help any more; the
next row-network gains must come from new kinds of training data (extreme photometric
degradations, 20 px/mm, sub-pixel targets). rowheavy3e stays the row network.

### Where PMcardio's 14.48 dB comes from (Nov 21)

- Our reproduction of its published method (fmenegui/pm-ecg-digitalization) read PM-ECG-ID at
  **3.1 dB** (315 images, run-pmid-05, Nov 14), ~6-8 dB on scenarios where nothing failed. That
  reproduction is **not** a fair test of the method, so it does not say where the gain comes from:
  - faithful only where the patent is precise: the Gridder and the per-cell Undistortion;
  - network depth and width are not published (ours: U-Net base 16, 4 levels); the Leader's
    training settings are not published either;
  - trained at about 0.2% of the paper's regime (Dotter 7 epochs x 1,000 patches on CPU vs 300 x
    10,000), on synthetic ecg-torch-renderer pages, not on annotated photos; the Leader used on
    PM-ECG-ID was fine-tuned on Kaggle labels;
  - the reconstruction (rows, columns, paths, format) is our own: the paper only names the steps;
  - in that run the geometry failed outright on many 16 Mpx photos (memory, Gridder coverage).
- Assisted inputs (format, speed, gain) explain at most ~0.2 dB (above).
- Hypothesis (not shown): the gain is mostly training data. A method trained on images from the
  same capture pipeline as its test reads far better there (Kaggle 2nd place: 30 dB on its training records, ~22.6 on the hidden
  test, 0 of 300 PM-ECG-ID pages). PMcardio built PM-ECG-ID (printed PTB-XL ECGs photographed with
  Samsung, iPhone and Doogee phones) and trained Dotter and Leader on 232 hand-annotated images of
  undisclosed origin; most likely the same kind of prints and phones.
- Our own evidence that real captures transfer: the row network with 35% real Kaggle strips
  (rowreal) moved PM-ECG-ID 8.52 → 9.46 dB on the whole database, though it never saw a PM image.
- A cheap discriminating test: train the reproduction's Dotter and Leader at the paper's regime on
  a GPU (300 x 10,000 patches, a few hours on a cheap card) with the same synthetic data. A large
  rise on PM-ECG-ID points to training amount and the method; little change points to data.
- **Paper-regime test launched (Nov 21, 14:40 UTC).** Kaggle GPU kernel `pm-ecg-paper-regime`
  (fmenegui/pm-ecg-digitalization, `kernel/paper_regime.py`, request run-paper-01; free Kaggle GPU:
  the code is in a private repository, which RunPod pods cannot clone without a secret). Same
  synthetic generator and architecture as the current weights; Dotter and Leader trained from
  scratch at 300 x 10,000 patches (up to 200 min each); the same 60 PM-ECG-ID images (one per
  scenario), same evaluator, arm A = current weights, arm B = new Dotter and Leader. Design and
  results in that repository's `docs/RESULTADOS.md`.
- Next for us either way: real photos of our own prints with labels fitted from the known signal (no hand annotation;
  `pm_ecg/labels_fit.py` and `hybrid/realstrips_build.py` already do the fitting), from PTB-XL
  records outside PM-ECG-ID's 100 ECGs.

### Assisted mode (what PMcardio's evaluation may have given its algorithm)

PMcardio's paper (medRxiv 2024.08.31.24312876) says paper speed and voltage gain are user inputs,
and format, lead order and rotation "can be provided as input by the user or determined
automatically"; it does not say which mode its PM-ECG-ID evaluation used (multi-photo ECGs have no
automatic detection). PM-ECG-ID's metadata has the printed format but no speed or gain; our reader
finds 25 mm/s and 10 mm/mV on 96% of the dev pages, so the assisted runs fix those values. All
three use rowheavy3e + page v3 on free runners (the automatic run is 44834958768):

| Run | Given to the reader | Result |
|---|---|---|
| 44846130588 | printed format + 25 mm/s + 10 mm/mV | 262 ECGs (shards 7 and 17 shut down): 9.75 → 9.87 dB (+0.12), failed leads 11.0% → 5.5%, 60-scenario mean 9.87 → 10.02 |
| 44917443473 | Kaggle dev (540 images), 25 mm/s + 10 mm/mV, with page v3 + photo_norm (reference 44856569576) | 17.96 → 17.97 dB (+0.01; holdout +0.01); at most +0.12 on one type (0012). Our speed and gain readings cost nothing on Kaggle |
| 44846134555 | printed format | 262 ECGs paired (18 of 20 shards; 7 and 17 runners shut down): 9.75 → 9.83 dB (+0.08), **failed leads 11.0% → 5.4%** (PMcardio 6.62%), 60-scenario mean 9.87 → 9.99. The format explains PMcardio's lower failure rate, not its SNR |
| 44846138808 | 25 mm/s + 10 mm/mV | 262 ECGs (shards 7 and 17 shut down again): 9.75 → 9.93 dB (+0.18), failed leads 11.0% → 10.1%, 60-scenario mean 9.87 → 10.06 |

**Conclusion.** Giving the reader the format, the speed and the gain brings our failed leads below
PMcardio's (5.4-5.5% vs 6.62%) but adds only 0.1-0.2 dB of SNR. Of the ~4.5 dB between us and
PMcardio's 14.48 dB, the inputs its evaluation may have had explain at most ~0.2 dB. The rest is
reading quality on degraded images (19 hardest scenarios: ours 1.8 dB vs its 11.1 dB "hard"
group; 29 middle: 10.8 vs 15.8). Their method (paper): a U-Net for grid intersections
("Dotter"), per-square 4-point undistortion ("Gridder"), and a U-Net trace mask per lead
("Leader") trained on 232 hand-annotated images of undisclosed origin.

### Photometric normalisation (Nov 21)

`Config.photo_norm`: a per-page tone map (`ecgdig/photonorm.py`) for pages whose grey levels span
less than 100 of 255: stretch between the 0.5th and 99.5th percentiles, then a gamma that puts the
paper (75th percentile) at 0.92; a white page with faint content is only stretched. On sample
pages it fires on brightness -160 (range 51-58) and contrast gamma 0.125 (range 53-57) and on one
of two opacity-0.15 pages; blur, low resolution and ordinary photos (range ~130) are left alone.
Runs (row3e + page v3, free runners): PM dev 44856538010, Kaggle dev 44856569576. Reference: 44834958768.
- **PM dev (44856538010, 243 of 6,000 ECGs paired, 4.0%; 17 of 20 shards): 9.73 → 10.35 dB (+0.62)**,
  PCC 0.725 → 0.747, failed leads 11.2% → 10.3%, 60-scenario mean 9.87 → 10.41. Contrast +3.16
  (failed 18.6% → 11.0%), brightness +0.79 (but failed leads 15.5% → 19.0%: to look at), digital
  +0.46; every other group unchanged (the gate did not fire). Candidate for the default.
  Per scenario: contrast gamma 0.25 7.08 → 15.49, gamma 8 1.25 → 9.85, gamma 0.125 7.15 → 12.32,
  brightness -120 2.78 → 7.54, +160 9.12 → 11.63 (failed leads 12 → 24); worse: brightness -160
  −0.33 → −1.77 and -40 5.92 → 4.95 (6 new failed leads). Next: tune the gate and the gamma limit
  on those two, then a whole-database run.
- Tuning (free runners): minimum gamma 0.35 (44859353824) and 0.5 (44859356037); only brightness
  -160 and -120 reach the 0.2 limit. The -40 pages sampled locally (range 145-176) do not fire the
  gate, so the -40 change comes from other pages.
  - Minimum gamma 0.5 (44859356037, all 20 shards, 280 ECGs): 9.59 → 10.17 dB (+0.58) over no
    normalisation; same as 0.2 on the 243 common ECGs (10.35 → 10.33). Brightness -160 −1.77 →
    +3.16 and -40 back to 5.93, but contrast gamma 8 9.85 → 7.25 and gamma 1.5 8.35 → 7.10: those
    pages need a gamma below 0.5.
  - **Minimum gamma 0.35 (44859353824, 300 ECGs, 5%): best.** 9.59 → 10.28 dB (+0.70) on 280
    paired with no normalisation; 60-scenario mean 9.65 → 10.34; vs gamma 0.5 on all 300: 10.23 →
    10.35. Brightness -120 2.78 → 10.15, -160 −2.53 → 2.81, gamma 0.125 8.42 → 14.38, 0.25 7.08 →
    15.49, 8 1.25 → 7.68; -40 5.07 → 4.35 (the one loss). Now the default minimum gamma.
  - **Whole database (Nov 21), paired, free runners.** rowheavy3e + page v3 (44875611556) vs the
    same + photo_norm (44875613583): 5,848 of 6,000 ECGs (97.5%) 9.89 → **10.41 dB (+0.52)**, PCC
    0.734 → 0.753, failed leads 10.0% → 9.3%, 60-scenario mean 9.94 → 10.44. Contrast +2.87,
    brightness +0.83, everything else within ±0.1. Against the previous whole-database best
    (rowheavy alone, pmcardio 44420734434, 9.71 dB): 5,918 ECGs (98.6%) 9.69 → **10.43 (+0.73)**,
    failed leads 12.8% → 9.2%, 60-scenario mean 9.72 → 10.45; page v3 alone +0.23. Kaggle metric on
    the same PM leads 13.01 → 12.93. Four shards hit the 300-min job limit (2 workers were too slow
    for 480 images with the page network); rescored from their per-image caches (partial, included
    above); the shards now stop at 280 min with partial results. Kaggle whole set for the same
    configuration: kaggle-full 44902767154 (stride 1) vs rowheavy 44420729713: 8,352 images (95%;
    one shard failed) 18.14 → **18.30 dB**, holdout 18.07 → 18.21. **Adopted (Nov 21): rowheavy3e +
    page v3 + photo_norm is the new default; `Config.photo_norm` is now True.**
- **Faint pen on white pages (Nov 21).** Opacity 0.15: 100 ECGs, 14 'no trace row found', wrong
  row counts on most 6x2 pages. The pen is mid-grey (~208 in every channel), the grid saturated
  orange-red (BGR 16/80/240): the grid is darker on average but brighter in its brightest channel,
  which is the one the ink map reads. `photo_norm_white_max=True` stretches the brightest
  channel's range on white pages (215-255 there): the pen turns black and the grid stays red. PM
  dev run 44905591721 (free), reference 44859353824 (photo_norm, gamma 0.35).
  - **Result: +0.19 dB** on all 300 PM dev ECGs (5%): 10.35 → 10.54, failed leads 10.5% → 9.2%.
    Opacity 0.15 2.98 → 7.28 (failed leads 36 → 24), 0.4 8.93 → 13.84, 0.2 9.26 → 8.39; brightness
    +120 9.67 → 13.22 (18 → 6 failed), +160 11.63 → 10.45. Kaggle metric on the PM leads 14.36 →
    14.33. Now on by default with photo_norm (Config.photo_norm_white_max=True).
    Whole database with it: pmcardio 44908014040 (free).
  - Whole database, paired (free runners, per_scenario=0, 6,000 ECGs): first try 44861704073 /
    44861706690 cancelled after 12 of 42 shards died (memory 15.5 of 16 GB with 3 workers; ECG 47's
    photos in flight each time). Now 2 workers with a 7 GB cap each (`ECGDIG_MEM_LIMIT_MB`: an image
    that needs more fails alone). Rerun: rowheavy3e + page v3 44875611556; + photo_norm 44875613583.
- Three shards' runners died again. The new progress line shows why: memory at 15.3-15.4 of 16 GB
  with 3-4 phone photos (16 Mpx, page network) in flight, no single culprit. `pmcardio.yml` now
  runs 3 images at a time.
- Kaggle dev (44856569576, 540 images, 6.1%; with page v3): 17.95 → 17.96 dB vs rowheavy3e alone;
  0006 +0.10, 0009 +0.07, 0010 +0.05, other types unchanged: no harm on normal pages.

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
- **Page v3 with rowheavy3e on free runners (Nov 20, pmcardio.yml 44834958768, 0 USD).** 19 of 20
  shards: shard 7's runner was shut down twice at 14 of its 20 images (likely memory: an image
  that the page network makes heavy; to find), so 280 of 6,000 ECGs are paired with
  rowheavy3e alone. PMcardio-style SNR 9.16 → 9.59 dB (+0.43), PCC 0.681 → 0.722, failed leads
  15.4% → 11.7%; 60-scenario mean 9.25 → 9.65. Gains: contrast +2.27, perspective +1.04, screens
  +1.42, resolution (failed leads 30% → 10%). Losses: rotation −0.64, padding −0.55, doogee and
  scans −0.7. The Kaggle metric on the same PM leads falls 15.62 → 14.22 (it is dominated by the
  best images: digital 21.7 → 19.6, scans 10.7 → 7.0), so the page network trades a little
  precision on clean pages for fewer failures. On Kaggle dev (with rowreal) it was neutral (+0.01).
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
| 44834954765 | rowheavy4 smoke: done, OK | | 0.08 USD |
| 44837522288 | rowheavy4: done (1.21 USD); dev benchmarks running | | |
| 44834958768 | PM dev, rowheavy3e + page v3, free runners: 19 of 20 shards (shard 7 runner shut down twice) | | free |
| 44834962498 | Open-ECG-Digitizer, Kaggle dev again: done, 10.35 dB | | 1.28 USD |
| 44834966288 / 44834969780 / 44834973129 | Open-ECG-Digitizer input size: none / 2200x1700 / 3300x2550, 1 PM-ECG-ID ECG per scenario | 1 h each | 1.5 USD |
| 44834976689 | 2nd place, PM-ECG-ID 3x4+1R pages: done (1 of 300 read) | | 0.16 USD |

