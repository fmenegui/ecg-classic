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
| PM-ECG-ID whole database | **10.49 dB** (rowheavy3e + page v3 + photo_norm with the faint-pen stretch, Nov 21) | 5,730 / 6,000 (95.5%) | PMcardio 14.48 dB (60-subset mean of its paper); 14.05 dB / PCC 0.885 / 6.62% failed in its supplementary table |
| PM-ECG-ID dev (300 ECGs) | 9.34 dB (rowheavy3, 44653806160) | 300 / 6,000 (5%) | |
| Kaggle training records, whole | **18.30 dB** all, 18.21 holdout (rowheavy3e + page v3 + photo_norm, Nov 21) | 8,352 / 8,793 (95%) | leaders 22.6-23.3 dB on the hidden test (private LB), not the same images |
| Kaggle dev (540 images) | 17.95 dB all, 17.67 holdout (rowheavy3) | 540 / 8,793 (6.1%) | |
| Kaggle hidden test (official) | 13.39 dB private / 13.10 public, hybrid late submission (Nov 22; classical 13.03 / 12.67) | ~1,000 hidden images | 1st place 23.33 dB |

The hybrid was submitted on Nov 22 (13.39 private). Our local numbers (~18 dB)
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
| 21/11 | 44930386428 | 4th place solution (tomoon33, grid-based rectification, 4 models x 5 folds), smoke: 18 PM-ECG-ID 3x4+1R + 18 Kaggle images (`benchmarks/topsol/run_4th.sh`) | 3090, 0.5 | 0.4 | 0.20 | did not run: its weights filled the 60 GB container disk at the 6th of 11 datasets; gpu-train.yml now has `disk_gb` |
| 21/11 | 44932673380 | 4th place solution, smoke again with a 150 GB disk | 3090, 0.5 | 1.2 | 0.59 | ran: 36 of 36 (1,255 s; grid model 4.1 GB, line-index 9.2 GB, ...). Kaggle dev, 18 images (in its training set): 27.95 dB, PCC 0.999. PM-ECG-ID 3x4+1R, 18 images: every lead returned (0% failed) but **SNR −2.35 dB, PCC 0.318** (best lead II 0.74 dB): its grid model rectifies the PM pages without failing, but the reading maps them onto the Kaggle sheet (1700x2200, fixed lead positions); ours on this format ~11 dB |
| 21/11 | 44930219937 | 3rd place solution (hirotetsu, hengck23 base), smoke: 18 PM-ECG-ID 3x4+1R + 18 Kaggle images | 4000 Ada, 0.28 | 0.3 | 0.08 | did not run: the converted notebook had an `except:` block of shell lines only (IndentationError); make_run.py now turns shell lines into `pass` |
| 21/11 | 44931957085 | 3rd place solution, smoke again | 3090, 0.5 | 0.3 | 0.15 | did not run: its model code imports `transformers` (not installed); added to run_hengck.sh |
| 21/11 | 44933687310 | 3rd place solution, smoke, third try | 3090, 0.5 | 0.45 | 0.22 | ran (586 s for 36 images, stages 0-2), but wrote to output-<MODE>/digitalised, not outputs/digitalised: nothing converted. run_hengck.sh now finds the output folder |
| 21/11 | 44936415142 | 3rd place solution, smoke, fourth try | 3090, 0.5 | 0.45 | 0.23 | ran: 36 of 36 converted (466 s). Kaggle dev, 18 images (2 records, in its training set): 27-30 dB per type, PCC 0.998 (memorised). PM-ECG-ID 3x4+1R: **0 of 18 read** (stage 0, the homography to the Kaggle reference page, failed on every page); PM SNR 0.00 dB |
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
| 22/11 | hybrid kernel check 1 (kaggle-submit.yml 44987230477, submit=false) | Kaggle CPU, free: ran, but `models: none`: the dataset folders arrive zipped and the image is Python 3.13 (wheels only ≤3.12) | — | see change |
| 22/11 | hybrid kernel check 2 (44987623093) | Kaggle CPU, free: models found; onnxruntime install failed (pip found no wheel for the running Python) | — | see change |
| 22/11 | hybrid kernel check 3 (44987916841) | Kaggle CPU, free: OK: both networks load, 2 public images ~100 s each (2 workers × 2 threads on 4 CPUs). Too slow for ~1,000 hidden images in 9 h, so the kernel got an 8.3 h budget: networks while the time left covers the rest classically, then classical, then zeros | — | see change |
| 22/11 | hybrid (rowheavy3e + page v3 + photo_norm) late submission (44988601240, submit=true) | Kaggle CPU, free: submitted (ref 58705110, kernel version with the 8.3 h budget); the workflow's score wait stopped at once (it matched the old COMPLETE row): fixed, and a status-only mode waits for the score (44991334960) | — | see change |

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
| 2nd place Kaggle | 0.00 (300, 3x4+1R only: 1 of 300 pages read) | 100% | 0.46 |
| 3rd place Kaggle | 0.00 (18, 3x4+1R only: 0 of 18 read) | 100% | 0.42 | 27-30 (18, in its training set) | same hengck23 template stage as the 2nd |
| 4th place Kaggle | −2.35 (18, 3x4+1R only; PCC 0.318) | 0% | −0.98 | 27.95 (18, in its training set) | grid rectification works; the reading assumes the Kaggle sheet | 30.07 (18) | Kaggle records are in its training set; fails the page template on PM |

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
  - **Result (Nov 21, ~10 h of free Kaggle GPU).** The full regime did not fit: 60 epochs x 10,000
    patches per network in 200 min (20% of the paper's, 85x the current weights). Synthetic
    validation IoU rose a lot (Dotter 0.865 → 0.957, Leader 0.731 → 0.906); PM-ECG-ID did not
    (60 images, one per scenario: SNR −0.76 → −1.42 dB, PCC 0.268 → 0.253, failed leads 24% → 22%).
    Training volume on synthetic pages is not what separates the reproduction from PMcardio. Both
    arms are below 0 dB: this reproduction fails earlier (geometry and reconstruction on 16 Mpx
    photos) and is too weak a proxy of PMcardio's method to attribute its gain. The PM-trained
    diagnostic (below) tests the data hypothesis directly.
- Next for us either way: real photos of our own prints with labels fitted from the known signal (no hand annotation;
  `pm_ecg/labels_fit.py` and `hybrid/realstrips_build.py` already do the fitting), from PTB-XL
  records outside PM-ECG-ID's 100 ECGs.

### DIAGNOSTIC: training on PM-ECG-ID itself (Nov 21; never a default, never released)

Question: does training on the benchmark's own kind of capture close the gap to PMcardio? Asked
for by the owner as a diagnostic only. Design: split PM-ECG-ID by ECG (Image ID even = train,
odd = test, so no test ECG's signal is seen); real strips from 20 even ECGs per scenario
(`hybrid/pmstrips_build.py`, same labelling as the Kaggle real strips); the row network continued
from rowheavy3e with those strips as its only real strips; both it and rowheavy3e evaluated on odd
ECGs only (`pmcardio.yml id_parity=odd`), with page v3 and photo_norm. The model never becomes a
default, is never released, and its numbers are never reported as our PM-ECG-ID result.

| Step | Run | Result |
|---|---|---|
| strips (32 vCPU pod, volume, ~1 h, ~1.1 USD) | 44963401452 | 1,355 pages of 20 even ECGs per scenario; **899 kept (66%)**, ~8,300 strips in 21 chunks, 888 s of labelling |
| training, first try | 44968424722 | no card free in EU-RO-1 with the volume; the pod started elsewhere and stopped at once (the guard: strips not there), cents. `gpu-train.yml volume_only` now waits for the volume's data center |
| training: rowheavy3e + 150 min with the PM strips as the only real strips (35%) | 44969275121 | 4000 Ada, 0.28 (EU-RO-1, volume), 2.6 h, ~0.7 USD; 11,240 steps; 7,544 PM strips in memory; held-out PM strip column error 0.54 px median, 2.46 px p90 (Kaggle real strips with rowheavy3e: 0.29 / 1.32) |
| training | | |
| evaluation, rowheavy3e (baseline), odd ECGs, 5 per scenario | 44963430509 | done (the 10 even shards had no odd ECG and failed on the empty list; fixed) |
| evaluation, diagnostic model, same ECGs | 44981336273 | 300 odd ECGs (5 per scenario): **9.95 → 10.30 dB (+0.35)**, PCC 0.748 → 0.754, failed leads 8.1% → 8.1%, Kaggle metric on the PM leads 12.67 → 12.94 |

**Reading of the diagnostic.** Training the row network on the benchmark's own captures (other
ECGs) gives only +0.35 dB: JPEG +0.85, scans +0.90, screens +0.71, digital +0.64, phones +0.5-0.6;
blur, rotation and brightness ±0.1-0.3; bent and crumpled photos −0.3. It does not close the ~4 dB
gap. Caveat: the strips are labelled with the classical reader's own geometry and kept only where
reading and truth agree (66% of the pages), so the pages we fail on (blur, low resolution, bent
photos) hardly enter the training; the row network also only reads strips around rows the reader
has already found. The gap therefore sits mostly before the row network (page geometry, rows and
layout on degraded images), and more real photos labelled the same way would help little. Real
photos would help if their labels did not depend on our reader: pages printed with fiducial
markers (known page geometry) can be labelled whatever our reader does.

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
    Whole database with it: pmcardio 44908014040 (free; shards 12 and 19 lost to Zenodo 500/504 at the
    ground-truth read, rerun). On 5,079 ECGs (84.7%) paired with the first photo_norm run: 10.53 →
    **10.63 dB (+0.10)**, digital +0.43, brightness +0.22, the rest within ±0.06. Against rowheavy alone
    (5,130 ECGs): 9.72 → 10.63 (+0.91), failed leads 13.0% → 9.2%, 60-scenario mean 9.82 → 10.70.
    Final, with the two shards rerun (one stopped at the 280-min deadline with partial results): 5,730
    of 6,000 ECGs (95.5%) against rowheavy alone 9.63 → **10.49 dB (+0.86)**, PCC 0.706 → 0.756,
    failed leads 13.0% → 9.1%, 60-scenario mean 9.71 → 10.55; against photo_norm without the stretch
    (5,679) 10.39 → 10.48 (+0.10). This is the current default.
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

- **PM-ECG-ID: where the gap to PMcardio is (Nov 22, whole run 44908014040, no new compute).**
  - The database is 100 ECGs × 60 scenarios, so a failure on one ECG repeats in up to 60 scenarios.
  - Our 19 worst scenarios average 4.46 dB, the middle 29 12.20, the best 12 16.20 (PMcardio's
    groups: 11.10 / 15.79 / 16.68). If the groups matched, the 60-mean gap (10.55 vs 14.48) splits
    into ~2.1 dB from the hard scenarios, ~1.7 from the middle ones and ~0.1 from the easy ones.
  - The same ECG swings by tens of dB between easy scenarios (ECG 59: 22 dB digital, −2 to +6 dB
    on its augmented photo). Per-ECG structure, not the trace shape, drives much of it.
  - `results.json` diag `rows_y_mm` divided rectified pixels by the source px/mm (wrong unit); now
    by `Config.R`. The diag also records the horizontal scale, the assumed duration and the grid
    residuals.
- **Time-scale oracle (Nov 22, local, free; classical reader).** Per lead, the best linear time map
  (scale ±6%, shift) against the truth; script in the session scratchpad, numbers here.
  - brightness +40 (one augmented photo per ECG, 45 ECGs, 534 leads): 14.11 → 15.06 dB.
  - Calibration on clean digital pages (opacity 1, 45 ECGs): 18.59 → 19.40 dB. The oracle fits
    +0.5-0.8 dB on correct pages, so the photo-specific part of the horizontal geometry is only
    ~+0.15 dB. A grid-based horizontal correction is not a large lever on PM-ECG-ID.
  - Exception: ECG 59's photo has a non-uniform horizontal scale (left column +3.2%, right −1%,
    0.18 s shift); its faint blue grid gave no local cell field.
- **Structural failures seen on single ECGs (Nov 22).**
  - ECG 15 (3x4+1R, pulses at the row ends): a faint row start (lighter left side of the photo)
    breaks the binarized trace into short pieces; the row starts 34 mm late and the time axis
    stretches to 20.8 mm/s (−2.1 dB on brightness +40).
  - ECG 35 (6x2): left column read 0.8 s short, plus a 1% stretch.
  - ECG 24 (3x4) and 23 (6x2): each column has its own row positions (rows placed by amplitude);
    rows found across the page mix the columns. Fails even on the digital page.
  - ECG 45 (6x2, page cut at the left, huge overlapping V2-V4): fails even on the digital page.
  - 12x1 pages: 7-10 dB with PCC 0.94 (shape, not gain).
- **Row start over a chain of beats (`digitize.EXT_ON_D`, Nov 22, local, free).**
  - v1, walk on the ink map: brightness +40 13.95 → 13.75 (ECG 15 +14.2; five 6x2/3x4 pages
    −1.7 to −9: the walk followed leaked grid lines to the grid border); digital 18.48 → 18.39.
    Dropped.
  - v2, chain of binarized pieces accepted when it holds two beats (≥5 mm tall, ≥8 mm apart):
    brightness +40 13.95 → 14.21 (ECG 15 +16.0, ECG 45 −4.5); digital 18.48 → 18.39 (ECG 39
    −3.3).
  - v3, kept only when at least half of the rows move to a common start: brightness +40
    13.95 → **14.31** (only ECG 15 changes, +16.0; no page worse); digital 18.39 → 18.39 (no page
    changes: no faint starts there). Off by default (`EXT_ON_D=False`) until a whole-database
    paired run.
  - Baselines were rerun with the flag off: brightness +40 reproduces 13.95 exactly. The first
    digital baseline (18.48) did not reproduce (18.39 twice with the same code, ECG 39 16.38 vs
    13.11); the digital comparison uses the rerun.
- **Local grid field on a flat digital page (Nov 22).** PM ECG 24, digital: the cell field came out
  at 1.35 mm rms (poly 0.36 mm; creases and photos give 0.09-0.7 mm) and visibly crumpled the
  flat page (dotted fine grid, dense trace). `geometry.CELL_MAX_MM` (0 = off) drops such a field;
  not measured yet. ECG 24 still fails without it (poly: −4.2 dB): its layout is read as 6x2+2
  (columns with their own row heights).
- **Geometry variants on the 45 digital pages (Nov 22, local, classical, free).** The digital pages
  are flat, yet the reading depends a lot on the geometry stage:
  - `nonrigid=poly`: 18.39 → 17.13; `nonrigid=none`: 18.39 → 16.26 (several pages lose their
    layout and score 0; a few gain +2 to +3.6). Best of cell/none per page (oracle): 18.87.
  - `geometry.H_SELECT_MM=0.15` (keep the homography only when it beats rotation + scale + shift
    by 0.15 mm; ECG 41 digital had a 1% top-to-bottom horizontal scale change from it): digital
    18.39 → 18.18 (+7.2 and +7.1 on two pages, −19.8 on one); brightness +40 unchanged (photos
    keep their homography). Not adopted.
  - `geometry.CELL_MAX_MM=0.8`: digital 18.39 → 17.11 (the same four pages fall as with poly);
    brightness +40 13.95 → 13.77. Not adopted.
  - Choosing between cell and the variant by the mean lead quality score: +0.10 (H_SELECT) or
    worse; the internal score does not track these errors.
  - Mixed swings of ±7-20 dB on single pages from small geometric changes: the structure stage
    (rows, layout) is fragile, and that, more than the geometry itself, is what to fix.
- **Orientation (Nov 22).** PM-ECG-ID pages are never turned, yet the default turned 183 of 6,199
  pages (whole run 44908014040): 96 by 180°, 87 by 90°; those ECGs average 0.9 dB (others 10.8).
  Mostly faint (opacity 0.15/0.2), blurred (21-41), 1/16 resolution, brightness ±160. On Kaggle dev
  4 of 540 images are turned by 90°, correctly (portrait photos of a landscape sheet). Flag
  `digitize.ORIENT_PRIOR` (off): a landscape sheet stays landscape unless standing rows win by
  1.5×, 180° needs more pulses or labels (not the T wave alone), the page network turns an upright
  reading only at 3× its score. Hybrid test on 78 pages the default turned: (running).
- **Blur (Nov 22).** On blur 21-31 the read leads' PCC falls to 0.57 / 0.34 (blur 11: 0.91), even
  where layout and rows are right. One page looked at (ECG 14, blur 21, hybrid, 2.45 dB): the wave
  shapes are right, but each QRS is placed 10-20 samples (20-40 ms) early or late, differently
  from beat to beat, sometimes split in two. The blur (σ ≈ 0.45 mm) smears the steep QRS edges over
  ~1.5 mm and the row network puts the jump anywhere inside it; its training blur stopped at
  σ 0.5 mm with rarely that much. rowheavy5 trains on blur up to σ 1 mm and on 1/24 resolution.
- **Why the Kaggle winners do not read common ECGs (Nov 22).** Their code: the hengck23 pipeline
  (1st, 2nd, 3rd) finds nine lead-name keypoints with a network and maps them by homography onto
  their places in ONE Kaggle image; snaps a 44 x 57 lattice of 5 mm grid points over the whole page
  onto that image's lattice; reads the signal at fixed places of the 2200 x 1700 result (zero mV at
  y = 703.5 / 987.5 / 1271.5 / 1531.5, 79 px/mV, 10 s from x = 118 to 2080, four 2.5 s columns;
  3x4 + II only). The 4th place classifies the absolute index of each grid line of the Kaggle sheet
  and reads 4 fixed bands. Probe pages (`benchmarks/topsol/probe_pages.py`, `probe_eik.py`,
  `score_probe.py`): 20 PTB-XL fold-10 records (no PM-ECG-ID ECG or patient), one change per
  variant; SNR in dB, PMcardio style (failed segment 0 dB):

  | Variant | 2nd | 3rd | ours (hybrid) |
  |---|---|---|---|
  | K0 template geometry, our drawing | 0/20 read | 0.00 | **20.85** |
  | K1 rows 25 mm apart | 0/20 | 0.00 | 20.52 |
  | K2 grid only in a box | 2/20, −0.98 | −1.60 | 22.08 |
  | K3 page 8% smaller | 0/20 | 0.00 | 20.84 |
  | K4 no labels | 0/20 | 0.00 | 21.09 |
  | K5 6x2 | 16/20, −6.55 | −9.46 | 20.47 |
  | K6 our print-kit page | 0/20 | 0.00 | 21.93 |
  | E0 ecg-image-kit default sheet (the Kaggle generator) | **29.86** | **27.11** | 25.17 |
  | E1 no lead names | 0/20 | 0.00 | 25.74 |
  | E2 blue grid | −9.47 | −16.65 | 21.57 |
  | E3 page 8% smaller | 28.39 | 27.18 | 22.65 |
  | E4 rotated 3°, JPEG 70 | 28.33 | 27.44 | 22.72 |
  | E5 6x2 (generator: both columns show 0-5 s) | 1/20, −0.22 | −0.43 | **9.24** (45% segments failed) |
  | E6 rows 30% closer | 0/20 | 0.00 | 23.87 |
  | E7 top 12% cut off | 29.91 | 25.29 | 25.09 |

  - On the Kaggle generator's own sheet the winners generalise to unseen ECGs (~27-30 dB; ours 25)
    and survive scale, rotation, JPEG and a cut page: the homography takes care of those.
  - Any one of these breaks them completely: no lead names or names drawn in another style (K0,
    E1: stage 0 finds no keypoints), a grid colour they did not train on (E2), any layout or row
    spacing other than the Kaggle 3x4 + II (E5, E6, K1, K5). PM-ECG-ID pages have at least one of
    them, hence 1 of 300 read.
  - Ours reads every variant but one at 20.5-25.7 dB. On the clean generator sheet we are 2-5 dB
    below the template readers: trace precision, not structure.
  - Ours fails half the leads of ecg-image-kit's 6x2 page (9.24 dB): there both columns show the
    same 0-5 s, while our reader (and PM-ECG-ID's 6x2) takes the second column as 5-10 s. Devices
    differ on this; a reader should try both time conventions for multi-column pages.
- **Print kit (Nov 22, `kit/`).** 40 training pages (PTB-XL folds 1-9) and 20 test pages (fold
  10, pages 100-119), A4, six layouts, six grid colours, pulse left/right/both, four ArUco markers
  per page. No PM-ECG-ID ECG or patient, no lead-II match with an evaluation ECG. `read_photo.py`
  gives page, homography and every trace sample in photo pixels; simulated perspective + blur +
  JPEG: 0.05 mm error. Regenerated exactly by the commands in `kit/README.md`. PDFs sent to the
  owner to print and photograph.

## 6. Running now (Nov 22, ~08:45 UTC)

| Run | What | Limit | Worst case |
|---|---|---|---|
| 44991334960 | Kaggle: waiting for the hybrid submission's score (ref 58705110) | 5.5 h | free |
| 45003441294 | env-check: RunPod, volume, proxy, Kaggle archive OK; tests failed only on test_kaggle_budget without pandas (fixed); balance 13.76 USD | done | cents |
| 45003632612 | rowheavy5 smoke (10 min): rowheavy3e + heavy_aug down to 1/24 and blur sigma to 10 px (`--heavy-down-max 24 --heavy-blur-max 10`), real strips from the volume | done: 4000 Ada 0.28 USD/h, ~0.4 h, ~0.12 USD; caches hit, 20,000 real strips, 0.9 s/step, data wait 1-5% | |
| 45004819752 | rowheavy5: same, 270 min, then dev benchmarks with page v3 (free runners) | done: 4000 Ada 0.28 USD/h, 4.6 h, ~1.28 USD; 20,327 steps; validation column error median 0.27 px, p90 1.47 (real strips 0.29 / 1.33); dev benchmarks 45027498706 (PM) and 45027500873 (Kaggle), free | |
| 45027498706 / 45027500873 | rowheavy5 dev benchmarks (with page v3 + photo_norm), vs 44905591721 / 44856569576 | done: PM dev 300 ECGs (5%) 10.54 → **10.79 dB** (+0.25), PCC 0.757 → 0.762, failed leads 9.2% → 9.4%, Kaggle metric on the PM leads 14.33 → 15.91; blur +0.92, resolution +0.80, digital +0.34, rotation +0.31, phones −0.8 to 0. Kaggle dev 540 images 17.96 → 17.96 (holdout 17.67 → 17.64; 0004 +0.20, 0006 −0.11). Candidate: whole-database pair next | free |
| 45014548754 / 45014555918 | Kaggle winners on probe pages K0-K6 (our drawing, template geometry): 2nd / 3rd place | done: RTX 4000 Ada 0.28 / A5000 0.27 USD/h, 0.33 / 0.78 h, ~0.09 / 0.21 USD (results §5) | |
| 45018566846 / 45018572676 / 45018578869 | same + ecg-image-kit pages E0-E7: 2nd / 3rd / 4th place | 2nd: 3090 0.5 USD/h, 0.76 h, 0.38 USD; 3rd: 3090, 1.55 h, 0.78 USD; 4th: 4000 Ada, 2.0 h, 0.56 USD **lost**: 20 pages unreadable aborted its whole submission (now patched, `patch_4th.py`) | |
| 45028920152 | 4th place on the probe pages again, with the patch | 4000 Ada 0.28 USD/h, 2.0 h, 0.58 USD **lost again**: the patch worked (20 pages written as NaN) but our converter failed on the empty fields; `to_npz_sub.py` fixed | |
| 45052758197 | 4th place, third try (K0 + E0-E7 only) | 2.5 h | 0.70 USD |
| 45028959725 | Kaggle: waiting for the hybrid's score again | done: **hidden test 13.10 public / 13.39 private** (classical 12.67 / 13.03: +0.36 private) | free |
| 45029638951 / 45029645225 | Kaggle whole training set (stride 1), default / + `EXT_ON_D` + `ORIENT_PRIOR` (pairs with the PM whole run 45010034441) | 5 h each | free |
| 45003726790 / 45003732873 / 45003738548 | PM-ECG-ID, 130 ECGs (40 the default turned + the 45 brightness +40 and 45 digital), hybrid default: base / `EXT_ON_D` / `ORIENT_PRIOR` v1 (free runners, `ecgs` list) | done: base 12.26; EXT_ON_D **12.36** (ECG 15 +16.1; 2 pages −2.5/−1.7 that change the same way in the orientation run too: run-to-run noise); ORIENT_PRIOR v1 **12.63** (the 40 turned ECGs 0.92 → 2.14, turned ECGs 38 → 29; the other 90 17.31 → 17.29) | free |
| 45006493923 | same 130 ECGs, ORIENT_PRIOR v2 (never 90° for a sheet lying landscape: rows run along the long side) | done: 12.26 → **12.70**; the 40 turned ECGs 0.92 → 2.39 (ECGs with a turned page 38 → 11), the other 90 17.31 → 17.29 | free |
| 45006500371 / 45006529376 | Kaggle dev (stride 20 = 60 images here), ORIENT_PRIOR v2 / base | done: 14.61 → 14.61, no image changed (none of them turned) | free |
| 45010034441 | PM-ECG-ID whole database, default + `EXT_ON_D` + `ORIENT_PRIOR` (pairs with 44908014040) | done: 5,670 of 6,000 ECGs (94.5%): 10.49 → **10.53 dB** (+0.04), PCC 0.756 → 0.759, failed leads 9.1% → 9.0%, Kaggle metric 13.04 → 13.05, 60-scenario mean 10.56 → 10.60; contrast +0.10, padding +0.09, Samsung +0.27, blur ±0, rotation −0.03 | free |
| 45008970196 / 45008976316 | Kaggle, the 20 records with the most turned images in the whole run (61 of 8,352 turned; 52 portrait photos, 9 landscape photos turned 180° that read well, up to 20.9 dB): base / ORIENT_PRIOR v2 | done: 180 images, 17.92 → 17.92; the same 21 images turned in both (12 by 90°, 9 by 180°): the prior keeps the genuine turns | free |

## 7. Nov 25: new repository, precision on clean pages

**Lost with the old repository (blocked by GitHub on Nov 22, ~18:20 UTC).** Their results and
artifacts are unreachable: 45052758197 (4th place on the probe pages, third try; 4000 Ada, up to
2.5 h, ≤ 0.70 USD; the pod was to be terminated by hand), 45029638951 / 45029645225 (Kaggle whole
training set, default vs `EXT_ON_D` + `ORIENT_PRIOR`, free). Every model artifact of the old runs
went too; rowheavy3e and page v3 survive in the private Kaggle dataset `ecgdig-src`.

| Run | What | Result | Cost |
|---|---|---|---|
| 45687811192 | env-check, first try in fmenegui/ecg-classic | tests OK (50); RunPod and Kaggle failed: the secrets were not set yet | free |
| 45695258992 | `models-from-kaggle.yml` | row network (row.onnx, sha256 b85b6c8f…) and page v3 (sha256 0a0aa979…, identical to the released page v3) copied from `ecgdig-src` into this run's artifacts: use 45695258992 as row/page model run (7 days) | free |
| 45695253156 | env-check | all OK (tests, Kaggle archive from GitHub and from the pod, RunPod volume, proxy); balance 19.98 USD (MCP get-balance) | cents |
| local | ecg-image-kit default sheet E0 (20 pages, `probe_eik.py`), hybrid default (fuse) | 20/20, segment SNR 25.20 dB, PCC 0.995, Kaggle-style 26.38 (pod run on Nov 22: 25.17) | free |
| local | same, `row_mode=replace` (rowheavy3e everywhere) | 23.48 dB, Kaggle-style 24.56: the 10 px/mm row network is 1.7 dB less precise than the classical centreline on clean pages | free |
| 45696445289 | row network at 20 px/mm across (`--xup 2`), smoke, 10 min, volume | trains: RTX 3090 (EU-RO-1, on the volume), batch 16 of 512 x 1024, 0.29 s/step, data wait 2%, 10.9 GB; 1,700 steps. Killed by its own 0.7 h limit just before export: setup took 18 min and the PTB-XL bank was rebuilt (30 min, now cached as bank-26dd5bd22762) | ~1.0 h, ~0.4 USD |
| 45706904250 | Kaggle dev baseline in the new repository (rowheavy3e as kept in `ecgdig-src` + page v3 + photo_norm; free runners, stride 20) | 540 of 8,793 images (6.1%): **17.59 dB** official metric (holes filled 17.76), PM-style SNR 15.82, PCC 0.957, failed leads 0.6%. Holdout records (117) 17.64 vs 17.67 on Nov 21, the other records 17.58 vs 18.04: `ecgdig-src` holds the best-validation checkpoint (unet_best.pt), the Nov 21 runs used the final one (trace_seg.onnx), which had seen those records' real strips longer. The holdout is the comparable number | free |
| 45706509393 / 45706519680 | row network from scratch, synthetic strips only (rowgen + bank, heavy 0.3, down to 1/24, blur 10 px), 270 min, batch 16: `--xup 1` / `--xup 2` (the A/B for horizontal resolution) | RTX A5000 each (EU-RO-1, volume), 0.27 USD/h. x1: 107,464 steps (0.151 s/step, 5.5 GB); x2: 56,630 steps (0.286 s/step, 10.9 GB). Final validation column error: x1 0.28 px median, 1.42 px p90; x2 0.27 / 1.43 (x2 measured on its stretched columns) | 2 × 4.7 h, ~1.27 USD each |
| local | E0 (20 pages, 20/20 read), `row_mode=replace`: x1 / x2 (`row_xup=2`) | segment SNR 22.04 / **23.24** (+1.20), Kaggle-style 22.39 / 23.33; after the time and gain oracles 24.16 / 25.78 (+1.6). Both below rowheavy3e in replace mode (23.48) and the default fuse (25.20): from scratch on synthetic strips only they lack rowheavy3e's real and rendered strips, but at equal training 20 px/mm wins on clean pages | free |
| 45742421025 | Kaggle dev, x1 replace + page v3 (540 of 540) | vs the baseline 45706904250: all 17.59 → 16.50, holdout 17.64 → 15.95; 0001 −2.94, but 0003 +0.50, 0004 +0.82, 0006 +0.85, 0009 +0.40, 0011 +0.27, 0012 +0.79 | free |
| 45742413858 | Kaggle dev, x2 replace (`row_xup=2`) + page v3 | 5 of 20 runners died while reading (memory: twice the activations with 4 images at a time; `kaggle-full.yml jobs` added). On the 405 images in both (45 records, 4.6%): **x2 vs x1 +0.62 dB** (holdout +0.48), 0001 21.58 → 23.05 (+1.46), degraded types ±0.1. x2 vs the baseline: all 17.42 → 17.07, holdout 17.72 → 16.65; 0001 −1.28; 0003 +0.75, 0004 +0.77, 0006 +1.01, 0009 +0.36, 0010 +0.18, 0011 +0.62, 0012 +0.65 | free |
| 45744820526 | real strips with labels at 20 px/mm (`realstrips_build.py --xup 2`), smoke, 40 pages | 37 of 40 pages kept, 292 strips; cpu5c 8 vCPU, EU-RO-1 | ~0.3 h, ~0.1 USD |
| 45748479274 | real strips `--xup 2`, whole training set | 6,447 of 6,894 pages kept (93.5%, as the 10 px/mm set), 126 chunks, 2,926 s of labelling; in the volume's cache (key with `xup2`) | 32 vCPU, ~1.0 h, ~1.1 USD |
| 45748476389 | Kaggle dev, x2 in fuse mode (classical centreline on clean ink, the network elsewhere), 2 images at a time | 540 of 540: all 17.59 → **17.88 (+0.29)**, holes filled 17.76 → 18.16; holdout 17.64 → 17.18, other records 17.58 → 18.06. 0001 +0.48, 0004 +0.21, 0006 +0.34, 0009 +0.22, 0012 +0.31, 0005 −0.24. Median image change 0.00; the holdout loss is two records (3687073465, 3646789790) where one lead goes wrong (V4 15 → 2 dB, V3 20 → 9 dB) and drags every type of those records down 6-8 dB: the network, trained on synthetic strips only, takes a wrong trace there | free |
| 45753798950 | row network at 20 px/mm with 35% real strips labelled at 20 px/mm, lr 5e-4, 270 min, on the volume (**rowx2real**). Meant to continue x2, but `--init /workspace/in/unet_best.pt` pointed at the wrong folder (the files arrive in `$IN_DIR`, `/workspace/runs/<run>/in`): it trained from the ImageNet encoder, no init | RTX 4000 Ada (EU-RO-1, volume), 0.28 USD/h; 35,868 steps; 49,764 real strips in memory (806 for validation). Final validation column error: synthetic 0.29 px median, 1.53 px p90; real strips 0.27 / 1.26 (rowheavy3e at 10 px/mm: 0.29 / 1.32) | ~5.1 h, ~1.4 USD |
| local | E0, rowx2real replace (`row_xup=2`) | 20/20: segment SNR **24.49**, Kaggle-style 25.88 (x2 synthetic only 23.24; rowheavy3e replace 23.48; default fuse 25.20 / 26.38); after the time and gain oracles 26.79 | free |
| 45795774190 | Kaggle dev, rowx2real **replace** (`row_mode=replace row_xup=2`, 2 images at a time) + page v3 | 540 of 540 (6.1%): **17.59 → 18.13 dB (+0.54)**, holdout 17.64 → **17.88 (+0.24)**, other records +0.62; holes filled 17.76 → 18.25. Every type up: 0001 +0.08, 0003 +1.15, 0004 +1.30, 0005 +0.43, 0006 +1.11, 0009 +0.83, 0010 +0.71, 0011 +0.96, 0012 +1.53. Median image +0.75 dB, 83% of the images better; worst 3633367021-0001 −7.1, 35187032-0003/0011 −4.3. The two records x2 broke are fine now (−0.8 to +1.6) | free |
| 45795763846 | Kaggle dev, rowx2real **fuse** (`row_xup=2`) | 540 of 540: 17.59 → 17.94 (+0.35), holdout −0.03; 0001 +0.44, other types +0.02 to +0.49. Replace wins: with real strips the network at 20 px/mm is as precise as the classical centreline on clean renders and better everywhere else | free |
| local | E0, rowx2real fuse | 24.96 / Kaggle-style 26.20 (replace 24.49 / 25.88; default 25.20 / 26.38) | free |
| 45802288083 / 45802311804 | PM dev (5 per scenario, free runners): baseline (rowheavy3e + page v3 + photo_norm) / rowx2real replace, 1 image at a time | 300 of 6,000 ECGs (5%): **10.54 → 11.03 dB (+0.49)**, PCC 0.757 → 0.757, failed leads 9.2% → 9.2%, Kaggle metric on the PM leads 14.35 → 14.85; 60-scenario mean 10.54 → 11.03. JPEG +0.81, blur +0.72 (failed leads 28 → 18%), resolution +0.78, padding +1.01, perspective +0.69, rotation +0.66, contrast +0.54, brightness +0.33 (failed 8 → 12%), digital +0.18, doogee/samsung +0.91, crumbles +0.76; iPhone 17.00 → 14.53 (5 ECGs), screens −0.26. (pmcardio.yml converts the best checkpoint, kaggle-full.yml uses the final weights) | free |
| 45808822516 | Kaggle whole training set (stride 1), rowx2real replace, 2 images at a time (row strips read in tiles); baseline: the Nov 21 whole run 44902767154 (rowheavy3e final weights + page v3 + photo_norm, 8,352 images; copy kept locally) | 11 of 20 shards hit the 300-min job limit (2 images at a time is too slow for 440 images). On the 3,951 images in both (44.9%, 439 records): official metric **18.31 → 18.30 (−0.01)**, holdout (918) 18.37 → 18.39; 0001 25.51 → 24.96 (−0.55); every degraded type up: 0003 +0.67, 0004 +0.84, 0005 +0.15, 0006 +0.89, 0009 +0.57, 0010 +0.33, 0011 +0.64, 0012 +0.97. The linear mean follows the clean renders, where the network is still 0.5 dB below the classical centreline (the baseline's final weights read 0001 1 dB better than the best-checkpoint baseline of the dev run) | free |
| 45808832833 | PM-ECG-ID whole database, rowx2real replace with `EXT_ON_D` + `ORIENT_PRIOR`; baseline: the Nov 22 whole run 45010034441 (same flags, rowheavy3e best checkpoint; copy kept locally), so the pair differs only in the row network | 5,116 of 6,000 ECGs in both (85.3%): **10.50 → 11.38 dB (+0.88)**, PCC 0.764 → 0.771, failed leads 8.5% → 8.3%, Kaggle metric on the PM leads 13.15 → 13.48; **60-scenario mean 10.67 → 11.52** (PMcardio 14.48). Every group up: contrast +1.13, rotation +1.16, brightness +1.00, JPEG +0.93, padding +0.93, resolution +0.87, digital +0.72, perspective +0.68, blur +0.43, phones +0.6-0.9, scans +0.70, screens +0.57, bent +0.43, crumpled +0.38 | free |

**Reading.** At equal training, 20 px/mm across beats 10 px/mm on clean renders (+1.5 dB on 0001,
+1.2 on E0) and ties elsewhere. Trained on synthetic strips only and used everywhere, the network
already beats the default on every degraded Kaggle type (+0.2 to +1.0 dB, photos +1.0) but loses
1.3 dB on clean renders, where the classical centreline is still the most precise reading, and the
linear mean follows the clean renders. Next: x2 in fuse mode (running), then x2 continued with the
real strips labelled at 20 px/mm.

**Where the error is on clean pages (`benchmarks/topsol/decompose_probe.py`, cumulative oracles,
segment dB / Kaggle-style).**

| Stage | E0, fuse | E0, replace | our synthetic pages (15, no degradation), fuse |
|---|---|---|---|
| reading | 25.20 / 26.38 | 23.48 / 24.56 | 20.88 / 22.71 |
| + sub-sample time shift | 26.91 / 28.38 | 24.36 / 25.57 | 21.47 / 23.42 |
| + time scale | 27.08 / 28.48 | 24.51 / 25.67 | 21.68 / 23.60 |
| + gain | 27.32 / 28.74 | 25.23 / 26.42 | 22.22 / 24.03 |
| ceiling, truth every 4 ms (10 px/mm) | 31.07 / 32.83 | | 27.39 / 29.37 |
| ceiling, every 2 ms (20 px/mm) | 34.00 / 35.80 | | 30.39 / 32.38 |

- On ecg-image-kit pages our reading lags the truth by a constant ~0.2 samples (0.4 ms, 0.1 px at
  10 px/mm) on almost every segment of every page. One constant shift of the whole reading:
  E0 25.20 → 26.45 (+1.25; Kaggle-style +1.63); Kaggle 0001 (5 local records) 26.05 → 27.10
  (+1.05); 0003 +0.15, 0011 +0.19. On our own cv2-drawn synthetic pages there is no such lag
  (best shift 0-0.05 samples): it belongs to the generator's drawing (anti-aliasing profile of
  the line start, likely), not to our estimator. A per-generator constant is a calibration to
  the Kaggle sheet, so it is not applied; worth about +0.65 dB on the Kaggle development score.
- After every oracle the shape error remains: 3.8 dB below the 10 px/mm ceiling, and the
  20 px/mm ceiling is 3 dB higher. Hence the row network at 20 px/mm (`--xup 2`).

**PM-ECG-ID: structure or trace? (Nov 25, no new compute; the whole run 45010034441, 5,908 images,
kept locally).** Per scenario, single pages whose layout was read as printed vs not:

| Scenario | SNR | layout right | SNR when right | SNR when wrong | failed leads | PCC when right |
|---|---|---|---|---|---|---|
| photos_crumbles | −1.23 | 37% | 0.61 | −2.09 | 25% | 0.554 |
| blur 31 | −0.50 | 49% | −0.30 | −0.68 | 32% | 0.433 |
| blur 41 | −0.50 | 31% | −0.56 | −0.43 | 45% | 0.342 |
| resolution 1/32 | −0.18 | 27% | −0.20 | −0.15 | 50% | 0.275 |
| resolution 1/16 | 0.75 | 55% | 1.30 | 0.09 | 23% | 0.484 |
| photos_bents | 0.81 | 52% | 3.14 | −1.71 | 15% | 0.589 |
| blur 21 | 1.21 | 55% | 2.25 | −0.02 | 27% | 0.596 |
| brightness −160 | 3.06 | 57% | 7.09 | −1.64 | 16% | 0.726 |
| resolution 1/8 | 3.31 | 80% | 4.27 | 0.12 | 12% | 0.687 |
| perspective 40 | 5.18 | 79% | 6.81 | −0.26 | 9% | 0.746 |
| brightness +160 | 6.22 | 70% | 8.51 | 0.81 | 15% | 0.771 |
| JPEG 2% | 6.88 | 77% | 9.16 | 0.51 | 2% | 0.857 |
| padding extra large | 7.40 | 71% | 10.31 | −0.19 | 16% | 0.866 |
| opacity 0.15 | 8.43 | 64% | 13.50 | −0.32 | 29% | 0.860 |
| opacity 0.2 | 9.20 | 65% | 14.36 | 0.48 | 26% | 0.873 |

- If every single page had its printed layout, at the SNR the scenario reaches when it does:
  60-scenario mean 10.59 → 11.86 (+1.3; hard 19: 4.46 → 6.20, middle 29: 12.26 → 13.34, easy 12:
  16.25 → 17.25). Structure is worth ~1.3 dB, mostly from the middle scenarios (opacity,
  brightness, padding, JPEG, perspective: 20-45% wrong layouts).
- The other ~2.6 dB to PMcardio is the reading on degraded pages even with the right layout: on
  blur ≥ 21, resolution ≤ 1/16 and crumpled or bent photos it stays at −0.6 to +3 dB (PCC
  0.3-0.6). The trace on degraded images is the shared lever of both benchmarks.
- **The page-network selection loses structure it had (same run).** On 5,415 single pages with
  both readings, 1,329 have exactly one of them with the printed layout: the classical one on 738,
  the page network's on 591. The current choice (`_reading_score`, internal quality + limb
  closure) is right on 956 (72%); shifting its margin moves that by at most ±30 pages (best
  961 at −0.1). An oracle choice would fix the other 373 pages (an upper bound of ~+0.6 dB on the
  whole database). The two scores barely separate the cases (score difference when the network
  is right: median −0.09, IQR −0.41..0.20; when the classical one is: 0.46, 0.03..0.83), so a
  better selector needs other evidence. It cannot be fitted on PM-ECG-ID (no training on the
  benchmark): synthetic pages of every layout, read both ways, would be the calibration set.

## 8. Nov 26: the four steps (owner's go, data kept as it is plus the new sources)

1. Strips with exact labels from ecg-image-kit pages (`hybrid/eikstrips_build.py`): matplotlib's
   pixel convention checked on the ink (the polyline at plotted_pixels − 0.5 overlaps the drawn
   trace best, correlation 0.714 vs 0.634 at 0 or 1 px); 1.2 s per page on one core; 12 test
   pages gave 114 strips, labels on the trace centre (visual check).
2. A larger encoder on the same data (next, after the smoke test of step 1).
3. Learned fine time and gain (design: a small network for the row start's sub-pixel position,
   trained on exact labels of several renderers).
4. The page-reading choice learned on synthetic pages (`benchmarks/selector`, `selector.yml`):
   PTB-XL signals, printed layouts, one generic degradation per page; both readings scored.

| Run | What | Result | Cost |
|---|---|---|---|
| 45815960826 | step 1 smoke: 200 ecg-image-kit pages + real strips + synthetic, 5 min | runs: 200 of 200 pages, 1,842 strips in 113 s; 49,764 real strips; trained and exported. The init did not load (same `$IN_DIR` mistake) | ~0.8 h, ~0.3 USD |
| 45817104093 | step 4: 600 synthetic pages read both ways (rowx2real replace + page v3), 10 shards | the two readings disagree much less than on PM-ECG-ID: classical layout right on 545 of 600, both right on 507, exactly one right on 62. Mean page SNR: current rule 16.67, fitted rule (5-fold CV) 16.68, oracle 16.75. On these pages the choice is already almost optimal; PM-ECG-ID's disagreements (1,329 of 5,415 pages) come from capture effects these pages lack. **Not adopted**; a calibration set needs photographed or photo-rendered pages | free |
| 45825904025 | step 3: row-start network, 6,000 pages (60% ecg-image-kit, 40% cv2), 30 epochs | validation error median **0.018 px**, p90 0.051 px, signed mean 0.000 (target: < 0.05 px) | ~1 h, ~0.3 USD |
| local | step 3: row-start network in the default (fuse) reader, `start_model` | E0 (20/20) 25.20 → **25.41** (+0.21), N0 (8/8, no pulse) 22.87 → **23.71** (+0.84). Its offsets go the right way (E0 +0.16..+0.23 px, N0 −0.05..−0.09 px) but not the right size: the best remaining constant shift is now −0.1 to −0.15 samples on both (E0 overcorrected, N0 undercorrected). Trained on resized pages, used on our rectified ones: the next version should take its crops from our own rectification of pages of known geometry | free |
| 45862574049 / 45862578574 | Kaggle dev / PM dev: rowx2real replace + row-start network (vs rowx2real replace without it) | Kaggle 540 of 540: 18.13 → **18.25 (+0.12)**, holdout 17.88 → 18.20 (+0.32); 0010 +0.40, 0011 +0.33, 0005 +0.29, 0001 +0.07, 0012 −0.13. PM 300 ECGs: SNR 11.03 → 11.07 (+0.04), PCC 0.757 → 0.759, failed leads 9.2 → 9.1%, but the Kaggle metric on the PM leads 14.85 → 13.39 (digital pages 19.72 → 16.63: a few clean digital pages read worse, which the linear mean amplifies). Not adopted yet: next version trained on crops of our own rectification | free |
| 45863126769 / 45863131345 | LITFL (102 real pages, no signal truth; structure against the hand-reviewed reference, internal consistency): default (rowheavy3e + page v3) / rowx2real replace | 102 of 102 read both times. Layout 92.2% → 91.2% (one page, 0003, 3x4+1 → 3x4: the page-network choice moved with the readings' scores), rhythm strips 92.9 → 91.9%, gains 97 → 95%, everything right 88.2 → 87.3%; coverage median 0.999 → 0.992; Einthoven residual median 0.350 → **0.328** (−6%), rhythm-vs-grid II NRMSE 0.338 → 0.335; 34 → 29 s per page | free |
| 45859972552 | step 1: rowx2real continued (`--init $IN_DIR/unet_best.pt`, lr 3e-4) with 35% real strips, 25% ecg-image-kit strips (20,000 pages), 40% synthetic, 240 min | (running) | ≤ 5.6 h × 0.5 USD |
| 45859986112 | step 2: resnet50 encoder (ImageNet), same data mix (its own 20,000 ecg-image-kit pages, seed 1), batch 12, 240 min | **lost**, same as 45859972552: RTX 4000 Ada 0.28 USD/h, 5.7 h; strips built to 11,230 pages at GPU 0%, then silent; no result archive | ~1.6 USD |
| 45817104093 | step 4: 600 synthetic pages read both ways with rowx2real replace + page v3, 10 shards | (queued) | free |
| local | step 3 diagnosis: 8 ecg-image-kit pages without the calibration pulse (N0), hybrid default | 22.87 dB; best constant shift **−0.2 samples** (+1.42 dB), the opposite of E0 with the pulse (+0.2): the origin from the start edge is biased by how the line starts (pulse joined or not, cap style), constant along the row (median lag per column 0.15-0.22 samples on E0) | free |
| local | step 3: row-start crops (2,500 pages: 1,491 ecg-image-kit, 1,009 cv2; 63,572 crops), first epoch on CPU | validation error median 0.27 px, p90 0.56 (signed −0.23): far from the 0.05 px needed; 30 min per epoch on this CPU, so training moved to a GPU | free |
| 45899861399 | env-check before an Open-ECG-Digitizer probe run | tests OK, Kaggle archive OK, balance 11.21 USD; **FAIL**: the 2-vCPU pod on the volume (EU-RO-1, 0.07 USD/h) never answered through the proxy in 50 min; terminated. RunPod batch on hold | ~0.06 USD |
| local | ecgdig **classical** (no network, default Config) on the 300 probe pages (K0-K6, E0-E7, same 20 fold-10 records as the winners' runs), `score_probe.py --ecgdig` | 300 of 300 read; mean of the 15 variants 21.04 dB (hybrid of Nov 22: 21.59; 2nd place 6.62; 3rd 5.26). K0 20.39, K1 19.69, K2 21.48, K3 20.53, K4 20.95, K5 19.83, K6 21.28; E0 24.87, E1 25.74, E2 21.52, E3 22.07, E4 22.50, E5 6.63 (45% segments failed), E6 23.32, E7 24.77 | free |
| 45904950943 | PM dev, ecgdig **classical** (no network), 5 per scenario | 300 of 6,000 ECGs (5%): SNR **9.41 dB**, failed leads 13.2% (hybrid rowx2real replace on the same ECGs: 11.03, 9.2%; Open-ECG-Digitizer 4.41, 24.7%) | free |
| 45906947336 | env-check again | **FAIL** again: the pod on the volume (EU-RO-1) did not answer through the proxy | ~0.06 USD |
| 45908147885 | dc-probe EU-RO-1, EU-CZ-1, US-KS-2, EUR-IS-1 (2-vCPU pods, no volume) | EU-CZ-1, US-KS-2, EUR-IS-1 answer and reach the Kaggle archive; **EU-RO-1 does not, even without the volume**: the outage is the data center, not the volume. `gpu-train.yml data_centers` added (d85c969): pods without the volume only in the listed data centers | cents |
| 45909192221 | Open-ECG-Digitizer on the probe pages, smoke (`TS_PROBE=1 TS_PROBE_N=2`, data centers EU-CZ-1, US-KS-2, EUR-IS-1) | 30 of 30 pages read and scored in 173 s; runs end to end | 3090 class, 0.5 USD/h, ~0.2 h, ~0.1 USD |
| 45910877080 | Open-ECG-Digitizer on the 300 probe pages (same 20 fold-10 records) | 300 of 300 read (1,686 s). Mean of the 15 variants **10.83 dB** (classical 21.04, hybrid of Nov 22 21.59, 2nd place 6.62). K0 14.07, K1 13.90, K2 12.16, K3 13.78, **K4 1.83** (no lead names, 62% segments failed), K5 13.31, K6 13.84; E0 14.57, **E1 0.65** (no lead names, 72%), E2 10.31, E3 14.71, E4 13.95, **E5 −0.64** (6x2, 52%), E6 12.12, E7 13.86. Template-free but needs lead names, and less precise than ours on clean pages | RTX A4000, 0.25 USD/h, 0.7 h, ~0.18 USD |
| 45918125712 | env-check with the data volume back on ecgdig-data (EUR-IS-1) | the mounted pod did not answer in 26 min (as on Nov 19-20): cancelled, pod terminated | cents |
| 45920513959 | env-check with a new data volume **ecgdig-data3** (40 GB, EU-CZ-1, id ibxoftvp2b; `runpod_v2.py` default now) | all OK: tests, Kaggle archive from GitHub and from the pod, volume writable, pod answered through the proxy | cents |
| 45921054335 | strip building smoke on 8 vCPU | the PTB-XL bank build was killed (exit 137, out of memory) | 8 vCPU, ~0.2 h, ~0.06 USD |
| 45922393687 | strip building smoke on 32 vCPU (bank, 100 ecg-image-kit pages, 40 real pages) | runs: bank cached (bank-26dd5bd22762), 100 eik pages in 12 s, 37 of 40 real pages kept | 32 vCPU, 1.12 USD/h, ~0.2 h, ~0.2 USD |
| 45923470743 | strips into ecgdig-data3: 20,000 ecg-image-kit pages (seed 0, xup 2) + real Kaggle strips at 20 px/mm (7,000 pages) | eik 20,000 of 20,000 pages in 1,658 s (26 GB, eik-d24e5423a038); real strips 6,894 pages, 4,400 s (5.3 GB, real-2c1233c68a07); volume 34 of 40 GB used | 32 vCPU, 1.12 USD/h, 1.7 h, ~1.9 USD |
| 45933739279 | step 1 smoke on the volume, 8 min (init rowx2real 45753798950) | caches hit (bank, eik, real), init loaded (all keys), 60,000 eik + 49,764 real strips in memory, 0.6 s/step, data wait 2%, 10.9 GB; VAL at 811 steps: col error 0.29 / 1.55 px, real 0.28 / 1.29, eik strips 0.16 / 0.58; ONNX exported | RTX 3090 (EU-CZ-1), 0.5 USD/h, ~0.4 h, ~0.2 USD |
| 45935753835 | **step 1 again**: rowx2real continued (lr 3e-4) with 35% real, 25% eik (20,000 pages from the cache), 40% synthetic, 220 min, on ecgdig-data3 | done: RTX 3090 (EU-CZ-1, volume), 0.5 USD/h; 43,196 steps; final validation column error synthetic 0.28 / 1.47 px (median / p90), real strips 0.27 / 1.25 (rowx2real 0.27 / 1.26), ecg-image-kit strips **0.16 / 0.42**. Dev benchmarks: Kaggle 45963250417, PM 45963261022 (free) | 3.75 h, ~1.9 USD |
| 45935763432 | **step 2 again**: resnet50 (ImageNet), same data and the **same** eik strips (seed 0, so the A/B differs only in the encoder; the lost run used seed 1), batch 12, lr 1e-3, 220 min | done: RTX 3090 (EU-CZ-1, volume), 0.5 USD/h; 41,334 steps; final validation synthetic 0.30 / 1.59 px, real 0.29 / 1.31, ecg-image-kit 0.16 / 0.43: worse than the resnet34 run on synthetic and real strips at the same time budget (fewer steps, from ImageNet instead of rowx2real). Dev benchmarks: Kaggle 45963271451, PM 45963281803 (free) | 3.75 h, ~1.9 USD |
| 45963250417 / 45963261022 | **step 1** (45935753835) dev benchmarks, `row_mode=replace row_xup=2` + page v3, vs rowx2real replace (45795774190 / 45802311804) | Kaggle dev 540 of 540 (6.1%): **18.13 → 19.46 dB (+1.33)**, holdout 17.88 → **18.97 (+1.09)**, holes filled 18.25 → 19.73; 0001 24.57 → **26.75 (+2.17)**, every degraded type +0.18 to +0.47. PM dev 300 ECGs (5%): **11.03 → 11.42 dB (+0.39)**, PCC 0.757 → 0.761, failed leads 9.2 → 9.5%, 60-scenario mean 11.03 → 11.42; Kaggle metric on the PM leads 14.85 → 13.46 (digital 19.72 → 16.35: as with the row-start network, a few clean digital pages read worse and the linear mean amplifies them). By group: digital +0.79, brightness +0.59, iPhone +1.59, scans +0.76, JPEG/contrast/rotation +0.4; blur −0.45 (failed leads 17.5 → 24.6%), screens −0.49. **Candidate**: whole-database pair next | free |
| 45963271451 / 45963281803 | **step 2** (resnet50, 45935763432) dev benchmarks, same setup | Kaggle dev 540: 18.13 → 19.14 (+1.01), holdout 17.88 → 18.65 (+0.77); 0001 +2.00 but every degraded type −0.10 to −0.28. PM dev 300: 11.03 → **10.63 (−0.41)**; screens −2.74, iPhone −1.33, crumpled −1.08, resolution −0.88. **Not adopted**: at the same time budget, a resnet50 from ImageNet loses to the resnet34 continued from rowx2real; the gain of both on clean pages comes from the exact ecg-image-kit labels, not the encoder | free |
| local | E0 (20 pages), step 1 replace (`row_xup=2`) vs rowx2real replace | 20/20: segment SNR 24.49 → **26.67 (+2.18)**, Kaggle-style 25.88 → **28.19**, PCC 0.994 → 0.995. First time above the default fuse reading (25.20 / 26.38) and close to the 2nd/3rd place on their own sheet (27-30) | free |
