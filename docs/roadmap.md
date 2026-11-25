# Roadmap (from Nov 20, 2026)

**Goal.** Be the best generic ECG-image digitizer:

- beat PMcardio on PM-ECG-ID (14.48 dB, 60-subset mean of its paper; 14.05 dB, PCC 0.885,
  6.62% failed leads in its supplementary table);
- beat the Kaggle leaders (22.6-23.3 dB private leaderboard) without a template of the Kaggle
  sheet.

**Where we are.**

- PM-ECG-ID: 9.71 dB, whole database.
- Kaggle training records: 18.12 dB (whole set). The Kaggle dev holdout is at 17.67 with the best row
  network.
- Classical only on the hidden test: 13.03 dB.

**What the measurements say.**

- **Kaggle:** the remaining error is the **shape of the trace on degraded images**. Clean renders
  are at 25 dB; the other types are at 12-16 dB, and 15-18.5 dB even with every oracle. Layout and
  time errors are small (≤ 0.45 dB per type, geometric part).
- **PM-ECG-ID:** **structure** (13% of single-page ECGs have the wrong row count, 12.9% failed
  leads) as well as the trace.

## Update (Nov 25): change of level, not more tuning

The repository moved to `fmenegui/ecg-classic` (the old one was blocked by GitHub on Nov 22 for
an unstated terms-of-service reason). Where we stand:

| | ours | target | gap |
|---|---|---|---|
| PM-ECG-ID whole database (5,670 of 6,000 ECGs, 94.5%) | 10.60 dB 60-scenario mean (SNR 10.53, Kaggle metric 13.05) | PMcardio 14.48 | −3.9 dB |
| Kaggle hidden test (private) | 13.39 | 1st place 23.33 | −10 dB |
| Kaggle training images (8,352 of 8,793, 95%) | 18.30 | ~27 (1st place: CV 26.8 → private 22.75) | −9 dB |

**The two metrics reward opposite things.**

- Kaggle averages linear SNR, so the best images dominate. Clean renders (0001) alone at 30 dB
  instead of 25 would take the dev score from 17.95 to 21.4; at 33 dB with +4 dB on the degraded
  types, 24.6. Our 0001 (25.2 dB) sits at the ceiling of a one-value-per-column reading at
  10 px/mm (26.6 dB; 32.9 at 20 px/mm). On ecg-image-kit's own clean sheet the winners read 30 dB,
  we read 25. The Kaggle gap is the representation of the reading, not the page structure.
- PM-ECG-ID averages dB with failed leads at 0, so the worst images dominate. On the 12 easy
  scenarios we are level with PMcardio (16.2 vs 16.7); on the 19 hard ones 4.5 vs 11.1 (~2.1 dB of
  the mean), on the 29 middle ones 12.2 vs 15.8 (~1.7 dB). Training on PM's own captures gave
  +0.35 dB and giving the reader the format +0.1 dB: the gap is geometry, rows and layout on
  degraded pages, and the trace on blur.

**Common cause.** Our real training labels come from our own reader and only where it is already
right (66% of PM pages kept, Kaggle strips kept at correlation ≥ 0.90), so the networks never
see the hard cases and inherit our geometry errors (rowheavy4, 2.5× the real strips: ±0). The last
round of classical flags gave +0.04 dB on the whole PM database; rowheavy5 +0.25 PM, 0 Kaggle.

**Plan.**

1. **Exact labels that do not depend on our reader** (the foundation):
   - synthetic pages of every layout and style with the generic degradations PM-ECG-ID uses (blur,
     low resolution, opacity, contrast, perspective, folds); labels: sub-pixel signal per column,
     lead instances, grid coordinates. ecg-image-kit stores the exact pixel of every plotted
     sample (`plotted_pixels` with `store_configs=2`); ecg-torch-renderer and `hybrid/rowgen.py`
     know theirs;
   - Kaggle training images: fit their known sheet offline, only to build labels (the reader stays
     template-free), so every image gets a label, including those we read badly today; PTB-XL's
     original 500 Hz signals as the truth (+0.8 dB for the 2nd place);
   - real photos of the print kit (`kit/`, ArUco markers), labelled by the markers.
2. **Trace network v2** (the Kaggle lever): 20-25 px/mm horizontally from the original image,
   sparse sub-pixel target (already) with a centroid or parabolic decoder, FFT resampling, context
   across leads, flip TTA. Target: 0001 25 → 30-33 dB; Kaggle training set 18 → 21-24.
3. **Learned page structure** (the PM lever): grid coordinates per pixel (a generic UV map, no
   template), row and lead instances, calibration pulses; the classical stages stay as verifier
   (physics, OCR) and fallback. Target: hard scenarios 4.5 → 8+ dB (~+1.5 dB on the mean).
4. Cheap: a GPU Kaggle kernel (the CPU kernel ran the networks on part of the hidden images only);
   both time conventions on multi-column pages (ecg-image-kit's 6x2 shows 0-5 s in both columns);
   zero Kaggle leads of low quality (0 dB beats a negative SNR).

Stopped: classical flag tuning, more real strips of the same kind, dewarp.

Infrastructure: fewer parallel free-runner shards (a possible, unconfirmed, reason for the block);
whole-database runs on fewer shards or on RunPod CPU pods. Kaggle-trained weights stay private
(the private Kaggle dataset `ecgdig-src` holds the row network and page v3 used by the
submission).

## Update (Nov 22): what PM-ECG-ID says after the per-ECG analysis

PM-ECG-ID is 100 ECGs × 60 scenarios, so one ECG's failure repeats in up to 60 scenarios. On the
whole default run (10.49 dB) the gap to PMcardio splits into ~2.1 dB from the 19 hardest scenarios
and ~1.7 dB from the 29 middle ones. Measured on local subsets (experiment log, §5):

- **Horizontal geometry from the grid is a small lever** (~+0.15 dB on photos beyond what a
  per-lead time-scale oracle also adds on flat digital pages).
- **The structure stage is fragile.** Small geometric changes swing single digital pages by
  ±7-20 dB (layout lost, rows mixed). Per-ECG structural failures dominate: a faint row start,
  columns with their own row heights, columns read short, wrong orientation (183 of 6,199 pages
  turned although none is), wrong rows on blur and low resolution.
- So the order becomes: (1) structure robustness (orientation, row starts, per-column rows,
  layout), each fix behind a flag, checked on local subsets and then together in one
  whole-database paired run; (2) rows and trace on blur and low resolution; (3) the print kit
  (`kit/`) for real photos with exact geometry, to train and to measure geometry and structure.

## Order of work

### 1. Measure against the winners on the same images

- Run the 2nd place (hengck23-based; 3rd uses the same runner) on:
  - the Kaggle dev set (in-sample for them: optimistic);
  - PM-ECG-ID dev;
  - every PM-ECG-ID image printed like the Kaggle sheet (`3x4+1R`, one page; ~28% of the database).
- Compare per Kaggle type and per PM scenario with the classical reader and the hybrid.
- Then the 4th place (its own runner) and the 5th (Docker).
- Submit the hybrid to the hidden Kaggle test, to know our real distance to 23.33 dB.

### 2. Row network: precision of the trace (largest Kaggle lever)

Each item is a separate experiment, judged on the development leaderboard:

1. **Horizontal resolution.**
   - Classical at 10 vs 20 px/mm is running.
   - If 20 px/mm wins, train the row network at 20 px/mm horizontally, read from the original image.
   - The simulated ceiling of a column reading rises from 26.6 to 32.9 dB, and the 3rd place found
     the same.
2. **Sub-pixel target and decoder.**
   - The target splits the trace between two pixels, the loss weighs positives ×20, and the decoder
     reads probabilities as weights, which may bias the position.
   - Test a sparse target with a matching decoder (2nd place: ≤ 2 px per column, weighted mean), or
     parabolic interpolation around the argmax (3rd place), against the current setup.
3. **Real strips at scale.**
   - The ~6,900 real training pages are now built once on the volume.
   - Train rowheavy4 on them, with checkpoint selection by trace error (already changed).
4. **Unbiased real strips.**
   - A real strip is kept only if the classical reading correlates ≥ 0.90 with the truth, which
     favours strips the classical reader already reads.
   - Label strips from the truth and the known geometry instead, so hard strips are kept.
5. **Context across leads.** A model that sees the simultaneous rows (2nd place's 2.5D series model)
   to resolve crossings and faint segments, respecting which segments share time in a generic
   layout.

### 3. PM-ECG-ID structure

- **Failures:** cut the 12.9% failed leads. Hard cases are folded, crumpled and overlapping
  traces.
- **Independent labels:** rows and leads labelled independently of the classical reader, for
  training.
- **Page network v3 with selection:** confirm on the whole database.
- **Rows that cross:** joint tracking.

### 4. Photos

- **Dewarp:** continue until the local error is < 1 mm, and judge it on PM photos (bent, crumpled,
  phone) only.
- **Horizontal rectification by the grid:** low priority. The geometric gain is ≤ 0.45 dB per
  Kaggle type.

### 5. Clinical validity

On patients held out from training, with clinical measures besides SNR:

- ST level error (µV);
- QRS duration (ms);
- R amplitude;
- lead swaps.

The Open-ECG-Digitizer (Akershus) is a reference to compare against.

## Infrastructure

- **Private home for Kaggle-trained weights** (a private repository or bucket). Today they expire
  with the run artifacts; rowheavy3e expires around Dec 20.
- **Old volume:** delete `ecgdig-data` (EUR-IS-1) once its rowheavy3d checkpoint is recovered or
  given up.
- **Live pod id for `gpu-train.yml`:** split start and follow, so a training's log can be read while
  it runs.
- **Results:** keep `docs/leaderboard.md`, `docs/experiments-log.md` and `docs/experiments.csv`
  current after every experiment.
