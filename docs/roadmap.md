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
