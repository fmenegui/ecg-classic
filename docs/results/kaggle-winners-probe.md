# Why the Kaggle winners do not read ordinary ECG pages (Nov 22, 2026)

The top PhysioNet/Kaggle 2025 solutions score 22.6-23.3 dB on the hidden test, yet 0-1 of 300
PM-ECG-ID pages (printed PTB-XL ECGs, photographed). This note finds out why, with their own
public code and pages made to change one thing at a time.

## What their code assumes

**hengck23 pipeline (1st, 2nd and 3rd place build on it).**

1. Stage 0: a network finds nine lead-name keypoints (aVR, aVL, aVF, V1-V6) and a homography
   maps them onto their positions in ONE Kaggle training image (640106434-0001,
   `stage0_common.make_ref_point`). No keypoints, no page.
2. Stage 1: a network finds a 44 x 57 lattice of 5 mm grid points that must cover the whole page;
   the lattice is mapped onto that same image's lattice (`stage1_common.rectify_image`).
3. Stage 2: the signal is read at fixed places of the 2200 x 1700 result: zero mV at
   y = 703.5 / 987.5 / 1271.5 / 1531.5 px, 79 px/mV, 10 s from x = 118 to 2080 px split into four
   2.5 s columns (3rd place `configs/stage2.py`). Only 3x4 + lead II rhythm exists.

**4th place (tomoon33).** A network classifies the absolute index of every grid line of the
Kaggle sheet, the page is warped onto a fixed 1700 x 2200 frame, the top 20% is cut off and four
fixed bands are read.

So every one of them reads the Kaggle sheet as a template: geometry, layout, and (stage 0) the
look of its lead names.

## Probe pages

20 PTB-XL records from strat fold 10 (no PM-ECG-ID ECG and no record of a PM-ECG-ID patient),
200 dpi, 2200 x 1700 like the Kaggle images. Each variant changes one thing.

- **K pages** (`benchmarks/topsol/probe_pages.py`): drawn by us in the exact template geometry
  (rows, scale, time span, labels at the reference keypoints, full-page grid on the reference
  lattice). K1-K6 change one thing from K0.
- **E pages** (`benchmarks/topsol/probe_eik.py`): drawn by ecg-image-kit, the generator of the
  competition's images, with its default sheet (no augmentation). E1-E7 change one thing from E0.

Scoring (`benchmarks/topsol/score_probe.py`): every printed segment is compared with the reading
over the same time window (shift up to ±0.5 s, vertical offset removed), PMcardio-style SNR, a
segment with no reading counts 0 dB. Runs: RunPod GPU pods, `TS_PROBE=1` in
`benchmarks/topsol/run_hengck.sh` / `run_4th.sh`; ours (rowheavy3e + page v3 + photo_norm, the
default) on CPU.

## Results (SNR, dB; "0/20" = no page read)

| Variant | 2nd place | 3rd place | 4th place | ours |
|---|---|---|---|---|
| K0 template geometry, our drawing | 0/20 | 0.00 | (running) | **20.85** |
| K1 rows 25 mm apart (PM-ECG-ID's 3x4+1R) | 0/20 | 0.00 | | 20.52 |
| K2 grid only in a box | 2/20, −0.98 | −1.60 | | 22.08 |
| K3 page 8% smaller | 0/20 | 0.00 | | 20.84 |
| K4 no lead names | 0/20 | 0.00 | | 21.09 |
| K5 6x2 | 16/20, −6.55 | −9.46 | | 20.47 |
| K6 our print-kit page (A4) | 0/20 | 0.00 | | 21.93 |
| **E0 ecg-image-kit default sheet** | **29.86** | **27.11** | (running) | 25.17 |
| E1 no lead names | 0/20 | 0.00 | | 25.74 |
| E2 blue grid | −9.47 | −16.65 | | 21.57 |
| E3 page 8% smaller | 28.39 | 27.18 | | 22.65 |
| E4 rotated 3°, JPEG 70 | 28.33 | 27.44 | | 22.72 |
| E5 6x2 (both columns 0-5 s) | 1/20, −0.22 | −0.43 | | 9.24 |
| E6 rows 30% closer | 0/20 | 0.00 | | 23.87 |
| E7 top 12% of the page cut off | 29.91 | 25.29 | | 25.09 |

## What it shows

- **Inside the template they are excellent and do generalise to unseen ECGs:** ~27-30 dB on the
  generator's own sheet, and the homography absorbs scale, rotation, JPEG and a cut page.
- **Each of these alone breaks them completely:**
  - lead names missing, or drawn in another font or position (K0-K6, E1): stage 0 finds no
    keypoints and nothing follows;
  - a grid colour they did not train on (E2: −9 to −17 dB);
  - any layout or row spacing other than the Kaggle 3x4 + II (E5, E6, K1, K5).
- **PM-ECG-ID pages have at least one of these** (own fonts and label places, grids in a box,
  25 mm rows, 6x2 and 12x1 formats), hence 1 of 300 read.
- **Ours reads every variant**, at 20.5-25.7 dB, except ecg-image-kit's 6x2 (9.24 dB), where both
  columns show the same 0-5 s while our reader (and PM-ECG-ID's 6x2) takes the second column as
  5-10 s. Devices differ on this; a reader should try both conventions on multi-column pages.
- **On the clean generator sheet we are 2-5 dB below the template readers.** Once the page
  structure is known exactly, their trace reading is more precise than ours: that gap is trace
  precision on clean pages, not structure.

## Cost

2nd place 0.33 + 0.76 h, 3rd 0.78 + 1.55 h, 4th 2.0 + 2.0 h (both lost to a solution abort and
then a converter bug, fixed) + the third run; cards 0.27-0.5 USD/h; ~2.8 USD in total so far.
