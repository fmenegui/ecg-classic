# Plan: format and structure first (from 16/11/2026)

**Goal.** Beat the published PMcardio numbers on the whole PM-ECG-ID database and then the Kaggle PhysioNet leaders, with a classical method only (the hybrid waits).

| Target | Now (`43cb3a6`) | Goal |
|---|---|---|
| PM-ECG-ID, whole database: SNR / PCC / failed leads | 7.91 dB / 0.656 / 15.3% → **8.51 / 0.686 / 13.6%** (`2c941fa`) | **> 12.5 dB / > 0.91 / < 6.6%** (PMcardio) |
| PhysioNet (Kaggle) training images, 135 local | 14.68 → 14.70 dB | > 19.7 dB (24th place), then ~22.7 dB (top reports) |

**Where the gap is.** Format read wrong on 27% of single-page ECGs costs ~3.2 dB: ~1.6 dB in normal scenarios (structural; 10 ECGs account for 304 of 861 failures) and ~1.5 dB in extreme degradations. With the format right the mean is 11.5 dB and 41% of leads are below 10 dB, so the tracing must also improve to pass 12.5 dB. On PhysioNet the format is already right on ~95% of images; its gap is the tracing.

**Rule for every change.** Paired whole-database PM-ECG-ID run (kept only if the SNR goes up), PhysioNet 135 images (no loss > 0.05 dB) and LITFL (no structural loss). One switch per change, so it can be turned off and measured.

## A. Measure the structure errors (day 1)

1. **Layout oracle:** whole database with the printed format forced (`--oracle-layout`). Ceiling of "format right" and split of the errors into *decision* (rows found right, wrong format chosen) vs *detection* (rows, columns or pulses found wrong).
2. **Per-page diagnosis in `results.json`:** rows found vs expected, columns and how they were found, labels read, orientation and how weak it was, pulses, ranking of the layout hypotheses with their scores. A taxonomy script sorts every wrong page into one cause.

## B. Rows (largest share)

3. **Rows whose traces cross** (12x1 and tight 6x2 read as `rows`, 6x1 or 3x4+k): rows from the expected count of each candidate format (6, 12, 3+k) fitted to the coverage profile and to the pulses, instead of free peak picking; joint tracking of neighbouring rows with "all ink explained".
4. **Invented rhythm strip** (3x4+1 → 3x4+2, 6x2 → 6x2+1): an extra row needs its own trace evidence (QRS simultaneous with the others, row step, length).
5. **Missing rows** (6x2 → 6x1-prec, 3x4 → 3x1): a column that the row finder missed is searched again with the format's expected row positions.

## C. Labels and columns

6. **Grey-level OCR** in the windows where each candidate format expects a label (light labels, labels crossed by the trace).
7. **Column boundaries** from separator marks, label positions and the change of baseline height, voted per format hypothesis.

## D. Layout decision

8. Calibrate the scoring weights on the oracle runs (logistic fit of OCR / boundary / physics / duplicate terms against "right format"), with the rule that a weight only changes if the whole-database run improves.

## E. Tracing on pages read right (to pass 12.5 dB)

9. PM-ECG-ID oracle decomposition (like `benchmarks/physionet/diagnose.py`): time origin, horizontal scale, gain and baseline per lead, to rank what costs SNR on correct pages.
10. Fix the top items (gain on photos in perspective, time origin, crossing QRS amplitude).

## F. PhysioNet / Kaggle

11. Same decomposition on the 135 images: on scans the gap is the QRS instant and peak amplitude (printing floor ±3 ms); on photos the geometry and lighting.

**Milestones:** A done → B → C → D measured on the whole database after each item; E starts when the format is right on ≥ 85% of single-page ECGs.

## Progress (16/11/2026)

- **A done.** Oracle run: forcing the printed format does not raise the SNR (7.91 → 7.91; −0.4 → −0.2 dB on the wrong-format pages). The wrong pages fail in the measurements under the format, so item D (weights) is dropped for now and the work moved to rows, pulses, gain, time origin and ink map. Details and fixes: [`evaluation-and-roadmap.md` §6b](evaluation-and-roadmap.md).
- **B partly done:** rows 3–8 mm apart, false half pulses, pulses between columns, wide margins, four columns that are two, tracking baseline off the pulse. Joint tracking of crossing traces is still open.
- **Next:** robustness of the same structure under degradation (JPEG, brightness, photos: desaturated coloured traces, black grid leaking into the binary map, labels touching the trace in the column gap), then the joint tracker.
