# Clean renders (Kaggle 0001): where the error is, and `edge_stroke`

## Error decomposition (5 local clean renders)

Each correction applied to the reading with the ground truth, one at a time:

| Correction | Gain |
|---|---|
| time shift, gain, baseline | +0.3 to +0.5 dB |
| per-image sub-sample lag (0.0 to 0.6 ms, not constant) | +1 to +2.6 dB on some images |

- Steep (QRS) segments hold 57-95% of the residual energy; content above 15 Hz holds 27-70%.
- Segment edges hold 13-72%: a QRS that runs into the lead-switch separator mark is cut with the mark and replaced by an extrapolation.

**Ceiling of a column reading.** Simulated on the ground truth of 15 records:

| Reading | 10 px/mm | 20 px/mm |
|---|---|---|
| ink extent mid-point per column, linear interpolation | 26.6 dB | 32.9 dB |
| true value at the column centre, linear interpolation | 29.4 dB | 35.9 dB |

At 10 px/mm with exact samples, cubic interpolation raises the first ceiling to 29.7 dB and the second to 37.1 dB.

The classical reader is at 25.2 dB on 0001, close to the first ceiling. Cubic or Akima interpolation of the real samples made it worse (25.0 -> 17.8 and 24.6 dB): the noise of each sample is amplified. More precision has to come from the reading itself (a row network that predicts the value at the column centre, at twice the horizontal resolution).

## `edge_stroke` (now on by default)

In the columns cut around the separator mark, the reader follows the far edge of the ink while it keeps moving at least 0.3 mm per column, and keeps it only if the tip passes the edge of the column where it stops by at least 0.5 mm. The value is then held instead of extrapolated.

Kaggle training set, 8,793 images of 977 records (100%), same commit `fd4aaaa`, paired. Raw metric = official metric; filled metric = holes filled as `kaggle/submission.py` does.

| Reader | Set | Raw metric (dB) | Filled metric (dB) |
|---|---|---|---|
| classical | all | 17.56 -> 17.59 | 17.71 -> 17.74 |
| classical | holdout (1,854 images, 206 records, 21%) | 17.41 -> 17.45 | 17.54 -> 17.58 |
| row network (real pages) | all | 18.04 -> 18.08 | 18.17 -> 18.21 |
| row network (real pages) | holdout | 17.88 -> 17.93 | 18.00 -> 18.05 |

The gain is spread over all capture types (+0.00 to +0.05 dB each); 0001 gains +0.04 to +0.06 dB.

PM-ECG-ID, 5,998 of 6,000 ECGs (99.97%), row network (real pages), paired:

| | Without | With |
|---|---|---|
| PM SNR | 9.46 dB | 9.47 dB |
| PCC | 0.706 | 0.706 |
| failed leads | 12.7% | 12.7% |
| Kaggle metric | 12.78 dB | 12.78 dB |

No scenario moves by more than 0.05 dB.

The local 5-image sample had promised +0.4 dB on 0001. One image accounts for it (426433979: 19.1 -> 23.6 dB).

The same code also gave different results on two clean renders locally and on GitHub runners:

| Image | Local | GitHub |
|---|---|---|
| 1257259783 | 25.23 dB | 24.95 dB |
| 3042142152 | 23.47 dB | 22.94 dB |

The likely cause is the OpenCV and NumPy versions; the anti-aliased ink edge is near the rule's thresholds.

Runs:

| Reader | Without | With |
|---|---|---|
| classical | 44309165597 | 44269875089 |
| row network | 44269851249 | 44269863071 |
| PM-ECG-ID | 44237642516 | 44269936154 |

## Merge fix

Re-running failed shards also downloaded the merged artifact of the first attempt, so three merges counted 17,145 images instead of 8,793. `merge_shards.py` now skips merged artifacts and keeps one result per image. The duplicates were identical, so the scores move by 0.01 dB (row network + edge 18.09 -> 18.08 dB). The tables above are computed on unique images.
