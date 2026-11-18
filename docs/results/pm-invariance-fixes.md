# Photo invariance fixes: 5 mm period rescue and lone-pulse gain (commit `48ca4ba`)

Two classical fixes for photographed pages (`ecgdig/geometry.py` `SCALE_MAJOR5`, `ecgdig/digitize.py` `LONE_PULSE`):

- **5 mm period rescue.** When blur hides the 1 mm lines, the grid period found is the 5 mm one. The scale used to come from the page width (about 10% off, no perspective correction). Now a period that gives a 150-450 mm plot width at 5 mm is refined and used.
- **Lone-pulse gain.** A gain other than 10 mm/mV read on a single row (a calibration pulse found only in part, e.g. 5.5-6 mm under blur) must agree with the standard value within 10%. Otherwise it is dropped.

The pulse-column gain (`PCOL_GAIN`, commit `60fb604`) is evaluated separately (run 44353561521).

## PM-ECG-ID, 5,997 of 6,000 ECGs (99.95%), row network (real pages), paired

Base: run 44269936154 (edge_stroke). Fixes: run 44325036056.

| | Base | With the fixes |
|---|---|---|
| PM SNR | 9.46 dB | 9.56 dB |
| PCC | 0.706 | 0.706 |
| Failed leads | 12.7% | 12.9% |
| Kaggle metric | 12.78 dB | 12.80 dB |

| Scenario group | PM SNR (dB) |
|---|---|
| padding (399) | 9.98 -> 10.40 |
| rotation (400) | 10.37 -> 10.61 |
| perspective (300) | 7.89 -> 8.07 |
| photos_iphone / scans (199) | +0.15 / +0.17 |
| other groups | within ±0.1 |

The local 18-image sample promised +2.1 dB; the whole database gives +0.10 dB. The cases these rules fix are few.

## Kaggle training set, 8,793 images of 977 records (100%), paired (runs 44269863071 -> 44325053591)

| Metric | Set | Base | With the fixes |
|---|---|---|---|
| Raw (official) | all | 18.08 dB | 18.08 dB |
| Raw (official) | holdout (1,854 images, 21%) | 17.93 dB | 17.93 dB |
| Filled | all | 18.21 dB | 18.21 dB |
| Filled | holdout | 18.05 dB | 18.05 dB |

No capture type moves by more than 0.03 dB. The fixes stay on.

## Where the PM-ECG-ID loss is

On the single-page images of the same run, the format is right on 75%. The rest:

| Error | Share of single-page images |
|---|---|
| wrong row count | 13% |
| no layout fits the rows found (`rows`) | 3.9% |
| reading failed | 2.1% |
| right rows, wrong columns or names | 5.6% |

On a random stratified sample of 122 pages (2 ECGs per scenario), the classical reader gets the format right on 88 (72%) and the row count on 99 (81%).

This led to the page-structure network (`hybrid/pagegen.py`, `hybrid/train_page.py`, `ecgdig/pagemodel.py`).
