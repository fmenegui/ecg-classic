# Development leaderboard

PM-ECG-ID: 300 ECGs common to all experiments (5.0% of 6000); Kaggle: 540 images (6.1% of 8793), holdout = id % 5 == 0 plus the 15 local records.

References: PMcardio 14.48 dB (60-subset mean of its paper); Kaggle leaders 22.6-22.9 dB (private leaderboard, test set: not the same images).

| Experiment | PM SNR (dB) | PM 60-scenario mean | PM failed leads | Kaggle all (dB) | Kaggle holdout | 0001 | 0003 | 0004 | 0005 | 0006 | 0009 | 0010 | 0011 | 0012 | What |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| dev2-row3e | 9.34 | 9.34 | 15.17% | 17.95 | 17.67 | 25.05 | 16.22 | 14.71 | 13.76 | 12.32 | 13.09 | 15.41 | 16.60 | 14.61 | row network rowheavy3 continued (44653806160) |
| dev3-row3e-dewarp2 | 9.26 | 9.26 | 15.89% | 17.95 | 17.67 | 25.05 | 16.22 | 14.71 | 13.75 | 12.32 | 13.09 | 15.41 | 16.60 | 14.61 | rowheavy3 + dewarp v2 (numpy flatten) |
| **dev2-base** | 9.12 | 9.12 | 15.17% | 17.71 | 17.13 | 24.62 | 16.20 | 14.89 | 13.74 | 12.25 | 13.02 | 15.41 | 16.55 | 14.60 | row network 44206699093 (rowreal); classical layout; no page or dewarp network |
| dev2-dewarp2 | 9.12 | 9.12 | 15.17% | 17.71 | 17.13 | 24.62 | 16.20 | 14.89 | 13.74 | 12.25 | 13.02 | 15.41 | 16.55 | 14.59 | base + dewarp v2 (44653802675); before Nov 20 the dewarp reading failed on hosts without torch |
| dev2-rowheavy | 8.95 | 8.95 | 15.22% | 17.41 | 16.71 | 24.31 | 15.95 | 14.50 | 13.40 | 12.31 | 12.86 | 15.05 | 16.23 | 14.18 | row network rowheavy3c (44523858950; the init of rowheavy3e) |
| dev2-page3 | - | - | -% | 17.72 | 17.14 | 24.62 | 16.20 | 14.89 | 13.77 | 12.35 | 13.07 | 15.45 | 16.55 | 14.59 | base + page network v3 (44560948929) with page select and the rhythm margin |
| dev3-classical-R20 | - | - | -% | 17.51 | 17.79 | 25.20 | 15.44 | 12.61 | 12.51 | 10.36 | 11.76 | 14.99 | 15.05 | 12.02 | classical reader with the rectified frame at 20 px/mm (instead of 10) |
| dev3-classical | - | - | -% | 17.22 | 17.15 | 24.38 | 15.70 | 14.09 | 13.18 | 10.53 | 12.17 | 14.86 | 15.73 | 13.25 | classical reader only (no network) |
| dev-base | 9.12 | 9.12 | 15.17% | - | - | - | - | - | - | - | - | - | - | - |  |
