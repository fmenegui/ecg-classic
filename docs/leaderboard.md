# Development leaderboard

PM-ECG-ID: 300 ECGs common to all experiments (5.0% of 6000); Kaggle: 540 images (6.1% of 8793), holdout = id % 5 == 0 plus the 15 local records.

References: PMcardio 14.48 dB (60-subset mean of its paper); Kaggle leaders 22.6-22.9 dB (private leaderboard, test set: not the same images).

| Experiment | PM SNR (dB) | PM 60-scenario mean | PM failed leads | Kaggle all (dB) | Kaggle holdout | 0001 | 0003 | 0004 | 0005 | 0006 | 0009 | 0010 | 0011 | 0012 | What |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| dev2-row3e | 9.34 | 9.34 | 15.17% | 17.95 | 17.67 | 25.05 | 16.22 | 14.71 | 13.76 | 12.32 | 13.09 | 15.41 | 16.60 | 14.61 | row network rowheavy3 continued (44653806160) |
| dev3-row3e-dewarp2 | 9.26 | 9.26 | 15.89% | 17.95 | 17.67 | 25.05 | 16.22 | 14.71 | 13.75 | 12.32 | 13.09 | 15.41 | 16.60 | 14.61 | rowheavy3 + dewarp v2 (numpy flatten) |
| dev-base | 9.12 | 9.12 | 15.17% | 17.96 | 17.67 | 25.05 | 16.22 | 14.71 | 13.76 | 12.42 | 13.16 | 15.45 | 16.60 | 14.61 |  |
| **dev2-base** | 9.12 | 9.12 | 15.17% | 17.71 | 17.13 | 24.62 | 16.20 | 14.89 | 13.74 | 12.25 | 13.02 | 15.41 | 16.55 | 14.60 | row network 44206699093 (rowreal); classical layout; no page or dewarp network |
| dev2-dewarp2 | 9.12 | 9.12 | 15.17% | 17.71 | 17.13 | 24.62 | 16.20 | 14.89 | 13.74 | 12.25 | 13.02 | 15.41 | 16.55 | 14.59 | base + dewarp v2 (44653802675); before Nov 20 the dewarp reading failed on hosts without torch |
| dev2-rowheavy | 8.95 | 8.95 | 15.22% | 17.41 | 16.71 | 24.31 | 15.95 | 14.50 | 13.40 | 12.31 | 12.86 | 15.05 | 16.23 | 14.18 | row network rowheavy3c (44523858950; the init of rowheavy3e) |
| dev3-photonorm-scale | - | - | -% | 17.97 | 17.68 | 25.05 | 16.22 | 14.71 | 13.82 | 12.43 | 13.22 | 15.45 | 16.60 | 14.73 |  |
| dev-orient2 | - | - | -% | 17.96 | 17.67 | 25.05 | 16.22 | 14.71 | 13.80 | 12.42 | 13.16 | 15.45 | 16.60 | 14.61 |  |
| dev3-photonorm | - | - | -% | 17.96 | 17.67 | 25.05 | 16.22 | 14.71 | 13.76 | 12.42 | 13.16 | 15.45 | 16.60 | 14.61 |  |
| dev-rowheavy5 | - | - | -% | 17.96 | 17.64 | 25.01 | 16.32 | 14.91 | 13.81 | 12.31 | 13.22 | 15.48 | 16.54 | 14.63 | rowheavy5 (45004819752): rowheavy3e + heavy_aug down to 1/24 and blur sigma 10 px; page v3; photo_norm |
| dev-rowheavy4 | - | - | -% | 17.93 | 17.58 | 24.98 | 16.22 | 14.79 | 13.89 | 12.45 | 13.07 | 15.46 | 16.48 | 14.74 |  |
| dev2-page3 | - | - | -% | 17.72 | 17.14 | 24.62 | 16.20 | 14.89 | 13.77 | 12.35 | 13.07 | 15.45 | 16.55 | 14.59 | base + page network v3 (44560948929) with page select and the rhythm margin |
| dev3-classical-R20 | - | - | -% | 17.51 | 17.79 | 25.20 | 15.44 | 12.61 | 12.51 | 10.36 | 11.76 | 14.99 | 15.05 | 12.02 | classical reader with the rectified frame at 20 px/mm (instead of 10) |
| dev3-classical | - | - | -% | 17.22 | 17.15 | 24.38 | 15.70 | 14.09 | 13.18 | 10.53 | 12.17 | 14.86 | 15.73 | 13.25 | classical reader only (no network) |
