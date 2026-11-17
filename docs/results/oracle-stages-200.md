# Stage oracle, run 44062555355 (commit 6c1b990)

Total: 200 pages = 100 standard (3x4 + rhythm, 25 mm/s, 10 mm/mV, sequential columns) + 100 generic. Synthetic pages from ecg-torch-renderer on PTB-XL fold 10, so the absolute numbers are not Kaggle numbers; the differences between readings are the point.

Caveat: 30% of the generic pages print every column over the same 2.5 s (renderer 'sequential' off), which no reader can tell from the image; part of the generic layout/calibration loss is that.


200 pages rendered by ecg-torch-renderer from PTB-XL fold 10 (never used in training), 0 pages failed to render. Each page is read six times, one stage at a time replaced by its truth (see `benchmarks/oracle/run.py`). Kaggle = official metric (mean of linear image SNRs); PM = PMcardio style (failure counts as 0 dB / PCC 0).

### All pages (200 pages)

| Reading | n | Kaggle score (dB) | median image (dB) | PM SNR (dB) | PM PCC | PM failed leads |
|---|---|---|---|---|---|---|
| photo | 200 | 9.96 | 0.00 | 4.07 | 0.420 | 34.6% |
| photo+cal | 200 | 9.92 | -0.27 | 3.99 | 0.452 | 24.1% |
| rectified | 200 | 10.78 | 0.90 | 6.28 | 0.575 | 22.6% |
| rectified+cal | 200 | 11.66 | 2.32 | 8.11 | 0.684 | 10.1% |
| document | 200 | 15.27 | 4.64 | 10.69 | 0.696 | 20.9% |
| document+cal | 200 | 16.00 | 12.07 | 12.62 | 0.779 | 10.1% |

| Stage replaced by its truth | Kaggle gain (dB) | PM SNR gain (dB) | PM failed leads |
|---|---|---|---|
| geometry (perspective, curl, folds, crop) | +0.82 | +2.21 | 34.6% -> 22.6% |
| capture appearance (light, blur, noise, screen) | +4.49 | +4.41 | 22.6% -> 20.9% |
| layout + calibration on the photo | -0.03 | -0.08 | 34.6% -> 24.1% |
| layout + calibration on the flat document | +0.73 | +1.92 | 20.9% -> 10.1% |
| everything (photo -> clean flat page with known calibration) | +6.04 | +8.55 | 34.6% -> 10.1% |

### Style: generic (100 pages)

| Reading | n | Kaggle score (dB) | median image (dB) | PM SNR (dB) | PM PCC | PM failed leads |
|---|---|---|---|---|---|---|
| photo | 100 | 7.21 | 0.00 | 1.15 | 0.220 | 54.7% |
| photo+cal | 100 | 7.08 | -0.67 | 1.00 | 0.266 | 39.4% |
| rectified | 100 | 7.38 | 0.01 | 2.13 | 0.319 | 47.6% |
| rectified+cal | 100 | 9.56 | 0.25 | 4.52 | 0.498 | 21.6% |
| document | 100 | 11.21 | 0.51 | 5.57 | 0.456 | 43.6% |
| document+cal | 100 | 13.80 | 2.80 | 8.66 | 0.612 | 21.7% |

| Stage replaced by its truth | Kaggle gain (dB) | PM SNR gain (dB) | PM failed leads |
|---|---|---|---|
| geometry (perspective, curl, folds, crop) | +0.17 | +0.99 | 54.7% -> 47.6% |
| capture appearance (light, blur, noise, screen) | +3.83 | +3.43 | 47.6% -> 43.6% |
| layout + calibration on the photo | -0.13 | -0.14 | 54.7% -> 39.4% |
| layout + calibration on the flat document | +2.59 | +3.09 | 43.6% -> 21.7% |
| everything (photo -> clean flat page with known calibration) | +6.59 | +7.51 | 54.7% -> 21.7% |

### Style: standard (100 pages)

| Reading | n | Kaggle score (dB) | median image (dB) | PM SNR (dB) | PM PCC | PM failed leads |
|---|---|---|---|---|---|---|
| photo | 100 | 11.63 | 0.77 | 6.61 | 0.593 | 17.1% |
| photo+cal | 100 | 11.63 | 1.01 | 6.58 | 0.613 | 10.8% |
| rectified | 100 | 12.67 | 4.71 | 9.88 | 0.798 | 0.8% |
| rectified+cal | 100 | 13.06 | 7.37 | 11.22 | 0.846 | 0.2% |
| document | 100 | 17.33 | 16.63 | 15.14 | 0.905 | 1.2% |
| document+cal | 100 | 17.45 | 16.88 | 16.06 | 0.924 | 0.0% |

| Stage replaced by its truth | Kaggle gain (dB) | PM SNR gain (dB) | PM failed leads |
|---|---|---|---|
| geometry (perspective, curl, folds, crop) | +1.04 | +3.27 | 17.1% -> 0.8% |
| capture appearance (light, blur, noise, screen) | +4.67 | +5.26 | 0.8% -> 1.2% |
| layout + calibration on the photo | -0.00 | -0.03 | 17.1% -> 10.8% |
| layout + calibration on the flat document | +0.12 | +0.91 | 1.2% -> 0.0% |
| everything (photo -> clean flat page with known calibration) | +5.82 | +9.45 | 17.1% -> 0.0% |

### Capture: paper (200 pages)

| Reading | n | Kaggle score (dB) | median image (dB) | PM SNR (dB) | PM PCC | PM failed leads |
|---|---|---|---|---|---|---|
| photo | 200 | 9.96 | 0.00 | 4.07 | 0.420 | 34.6% |
| photo+cal | 200 | 9.92 | -0.27 | 3.99 | 0.452 | 24.1% |
| rectified | 200 | 10.78 | 0.90 | 6.28 | 0.575 | 22.6% |
| rectified+cal | 200 | 11.66 | 2.32 | 8.11 | 0.684 | 10.1% |
| document | 200 | 15.27 | 4.64 | 10.69 | 0.696 | 20.9% |
| document+cal | 200 | 16.00 | 12.07 | 12.62 | 0.779 | 10.1% |

| Stage replaced by its truth | Kaggle gain (dB) | PM SNR gain (dB) | PM failed leads |
|---|---|---|---|
| geometry (perspective, curl, folds, crop) | +0.82 | +2.21 | 34.6% -> 22.6% |
| capture appearance (light, blur, noise, screen) | +4.49 | +4.41 | 22.6% -> 20.9% |
| layout + calibration on the photo | -0.03 | -0.08 | 34.6% -> 24.1% |
| layout + calibration on the flat document | +0.73 | +1.92 | 20.9% -> 10.1% |
| everything (photo -> clean flat page with known calibration) | +6.04 | +8.55 | 34.6% -> 10.1% |

### Layout with 1 rows (8 pages)

| Reading | n | Kaggle score (dB) | median image (dB) | PM SNR (dB) | PM PCC | PM failed leads |
|---|---|---|---|---|---|---|
| photo | 8 | 11.98 | 0.26 | 0.39 | 0.019 | 96.2% |
| photo+cal | 8 | 11.95 | 0.27 | 0.30 | 0.024 | 94.2% |
| rectified | 8 | 11.99 | 0.29 | 1.04 | 0.057 | 94.2% |
| rectified+cal | 8 | 12.38 | 2.43 | 3.66 | 0.429 | 38.5% |
| document | 8 | 0.56 | 0.13 | 3.28 | 0.173 | 80.8% |
| document+cal | 8 | 15.88 | 8.93 | 9.55 | 0.545 | 38.5% |

| Stage replaced by its truth | Kaggle gain (dB) | PM SNR gain (dB) | PM failed leads |
|---|---|---|---|
| geometry (perspective, curl, folds, crop) | +0.01 | +0.64 | 96.2% -> 94.2% |
| capture appearance (light, blur, noise, screen) | -11.42 | +2.24 | 94.2% -> 80.8% |
| layout + calibration on the photo | -0.03 | -0.10 | 96.2% -> 94.2% |
| layout + calibration on the flat document | +15.31 | +6.27 | 80.8% -> 38.5% |
| everything (photo -> clean flat page with known calibration) | +3.90 | +9.15 | 96.2% -> 38.5% |

### Layout with 2 rows (10 pages)

| Reading | n | Kaggle score (dB) | median image (dB) | PM SNR (dB) | PM PCC | PM failed leads |
|---|---|---|---|---|---|---|
| photo | 10 | 9.34 | 0.00 | -0.28 | 0.133 | 70.0% |
| photo+cal | 10 | -1.14 | -0.57 | 1.10 | 0.233 | 56.7% |
| rectified | 10 | 9.39 | 0.01 | 0.26 | 0.118 | 86.7% |
| rectified+cal | 10 | 10.47 | 1.18 | 7.83 | 0.657 | 16.7% |
| document | 10 | 11.09 | 0.00 | 3.50 | 0.287 | 68.3% |
| document+cal | 10 | 13.55 | 6.07 | 8.98 | 0.625 | 18.3% |

| Stage replaced by its truth | Kaggle gain (dB) | PM SNR gain (dB) | PM failed leads |
|---|---|---|---|
| geometry (perspective, curl, folds, crop) | +0.05 | +0.54 | 70.0% -> 86.7% |
| capture appearance (light, blur, noise, screen) | +1.70 | +3.24 | 86.7% -> 68.3% |
| layout + calibration on the photo | -10.48 | +1.38 | 70.0% -> 56.7% |
| layout + calibration on the flat document | +2.46 | +5.48 | 68.3% -> 18.3% |
| everything (photo -> clean flat page with known calibration) | +4.21 | +9.26 | 70.0% -> 18.3% |

### Layout with 3 rows (16 pages)

| Reading | n | Kaggle score (dB) | median image (dB) | PM SNR (dB) | PM PCC | PM failed leads |
|---|---|---|---|---|---|---|
| photo | 16 | -0.73 | -0.08 | -1.08 | 0.267 | 54.3% |
| photo+cal | 16 | 7.92 | -0.71 | -0.24 | 0.251 | 53.6% |
| rectified | 16 | 0.36 | 0.12 | 0.59 | 0.429 | 38.4% |
| rectified+cal | 16 | 9.17 | 0.03 | 3.55 | 0.448 | 26.8% |
| document | 16 | 11.30 | 0.34 | 5.27 | 0.510 | 42.8% |
| document+cal | 16 | 13.22 | 2.18 | 6.36 | 0.548 | 26.8% |

| Stage replaced by its truth | Kaggle gain (dB) | PM SNR gain (dB) | PM failed leads |
|---|---|---|---|
| geometry (perspective, curl, folds, crop) | +1.09 | +1.68 | 54.3% -> 38.4% |
| capture appearance (light, blur, noise, screen) | +10.94 | +4.68 | 38.4% -> 42.8% |
| layout + calibration on the photo | +8.65 | +0.84 | 54.3% -> 53.6% |
| layout + calibration on the flat document | +1.92 | +1.09 | 42.8% -> 26.8% |
| everything (photo -> clean flat page with known calibration) | +13.95 | +7.44 | 54.3% -> 26.8% |

### Layout with 4 rows (113 pages)

| Reading | n | Kaggle score (dB) | median image (dB) | PM SNR (dB) | PM PCC | PM failed leads |
|---|---|---|---|---|---|---|
| photo | 113 | 11.60 | 0.28 | 6.50 | 0.572 | 19.1% |
| photo+cal | 113 | 11.49 | 0.26 | 6.36 | 0.587 | 14.1% |
| rectified | 113 | 12.54 | 4.39 | 9.46 | 0.757 | 5.5% |
| rectified+cal | 113 | 12.83 | 5.61 | 10.80 | 0.815 | 3.3% |
| document | 113 | 17.00 | 16.43 | 14.39 | 0.868 | 4.9% |
| document+cal | 113 | 17.12 | 16.44 | 15.49 | 0.895 | 3.2% |

| Stage replaced by its truth | Kaggle gain (dB) | PM SNR gain (dB) | PM failed leads |
|---|---|---|---|
| geometry (perspective, curl, folds, crop) | +0.94 | +2.96 | 19.1% -> 5.5% |
| capture appearance (light, blur, noise, screen) | +4.47 | +4.92 | 5.5% -> 4.9% |
| layout + calibration on the photo | -0.10 | -0.15 | 19.1% -> 14.1% |
| layout + calibration on the flat document | +0.11 | +1.10 | 4.9% -> 3.2% |
| everything (photo -> clean flat page with known calibration) | +5.52 | +8.98 | 19.1% -> 3.2% |

### Layout with 5 rows (17 pages)

| Reading | n | Kaggle score (dB) | median image (dB) | PM SNR (dB) | PM PCC | PM failed leads |
|---|---|---|---|---|---|---|
| photo | 17 | 3.99 | -0.31 | 1.19 | 0.224 | 50.0% |
| photo+cal | 17 | 2.90 | -1.02 | 1.26 | 0.262 | 36.8% |
| rectified | 17 | 4.04 | -0.86 | 1.41 | 0.302 | 37.3% |
| rectified+cal | 17 | 7.93 | -1.30 | 2.54 | 0.388 | 24.0% |
| document | 17 | 11.02 | 1.10 | 5.53 | 0.467 | 39.2% |
| document+cal | 17 | 12.59 | 2.99 | 8.15 | 0.594 | 20.6% |

| Stage replaced by its truth | Kaggle gain (dB) | PM SNR gain (dB) | PM failed leads |
|---|---|---|---|
| geometry (perspective, curl, folds, crop) | +0.05 | +0.22 | 50.0% -> 37.3% |
| capture appearance (light, blur, noise, screen) | +6.98 | +4.12 | 37.3% -> 39.2% |
| layout + calibration on the photo | -1.09 | +0.08 | 50.0% -> 36.8% |
| layout + calibration on the flat document | +1.57 | +2.62 | 39.2% -> 20.6% |
| everything (photo -> clean flat page with known calibration) | +8.59 | +6.96 | 50.0% -> 20.6% |

### Layout with 6 rows (15 pages)

| Reading | n | Kaggle score (dB) | median image (dB) | PM SNR (dB) | PM PCC | PM failed leads |
|---|---|---|---|---|---|---|
| photo | 15 | 1.56 | -0.10 | 1.66 | 0.267 | 48.9% |
| photo+cal | 15 | -0.23 | -0.41 | 0.12 | 0.240 | 41.7% |
| rectified | 15 | 3.08 | 0.71 | 3.09 | 0.476 | 26.7% |
| rectified+cal | 15 | 4.88 | 0.72 | 4.00 | 0.519 | 23.9% |
| document | 15 | 10.78 | 3.12 | 6.47 | 0.627 | 25.0% |
| document+cal | 15 | 11.74 | 0.72 | 5.62 | 0.517 | 27.2% |

| Stage replaced by its truth | Kaggle gain (dB) | PM SNR gain (dB) | PM failed leads |
|---|---|---|---|
| geometry (perspective, curl, folds, crop) | +1.52 | +1.43 | 48.9% -> 26.7% |
| capture appearance (light, blur, noise, screen) | +7.69 | +3.37 | 26.7% -> 25.0% |
| layout + calibration on the photo | -1.79 | -1.54 | 48.9% -> 41.7% |
| layout + calibration on the flat document | +0.96 | -0.85 | 25.0% -> 27.2% |
| everything (photo -> clean flat page with known calibration) | +10.18 | +3.96 | 48.9% -> 27.2% |

### Layout with 7 rows (6 pages)

| Reading | n | Kaggle score (dB) | median image (dB) | PM SNR (dB) | PM PCC | PM failed leads |
|---|---|---|---|---|---|---|
| photo | 6 | 8.89 | -0.76 | 2.07 | 0.307 | 37.5% |
| photo+cal | 6 | 11.20 | -0.67 | 5.07 | 0.383 | 34.7% |
| rectified | 6 | 9.01 | 0.57 | 3.29 | 0.391 | 20.8% |
| rectified+cal | 6 | 11.25 | 0.84 | 5.98 | 0.541 | 16.7% |
| document | 6 | 16.78 | 10.39 | 11.21 | 0.648 | 20.8% |
| document+cal | 6 | 17.50 | 17.33 | 13.95 | 0.778 | 16.7% |

| Stage replaced by its truth | Kaggle gain (dB) | PM SNR gain (dB) | PM failed leads |
|---|---|---|---|
| geometry (perspective, curl, folds, crop) | +0.12 | +1.22 | 37.5% -> 20.8% |
| capture appearance (light, blur, noise, screen) | +7.78 | +7.92 | 20.8% -> 20.8% |
| layout + calibration on the photo | +2.31 | +3.00 | 37.5% -> 34.7% |
| layout + calibration on the flat document | +0.71 | +2.74 | 20.8% -> 16.7% |
| everything (photo -> clean flat page with known calibration) | +8.61 | +11.88 | 37.5% -> 16.7% |

### Layout with 8 rows (1 pages)

| Reading | n | Kaggle score (dB) | median image (dB) | PM SNR (dB) | PM PCC | PM failed leads |
|---|---|---|---|---|---|---|
| photo | 1 | -5.14 | -5.14 | -6.14 | 0.164 | 0.0% |
| photo+cal | 1 | -3.37 | -3.37 | -2.42 | 0.124 | 8.3% |
| rectified | 1 | -1.16 | -1.16 | -1.47 | 0.244 | 0.0% |
| rectified+cal | 1 | -1.18 | -1.18 | -0.69 | 0.358 | 0.0% |
| document | 1 | -1.21 | -1.21 | -1.53 | 0.241 | 0.0% |
| document+cal | 1 | -5.02 | -5.02 | -2.67 | 0.290 | 0.0% |

| Stage replaced by its truth | Kaggle gain (dB) | PM SNR gain (dB) | PM failed leads |
|---|---|---|---|
| geometry (perspective, curl, folds, crop) | +3.98 | +4.66 | 0.0% -> 0.0% |
| capture appearance (light, blur, noise, screen) | -0.06 | -0.06 | 0.0% -> 0.0% |
| layout + calibration on the photo | +1.77 | +3.72 | 0.0% -> 8.3% |
| layout + calibration on the flat document | -3.81 | -1.14 | 0.0% -> 0.0% |
| everything (photo -> clean flat page with known calibration) | +0.11 | +3.46 | 0.0% -> 0.0% |

### Layout with 12 rows (4 pages)

| Reading | n | Kaggle score (dB) | median image (dB) | PM SNR (dB) | PM PCC | PM failed leads |
|---|---|---|---|---|---|---|
| photo | 4 | -1.02 | 0.00 | -1.53 | -0.005 | 85.4% |
| photo+cal | 4 | -2.06 | -1.58 | -3.29 | 0.133 | 14.6% |
| rectified | 4 | 0.66 | 0.00 | 0.78 | 0.314 | 50.0% |
| rectified+cal | 4 | 1.13 | 0.18 | 2.28 | 0.522 | 8.3% |
| document | 4 | 1.56 | 0.38 | 3.05 | 0.376 | 27.1% |
| document+cal | 4 | 1.67 | 0.27 | 5.89 | 0.619 | 12.5% |

| Stage replaced by its truth | Kaggle gain (dB) | PM SNR gain (dB) | PM failed leads |
|---|---|---|---|
| geometry (perspective, curl, folds, crop) | +1.68 | +2.31 | 85.4% -> 50.0% |
| capture appearance (light, blur, noise, screen) | +0.90 | +2.27 | 50.0% -> 27.1% |
| layout + calibration on the photo | -1.04 | -1.76 | 85.4% -> 14.6% |
| layout + calibration on the flat document | +0.10 | +2.83 | 27.1% -> 12.5% |
| everything (photo -> clean flat page with known calibration) | +2.69 | +7.42 | 85.4% -> 12.5% |

### Layout with 13 rows (1 pages)

| Reading | n | Kaggle score (dB) | median image (dB) | PM SNR (dB) | PM PCC | PM failed leads |
|---|---|---|---|---|---|---|
| photo | 1 | 0.00 | 0.00 | 0.00 | 0.000 | 100.0% |
| photo+cal | 1 | -1.49 | -1.49 | -1.25 | 0.204 | 0.0% |
| rectified | 1 | 0.00 | 0.00 | 0.00 | 0.000 | 100.0% |
| rectified+cal | 1 | 8.04 | 8.04 | 13.89 | 0.979 | 0.0% |
| document | 1 | 10.96 | 10.96 | 10.99 | 0.956 | 0.0% |
| document+cal | 1 | 19.09 | 19.09 | 19.07 | 0.993 | 0.0% |

| Stage replaced by its truth | Kaggle gain (dB) | PM SNR gain (dB) | PM failed leads |
|---|---|---|---|
| geometry (perspective, curl, folds, crop) | +0.00 | +0.00 | 100.0% -> 100.0% |
| capture appearance (light, blur, noise, screen) | +10.96 | +10.99 | 100.0% -> 0.0% |
| layout + calibration on the photo | -1.49 | -1.25 | 100.0% -> 0.0% |
| layout + calibration on the flat document | +8.13 | +8.08 | 0.0% -> 0.0% |
| everything (photo -> clean flat page with known calibration) | +19.09 | +19.07 | 100.0% -> 0.0% |

### Layout with 14 rows (3 pages)

| Reading | n | Kaggle score (dB) | median image (dB) | PM SNR (dB) | PM PCC | PM failed leads |
|---|---|---|---|---|---|---|
| photo | 3 | -1.39 | -1.22 | -1.86 | 0.034 | 47.2% |
| photo+cal | 3 | -2.91 | -2.29 | -3.98 | 0.039 | 13.9% |
| rectified | 3 | -0.01 | 0.00 | -0.03 | 0.028 | 83.3% |
| rectified+cal | 3 | -2.11 | -1.97 | -1.78 | 0.220 | 8.3% |
| document | 3 | 0.01 | 0.00 | 0.00 | 0.000 | 100.0% |
| document+cal | 3 | 12.67 | -1.37 | 6.30 | 0.566 | 11.1% |

| Stage replaced by its truth | Kaggle gain (dB) | PM SNR gain (dB) | PM failed leads |
|---|---|---|---|
| geometry (perspective, curl, folds, crop) | +1.38 | +1.83 | 47.2% -> 83.3% |
| capture appearance (light, blur, noise, screen) | +0.02 | +0.03 | 83.3% -> 100.0% |
| layout + calibration on the photo | -1.51 | -2.12 | 47.2% -> 13.9% |
| layout + calibration on the flat document | +12.67 | +6.30 | 100.0% -> 11.1% |
| everything (photo -> clean flat page with known calibration) | +14.07 | +8.16 | 47.2% -> 11.1% |

### Layout with 15 rows (6 pages)

| Reading | n | Kaggle score (dB) | median image (dB) | PM SNR (dB) | PM PCC | PM failed leads |
|---|---|---|---|---|---|---|
| photo | 6 | -0.14 | 0.00 | -0.31 | 0.013 | 86.1% |
| photo+cal | 6 | 1.24 | -0.79 | -0.45 | 0.437 | 5.6% |
| rectified | 6 | 0.00 | 0.00 | 0.00 | 0.000 | 100.0% |
| rectified+cal | 6 | 13.39 | 3.00 | 6.94 | 0.627 | 5.6% |
| document | 6 | 0.00 | 0.00 | 0.00 | 0.000 | 100.0% |
| document+cal | 6 | 16.25 | 16.19 | 13.79 | 0.775 | 2.8% |

| Stage replaced by its truth | Kaggle gain (dB) | PM SNR gain (dB) | PM failed leads |
|---|---|---|---|
| geometry (perspective, curl, folds, crop) | +0.14 | +0.31 | 86.1% -> 100.0% |
| capture appearance (light, blur, noise, screen) | +0.00 | +0.00 | 100.0% -> 100.0% |
| layout + calibration on the photo | +1.38 | -0.15 | 86.1% -> 5.6% |
| layout + calibration on the flat document | +16.25 | +13.79 | 100.0% -> 2.8% |
| everything (photo -> clean flat page with known calibration) | +16.39 | +14.10 | 86.1% -> 2.8% |

### Calibration read on the flat document (auto)

speed wrong on 52 pages, gain wrong on 49, no reading on 0 of 200.
