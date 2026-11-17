# Classical reader under both reported metrics

Local Kaggle regression set: 135 images (15 training records × 9 capture types), classical reader at `61324b1` (no network), 16/11/2026. Kaggle = official metric code (`benchmarks/metrics.py`); PMcardio = per-lead ±0.5 s alignment, PCC/RMSE/SNR, failure = missing or < 50% covered. Not the hidden Kaggle test: 15 records only. Old convention (mean of per-image dB): 14.70 dB.

| Type | n | Kaggle score (dB) | PM SNR (dB) | PM PCC | PM failed leads | PM SNR w/o failures | PM RMSE w/o failures (mV) |
|---|---|---|---|---|---|---|---|
| 0001 | 15 | 26.20 | 25.39 | 0.995 | 0.0% | 25.39 | 0.010 |
| 0003 | 15 | 17.12 | 17.00 | 0.984 | 0.0% | 17.00 | 0.025 |
| 0004 | 15 | 14.35 | 14.70 | 0.970 | 0.0% | 14.70 | 0.034 |
| 0005 | 15 | 13.65 | 14.76 | 0.963 | 0.0% | 14.76 | 0.033 |
| 0006 | 15 | 11.55 | 11.51 | 0.891 | 5.0% | 12.12 | 0.046 |
| 0009 | 15 | 12.80 | 13.57 | 0.958 | 0.6% | 13.65 | 0.041 |
| 0010 | 15 | 15.75 | 14.78 | 0.964 | 0.0% | 14.78 | 0.045 |
| 0011 | 15 | 16.56 | 16.50 | 0.978 | 0.0% | 16.50 | 0.029 |
| 0012 | 15 | 14.93 | 15.02 | 0.972 | 0.0% | 15.02 | 0.033 |
| **all** | 135 | 18.70 | 15.91 | 0.964 | 0.6% | 16.01 | 0.033 |

References: Kaggle private leaderboard 1st 22.75 dB (single model 22.98 public), 7th 22.48, 14th 21.37, 24th 19.7. PMcardio paper on PM-ECG-ID: PCC > 0.91, failed leads 6.62%.
