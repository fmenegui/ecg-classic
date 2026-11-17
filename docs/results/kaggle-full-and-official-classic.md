# Kaggle: whole training set and official late submission (classical, 17/11/2026)

- Official late submission (CPU notebook, hidden test ~1,000 images): **public 12.67 dB, private 13.03 dB**.
- Whole training set, kaggle-full.yml run 44011597597: 928 of 977 records (95%; one of 20 shards could not open the archive), 8,352 images.
- Top teams report a similar gap between their training CV and the private leaderboard (1st place: CV 26.8, private 22.75).

# Kaggle training records: 928 records, 8352 images

Config: `{}`

## All records

### Kaggle metric (official) and PMcardio-style metrics

| Type | n | Kaggle score (dB) | PM SNR (dB) | PM PCC | PM failed leads | PM SNR w/o failures | PM RMSE w/o failures (mV) |
|---|---|---|---|---|---|---|---|
| 0001 | 928 | 24.86 | 24.23 | 0.989 | 0.0% | 24.23 | 0.018 |
| 0003 | 928 | 16.12 | 16.33 | 0.975 | 0.0% | 16.33 | 0.033 |
| 0004 | 928 | 14.04 | 14.25 | 0.955 | 0.3% | 14.29 | 0.044 |
| 0005 | 928 | 13.36 | 13.94 | 0.924 | 1.6% | 14.17 | 0.054 |
| 0006 | 928 | 11.11 | 11.04 | 0.901 | 1.7% | 11.23 | 0.061 |
| 0009 | 928 | 12.08 | 12.57 | 0.933 | 0.6% | 12.65 | 0.053 |
| 0010 | 928 | 14.79 | 14.57 | 0.940 | 0.4% | 14.62 | 0.047 |
| 0011 | 928 | 16.00 | 16.03 | 0.968 | 0.1% | 16.04 | 0.035 |
| 0012 | 928 | 13.38 | 13.08 | 0.930 | 0.5% | 13.14 | 0.057 |
| **all** | 8352 | 17.58 | 15.11 | 0.946 | 0.6% | 15.20 | 0.045 |

Kaggle: 10·log10 of the mean LINEAR image SNR (official metric code, `benchmarks/metrics.py`). PMcardio: per lead ±0.5 s Pearson alignment, median offset; failure = missing or < 50% covered (PCC 0, SNR 0 dB).

## Local regression subset (135 images)

### Kaggle metric (official) and PMcardio-style metrics

| Type | n | Kaggle score (dB) | PM SNR (dB) | PM PCC | PM failed leads | PM SNR w/o failures | PM RMSE w/o failures (mV) |
|---|---|---|---|---|---|---|---|
| 0001 | 15 | 26.26 | 25.42 | 0.995 | 0.0% | 25.42 | 0.010 |
| 0003 | 15 | 17.12 | 16.97 | 0.984 | 0.0% | 16.97 | 0.025 |
| 0004 | 15 | 14.31 | 14.51 | 0.963 | 0.0% | 14.51 | 0.035 |
| 0005 | 15 | 13.65 | 14.81 | 0.964 | 0.0% | 14.81 | 0.033 |
| 0006 | 15 | 11.58 | 11.50 | 0.891 | 5.0% | 12.10 | 0.046 |
| 0009 | 15 | 12.84 | 13.65 | 0.957 | 0.6% | 13.72 | 0.041 |
| 0010 | 15 | 15.97 | 14.90 | 0.965 | 0.0% | 14.90 | 0.044 |
| 0011 | 15 | 16.51 | 16.50 | 0.978 | 0.0% | 16.50 | 0.029 |
| 0012 | 15 | 15.00 | 15.02 | 0.972 | 0.0% | 15.02 | 0.033 |
| **all** | 135 | 18.74 | 15.92 | 0.963 | 0.6% | 16.02 | 0.033 |

Kaggle: 10·log10 of the mean LINEAR image SNR (official metric code, `benchmarks/metrics.py`). PMcardio: per lead ±0.5 s Pearson alignment, median offset; failure = missing or < 50% covered (PCC 0, SNR 0 dB).
