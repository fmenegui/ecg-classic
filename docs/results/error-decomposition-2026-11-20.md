# Where the Kaggle score is lost (Nov 20, 2026)

Development set: every 20th Kaggle training record of each of the 20 shards, 60 records, 540 images
(6.1% of 8,793). Row network rowheavy3 (run 44653806160). Kaggle-full runs 44754243797 (rowheavy3)
and 44754249172 (row network 44206699093); `benchmarks/physionet/diagnose.py` inside each shard,
`decomp_summary.py` on the merge. Signals at 250 Hz, so "metric" differs by ~0.1 dB from the
evaluation (17.95).

Score after each cumulative oracle correction (mean linear SNR over images, then dB); in brackets
what that error costs.

| Type | Images | metric | labels | coverage | origin | drift | local drift | gain | baseline |
|---|---|---|---|---|---|---|---|---|---|
| all | 540 | 17.85 | 17.90 (+0.04) | 18.03 (+0.14) | 17.98 (-0.05) | 18.40 (+0.42) | 19.09 (+0.69) | 19.27 (+0.18) | 19.45 (+0.18) |
| 0001 | 60 | 25.22 | 25.22 (+0.00) | 25.40 (+0.18) | 25.32 (-0.08) | 25.36 (+0.04) | 25.45 (+0.09) | 25.59 (+0.14) | 25.71 (+0.12) |
| 0003 | 60 | 15.50 | 15.60 (+0.10) | 15.61 (+0.01) | 15.61 (-0.00) | 16.63 (+1.02) | 17.86 (+1.24) | 18.03 (+0.17) | 18.12 (+0.10) |
| 0004 | 60 | 14.37 | 14.47 (+0.10) | 14.50 (+0.03) | 14.49 (-0.01) | 15.56 (+1.07) | 16.98 (+1.42) | 17.14 (+0.15) | 17.25 (+0.12) |
| 0005 | 60 | 13.51 | 13.62 (+0.11) | 13.72 (+0.10) | 13.72 (-0.00) | 14.60 (+0.88) | 15.91 (+1.31) | 16.13 (+0.21) | 16.73 (+0.60) |
| 0006 | 60 | 12.11 | 12.19 (+0.08) | 12.49 (+0.30) | 12.52 (+0.03) | 13.31 (+0.79) | 14.67 (+1.36) | 15.03 (+0.35) | 15.22 (+0.20) |
| 0009 | 60 | 12.83 | 12.97 (+0.13) | 12.98 (+0.02) | 12.97 (-0.01) | 13.85 (+0.87) | 15.38 (+1.54) | 15.60 (+0.22) | 15.80 (+0.20) |
| 0010 | 60 | 14.79 | 14.83 (+0.04) | 14.93 (+0.10) | 14.92 (-0.01) | 15.88 (+0.96) | 17.12 (+1.24) | 17.36 (+0.24) | 17.72 (+0.36) |
| 0011 | 60 | 15.91 | 15.94 (+0.03) | 15.97 (+0.03) | 15.95 (-0.02) | 16.81 (+0.86) | 18.24 (+1.43) | 18.41 (+0.16) | 18.49 (+0.08) |
| 0012 | 60 | 14.29 | 14.54 (+0.25) | 14.66 (+0.12) | 14.64 (-0.02) | 15.55 (+0.91) | 16.74 (+1.20) | 16.95 (+0.21) | 17.32 (+0.37) |

Reading:

- **Clean render (0001), 25.2 dB**: almost nothing is left to oracles (+0.5 dB in all); its remaining error is shape.
- **Horizontal time errors are smaller than the per-lead oracles suggest.** Per lead, drift (time scale and shift) gains +0.8-1.1 dB and local drift (0.5 s windows) +1.2-1.5 dB on every degraded type, but these oracles also fit noise: on a synthetic page with noise only they gain +0.34 dB. With one shift per window shared by the leads of a page column (a geometric distortion moves them together), local drift is worth only +0.3-0.45 dB per type (run 44776638008):

| Type | drift | local drift, per lead | local drift, shared by the leads of a page column |
|---|---|---|---|
| all | 18.40 | 19.09 (+0.69) | 18.57 (+0.17) |
| 0001 | 25.36 | 25.45 (+0.09) | 25.37 (+0.00) |
| 0003 | 16.63 | 17.86 (+1.24) | 16.95 (+0.32) |
| 0004 | 15.56 | 16.98 (+1.43) | 16.01 (+0.45) |
| 0005 | 14.60 | 15.92 (+1.31) | 14.96 (+0.36) |
| 0006 | 13.31 | 14.66 (+1.36) | 13.63 (+0.33) |
| 0009 | 13.85 | 15.38 (+1.54) | 14.20 (+0.36) |
| 0010 | 15.88 | 17.12 (+1.24) | 16.22 (+0.34) |
| 0011 | 16.81 | 18.24 (+1.43) | 17.22 (+0.41) |
| 0012 | 15.55 | 16.74 (+1.20) | 15.88 (+0.33) |

  The per-lead time-scale oracle (drift) has the same weakness; its geometric part is not measured yet.
- **Layout, names, coverage, time origin cost little** (≤0.3 dB per type): failures are not the bottleneck on Kaggle.
- **Gain and baseline**: 0.3-0.8 dB per type.
- **Shape is the main lever**: degraded types still end at 15.2-18.5 dB with every oracle (and the oracles flatter them), against 25.7 for 0001. Tracing precision on degraded images is where the score is.

The leaders score 22.6-22.9 dB on the private leaderboard. With the competition metric dominated by the
linear mean, that requires the degraded types around 20 dB and the clean render near 28.
