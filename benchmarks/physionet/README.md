# PhysioNet (Kaggle) benchmark

Training set of the Kaggle competition *PhysioNet - Digitization of ECG Images*: each record has the true signal (`<id>.csv`, 12 leads, 250–1025 Hz, 10 s; empty where the paper does not show the lead) and 9 images of the same ECG in different capture types (`<id>-NNNN.png`). All pages are 3x4 + rhythm II.

| Type | Capture |
|---|---|
| 0001 | digital render |
| 0003 / 0004 | colour / black-and-white scan |
| 0011 / 0012 | dirty / darkened scan |
| 0005 | phone photo |
| 0006 | photo of a screen |
| 0009 / 0010 | crumpled photo / photo with a shadow |

It runs on GitHub Actions with the repository secret `KAGGLE_API_TOKEN` and the variable `KAGGLE_USERNAME`:

- `physionet-kaggle.yml` (manual): `mode=inspect` lists the files and downloads one example record; `mode=eval`, `n=20` draws N records (seed 2026), digitizes the 9 images of each and measures the SNR; `config=layout=3x4+1` forces the layout; `ids=a,b,c` evaluates specific records and keeps their raw images.
- `physionet-shards.yml` (manual): downloads the 20 records once (shared cache, slowly because of the Kaggle request limit) and evaluates them in 4 parallel shards.

**Metric:** SNR per image = 10·log10(Σ signal² / Σ noise²) summed over the 12 leads, only on the samples shown. "Aligned": per lead, a shift of up to ±0.2 s (1-sample step) and the mean difference removed (the competition metric tolerates the same). With a coarse step (4 ms) a render varied by ~6 dB with a 1 ms shift, so the step must be one sample. This is a reimplementation: the official leaderboard may differ in details. External reference: a deep-learning solution ranked 24th of 1,425 teams reports 19.7 dB on the private leaderboard.

## Diagnosis and report

Each run's artifact has `results.json` (per-image diagnosis: layout, gains, duration, ppmm, grid residuals, per-row origins), `sig/*.npz` (ground truth and prediction at 250 Hz), `img/*.jpg` (overlays) and `raw/` (raw images and ground truth of the first 5 records, for local debugging).

```bash
gh run download <run_id> -R fmenegui/ecg-digitalization-classic -D art
python benchmarks/physionet/diagnose.py art/physionet-eval-<run_id>        # -> decomposition.json
python benchmarks/physionet/report_html.py art/physionet-eval-<run_id> --before <previous_artifact> -o report.html
```

`diagnose.py` measures how much SNR each kind of error costs. It applies cumulative "oracle" corrections, each using the ground truth only for its own error, in this order: swapped layout/lead → empty stretches → time origin → horizontal scale → local horizontal distortion (0.5 s windows) → gain → baseline. What remains is shape error.

## Results

The local regression set is 135 images (15 records × 9 captures) saved from these runs. The history of the rounds (3.3 → 14.7 dB) is in [`docs/evaluation-and-roadmap.md`](../../docs/evaluation-and-roadmap.md).

The printing floor: on scans, the instant of each QRS varies by ±3 ms per beat relative to the ground truth, the same in every row and in several captures of the same sheet, while the grid is right to ±0.02 mm. The printed signal itself came out shifted (rasterization before printing), and no method can recover that from the image.
