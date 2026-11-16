# PM-ECG-ID (PMcardio ECG Image Database) benchmark

[PM-ECG-ID](https://doi.org/10.5281/zenodo.13617673) (Iring et al., 2024, GPL-3.0-or-later): 6,600 images of 100 PTB-XL ECGs in 60 scenarios: phone photos and scans (`photos_*`), JPEG compression, blur, brightness, contrast/gamma, padding, perspective, rotation, resolution down-scaling and digital variants with noise or low opacity. Formats: 6x2, 3x4, 3x4 + 1 or 3 rhythm strips, 6x2 + 1 strip and 12x1; a few ECGs span 2 or 4 pages.

**Metric:** the one of the PMcardio paper (Demolder et al., medRxiv 2024.08.31.24312876). Per lead: alignment of up to ±0.5 s, vertical offset removed by the median of the difference, then PCC, RMSE (mV) and SNR (dB). A lead that is missing or less than 50% covered is a **failure** and counts as SNR 0, PCC 0. Pages of the same ECG are merged per lead before the comparison.

**Published reference** (PMcardio, whole database): SNR > 12.5 dB, PCC > 0.91, 6.62% of leads failed.

## Run

The zip has 34 GB; `fetch.py` downloads only the needed files by HTTP ranges.

```bash
pip install -e . remotezip pandas
python benchmarks/pmcardio/fetch.py --out data/pm --per-scenario 2        # stratified sample (2 ECGs per scenario)
python benchmarks/pmcardio/eval_pm.py --root data/pm/visual_data --gt data/pm/final_data/data \
    --meta data/pm/final_data/metadata.csv --list data/pm/sel.csv --out results/pm --jobs 2
python benchmarks/pmcardio/compare.py results/pm/results.json --gt data/pm/final_data/data --out results/pm
```

The whole database runs on GitHub Actions (`.github/workflows/pmcardio.yml`, manual): 20 shards of ~330 images, about 1h40. Inputs:

- `per_scenario`: ECGs per scenario (0 = whole database; 40 gives a 2,400-ECG sample);
- `config`: `Config` fields, e.g. `synth=False`;
- `flags`: ablation switches over the module-level 1-element lists in `ecgdig.*`, e.g. `digitize.GAP_IGNORE_TEXT=False;colalign.STRUCT=False`. This is how each correction was kept or dropped: the same code with one switch off, on the same ECGs.

```bash
gh run download <run_id> -D art
python benchmarks/pmcardio/merge.py art merged          # -> merged/results.json, summary.md
python benchmarks/pmcardio/compare.py merged/results.json --gt data/pm/final_data/data --out merged
python benchmarks/pmcardio/ab.py merged_before merged_after --min-delta 1.0   # ECG by ECG
```

## Reading the numbers

A single page can move by several dB with a 1% resize, and every change moves hundreds of ECGs in both directions. The paired standard error of the mean on the whole database is about 0.05 dB; a 120-ECG sample misled us more than once (+0.5 dB on the sample, 0 on the database). Decisions are made on whole-database or 2,400-ECG paired runs.

Results and history: [`docs/evaluation-and-roadmap.md`](../../docs/evaluation-and-roadmap.md) and the per-scenario tables in [`docs/results/`](../../docs/results/).
