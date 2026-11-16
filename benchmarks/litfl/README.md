# LITFL benchmark

102 real ECGs from the [Life in the Fast Lane](https://litfl.com/ecg-library/) library (scans and photos from books, websites and various devices), listed in `metadata.csv` (the same list as the `image-based-ecg-classification-emergency` repository). There is no true signal, so the benchmark measures two things.

**1. Structure, against a hand-reviewed reference** (`reference.json`)

| Field | Meaning | How it was annotated |
|---|---|---|
| `layout` | printed family: `3x4`, `3x4+1`, `3x4+2`, `3x4+3`, `6x2`, `rows` | visual review of the whole page |
| `rhythm` | rhythm-strip leads, top to bottom | label printed at the start of the strip, checked on a crop; `null` when unreadable |
| `speed` | mm/s | 25 everywhere (printed text where present) |
| `gain` | distinct gains on the page, mm/mV | calibration-pulse height and header text; `null` when the pulse is cut |
| `duration` | seconds per row | 10 (12-lead page) |
| `review` | `good` / `attention` / `failure` | visual review of 12/11/2026, context only |

Image 0043 is marked `unavailable` (the site refuses the download) and is left out.

**2. Internal consistency, without a reference**

- **Coverage:** fraction of the samples the page shows that were extracted.
- **Einthoven:** |I − II + III| / RMS(I, II, III) on the grid measurements before fusion. On a 3x4 page the three limb leads are in the same column at the same instant, so the residual measures tracing, gain and alignment errors.
- **Duplicate:** NRMSE between the rhythm strip II and the grid II over the same stretch. In several LITFL montages the strip is not simultaneous, so look at the median and at the change between versions, not at the absolute value.

## Run

```bash
pip install -e .
python benchmarks/litfl/run.py --jobs 2                     # downloads the images to benchmarks/litfl/cache/
python benchmarks/litfl/run.py --compare benchmarks/litfl/baseline.json
python benchmarks/litfl/run.py --config nonrigid=poly       # any Config field
```

Output in `results/litfl/`: `results.json` (per image and summary), `report.md` (metrics table and images with errors) and `report.html` with `img/` (summary + gallery of the rectified pages with the extracted trace; opens straight from disk). `--no-html` skips the images. The workflow `.github/workflows/litfl-benchmark.yml` runs the benchmark on every pull request that touches `ecgdig/` (and on demand) and publishes the report in the job summary, compared with `baseline.json`.

## Limits

The reference was made by looking at the program's own output; subtle label errors may have slipped through. Speed and duration hardly discriminate (almost everything is 25 mm/s and 10 s). The consistency metrics detect a worse tracing, but they do not prove the signal is right.
