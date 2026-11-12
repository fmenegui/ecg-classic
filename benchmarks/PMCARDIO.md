# PM-ECG-ID pilot

Dataset: https://zenodo.org/records/13617673 (33.9 GB ZIP). Unpack it locally on a machine with sufficient disk. No dataset images or ground truth are committed to Git.

Run:

```bash
pip install -e .
python benchmarks/pmcardio_pilot.py --dataset /path/to/unpacked/pmcardio --n 100
```

The sampler round-robins over image directory groups with seed 2026. It records the sampled metadata, per-image exceptions, runtime, extracted signals, and overall success rate. Sampling is reproducible, but does not guarantee 100 distinct patients or balanced ECG formats.

**Important:** This is an execution and coverage pilot, not a signal-accuracy benchmark. PMcardio `leads.npz` and `rhythms.npz` hold layout-specific cutouts. Computing SNR requires verifying lead order, units, visibility masks, and the time origin for each page. Do not report an SNR until these have been checked against representative metadata and ground-truth arrays.

The optional GitHub Actions workflow requires a self-hosted Linux x64 runner labeled `pmcardio` with the dataset already unpacked. GitHub-hosted runners are intentionally not used for a 33.9 GB download.
