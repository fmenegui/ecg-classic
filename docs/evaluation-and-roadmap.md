# ecgdig: evaluation and roadmap

*Classical ECG digitizer (no deep learning). History of the evaluations from 12/11/2026 to 16/11/2026, what each change bought, what still fails and what comes next.*

## 1. Where it stands

| Benchmark | Start | Now | Reference |
|---|---|---|---|
| PM-ECG-ID, whole database (5,994–6,000 ECGs): SNR / PCC / failed leads | 6.99 dB / 0.591 / 22.2% (`7a60059`) | **7.91 dB / 0.656 / 15.3%** (`43cb3a6`) | PMcardio, published: > 12.5 dB / > 0.91 / 6.62% |
| PM-ECG-ID, `photos_scans` (100 ECGs, desk scanner) | 11.5 dB / PCC 0.752 | 14.1 dB / PCC 0.875 | (whole-database reference above) |
| PhysioNet (Kaggle) training images, 135 local (15 records × 9 captures) | 3.3 dB | **14.68 dB** | a deep-learning entry ranked 24th of 1,425: 19.7 dB (private leaderboard) |
| LITFL, 102 real pages: layout / rhythm strips / layout+speed+duration+gain | 92.2 / 87.9 / 83.3% | 92–93 / 93–97 / 91% | hand-reviewed reference |
| Synthetic (exact ground truth): medium / heavy / fold | 21.6 / 16.7 / 9.7 dB | 21.6 / 19.4 / 18.1 dB | — |

**Where the PM-ECG-ID gap is.** When the page format is read correctly (72% of the single-page ECGs), the classical pipeline reaches 11.5 dB and PCC 0.86, close to PMcardio's whole-database numbers. When it is read wrong, the ECG is near 0 dB. On the first whole-database run the format was wrong or unreadable on 37% of the single-page ECGs; most of the work of 15–16/11 went into that, and the share fell to 28%.

## 2. Method choices that held up

- **Geometry from the grid:** local Fourier phase of the grid + quality-guided 2D unwrapping + RANSAC homography + a cubic polynomial and a local field on 5 mm cells, then the major (5 mm) and minor (1 mm) lines located one by one. Sub-millimetre rectification (0.02–0.05 mm on scans, 0.1–0.3 mm on photos).
- **Centre line by morphological inversion** (elliptical pen; opening of the top edge / closing of the bottom edge), refined by analysis-by-synthesis on thin, blurred traces: keeps the R peak.
- **Physics used explicitly:** Einthoven/Goldberger, duplicate copies of the same lead, simultaneity of the rows (beat registration, common rhythm), confidence-weighted fusion where traces merge.
- **Layout by hypothesis and verification:** every candidate format is scored by classical OCR of the labels, boundary marks, the Einthoven/Goldberger residual and duplicate agreement.

**Physical models: where they help.** As a prior where the pixel is ambiguous (occlusions, merges, boundaries, clipped peaks) and as a global consistency test, never overriding reliable pixels: 87–95% of the samples are already at the quantization limit, and a prior there could only erase pathology (ST, delta wave, notches, pacemaker spikes).

| Model | Worth it? | Why |
|---|---|---|
| Exact linear constraints (Einthoven/Goldberger, duplicates) | Yes (done) | Zero model error |
| The patient's own beat template (median beat, restricted DTW) | Tested as a global prior: −0.1 to −0.9 dB | A prior over the whole trace pulls good samples |
| Dipole / rank 3 (12 leads ≈ L·d(t)) | Open | Would separate merged traces and validate the layout |
| Sum of Gaussians (McSharry, Sameni) | Only as a template representation | As a generic prior it biases arrhythmias |
| Cylindrical torso | No | One image has no data to fit a torso geometry |
| EKF (Sameni et al. 2007) | Right idea, wrong tool | Offline needs a smoother; the image observation is multimodal |

## 3. PhysioNet (Kaggle), 12–13/11/2026

20 training records × 9 captures with the true signal; SNR with ±0.2 s alignment at a 1-sample step (with a 4 ms step a render varied by ~6 dB with a 1 ms shift).

| Round | Overall | What changed |
|---|---|---|
| Start (12/11) | 3.3 dB | layout 3x4+1 read on 44% of the images |
| 1st (13/11) | 11.0 dB | black grid on B&W scans reached by the template; paper slipped in the scanner fixed by locating the 5 mm lines (`_major_snap`); the pulse as a weak time anchor; spurious rows in photos (border, header, ruler); duration from the standard length most row ends confirm. Layout 3x4+1 read on 92% |
| 2nd (13/11) | 13.2 dB | each 1 mm grid line located with the trace masked out by colour (the classical equivalent of the intersection-indexed grid of the top Kaggle entries); separator mark cut as a rectangle; return-to-baseline and pulse-anchored origin in the Viterbi; robust pulse and duration on photos. Phone photos 3.7 → 11.3 dB, colour scans 12.1 → 15.2 dB |
| 3rd (13/11) | 13.6 dB (135 local) | analysis-by-synthesis fit of the trace (`ecgdig/synthesis.py`): M = 1 − exp(−Σ a·g_σ) with pen and blur estimated on the image, points moved along the normal until the drawing matches the ink; on thin traces only (radius ≤ 2 px) |
| 4th (13/11) | 14.57 dB | rigid translation in the Viterbi (a giant QRS crossing rows does not pay edge continuity: one render 10.3 → 29.9 dB); common rhythm between rows (an edge row without simultaneous QRS is dropped); ink bridge between segments; second pulse search |
| 16/11 | **14.68 dB** | the column, label and structure fixes of §5–§6; the merge first cost three images (14.47 dB), fixed by the straight-line and tracks guards of §6; one image stays 1.4 dB lower |

**Floor of the printing.** On scans, the instant of each QRS varies by ±3 ms per beat relative to the ground truth, the same in every row and in several captures of the same sheet, while the grid is right to ±0.02 mm. The printed signal itself is shifted; no method recovers that.

**Hybrid, first test (14/11).** A small U-Net (386k parameters) trained only on generic synthetic crops (`hybrid/gen.py`: paper, grid, pen, text, stains, moiré and degradations drawn at random; no PhysioNet/Kaggle image) used as the ink map, everything else classical. Five hours of CPU training (~4,700 steps): 14.57 → 13.69 dB on the 135 images (wins on screen and crumpled photos, loses on renders and phone photos). The best of the two per image would give 15.49 dB. The hybrid stays in its own branch until it is clearly better; the classical mode stays available.

## 4. PM-ECG-ID, first rounds (14–15/11/2026)

**The database:** Zenodo 13617673 (GPL-3.0-or-later). 6,600 images of 100 PTB-XL ECGs in 60 scenarios. **The metric:** the PMcardio paper's: ±0.5 s alignment, median removed, a failed lead counts as 0 (`benchmarks/pmcardio/`).

| Version | Set | SNR | PCC | failed |
|---|---|---|---|---|
| dev4 | 120 ECGs (2 per scenario) | 4.60 | — | — |
| dev20 | 120 ECGs | 7.50 | 0.653 | 13.1% |
| `7a60059` | whole database | 6.99 | 0.591 | 22.2% |

The 120-ECG sample was easier than the database and misled us (dev20 → dev21: +0.5 dB on the sample, 0 on the database). From then on every decision came from whole-database runs on GitHub Actions (~1h40 each, 20 shards), or from 2,400-ECG paired samples with one switch flipped. The paired standard error on the whole database is about 0.05 dB.

Fixes of these rounds: the PM generator's column gap removed from the image; a pulse column detached from the trace erased and used for the gain; the end of the row as anchor for 10 s pages; boundary evidence relative to the row background and weighted by agreement between rows; physics as a floored log-likelihood; Cabrera only with a "−aVR" label; the T wave after the QRS to break 0/180° ties; the rescued scale (5 mm taken for 1 mm) keeps the homography (black dotted grids −0.7 → +22 dB); the non-rigid polynomial kept within the observed range when it distorts the sheet; small sheet on cardboard (Otsu on the background colour, crop when the sheet fills < 30% of the frame); a light-trace threshold relative to the trace contrast; pulses on alternate rows; baseline drift > 1.5 mV removed where the tracker switched rows; blur (grid level recomputed without the blurred trace's halo; rare gains only from a printed left pulse).

## 5. Speed, determinism and quality score (15/11/2026)

Branch `claude/classico-melhorias`, merged on 16/11.

**1.9× faster with the same output** (8 pages, one process, 2 idle cores: 282.9 → 149.1 s): per-band medians by a single sort (`_nanmedian_axis`), vectorized grid-line loops and 3×3 windows (`rectify` 6.6 → 3.4 s); the 4D median filter of the ink map as `cv2.medianBlur` per template phase (3.6 → 0.8 s); all columns of `column_runs` at once; Python scalars in the Viterbi; the 2D Gaussian of the synthesis as a product of two 1D ones, with blocks computed once per iteration (3.3× faster, max difference 6e-13 px). Synthetic, PM-ECG-ID sample and LITFL outputs matched the previous version.

**The synthesis depended on the clock.** `synth_refine` stopped after 4 s of wall clock per segment, so the number of iterations, and the signal, depended on the machine and its load. The limit became a work budget (`SYNTH_WORK`, point-pixel pairs × iterations). On 16/11 the remaining 4 s emergency brake was still cutting long rhythm strips (~2,500 points, ~4–5 s) on loaded machines; it is now 30 s, a real emergency brake only.

**Quality score** (`res["quality"]`, `ecgdig/quality.py`): 0–1 per lead and per page, without ground truth (coverage × beat repetition × layout confidence). Against PM-ECG-ID ground truth (660 leads, "bad" = PCC < 0.5): AUC 0.75. Too little to refuse safely, because with a swapped layout each trace is a well-formed ECG with the wrong name. `Config(refuse_below=...)` exists but is off.

## 6. Page structure on PM-ECG-ID (15–16/11/2026)

Two sessions attacked the wrong-format pages, in parallel, and were merged on 16/11.

| Cause (seen on the debug images) | Fix | Example |
|---|---|---|
| Phase-unwrapping seed on an island (margin, footer): the homography came from a dozen clustered points | seed = best block of the largest connected region of good blocks; other regions join with the integer of the fitted plane (`geometry._unwrap_quality`) | scans 8, 10, 20, 47: −0.1 / 0.0 / −4.7 / 0.0 → 10.1 / 20.2 / 25.7 / 11.1 dB |
| GrabCut paper mask cutting the grid when the sheet touches the frame | border pixels with the paper colour are not background seeds (`PAPER_SEEDFIX`) | doogee −4.7 → 20.1 dB |
| Black or grey dashed grid left in the binary map (12 false rows on a 6x2 page) | straight pieces with neighbours 5 or 10 mm away removed for row finding, beat counting and OCR (`GRID_SEGS`) | |
| Curved grid border traced as the trace | vertical lines with trace on one side only removed (`LONG_VLINES`, `LONG_VLINES_EDGE`); chains of stacked QRS stay | scan 14: 12.7 → 19.3 dB |
| A ruled line (fold, curved grid line) with trace on both sides kept by the one-sided rule | a straight, thin line (centre within 0.12 mm of a fit, median width ≤ 0.5 mm) is removed anyway; QRS chains wiggle 0.15–0.25 mm and are 0.6–1 mm wide (`LONG_VLINES_STRAIGHT`) | PhysioNet 119693645-0005: −1.6 → 15.2 dB |
| Labels touching the column separator ("\|V2") | OCR variants without thin end bars; vertical lines > 6 mm are not letters; repeated "I" bars along a row are QRS (`OCR_QRS_BARS`, `OCR_TRIM_BARS`) | |
| Trace contrast falling across a photo | ink boosted per 5 cm block up to the page level, below the grid leakage (`LOCAL_CONTRAST`) | |
| Columns with their own row heights (6x2 with a tight left column and a spread right column; 3x4 with the four columns offset) | columns from blank gaps, from equal-width hypotheses (`EQUAL_SPLIT`) or from baseline tracks that end and start at the column changes (`col_struct`); each (column, row) band moved to stacked reference rows with the largest interval across columns; rows and column count passed downstream as hints (`RELAYOUT`) | img_23, 49, 76 (6x2) and 32, 50 (3x4): −2 to 0 → 9–24 dB |
| One header text line at a different height in each column taken for a column structure | the tracks structure needs at least max(2, n/3) rows that change height (`TRACKS_MANY_ROWS`) | PhysioNet 426433979-0004: 0.6 → 12.7 dB |
| QR code and "id: … page: …" footer counted as rows when matching columns | peaks with ink on < 50% of the column width dropped (`ROW_FILTER`) | img_32 −0.2 → 14.5 dB |
| Lead label printed inside the column gap | a gap filled only by letter-sized components still counts (`GAP_LABELS`) | img_44 −2.8 → 14.9 dB (with the next row) |
| A loose step at the end of the row read as a 5 mm/mV calibration | a half pulse on the right only counts if it gives 10 mm/mV (`RPULSE_HALF`) | |
| Wrong pulse feet replacing 12 regular projection rows by 3 | a regular projection block wins (`PULSE_ROWS_GUARD`) | |
| Upright page flipped 180° without pulse or label to break the tie | the other orientation is read up to the layout and wins only with a clear margin and closing physics (`orient_check`) | scan 33: −2.8 → 11.2 dB |
| Grid frame read as a lead | hypotheses that drop an edge row, only with calibration pulses as witnesses (`ROW_DROP`) | scan 66: −1.6 → 5.8 dB |

Whole-database runs (paired on the same ECGs; each fix was also checked by turning it off on a 2,400-ECG sample):

| Run | Commit | SNR | Notes |
|---|---|---|---|
| full2 | `e9324ef` | 6.95 | |
| seedfix | `5dc4a5f` | 7.02 | paper mask |
| head2 | `0fd51d6` | 7.08 | grid segments, OCR, long lines, equal split, local contrast, scale and span sanity |
| head3 | `1456ce0` | 7.30 | label in the gap, end-of-row step, relayout of per-column rows |
| head4 | `f2b1c08` | 7.43 | non-trace rows filtered |
| merge1 | `da0da70` | 7.91 | + baseline tracks, unwrap seed, orientation check, row drop, pulse-row guard (801 ECGs up > 1 dB, 382 down) |
| final | `43cb3a6` | **7.91** (PCC 0.656, 15.3% failed) | cleanup, deterministic synthesis brake, straight-line and tracks guards (PhysioNet back to 14.68 dB); 24.6 s per page on CI against 44.9 s before the merge |

Switches tested and left off: the row's own QRS masked as boundary evidence (`BND_OWN_QRS`: +0.08 dB on the pre-merge sample, but it also masks separator marks the tracker followed, and a 6x2 page fell from 10.9 to −0.9 dB after the merge); a loose "I" bar never contradicting a lead name (+0.02 dB on PM-ECG-ID, −0.03 dB on PhysioNet; removed); erasing pulse columns in the middle of the trace (rotation_35 img_41: 7.6 → −0.2 dB; removed); grid-segment cleanup also for tracking (removed); extending the paper mask to the gridded area (−0.13 dB).

## 7. What still fails

| Family | Symptom |
|---|---|
| Blur 31/41, resolution 1/16–1/32 | trace and grid merge; rows or pulses are lost (≈ 0 dB) |
| Rows 2–8 mm apart whose traces cross (PM-ECG-ID img_70, img_45, img_6) | separating rows by horizontal bands cannot untangle crossing curves; it needs tracking that follows crossing curves |
| Gray or touching labels (img_23 left column, img_39) | the OCR reads nothing and the format is decided by physiology alone |
| Crumpled and bent photos | local grid residual 0.35–0.5 mm per cell; shadows erase the grid |
| Digital opacity 0.15, brightness −160 | the trace is almost invisible |
| Gain on photos in perspective | the pulse shortens by up to ~20% |

## 8. Next steps

1. **Rows whose traces cross:** a joint tracker (k disjoint paths, or a coupled HMM with "all ink explained") instead of one Viterbi per band.
2. **OCR on light labels and labels crossed by the trace:** a grey-level OCR pass in the expected label windows.
3. **Hybrid, generic:** use a learned ink map only for reading the trace, keep rows, layout, pulses and time classical, and choose per image with internal criteria (Einthoven, duplicate II, common rhythm). No template and no known grid.
4. **Validation:** every change goes through the whole PM-ECG-ID database (paired), the 135 PhysioNet images and LITFL, because a single page moves by several dB with a 1% resize and every change moves hundreds of ECGs in both directions.

The per-scenario tables of each whole-database run are in [`docs/results/`](results/).
