# Hybrid mode: a learned ink map inside the classical pipeline

**Why.** The layout oracle on PM-ECG-ID (see [`evaluation-and-roadmap.md` §6b](evaluation-and-roadmap.md)) showed that the format decision is not the bottleneck: the pages that fail do so because the trace is not separated from the grid, the background or the degradation (blur, low resolution, light or desaturated ink, a black grid as dark as the pen). The same ECG reads at 20 dB on the clean page and at 3–6 dB after JPEG, a brightness change or a phone photo. On PhysioNet the format is already right and the gap to the leaders (14.7 vs 19.7–22.7 dB) is in the trace itself. That is exactly what a pixel classifier learns well and what hand-written rules learn badly.

**What stays classical.** Everything that can be checked: grid geometry and rectification, rows, columns, layout hypotheses, OCR of the labels, calibration pulses, speed, gain, time origin, tracking, analysis-by-synthesis and the physiological constraints. The network only answers one question: *where is the pen ink* in the rectified image (10 px/mm).

**No template, no known page.** The network is trained on generic synthetic crops (`hybrid/gen.py`): paper, grid style and colour, pen colour and width, text, stamps, stains, moiré, blur, JPEG, shadows and noise drawn at random, with no page layout, no fixed grid and no image from PhysioNet/Kaggle or PM-ECG-ID. The label is the anti-aliased pen coverage before the degradations, so the network learns to undo them.

## How it is used

```python
from ecgdig import digitize, Config
res = digitize("ecg.jpg", Config(ink_model="trace_seg.onnx"))   # hybrid="select" by default
res["hybrid"]   # {"chosen": "model" | "classical", "score_classical": ..., "score_model": ..., ...}
```

- `hybrid="select"` (default with a model): the page is read twice, with the classical ink map and with the network's, and the reading with the better internal quality is kept. The score uses no ground truth: page quality (coverage × beat repetition × layout confidence) plus the Einthoven/Goldberger closure of the chosen layout (`digitize._reading_score`). Reading twice doubles the time.
- `hybrid="model"`: the network's map only. `ink_mix < 1` blends it with the classical map.
- Without `ink_model` nothing changes: the classical digitizer is the default.

## Components

| File | Role |
|---|---|
| `hybrid/gen.py` | generator of training crops and labels |
| `hybrid/train.py` | small U-Net (~0.4 M parameters), BCE + Dice, cosine schedule, ONNX export; `--init` continues from a previous run |
| `.github/workflows/hybrid-train.yml` | CPU training on GitHub Actions (~5 h per run); `init_run` chains runs |
| `ecgdig/inkmodel.py` | tiled ONNX inference (onnxruntime) |
| `ecgdig/digitize.py` | `Config.ink_model`, `Config.hybrid`, `_reading_score` |
| `.github/workflows/pmcardio.yml` | `ink_model_run` evaluates a trained model on PM-ECG-ID |

## Results so far

| Date | Model | PhysioNet 135 | Notes |
|---|---|---|---|
| 14/11 | 4,700 CPU steps (loss 0.42, still falling) | classical 14.57 → network only 13.69 dB; best of the two per image 15.49 dB | wins on screens and crumpled photos, loses on renders and phone photos |
| 16/11 | same model, current classical pipeline | classical 14.70, network only 12.16, best of the two per image 15.37 dB; automatic choice (page score + 1.0 × limb closure, margin 0.2) 14.80 dB | the internal score finds only a fraction of the oracle gain; the downstream rules (row filters) were tuned on the classical map and drop rows the network draws well (12x1 photo, ECG 70) |

## Comparison with the Kaggle leaders

| | ecgdig hybrid (now) | Kaggle 5th place (22.6 dB private LB) | Oxford, CinC 2026 (16.9 dB, own split) |
|---|---|---|---|
| Page geometry | classical grid detection and rectification, any layout | keypoint U-Net (2,422 points) registered to the competition's reference template | grid-intersection U-Net, warp to a standard frame |
| Layout / leads | classical hypotheses, OCR, pulses | known from the template | 4 trace channels of the competition format |
| Network output | one ink map for the whole page (where is pen ink) | per lead, a column-wise 1-D heatmap of the signal height | 4 soft trace maps, then a BiLSTM reads the offset per column |
| Trace to signal | classical Viterbi tracking | argmax / expectation of the heatmap | learned (attention + BiLSTM) |
| Training data | generic synthetic crops, no test-domain image | competition images (PTB-XL rendered by ecg-image-kit), pseudo-labels | competition images |
| Generic? | yes (no template, no known grid) | no: relies on the single competition template | partly (fixed 4-row format) |

The leaders' gain comes from two things: a network that outputs the signal height per column for each lead (it resolves overlaps and faint ink itself), and training on images of the test distribution. The plan keeps the generic classical structure and adds both: a per-row height network on the rows found by the classical pipeline, trained on PTB-XL renders (ecg-image-kit), the Kaggle training images and our generator, split by record.

Sources: [5th place code](https://github.com/dangnh0611/kaggle_ecg_digitization), [McIntosh et al., CinC 2026 preprint](https://cinc.org/2026/Program/accepted/550_Preprint.pdf).

## Plan

1. **Train on a GPU.** The first model is tiny and stopped at a loss still falling. `.github/workflows/gpu-train.yml` starts a RunPod pod (secret `RUNPOD_API_KEY`), trains a U-Net with a pretrained encoder (`hybrid/train_gpu.py`), returns the ONNX and always terminates the pod; chained CPU runs (`init_run`) remain as a fallback.
2. **Selection without ground truth.** Calibrate `_reading_score` on PhysioNet (where both readings are known) so that the per-image choice gets close to the 15.49 dB oracle; then confirm on PM-ECG-ID, paired.
3. **Closer to real captures.** Phone-photo optics (perspective blur, vignetting, uneven light), printer dot patterns and the black/grey grids where the classical map fails; pseudo-labels from pages the classical reader gets right (high internal quality) to adapt to real paper and cameras, still without any test image.
4. **Per-lead choice.** When both readings agree on the layout and time axis, keep each lead from the reading with the better lead score.
5. **Validation rule** as for the classical work: whole PM-ECG-ID database paired, PhysioNet 135, LITFL; the hybrid becomes the recommended mode only when it wins on all three.
