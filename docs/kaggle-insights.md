# Kaggle "PhysioNet - Digitization of ECG Images": what the top solutions did
## 1st (outrunner et al., 22.98 public / 22.75 private single; ensemble 10)
- uses hengck23 public rectification (stage0 rotation+homography, stage1 44x57 grid point remap)
- direct regression of 4 rows x 20000 values from rectified grayscale + coord channels, huge input 5056x1280, SmoothL1
- scipy.signal.resample vs linear: 23.83 vs 22.18 (1 image); remap from high-res vs low-res: 23.83 vs 21.27
- Einthoven blending +0.06, TTA +0.07; blank synthetic images, moire, flare
- grid point subpixel by weighted average of feature map, surface fitting outliers
- scan 0003 grid-signal offset differs from 0001
## 2nd (Someya, 23.27 private)
- hengck23 stage0/1 unchanged; replace Kaggle signals by PTB-XL 500 Hz originals (match by correlation): 21.67 -> 22.49 single fold
- metric: mean SNR in LINEAR domain then dB -> push medium/easy images
- SPARSE mask: <=2 px per column, weight split by fractional y; decode weighted mean per column; dense masks had low reconstruction SNR
- mask width 5600 for 5000 samples (no resampling); BCE pos_weight 20
- 2.5D series model: 4 rows (crop +-3 mV around 0 mV) shared encoder, fused features (conv2d) -> shares phase/amplitude across leads; recovers masked regions
- scipy.signal.resample best; hflip TTA
## 3rd (Tang)
- params at low res, image warp at high res (original image) -> big gain
- label = 1-px line of the SIGNAL (not ink); gaussian soft label did not help; overfit to thin lines
- mask reconstruction ceiling: width 2200 -> 24.2 dB, width 4400 (height same) -> 31.2 dB: horizontal resolution limits SNR
- subpixel: parabolic interpolation around argmax beats top-K centroid
- ConvNeXtV2 > HRNet > EffNetV2; signal-level ensemble; heavy aug incl ElasticTransform (rectification residuals)
- lead II fusion with rhythm strip; Einthoven correction (II weight 2); TTA flips
## 4th (tomoon33, 22.69 private)
- stage1 sliding 704 window multitask: grid intersection heatmap + layout seg (separators, calibration pulses) -> orientation from layout
- stage2 line index per pixel (55/43 classes), majority vote near intersection
- rectify with grid_sample to fixed 1700x2200 (template); split width in 4 segments (3 short + rhythm stacked) -> model uses rows above/below to fill gaps
- regression model: asymmetric decoder (only horizontal upsampling after 1/8), vertical GAP + 1D conv; aux seg head; SNR-based loss
- external: PTB-XL and CODE-15% rendered with ecg-image-kit
- FCMAE self-supervised pretraining on ECG images; ~1000 manually corrected annotations
- weighted MEDIAN ensemble (robust to spikes)
## 6th (AnnieGo, ~22.4)
- hengck23 baseline stage 2 replaced by direct regression: per-row softmax over y (soft-argmax, temperature 0.5) -> y -> mV with known zero line and mm/mV
- FFT resample (scipy.signal.resample) >> linear; intermediate length 10250 > 5120 > 2560 (2560 -> 21.36, 5120 -> 22.20)
- 24x RTX3090 for 10 h (!)
## 7th (Bartley + Harshit, 22.48 private) - the least template-bound
- NO grid rectification: rotation classifier (0/90/180/270) -> lead detection boxes + 13-class lead segmentation (trained on modified ecg-image-kit: PTB-XL + DTD textures backgrounds, shadows) -> crop per lead
- digitization input 5 channels: RGB + target-lead probability + max prob of other leads (overlap robustness); 1D unet decoder on pooled encoder features
- SNR loss with learned/optimal vertical shift; heavy aug (color, distortion, rotation, flips, thin dropout as pen marks, ECG text)
- ViT (dinov3 small) patch->pixel 1D output; progressive width 896 -> 3584
- OOD detector (ArcFace embeddings) to gate specialist models: +0.4
- Muon optimizer clearly better
- 100 validation samples enough for CV/LB correlation
- lead detection trained only on 3x4+1R; 6x2 needs fine-tuning (not generic in layout)
## hengck23 "from 0.16 to 0.20" (16th; author of the public rectification everyone used)
- resolution x2 helps; resolution limited by alignment; simulation: <1 px alignment error -> ~21 dB
- rectification had 1-2 px (up to 5) error; subpixel corner/line heuristics; only need to rectify cells containing signal (move corners to nearest reference grid, barycentric per quad)
- periodicity of ECG to fill occlusions/dirt (+3 dB local, +0.5 LB); host (Clifford): also reconstruct from simultaneous leads; combine and check agreement
- 0001 benefits most from mask at high res; others limited by rectification error -> refine target masks by nudging
- private set harder than public; robust > tuned heuristics
## hengck23 "design winning solution"
- GT label must be subpixel: draw 1px antialiased line on big canvas, INTER_AREA down; post-processing from truth mask -> signal must reach ~25 dB
- coarse-to-fine; flip TTA averages pixel drift; train with infinite synthetic data with occlusion
- truth itself is imperfect (1 px shift): iterate "model, better truth = learn(data, truth)"
- stretch input horizontally for sharp peaks -> better SNR
- shortcut idea (GENERIC): per-pixel UV map = predict mV and seconds for each pixel directly (no unwarping), needs lots of data
- Open-ECG-Digitizer (Ahus-AIM, Stenhede; arXiv 2510.19590): open U-Net weights, 4 classes background/text/signal/grid
- "II appears twice" (rhythm strip + short lead): fuse
## 13th (21.46): E2E seg -> differentiable centroid -> 1D ResUNet refinement, L1+SNR; hflip aug +0.6 dB; resampling ensemble (fir, polyphase, linear); BiLSTM worse than 1D ResUNet
## 14th (21.37): HRNet-W48, CoordConv, high res 1696x4352 (+3 dB, biggest); BCE pos_weight 10 beat MSE/SNR losses; segment-type-specific aug (moire, water stains, folds, mold); Einthoven alpha 0.33
- hengck23: direct regression tolerates misalignment (crooked lines are cues)
## Host (Sameni): zero out leads you suspect are wrong (0 dB beats negative); linear-phase resampling same for all leads; diagnose outliers; QRS first, then low-power segments; overlaps are the main bottleneck of open models

## Metric note (checked on the competition's Evaluation page, 16/11/2026)

"The competition score is the average of these individual ECG SNRs, converted to decibel scale": the per-image SNR is averaged in the LINEAR domain, then converted to dB. Our reports averaged dB per image. On the local 135-image set (15 records x 9 captures) the classical reader is **14.70 dB as a mean of dB but 18.77 dB on the Kaggle scale** (network-only ink model: 12.16 / 17.22). `eval_snr.py` now prints both.

## What it means for ecgdig (generic: no template)

1. **Read the signal, not the ink.** 2nd/3rd place train on a sparse 1-2 px label of the signal height per column (sub-pixel by splitting the weight), decoded by weighted centroid or parabolic peak; dense ink masks cap the reconstruction SNR. The row network should predict this (it is template-free: it lives in a row strip).
2. **Horizontal resolution and resampling.** Width 2200 -> 24.2 dB, 4400 -> 31.2 dB reconstruction ceiling; FFT resampling (scipy.signal.resample) beats linear by up to ~1.6 dB; warp from the original high-resolution image, never from a downscaled one.
3. **Geometry precision.** Sub-pixel grid alignment is worth several dB (< 1 px error ~ 21 dB). Generic route: predict local grid phase (UV map: mV and seconds per pixel) instead of template line indices.
4. **Use the other leads.** 2.5D models sharing the 4 rows, Einthoven/Goldberger blending (+0.06), lead II short + rhythm fusion, ECG periodicity to fill occlusions (+0.5 LB).
5. **Zero a lead you do not trust** (0 dB beats a negative SNR), as the host suggests; our per-lead quality score can drive it.
6. Training details that paid off: hflip aug/TTA (+0.3-0.6), median ensembles, Muon optimizer, EMA, PTB-XL/CODE-15% renders via ecg-image-kit, original 500 Hz PTB-XL signals as labels (+0.8).
