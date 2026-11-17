# PM-ECG-ID, row network v2 (GPU run 44069775455) vs classical

Total: 6000/6000 ECGs (100%), paired image by image. Base run 44075875449, row network run 44132768955 (Config.row_model, fuse mode). Row network v2: rowgen strips + GPU-rendered pages, 70% of them photos brought back onto the flat page with the exact inverse field (hybrid/rendergen.py); trained 31k steps.

Columns: PMcardio-style mean SNR (failure = 0 dB), PCC, failed leads, and the official Kaggle score. Grid-phase v3 (run 44101364431) was neutral on the same database (SNR 8.53 -> 8.63 on 5100 ECGs) and stays off by default.

```
ALL                    n=6000 SNR   8.52 ->   9.41 | PCC 0.687 -> 0.705 | fail  13.4% ->  12.9% | Kaggle  11.66 ->  12.56
aug_JPEG               n= 600 SNR  11.36 ->  12.13 | PCC 0.851 -> 0.865 | fail   3.5% ->   3.2% | Kaggle  12.05 ->  12.58
aug_blur               n= 400 SNR   1.57 ->   1.57 | PCC 0.309 -> 0.308 | fail  42.0% ->  40.3% | Kaggle   2.52 ->   2.90
aug_brightness         n= 800 SNR   7.49 ->   8.67 | PCC 0.674 -> 0.701 | fail  11.7% ->  10.8% | Kaggle  10.18 ->  11.04
aug_contrast           n= 800 SNR   7.76 ->   9.12 | PCC 0.682 -> 0.710 | fail  10.6% ->  10.2% | Kaggle  10.21 ->  11.35
aug_padding            n= 400 SNR   8.93 ->  10.01 | PCC 0.783 -> 0.802 | fail   7.8% ->   7.7% | Kaggle   9.56 ->  10.71
aug_perspective        n= 300 SNR   7.34 ->   8.07 | PCC 0.740 -> 0.753 | fail   6.1% ->   5.6% | Kaggle   8.04 ->   9.00
aug_resolution         n= 500 SNR   5.46 ->   5.98 | PCC 0.519 -> 0.533 | fail  28.1% ->  27.6% | Kaggle   8.88 ->   9.14
aug_rotation           n= 400 SNR   9.28 ->  10.37 | PCC 0.811 -> 0.830 | fail   6.3% ->   6.0% | Kaggle   9.32 ->  10.55
digital                n=1100 SNR  11.86 ->  12.72 | PCC 0.709 -> 0.730 | fail  14.8% ->  14.4% | Kaggle  15.37 ->  16.40
photos_bents           n= 100 SNR  -0.11 ->   0.04 | PCC 0.393 -> 0.403 | fail  14.5% ->  14.2% | Kaggle   0.23 ->   0.41
photos_crumbles        n= 100 SNR  -2.67 ->  -1.83 | PCC 0.303 -> 0.329 | fail  19.8% ->  22.3% | Kaggle  -2.04 ->  -1.36
photos_doogee          n= 100 SNR  14.78 ->  15.93 | PCC 0.918 -> 0.933 | fail   2.2% ->   2.2% | Kaggle  13.98 ->  14.80
photos_iphone          n= 100 SNR  13.97 ->  14.74 | PCC 0.878 -> 0.886 | fail   4.2% ->   4.2% | Kaggle  13.90 ->  14.13
photos_samsung         n= 100 SNR  13.62 ->  14.39 | PCC 0.913 -> 0.923 | fail   1.2% ->   0.9% | Kaggle  13.02 ->  13.70
photos_scans           n= 100 SNR  14.69 ->  16.06 | PCC 0.910 -> 0.921 | fail   2.2% ->   2.1% | Kaggle  13.58 ->  14.47
photos_screens         n= 100 SNR   7.89 ->   8.49 | PCC 0.712 -> 0.722 | fail  11.7% ->  11.5% | Kaggle   8.83 ->   9.38
```
