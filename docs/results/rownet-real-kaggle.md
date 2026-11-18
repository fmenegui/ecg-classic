# Row network trained with real Kaggle pages (GPU run 44206699093)

Strips: 35% real Kaggle training pages (3,755 of 4,000 pages kept; holdout records id % 5 == 0 and the local 15 never read), 40% rendered (70% photos rectified with the exact field), rest rowgen; 17k steps. Fuse mode.

## Kaggle, holdout records only (never used for training)

Total: 1,854 images of 206 records (21% of the training set), paired with the classical run 44011597597; network run 44237830781.

```
holdout: 1854 images, 206 records | Kaggle 17.42 -> 17.88
  0001  206   24.68 ->  24.96
  0003  206   16.03 ->  16.51
  0004  206   13.78 ->  14.74
  0005  206   12.90 ->  13.51
  0006  206   10.68 ->  12.27
  0009  206   11.82 ->  12.49
  0010  206   14.99 ->  15.39
  0011  206   15.87 ->  16.43
  0012  206   13.43 ->  14.71
```

Training records (seen by the network): train: 6939 images, 771 records | Kaggle 17.62 -> 18.08 - same gain as on the holdout, so no memorisation.

## PM-ECG-ID, 5,990 of 6,000 ECGs (99.8%), paired with the classical run 44075875449; network run 44237642516

```
== PM real vs base
ALL                    n=5990 SNR   8.52 ->   9.46 | PCC 0.687 -> 0.706 | fail  13.4% ->  12.7% | Kaggle  11.66 ->  12.78
aug_JPEG               n= 600 SNR  11.36 ->  12.14 | PCC 0.851 -> 0.866 | fail   3.5% ->   3.1% | Kaggle  12.05 ->  12.53
aug_blur               n= 400 SNR   1.57 ->   1.63 | PCC 0.309 -> 0.309 | fail  42.0% ->  40.4% | Kaggle   2.52 ->   2.89
aug_brightness         n= 800 SNR   7.49 ->   8.59 | PCC 0.674 -> 0.702 | fail  11.7% ->  10.6% | Kaggle  10.18 ->  11.09
aug_contrast           n= 800 SNR   7.76 ->   9.12 | PCC 0.682 -> 0.711 | fail  10.6% ->   9.8% | Kaggle  10.21 ->  11.33
aug_padding            n= 400 SNR   8.93 ->   9.99 | PCC 0.783 -> 0.801 | fail   7.8% ->   7.6% | Kaggle   9.56 ->  10.71
aug_perspective        n= 300 SNR   7.34 ->   7.89 | PCC 0.740 -> 0.748 | fail   6.1% ->   5.5% | Kaggle   8.04 ->   9.12
aug_resolution         n= 500 SNR   5.46 ->   6.03 | PCC 0.519 -> 0.531 | fail  28.1% ->  27.6% | Kaggle   8.88 ->   9.26
aug_rotation           n= 400 SNR   9.28 ->  10.36 | PCC 0.811 -> 0.830 | fail   6.3% ->   5.9% | Kaggle   9.32 ->  10.52
digital                n=1093 SNR  11.87 ->  13.08 | PCC 0.709 -> 0.733 | fail  14.8% ->  14.3% | Kaggle  15.38 ->  16.89
photos_bents           n=  98 SNR  -0.07 ->  -0.08 | PCC 0.394 -> 0.402 | fail  14.3% ->  13.4% | Kaggle   0.26 ->   0.30
photos_crumbles        n= 100 SNR  -2.67 ->  -2.36 | PCC 0.303 -> 0.325 | fail  19.8% ->  21.3% | Kaggle  -2.04 ->  -1.67
photos_doogee          n= 100 SNR  14.78 ->  15.80 | PCC 0.918 -> 0.933 | fail   2.2% ->   2.0% | Kaggle  13.98 ->  14.57
photos_iphone          n=  99 SNR  14.06 ->  15.03 | PCC 0.879 -> 0.891 | fail   4.3% ->   4.2% | Kaggle  13.94 ->  14.10
photos_samsung         n= 100 SNR  13.62 ->  14.48 | PCC 0.913 -> 0.925 | fail   1.2% ->   0.9% | Kaggle  13.02 ->  13.51
photos_scans           n= 100 SNR  14.69 ->  16.14 | PCC 0.910 -> 0.927 | fail   2.2% ->   1.6% | Kaggle  13.58 ->  14.47
photos_screens         n= 100 SNR   7.89 ->   8.58 | PCC 0.712 -> 0.736 | fail  11.7% ->   9.2% | Kaggle   8.83 ->   9.60
== PM real vs row v2
ALL                    n=5990 SNR   9.41 ->   9.46 | PCC 0.705 -> 0.706 | fail  12.9% ->  12.7% | Kaggle  12.57 ->  12.78

```
