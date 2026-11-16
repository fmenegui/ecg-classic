# ecgdig (classical) on PM-ECG-ID

Commit `43cb3a6` (16/11/2026): both classical branches merged, cleanup and the PhysioNet fixes. GitHub Actions run 43837701674, 6,000 ECGs, 24.6 s per page on average (44.9 s before the merge).

Failure = lead missing or < 50% covered; it counts as SNR 0, PCC 0 and the RMSE of a null signal.

## Whole database

| set | ECGs | SNR | PCC | RMSE | failure | SNR (no failures) | PCC (no failures) | RMSE (no failures) |
|---|---|---|---|---|---|---|---|---|
| ecgdig | 6000 | 7.91 | 0.656 | 0.386 | 15.3% | 9.34 | 0.774 | 0.415 |
| PMcardio (published) | 100 | > 12.5 | > 0.91 | — | 6.62% | — | — | — |

Groups of the PMcardio paper (no scenario list): easy (12): PCC 0.977, SNR 16.68 dB; medium (29): PCC 0.970, SNR 15.79 dB; hard (19): PCC 0.817, SNR 11.10 dB

## By scenario family

| set | ECGs | SNR | PCC | RMSE | failure | SNR (no failures) | PCC (no failures) | RMSE (no failures) |
|---|---|---|---|---|---|---|---|---|
| JPEG_compression | 600 | 10.87 | 0.818 | 0.103 | 5.4% | 11.49 | 0.864 | 0.097 |
| blur_kernel | 400 | 1.29 | 0.284 | 0.214 | 43.7% | 2.29 | 0.505 | 0.208 |
| brightness | 800 | 6.60 | 0.636 | 0.254 | 13.6% | 7.63 | 0.736 | 0.258 |
| contrast_gamma | 800 | 7.19 | 0.658 | 0.163 | 11.1% | 8.08 | 0.740 | 0.157 |
| digital: noise | 600 | 12.30 | 0.751 | 2.302 | 8.9% | 13.50 | 0.824 | 2.495 |
| digital: opacity | 500 | 9.46 | 0.587 | 0.259 | 28.0% | 13.13 | 0.816 | 0.273 |
| padding | 400 | 8.43 | 0.753 | 0.122 | 9.4% | 9.30 | 0.831 | 0.114 |
| perspective | 300 | 6.63 | 0.702 | 0.145 | 8.3% | 7.23 | 0.766 | 0.139 |
| photos | 700 | 8.47 | 0.698 | 0.149 | 9.7% | 9.37 | 0.773 | 0.142 |
| resolution_scale_down | 500 | 5.04 | 0.485 | 0.167 | 30.9% | 7.29 | 0.702 | 0.143 |
| rotation | 400 | 8.76 | 0.776 | 0.114 | 8.3% | 9.55 | 0.846 | 0.106 |

## By scenario

| set | ECGs | SNR | PCC | RMSE | failure | SNR (no failures) | PCC (no failures) | RMSE (no failures) |
|---|---|---|---|---|---|---|---|---|
| augmentation_JPEG_compression_11_percent | 100 | 11.74 | 0.837 | 0.091 | 4.8% | 12.33 | 0.879 | 0.086 |
| augmentation_JPEG_compression_22_percent | 100 | 12.64 | 0.871 | 0.081 | 2.9% | 13.02 | 0.897 | 0.079 |
| augmentation_JPEG_compression_2_percent | 100 | 5.16 | 0.665 | 0.168 | 10.8% | 5.79 | 0.745 | 0.160 |
| augmentation_JPEG_compression_45_percent | 100 | 12.68 | 0.868 | 0.085 | 3.9% | 13.20 | 0.903 | 0.081 |
| augmentation_JPEG_compression_5_percent | 100 | 10.07 | 0.803 | 0.111 | 5.7% | 10.67 | 0.851 | 0.103 |
| augmentation_JPEG_compression_90_percent | 100 | 12.95 | 0.863 | 0.084 | 4.2% | 13.52 | 0.901 | 0.075 |
| augmentation_blur_kernel_11 | 100 | 5.16 | 0.633 | 0.152 | 14.6% | 6.04 | 0.741 | 0.144 |
| augmentation_blur_kernel_21 | 100 | 0.47 | 0.228 | 0.229 | 42.7% | 0.82 | 0.398 | 0.226 |
| augmentation_blur_kernel_31 | 100 | -0.26 | 0.183 | 0.241 | 43.3% | -0.46 | 0.323 | 0.262 |
| augmentation_blur_kernel_41 | 100 | -0.22 | 0.092 | 0.236 | 74.3% | -0.85 | 0.358 | 0.261 |
| augmentation_brightness_-120 | 100 | 4.56 | 0.590 | 0.196 | 15.8% | 5.42 | 0.701 | 0.193 |
| augmentation_brightness_-160 | 100 | -2.44 | 0.322 | 0.371 | 24.2% | -3.22 | 0.425 | 0.413 |
| augmentation_brightness_-40 | 100 | 11.45 | 0.817 | 0.104 | 5.1% | 12.06 | 0.861 | 0.098 |
| augmentation_brightness_-80 | 100 | 9.50 | 0.760 | 0.125 | 9.2% | 10.46 | 0.837 | 0.116 |
| augmentation_brightness_120 | 100 | 5.96 | 0.641 | 0.523 | 8.6% | 6.52 | 0.701 | 0.553 |
| augmentation_brightness_160 | 100 | 0.84 | 0.312 | 0.519 | 32.9% | 1.26 | 0.466 | 0.656 |
| augmentation_brightness_40 | 100 | 11.99 | 0.833 | 0.091 | 5.0% | 12.62 | 0.877 | 0.085 |
| augmentation_brightness_80 | 100 | 10.92 | 0.815 | 0.100 | 7.8% | 11.84 | 0.884 | 0.089 |
| augmentation_contrast_gamma_0.125 | 100 | 1.15 | 0.333 | 0.223 | 28.8% | 1.61 | 0.467 | 0.224 |
| augmentation_contrast_gamma_0.25 | 100 | 5.82 | 0.603 | 0.161 | 13.2% | 6.70 | 0.694 | 0.153 |
| augmentation_contrast_gamma_0.5 | 100 | 10.95 | 0.814 | 0.102 | 5.7% | 11.61 | 0.863 | 0.096 |
| augmentation_contrast_gamma_0.666 | 100 | 12.27 | 0.858 | 0.085 | 3.7% | 12.74 | 0.891 | 0.081 |
| augmentation_contrast_gamma_1.5 | 100 | 12.52 | 0.850 | 0.087 | 6.5% | 13.39 | 0.909 | 0.082 |
| augmentation_contrast_gamma_2 | 100 | 10.92 | 0.782 | 0.109 | 6.6% | 11.69 | 0.838 | 0.100 |
| augmentation_contrast_gamma_4 | 100 | 5.75 | 0.634 | 0.183 | 10.2% | 6.40 | 0.706 | 0.179 |
| augmentation_contrast_gamma_8 | 100 | -1.87 | 0.391 | 0.354 | 14.0% | -2.18 | 0.454 | 0.377 |
| augmentation_padding_extra_large | 100 | 5.63 | 0.601 | 0.168 | 17.7% | 6.84 | 0.730 | 0.156 |
| augmentation_padding_large | 100 | 9.33 | 0.786 | 0.111 | 7.3% | 10.07 | 0.848 | 0.103 |
| augmentation_padding_medium | 100 | 9.57 | 0.808 | 0.099 | 7.0% | 10.29 | 0.869 | 0.094 |
| augmentation_padding_small | 100 | 9.18 | 0.816 | 0.110 | 5.6% | 9.72 | 0.865 | 0.106 |
| augmentation_perspective_transformation_20 | 100 | 8.24 | 0.770 | 0.123 | 5.6% | 8.73 | 0.816 | 0.118 |
| augmentation_perspective_transformation_30 | 100 | 7.25 | 0.755 | 0.138 | 5.1% | 7.63 | 0.796 | 0.134 |
| augmentation_perspective_transformation_40 | 100 | 4.39 | 0.581 | 0.173 | 14.2% | 5.13 | 0.677 | 0.168 |
| augmentation_resolution_scale_down_factor_16 | 100 | -0.17 | 0.213 | 0.237 | 41.1% | -0.28 | 0.361 | 0.247 |
| augmentation_resolution_scale_down_factor_2 | 100 | 13.28 | 0.869 | 0.072 | 5.1% | 13.99 | 0.916 | 0.065 |
| augmentation_resolution_scale_down_factor_32 | 100 | -0.32 | 0.030 | 0.237 | 85.1% | -2.16 | 0.201 | 0.284 |
| augmentation_resolution_scale_down_factor_4 | 100 | 10.49 | 0.844 | 0.095 | 4.3% | 10.96 | 0.882 | 0.092 |
| augmentation_resolution_scale_down_factor_8 | 100 | 1.92 | 0.470 | 0.196 | 18.7% | 2.36 | 0.578 | 0.192 |
| augmentation_rotation_15 | 100 | 9.10 | 0.808 | 0.105 | 6.6% | 9.74 | 0.865 | 0.099 |
| augmentation_rotation_25 | 100 | 8.78 | 0.762 | 0.119 | 10.1% | 9.76 | 0.847 | 0.110 |
| augmentation_rotation_35 | 100 | 7.89 | 0.719 | 0.126 | 10.5% | 8.82 | 0.803 | 0.117 |
| augmentation_rotation_5 | 100 | 9.27 | 0.815 | 0.105 | 5.9% | 9.86 | 0.866 | 0.099 |
| digital_data_high_freq_noise_large | 100 | 8.91 | 0.775 | 0.958 | 7.5% | 9.63 | 0.838 | 1.014 |
| digital_data_high_freq_noise_low | 100 | 14.57 | 0.853 | 0.386 | 5.2% | 15.37 | 0.899 | 0.398 |
| digital_data_high_freq_noise_middle | 100 | 12.16 | 0.843 | 0.328 | 3.4% | 12.59 | 0.873 | 0.334 |
| digital_data_low_freq_noise_large | 100 | 7.35 | 0.465 | 11.888 | 20.0% | 9.19 | 0.581 | 14.750 |
| digital_data_low_freq_noise_low | 100 | 16.60 | 0.847 | 0.094 | 5.8% | 17.63 | 0.899 | 0.083 |
| digital_data_low_freq_noise_middle | 100 | 14.21 | 0.723 | 0.159 | 11.3% | 16.03 | 0.816 | 0.139 |
| digital_data_opacity_015 | 100 | 1.72 | 0.220 | 0.216 | 65.7% | 5.01 | 0.641 | 0.201 |
| digital_data_opacity_02 | 100 | 5.09 | 0.447 | 0.189 | 37.8% | 8.19 | 0.720 | 0.168 |
| digital_data_opacity_04 | 100 | 9.28 | 0.611 | 0.545 | 23.1% | 12.07 | 0.794 | 0.635 |
| digital_data_opacity_06 | 100 | 13.97 | 0.778 | 0.272 | 9.2% | 15.39 | 0.858 | 0.277 |
| digital_data_opacity_1 | 100 | 17.21 | 0.880 | 0.075 | 4.1% | 17.95 | 0.918 | 0.071 |
| photos_bents | 100 | -0.38 | 0.375 | 0.266 | 16.8% | -0.45 | 0.451 | 0.284 |
| photos_crumbles | 100 | -2.84 | 0.287 | 0.332 | 25.4% | -3.81 | 0.385 | 0.364 |
| photos_doogee | 100 | 14.24 | 0.881 | 0.070 | 5.4% | 15.06 | 0.932 | 0.064 |
| photos_iphone | 100 | 12.92 | 0.832 | 0.090 | 6.3% | 13.80 | 0.888 | 0.079 |
| photos_samsung | 100 | 13.54 | 0.899 | 0.071 | 2.9% | 13.95 | 0.926 | 0.067 |
| photos_scans | 100 | 14.19 | 0.881 | 0.076 | 4.1% | 14.80 | 0.918 | 0.070 |
| photos_screens | 100 | 7.59 | 0.733 | 0.140 | 6.8% | 8.14 | 0.786 | 0.134 |
