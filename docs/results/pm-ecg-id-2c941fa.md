# ecgdig (classical) on PM-ECG-ID

Commit `2c941fa` (16/11/2026): structure and measurement fixes on top of `43cb3a6` (coloured trace on an achromatic grid, rows 3-8 mm apart, false half pulses, pulses between columns, wide column margins, tracking baseline off the pulse, orientation and degenerate-homography guards). GitHub Actions run 43902500481, 6,000 ECGs.

Failure = lead missing or < 50% covered; it counts as SNR 0, PCC 0 and the RMSE of a null signal.

## Whole database

| set | ECGs | SNR | PCC | RMSE | failure | SNR (no failures) | PCC (no failures) | RMSE (no failures) |
|---|---|---|---|---|---|---|---|---|
| ecgdig | 6000 | 8.51 | 0.686 | 0.370 | 13.6% | 9.84 | 0.793 | 0.391 |
| PMcardio (published) | 100 | > 12.5 | > 0.91 | — | 6.62% | — | — | — |

Groups of the PMcardio paper (no scenario list): easy (12): PCC 0.977, SNR 16.68 dB; medium (29): PCC 0.970, SNR 15.79 dB; hard (19): PCC 0.817, SNR 11.10 dB

## By scenario family

| set | ECGs | SNR | PCC | RMSE | failure | SNR (no failures) | PCC (no failures) | RMSE (no failures) |
|---|---|---|---|---|---|---|---|---|
| JPEG_compression | 600 | 11.36 | 0.852 | 0.098 | 3.5% | 11.77 | 0.883 | 0.092 |
| blur_kernel | 400 | 1.58 | 0.311 | 0.209 | 41.4% | 2.69 | 0.530 | 0.196 |
| brightness | 800 | 7.48 | 0.675 | 0.208 | 11.5% | 8.45 | 0.762 | 0.206 |
| contrast_gamma | 800 | 7.81 | 0.681 | 0.151 | 10.8% | 8.75 | 0.763 | 0.142 |
| digital: noise | 600 | 13.01 | 0.774 | 2.167 | 7.5% | 14.07 | 0.837 | 2.316 |
| digital: opacity | 500 | 10.20 | 0.614 | 0.355 | 25.9% | 13.78 | 0.829 | 0.401 |
| padding | 400 | 8.93 | 0.784 | 0.116 | 7.8% | 9.69 | 0.850 | 0.108 |
| perspective | 300 | 7.30 | 0.739 | 0.132 | 6.1% | 7.77 | 0.787 | 0.127 |
| photos | 700 | 8.91 | 0.716 | 0.142 | 8.5% | 9.74 | 0.782 | 0.133 |
| resolution_scale_down | 500 | 5.49 | 0.521 | 0.161 | 28.2% | 7.64 | 0.725 | 0.136 |
| rotation | 400 | 9.29 | 0.814 | 0.107 | 6.0% | 9.88 | 0.866 | 0.101 |

## By scenario

| set | ECGs | SNR | PCC | RMSE | failure | SNR (no failures) | PCC (no failures) | RMSE (no failures) |
|---|---|---|---|---|---|---|---|---|
| augmentation_JPEG_compression_11_percent | 100 | 12.37 | 0.874 | 0.085 | 2.8% | 12.73 | 0.899 | 0.082 |
| augmentation_JPEG_compression_22_percent | 100 | 13.02 | 0.888 | 0.079 | 2.9% | 13.41 | 0.914 | 0.073 |
| augmentation_JPEG_compression_2_percent | 100 | 5.50 | 0.690 | 0.164 | 9.7% | 6.09 | 0.764 | 0.152 |
| augmentation_JPEG_compression_45_percent | 100 | 13.20 | 0.906 | 0.080 | 1.5% | 13.40 | 0.920 | 0.077 |
| augmentation_JPEG_compression_5_percent | 100 | 10.38 | 0.844 | 0.106 | 2.9% | 10.70 | 0.869 | 0.100 |
| augmentation_JPEG_compression_90_percent | 100 | 13.66 | 0.910 | 0.075 | 1.1% | 13.81 | 0.920 | 0.072 |
| augmentation_blur_kernel_11 | 100 | 5.56 | 0.657 | 0.147 | 12.3% | 6.34 | 0.750 | 0.139 |
| augmentation_blur_kernel_21 | 100 | 0.92 | 0.274 | 0.218 | 37.8% | 1.47 | 0.440 | 0.204 |
| augmentation_blur_kernel_31 | 100 | 0.03 | 0.213 | 0.237 | 44.6% | 0.05 | 0.385 | 0.251 |
| augmentation_blur_kernel_41 | 100 | -0.20 | 0.098 | 0.235 | 71.1% | -0.70 | 0.340 | 0.251 |
| augmentation_brightness_-120 | 100 | 6.56 | 0.686 | 0.170 | 9.9% | 7.28 | 0.761 | 0.163 |
| augmentation_brightness_-160 | 100 | -1.26 | 0.323 | 0.315 | 29.8% | -1.79 | 0.460 | 0.361 |
| augmentation_brightness_-40 | 100 | 12.06 | 0.847 | 0.092 | 2.8% | 12.40 | 0.871 | 0.086 |
| augmentation_brightness_-80 | 100 | 10.82 | 0.824 | 0.104 | 4.2% | 11.29 | 0.859 | 0.098 |
| augmentation_brightness_120 | 100 | 6.34 | 0.659 | 0.514 | 8.1% | 6.90 | 0.717 | 0.541 |
| augmentation_brightness_160 | 100 | 0.98 | 0.318 | 0.294 | 31.5% | 1.44 | 0.464 | 0.320 |
| augmentation_brightness_40 | 100 | 13.11 | 0.903 | 0.078 | 1.6% | 13.32 | 0.917 | 0.076 |
| augmentation_brightness_80 | 100 | 11.24 | 0.839 | 0.101 | 4.1% | 11.72 | 0.875 | 0.092 |
| augmentation_contrast_gamma_0.125 | 100 | 1.31 | 0.346 | 0.220 | 27.0% | 1.79 | 0.474 | 0.220 |
| augmentation_contrast_gamma_0.25 | 100 | 6.01 | 0.624 | 0.160 | 11.4% | 6.79 | 0.704 | 0.153 |
| augmentation_contrast_gamma_0.5 | 100 | 11.58 | 0.848 | 0.093 | 3.8% | 12.04 | 0.882 | 0.087 |
| augmentation_contrast_gamma_0.666 | 100 | 12.81 | 0.890 | 0.081 | 1.5% | 13.01 | 0.903 | 0.079 |
| augmentation_contrast_gamma_1.5 | 100 | 13.02 | 0.888 | 0.084 | 2.8% | 13.40 | 0.914 | 0.079 |
| augmentation_contrast_gamma_2 | 100 | 11.81 | 0.835 | 0.098 | 3.0% | 12.18 | 0.861 | 0.096 |
| augmentation_contrast_gamma_4 | 100 | 6.64 | 0.679 | 0.171 | 7.1% | 7.15 | 0.731 | 0.166 |
| augmentation_contrast_gamma_8 | 100 | -0.74 | 0.337 | 0.304 | 29.5% | -1.05 | 0.478 | 0.333 |
| augmentation_padding_extra_large | 100 | 6.14 | 0.626 | 0.163 | 16.8% | 7.38 | 0.752 | 0.151 |
| augmentation_padding_large | 100 | 9.80 | 0.812 | 0.106 | 6.3% | 10.46 | 0.867 | 0.098 |
| augmentation_padding_medium | 100 | 10.16 | 0.857 | 0.093 | 4.2% | 10.61 | 0.895 | 0.089 |
| augmentation_padding_small | 100 | 9.63 | 0.839 | 0.103 | 3.8% | 10.01 | 0.873 | 0.099 |
| augmentation_perspective_transformation_20 | 100 | 9.25 | 0.820 | 0.106 | 3.9% | 9.63 | 0.853 | 0.102 |
| augmentation_perspective_transformation_30 | 100 | 7.89 | 0.779 | 0.125 | 5.1% | 8.31 | 0.821 | 0.121 |
| augmentation_perspective_transformation_40 | 100 | 4.76 | 0.619 | 0.166 | 9.3% | 5.25 | 0.682 | 0.161 |
| augmentation_resolution_scale_down_factor_16 | 100 | -0.01 | 0.249 | 0.236 | 39.2% | -0.02 | 0.410 | 0.245 |
| augmentation_resolution_scale_down_factor_2 | 100 | 14.02 | 0.911 | 0.064 | 3.3% | 14.50 | 0.942 | 0.059 |
| augmentation_resolution_scale_down_factor_32 | 100 | -0.24 | 0.034 | 0.233 | 84.8% | -1.55 | 0.222 | 0.245 |
| augmentation_resolution_scale_down_factor_4 | 100 | 11.39 | 0.883 | 0.082 | 2.5% | 11.68 | 0.906 | 0.079 |
| augmentation_resolution_scale_down_factor_8 | 100 | 2.27 | 0.526 | 0.191 | 11.3% | 2.56 | 0.593 | 0.191 |
| augmentation_rotation_15 | 100 | 9.53 | 0.838 | 0.100 | 5.2% | 10.05 | 0.883 | 0.095 |
| augmentation_rotation_25 | 100 | 9.34 | 0.804 | 0.110 | 6.8% | 10.03 | 0.863 | 0.104 |
| augmentation_rotation_35 | 100 | 8.47 | 0.765 | 0.119 | 8.2% | 9.22 | 0.833 | 0.111 |
| augmentation_rotation_5 | 100 | 9.82 | 0.850 | 0.097 | 3.8% | 10.20 | 0.883 | 0.093 |
| digital_data_high_freq_noise_large | 100 | 8.61 | 0.777 | 1.037 | 6.2% | 9.18 | 0.828 | 1.085 |
| digital_data_high_freq_noise_low | 100 | 16.09 | 0.896 | 0.240 | 3.0% | 16.59 | 0.924 | 0.241 |
| digital_data_high_freq_noise_middle | 100 | 13.12 | 0.872 | 0.085 | 4.1% | 13.68 | 0.909 | 0.081 |
| digital_data_low_freq_noise_large | 100 | 7.16 | 0.473 | 11.409 | 18.2% | 8.76 | 0.578 | 13.858 |
| digital_data_low_freq_noise_low | 100 | 17.90 | 0.874 | 0.080 | 6.1% | 19.06 | 0.930 | 0.067 |
| digital_data_low_freq_noise_middle | 100 | 15.15 | 0.751 | 0.153 | 7.7% | 16.40 | 0.814 | 0.139 |
| digital_data_opacity_015 | 100 | 1.84 | 0.225 | 0.215 | 63.4% | 5.03 | 0.616 | 0.204 |
| digital_data_opacity_02 | 100 | 5.58 | 0.457 | 0.182 | 39.0% | 9.14 | 0.749 | 0.152 |
| digital_data_opacity_04 | 100 | 10.10 | 0.651 | 0.397 | 19.1% | 12.48 | 0.805 | 0.436 |
| digital_data_opacity_06 | 100 | 14.87 | 0.815 | 0.887 | 5.8% | 15.78 | 0.865 | 0.928 |
| digital_data_opacity_1 | 100 | 18.62 | 0.922 | 0.096 | 2.4% | 19.09 | 0.944 | 0.093 |
| photos_bents | 100 | -0.16 | 0.383 | 0.258 | 15.8% | -0.19 | 0.454 | 0.267 |
| photos_crumbles | 100 | -2.52 | 0.291 | 0.317 | 22.4% | -3.25 | 0.375 | 0.337 |
| photos_doogee | 100 | 14.78 | 0.918 | 0.063 | 2.2% | 15.12 | 0.939 | 0.060 |
| photos_iphone | 100 | 13.94 | 0.878 | 0.080 | 4.2% | 14.56 | 0.917 | 0.070 |
| photos_samsung | 100 | 13.60 | 0.913 | 0.072 | 1.2% | 13.76 | 0.924 | 0.070 |
| photos_scans | 100 | 14.79 | 0.917 | 0.068 | 1.9% | 15.08 | 0.935 | 0.064 |
| photos_screens | 100 | 7.93 | 0.712 | 0.132 | 11.7% | 8.97 | 0.806 | 0.120 |
