# PM-ECG-ID, whole database, classical reader `61324b1`, both metrics

6,000 images (60 scenarios × 100 ECGs), GitHub run 43989319488, 16/11/2026. PMcardio-style: per-lead ±0.5 s Pearson alignment, median offset, failure = missing or < 50% covered (PCC 0, SNR 0). Kaggle: official metric per ECG image (12 lead cuts, ±0.2 s, powers summed), linear mean, dB. PMcardio paper (same database): PCC > 0.91, failed leads 6.62%.

| scenario | ECGs | SNR (dB) | PCC | failure | SNR without failures | RMSE without failures (mV) | Kaggle score (dB) |
|---|---|---|---|---|---|---|---|
| ALL | 6000 | 8.52 | 0.688 | 13.2% | 9.82 | 0.375 | 11.66 |
| augmentation_JPEG_compression_11_percent | 100 | 12.30 | 0.872 | 2.8% | 12.66 | 0.082 | 11.99 |
| augmentation_JPEG_compression_22_percent | 100 | 13.03 | 0.888 | 2.9% | 13.43 | 0.073 | 12.82 |
| augmentation_JPEG_compression_2_percent | 100 | 5.53 | 0.691 | 9.7% | 6.13 | 0.149 | 6.52 |
| augmentation_JPEG_compression_45_percent | 100 | 13.21 | 0.907 | 1.5% | 13.41 | 0.076 | 13.20 |
| augmentation_JPEG_compression_5_percent | 100 | 10.36 | 0.845 | 2.9% | 10.67 | 0.101 | 10.91 |
| augmentation_JPEG_compression_90_percent | 100 | 13.69 | 0.911 | 1.1% | 13.84 | 0.072 | 13.66 |
| augmentation_blur_kernel_11 | 100 | 5.55 | 0.657 | 12.3% | 6.34 | 0.139 | 5.77 |
| augmentation_blur_kernel_21 | 100 | 0.90 | 0.272 | 38.8% | 1.48 | 0.204 | 1.09 |
| augmentation_blur_kernel_31 | 100 | 0.05 | 0.212 | 45.1% | 0.09 | 0.249 | 0.25 |
| augmentation_blur_kernel_41 | 100 | -0.22 | 0.098 | 71.1% | -0.77 | 0.253 | 0.15 |
| augmentation_brightness_-120 | 100 | 6.41 | 0.687 | 8.2% | 6.99 | 0.169 | 7.87 |
| augmentation_brightness_-160 | 100 | -1.51 | 0.318 | 30.8% | -2.19 | 0.380 | -0.73 |
| augmentation_brightness_-40 | 100 | 12.12 | 0.847 | 2.8% | 12.46 | 0.086 | 13.17 |
| augmentation_brightness_-80 | 100 | 10.77 | 0.821 | 4.4% | 11.27 | 0.099 | 11.06 |
| augmentation_brightness_120 | 100 | 6.34 | 0.659 | 8.1% | 6.90 | 0.541 | 8.14 |
| augmentation_brightness_160 | 100 | 0.99 | 0.318 | 31.5% | 1.45 | 0.318 | 2.20 |
| augmentation_brightness_40 | 100 | 13.13 | 0.903 | 1.6% | 13.34 | 0.075 | 12.93 |
| augmentation_brightness_80 | 100 | 11.23 | 0.839 | 4.1% | 11.71 | 0.092 | 11.76 |
| augmentation_contrast_gamma_0.125 | 100 | 1.30 | 0.346 | 27.2% | 1.78 | 0.220 | 3.48 |
| augmentation_contrast_gamma_0.25 | 100 | 6.02 | 0.624 | 11.4% | 6.80 | 0.153 | 7.55 |
| augmentation_contrast_gamma_0.5 | 100 | 11.57 | 0.848 | 3.8% | 12.03 | 0.087 | 11.74 |
| augmentation_contrast_gamma_0.666 | 100 | 12.81 | 0.890 | 1.5% | 13.01 | 0.079 | 12.33 |
| augmentation_contrast_gamma_1.5 | 100 | 13.03 | 0.889 | 2.8% | 13.41 | 0.079 | 13.11 |
| augmentation_contrast_gamma_2 | 100 | 11.74 | 0.838 | 2.0% | 11.98 | 0.098 | 12.05 |
| augmentation_contrast_gamma_4 | 100 | 6.62 | 0.677 | 7.3% | 7.14 | 0.166 | 8.14 |
| augmentation_contrast_gamma_8 | 100 | -0.88 | 0.350 | 27.5% | -1.22 | 0.332 | 0.28 |
| augmentation_padding_extra_large | 100 | 6.15 | 0.626 | 16.8% | 7.39 | 0.151 | 9.16 |
| augmentation_padding_large | 100 | 9.80 | 0.812 | 6.4% | 10.48 | 0.098 | 9.47 |
| augmentation_padding_medium | 100 | 10.14 | 0.857 | 4.2% | 10.59 | 0.089 | 9.71 |
| augmentation_padding_small | 100 | 9.63 | 0.839 | 3.8% | 10.01 | 0.099 | 9.87 |
| augmentation_perspective_transformation_20 | 100 | 9.17 | 0.820 | 3.4% | 9.50 | 0.109 | 9.01 |
| augmentation_perspective_transformation_30 | 100 | 7.84 | 0.779 | 4.6% | 8.22 | 0.122 | 8.69 |
| augmentation_perspective_transformation_40 | 100 | 4.78 | 0.618 | 9.7% | 5.29 | 0.161 | 5.73 |
| augmentation_resolution_scale_down_factor_16 | 100 | -0.02 | 0.250 | 39.0% | -0.03 | 0.245 | 0.04 |
| augmentation_resolution_scale_down_factor_2 | 100 | 14.02 | 0.911 | 3.3% | 14.51 | 0.059 | 13.64 |
| augmentation_resolution_scale_down_factor_32 | 100 | -0.24 | 0.034 | 84.8% | -1.55 | 0.245 | 0.05 |
| augmentation_resolution_scale_down_factor_4 | 100 | 11.38 | 0.883 | 2.3% | 11.65 | 0.080 | 10.70 |
| augmentation_resolution_scale_down_factor_8 | 100 | 2.27 | 0.526 | 11.3% | 2.56 | 0.191 | 2.29 |
| augmentation_rotation_15 | 100 | 9.54 | 0.838 | 5.2% | 10.06 | 0.095 | 9.28 |
| augmentation_rotation_25 | 100 | 9.35 | 0.804 | 6.8% | 10.03 | 0.104 | 9.70 |
| augmentation_rotation_35 | 100 | 8.48 | 0.765 | 8.2% | 9.23 | 0.111 | 9.12 |
| augmentation_rotation_5 | 100 | 9.82 | 0.850 | 3.8% | 10.20 | 0.094 | 9.16 |
| digital_data_high_freq_noise_large | 100 | 8.71 | 0.786 | 5.2% | 9.19 | 1.075 | 10.19 |
| digital_data_high_freq_noise_low | 100 | 16.39 | 0.911 | 1.5% | 16.64 | 0.238 | 15.98 |
| digital_data_high_freq_noise_middle | 100 | 13.25 | 0.877 | 3.6% | 13.74 | 0.080 | 12.86 |
| digital_data_low_freq_noise_large | 100 | 7.22 | 0.476 | 17.2% | 8.73 | 13.694 | 7.87 |
| digital_data_low_freq_noise_low | 100 | 17.91 | 0.878 | 5.1% | 18.87 | 0.070 | 19.91 |
| digital_data_low_freq_noise_middle | 100 | 15.25 | 0.756 | 7.6% | 16.50 | 0.137 | 14.30 |
| digital_data_opacity_015 | 100 | 2.00 | 0.235 | 62.4% | 5.32 | 0.201 | 3.91 |
| digital_data_opacity_02 | 100 | 5.89 | 0.477 | 36.0% | 9.20 | 0.151 | 9.93 |
| digital_data_opacity_04 | 100 | 10.21 | 0.657 | 18.6% | 12.54 | 0.433 | 13.07 |
| digital_data_opacity_06 | 100 | 14.88 | 0.818 | 4.8% | 15.63 | 0.088 | 18.26 |
| digital_data_opacity_1 | 100 | 18.98 | 0.936 | 0.9% | 19.15 | 0.090 | 18.96 |
| photos_bents | 100 | -0.17 | 0.388 | 15.0% | -0.19 | 0.271 | 0.30 |
| photos_crumbles | 100 | -2.63 | 0.309 | 18.6% | -3.23 | 0.337 | -2.02 |
| photos_doogee | 100 | 14.77 | 0.918 | 2.2% | 15.11 | 0.061 | 13.97 |
| photos_iphone | 100 | 13.98 | 0.879 | 4.4% | 14.62 | 0.068 | 13.90 |
| photos_samsung | 100 | 13.61 | 0.913 | 1.2% | 13.77 | 0.070 | 13.02 |
| photos_scans | 100 | 14.67 | 0.914 | 1.7% | 14.92 | 0.066 | 13.57 |
| photos_screens | 100 | 7.88 | 0.712 | 11.7% | 8.92 | 0.120 | 9.05 |
