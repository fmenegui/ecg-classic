# ecgdig (clássico) no PM-ECG-ID

Falha = derivação ausente ou < 50% coberta; entra como SNR 0, PCC 0 e RMSE do sinal nulo.

## Base inteira

| conjunto | ECGs | SNR | PCC | RMSE | falha | SNR (sem falhas) | PCC (sem falhas) | RMSE (sem falhas) |
|---|---|---|---|---|---|---|---|---|
| ecgdig | 5994 | 6.99 | 0.591 | 0.216 | 22.2% | 8.99 | 0.760 | 0.212 |
| PMcardio (publicado) | 100 | > 12.5 | > 0.91 | — | 6.62% | — | — | — |

Grupos do artigo da PMcardio (sem lista de cenários): fácil (12): PCC 0.977, SNR 16.68 dB; médio (29): PCC 0.970, SNR 15.79 dB; difícil (19): PCC 0.817, SNR 11.10 dB

## Por família de cenário

| conjunto | ECGs | SNR | PCC | RMSE | falha | SNR (sem falhas) | PCC (sem falhas) | RMSE (sem falhas) |
|---|---|---|---|---|---|---|---|---|
| JPEG_compression | 600 | 9.10 | 0.741 | 0.130 | 10.0% | 10.10 | 0.823 | 0.119 |
| blur_kernel | 400 | 1.07 | 0.257 | 0.219 | 47.5% | 2.04 | 0.490 | 0.215 |
| brightness | 800 | 5.98 | 0.575 | 0.177 | 20.9% | 7.55 | 0.726 | 0.163 |
| contrast_gamma | 800 | 6.25 | 0.596 | 0.173 | 17.6% | 7.59 | 0.724 | 0.163 |
| digital: opacidade | 497 | 9.28 | 0.559 | 0.410 | 33.8% | 14.01 | 0.843 | 0.511 |
| digital: ruído | 600 | 12.02 | 0.705 | 0.449 | 15.2% | 14.18 | 0.832 | 0.475 |
| fotos | 697 | 7.00 | 0.622 | 0.172 | 15.3% | 8.26 | 0.734 | 0.164 |
| padding | 400 | 6.93 | 0.659 | 0.152 | 17.6% | 8.41 | 0.800 | 0.135 |
| perspective | 300 | 5.69 | 0.643 | 0.159 | 15.5% | 6.74 | 0.761 | 0.148 |
| resolution_scale_down | 500 | 4.44 | 0.406 | 0.179 | 44.9% | 8.06 | 0.737 | 0.139 |
| rotation | 400 | 7.09 | 0.665 | 0.141 | 16.4% | 8.48 | 0.796 | 0.122 |

## Por cenário

| conjunto | ECGs | SNR | PCC | RMSE | falha | SNR (sem falhas) | PCC (sem falhas) | RMSE (sem falhas) |
|---|---|---|---|---|---|---|---|---|
| augmentation_JPEG_compression_11_percent | 100 | 10.09 | 0.784 | 0.114 | 7.8% | 10.94 | 0.850 | 0.106 |
| augmentation_JPEG_compression_22_percent | 100 | 10.67 | 0.798 | 0.113 | 7.0% | 11.47 | 0.859 | 0.105 |
| augmentation_JPEG_compression_2_percent | 100 | 4.15 | 0.578 | 0.189 | 15.6% | 4.92 | 0.684 | 0.180 |
| augmentation_JPEG_compression_45_percent | 100 | 10.80 | 0.796 | 0.114 | 8.3% | 11.79 | 0.868 | 0.104 |
| augmentation_JPEG_compression_5_percent | 100 | 8.25 | 0.718 | 0.139 | 12.2% | 9.39 | 0.818 | 0.123 |
| augmentation_JPEG_compression_90_percent | 100 | 10.61 | 0.772 | 0.111 | 8.9% | 11.65 | 0.848 | 0.099 |
| augmentation_blur_kernel_11 | 100 | 4.33 | 0.578 | 0.169 | 17.1% | 5.22 | 0.697 | 0.157 |
| augmentation_blur_kernel_21 | 100 | 0.35 | 0.205 | 0.229 | 46.2% | 0.66 | 0.381 | 0.226 |
| augmentation_blur_kernel_31 | 100 | -0.25 | 0.183 | 0.241 | 47.4% | -0.48 | 0.348 | 0.273 |
| augmentation_blur_kernel_41 | 100 | -0.15 | 0.063 | 0.236 | 79.2% | -0.73 | 0.304 | 0.275 |
| augmentation_brightness_-120 | 100 | 4.02 | 0.528 | 0.199 | 21.9% | 5.14 | 0.676 | 0.193 |
| augmentation_brightness_-160 | 100 | -1.21 | 0.296 | 0.313 | 38.7% | -1.98 | 0.482 | 0.369 |
| augmentation_brightness_-40 | 100 | 9.57 | 0.702 | 0.131 | 14.4% | 11.18 | 0.820 | 0.114 |
| augmentation_brightness_-80 | 100 | 7.94 | 0.670 | 0.147 | 14.0% | 9.23 | 0.779 | 0.135 |
| augmentation_brightness_120 | 100 | 5.64 | 0.585 | 0.167 | 17.4% | 6.83 | 0.709 | 0.149 |
| augmentation_brightness_160 | 100 | 1.15 | 0.303 | 0.230 | 36.7% | 1.82 | 0.479 | 0.231 |
| augmentation_brightness_40 | 100 | 11.04 | 0.767 | 0.105 | 12.4% | 12.60 | 0.876 | 0.089 |
| augmentation_brightness_80 | 100 | 9.69 | 0.748 | 0.122 | 11.3% | 10.93 | 0.844 | 0.107 |
| augmentation_contrast_gamma_0.125 | 100 | 0.92 | 0.311 | 0.226 | 31.6% | 1.35 | 0.455 | 0.232 |
| augmentation_contrast_gamma_0.25 | 100 | 4.86 | 0.539 | 0.178 | 20.2% | 6.10 | 0.676 | 0.166 |
| augmentation_contrast_gamma_0.5 | 100 | 9.16 | 0.728 | 0.133 | 11.8% | 10.38 | 0.825 | 0.119 |
| augmentation_contrast_gamma_0.666 | 100 | 10.61 | 0.776 | 0.115 | 9.5% | 11.73 | 0.857 | 0.106 |
| augmentation_contrast_gamma_1.5 | 100 | 10.51 | 0.762 | 0.119 | 8.9% | 11.54 | 0.837 | 0.113 |
| augmentation_contrast_gamma_2 | 100 | 8.81 | 0.685 | 0.138 | 15.5% | 10.42 | 0.810 | 0.123 |
| augmentation_contrast_gamma_4 | 100 | 5.94 | 0.602 | 0.184 | 16.2% | 7.09 | 0.719 | 0.174 |
| augmentation_contrast_gamma_8 | 100 | -0.79 | 0.367 | 0.288 | 26.9% | -1.09 | 0.503 | 0.309 |
| augmentation_padding_extra_large | 100 | 4.11 | 0.485 | 0.197 | 28.1% | 5.71 | 0.674 | 0.184 |
| augmentation_padding_large | 100 | 7.78 | 0.692 | 0.155 | 15.2% | 9.17 | 0.816 | 0.144 |
| augmentation_padding_medium | 100 | 7.59 | 0.720 | 0.133 | 13.7% | 8.79 | 0.834 | 0.116 |
| augmentation_padding_small | 100 | 8.25 | 0.738 | 0.121 | 13.6% | 9.54 | 0.854 | 0.103 |
| augmentation_perspective_transformation_20 | 100 | 7.21 | 0.717 | 0.136 | 12.2% | 8.21 | 0.816 | 0.125 |
| augmentation_perspective_transformation_30 | 100 | 6.17 | 0.675 | 0.147 | 14.0% | 7.17 | 0.784 | 0.135 |
| augmentation_perspective_transformation_40 | 100 | 3.70 | 0.538 | 0.193 | 20.3% | 4.64 | 0.675 | 0.186 |
| augmentation_resolution_scale_down_factor_16 | 100 | -0.05 | 0.088 | 0.231 | 74.4% | -0.20 | 0.344 | 0.217 |
| augmentation_resolution_scale_down_factor_2 | 100 | 11.71 | 0.806 | 0.099 | 7.5% | 12.66 | 0.871 | 0.092 |
| augmentation_resolution_scale_down_factor_32 | 100 | -0.01 | 0.017 | 0.229 | 95.2% | -0.17 | 0.347 | 0.331 |
| augmentation_resolution_scale_down_factor_4 | 100 | 9.72 | 0.775 | 0.113 | 9.9% | 10.79 | 0.861 | 0.103 |
| augmentation_resolution_scale_down_factor_8 | 100 | 0.85 | 0.343 | 0.223 | 37.5% | 1.35 | 0.549 | 0.212 |
| augmentation_rotation_15 | 100 | 7.07 | 0.708 | 0.135 | 11.2% | 7.96 | 0.797 | 0.125 |
| augmentation_rotation_25 | 100 | 7.09 | 0.636 | 0.148 | 19.0% | 8.75 | 0.786 | 0.126 |
| augmentation_rotation_35 | 100 | 6.36 | 0.583 | 0.160 | 24.7% | 8.44 | 0.774 | 0.129 |
| augmentation_rotation_5 | 100 | 7.85 | 0.735 | 0.122 | 10.9% | 8.81 | 0.825 | 0.109 |
| digital_data_high_freq_noise_large | 100 | 8.44 | 0.712 | 0.139 | 13.7% | 9.77 | 0.824 | 0.124 |
| digital_data_high_freq_noise_low | 100 | 13.83 | 0.799 | 0.343 | 12.5% | 15.80 | 0.914 | 0.363 |
| digital_data_high_freq_noise_middle | 100 | 11.91 | 0.790 | 0.290 | 11.5% | 13.46 | 0.893 | 0.297 |
| digital_data_low_freq_noise_large | 100 | 7.68 | 0.454 | 1.117 | 24.0% | 10.11 | 0.598 | 1.326 |
| digital_data_low_freq_noise_low | 100 | 15.39 | 0.764 | 0.657 | 13.7% | 17.83 | 0.884 | 0.724 |
| digital_data_low_freq_noise_middle | 100 | 14.88 | 0.711 | 0.149 | 16.1% | 17.73 | 0.847 | 0.116 |
| digital_data_opacity_015 | 100 | 1.89 | 0.220 | 0.236 | 68.3% | 5.96 | 0.696 | 0.282 |
| digital_data_opacity_02 | 99 | 5.77 | 0.476 | 0.172 | 38.5% | 9.38 | 0.773 | 0.147 |
| digital_data_opacity_04 | 100 | 9.92 | 0.580 | 1.445 | 31.2% | 14.43 | 0.844 | 2.000 |
| digital_data_opacity_06 | 100 | 13.40 | 0.724 | 0.102 | 17.6% | 16.26 | 0.879 | 0.082 |
| digital_data_opacity_1 | 98 | 15.51 | 0.797 | 0.086 | 12.8% | 17.80 | 0.914 | 0.070 |
| photos_bents | 99 | -0.16 | 0.342 | 0.254 | 23.0% | -0.21 | 0.445 | 0.269 |
| photos_crumbles | 100 | -3.09 | 0.272 | 0.341 | 30.6% | -4.45 | 0.391 | 0.391 |
| photos_doogee | 99 | 11.07 | 0.740 | 0.114 | 11.9% | 12.56 | 0.839 | 0.102 |
| photos_iphone | 100 | 11.65 | 0.775 | 0.112 | 9.2% | 12.84 | 0.854 | 0.098 |
| photos_samsung | 99 | 10.73 | 0.784 | 0.122 | 6.7% | 11.51 | 0.840 | 0.117 |
| photos_scans | 100 | 11.89 | 0.776 | 0.104 | 11.4% | 13.42 | 0.876 | 0.091 |
| photos_screens | 100 | 6.89 | 0.666 | 0.156 | 14.1% | 8.02 | 0.775 | 0.146 |
