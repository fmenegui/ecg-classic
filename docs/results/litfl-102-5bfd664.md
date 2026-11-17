# LITFL benchmark

102/102 images processed (0 errors), 10.2 s/image.

| Metric | Value | n |
|---|---|---|
| Correct layout | 0.931 (before 0.922) | 102 |
| Rhythm strips (all right) | 0.909 (before 0.929) | 99 |
| Rhythm strips (per strip) | 0.893 (before 0.917) |  |
| Speed | 0.98 (before 0.99) | 102 |
| Duration | 0.99 | 102 |
| Gain | 0.97 | 100 |
| Layout+speed+duration+gain | 0.902 (before 0.912) |  |
| Coverage (median) | 0.997 (before 0.993) | 102 |
| Einthoven |I−II+III|/RMS(I,II,III) (median) | 0.499 (before 0.473) | 100 |
| Rhythm II vs grid II, NRMSE (median) | 0.444 (before 0.449) | 71 |

## Errors per image

| Image | Failed on | Detected | Reference |
|---|---|---|---|
| 0007 | layout, rhythm | free  25 mm/s 10 s [10.0] | 3x4+1 II 25.0 10.0 [10.0] |
| 0021 | rhythm | 3x4+1 I 25 mm/s 10 s [10.0] | 3x4+1 II 25.0 10.0 [10.0] |
| 0023 | rhythm | 3x4+1 I 25 mm/s 10 s [10.0] | 3x4+1 II 25.0 10.0 [10.0] |
| 0030 | layout, rhythm | 3x4  25 mm/s 10 s [10.0] | 3x4+1 II 25.0 10.0 None |
| 0033 | layout, gain | 3x4+3 II/aVF/V5 25 mm/s 10 s [5.0] | 6x2  25.0 10.0 [10.0] |
| 0034 | rhythm, speed, duration | 3x4+1 V1 50 mm/s 5 s [10.0] | 3x4+1 II 25.0 10.0 [10.0] |
| 0035 | rhythm, speed | 3x4+1 III 50 mm/s 10 s [10.0] | 3x4+1 II 25.0 10.0 [10.0] |
| 0036 | layout | 3x4+2 II/V1 25 mm/s 10 s [10.0] | rows  25.0 10.0 [10.0] |
| 0048 | gain | 3x4+1 II 25 mm/s 10 s [10.0] | 3x4+1 II 25.0 10.0 [5.0, 10.0] |
| 0058 | layout, rhythm | 3x4  25 mm/s 10 s [10.0] | 3x4+1 II 25.0 10.0 [10.0] |
| 0065 | layout, rhythm | 3x4+2-cabrera V6/II 25 mm/s 10 s [10.0] | 3x4+1 II 25.0 10.0 [10.0] |
| 0070 | rhythm | 3x4+1 aVF 25 mm/s 10 s [10.0] | 3x4+1 II 25.0 10.0 [10.0] |
| 0090 | layout, gain | 3x4+3 II/aVF/V6 25 mm/s 10 s [5.0] | 6x2  25.0 10.0 [10.0] |
