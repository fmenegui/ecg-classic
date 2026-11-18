# Kaggle: does the submission path lose dB? (run 44206754940, 18/11/2026)

`kaggle/submission.py` (the CPU notebook that made the official late submission) run on the training
records as a fake test set (one id per image, `number_of_rows` from the truth), scored with the
official metric, paired image by image with the evaluation path (kaggle-full run 44011597597).

Total: 8,352 images of 928 records (95% of the training set; one shard could not open the archive).

| Type | n | evaluation path | notebook path |
|---|---|---|---|
| 0001 | 928 | 24.90 | 25.07 |
| 0003 | 928 | 16.13 | 16.17 |
| 0004 | 928 | 14.09 | 14.07 |
| 0005 | 928 | 13.41 | 13.41 |
| 0006 | 928 | 11.15 | 11.06 |
| 0009 | 928 | 12.07 | 11.98 |
| 0010 | 928 | 14.80 | 14.79 |
| 0011 | 928 | 16.02 | 16.05 |
| 0012 | 928 | 13.43 | 13.41 |
| **all** | 8352 | **17.61** | **17.71** |

- The notebook path loses nothing (median per-image difference 0.00 dB). Time per image on a 4-CPU
  runner: median 16 s, p99 32 s, max 77 s; no image near the 300 s limit.
- So the gap to the official private score (13.03 dB) is the hidden test set being harder than the
  training images, as for the leaders (1st place: CV 26.8, private 22.75). To lead the private
  board, a training-set score around 27 dB is needed.
- The public test files in the archive: 2 images, 2200x1700, clean renders; test.csv gives fs and
  number_of_rows per lead (lead II 10 s, others 2.5 s).
