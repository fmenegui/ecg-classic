# Kaggle late submission

`submission.py` is the competition kernel: CPU only, no internet, the ecgdig source attached as a
private dataset. `.github/workflows/kaggle-submit.yml` uploads the dataset, pushes the kernel,
waits for its run on the public test images and submits that version (5 submissions per day).

Local check on a fake competition folder (test.csv + test/<id>.png):

    ECG_COMP_DIR=/path/fake ECG_OUT=/path/fake/submission.csv python kaggle/submission.py

On 6 local training images the submission scored 22.01 dB with the official metric against the
ground truth, the local evaluation of the same images 20.37 dB (the submission fills unknown
samples from the nearest read ones instead of leaving them empty).

The competition only accepts an output named `submission.csv` (not parquet).
