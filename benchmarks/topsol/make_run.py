"""Turns a Kaggle submission notebook built on hengck23's demo (2nd and 3rd place) into a script
that runs on one GPU of a pod, on a test set made by prepare.py.

    python benchmarks/topsol/make_run.py NOTEBOOK.ipynb OUT.py

Patches only paths and the device count: /kaggle/input/physionet-ecg-image-digitization -> $TS_DATA,
/kaggle/input -> $TS_INPUT, /kaggle/working -> $TS_WORK; the second GPU of the 2-GPU notebooks
gets no image; shell lines (!pip, !ls) and the final clean-up are dropped. The digitised series
stay in $TS_WORK/outputs/digitalised/<id>.series.npy for to_npz.py.
"""
from __future__ import annotations

import json
import re
import sys


def convert(nb_path, out_path):
    nb = json.load(open(nb_path))
    code = []
    for c in nb["cells"]:
        if c["cell_type"] != "code":
            continue
        src = "".join(c["source"])
        lines = []
        for ln in src.split("\n"):
            s = ln.strip()
            if s.startswith("!") or s.startswith("%"):
                continue
            if "shutil.rmtree(OUT_DIR)" in ln:
                continue
            lines.append(ln)
        code.append("\n".join(lines))
    s = "\n\n".join(code)
    s = s.replace("'/kaggle/input/physionet-ecg-image-digitization'", "os.environ['TS_DATA']")
    s = s.replace('"/kaggle/input/physionet-ecg-image-digitization"', "os.environ['TS_DATA']")
    s = re.sub(r"(['\"])/kaggle/input/", r"os.environ['TS_INPUT'] + \1/", s)
    s = re.sub(r"f(['\"])/kaggle/working", r"os.environ['TS_WORK'] + f\1", s)
    s = re.sub(r"(['\"])/kaggle/working", r"os.environ['TS_WORK'] + \1", s)
    # one GPU: the first process gets every image, the second none (it starts and exits)
    s = re.sub(r"ids_gpu0 = valid_id\[:mid_idx\]", "ids_gpu0 = valid_id", s)
    s = re.sub(r"ids_gpu1 = valid_id\[mid_idx:\]", "ids_gpu1 = []", s)
    s = re.sub(r"args=\(1,", "args=(0,", s)
    # the notebooks draw their first 10 images; on an image the solution fails on, the drawing
    # itself crashes the whole stage (the failure was meant to be skipped): no drawing
    s = re.sub(r"if n\s*<\s*10 and gpu_id\s*==\s*0\s*:", "if False:", s)
    s = re.sub(r"\bp1\.start\(\)", "pass", s)
    s = re.sub(r"\bp1\.join\(\)", "pass", s)
    head = ("import os, sys\nimport matplotlib\nmatplotlib.use('Agg')\n"
            "import matplotlib.pyplot as plt\nplt.show = lambda *a, **k: None\n"
            "from timeit import default_timer as timer\n")
    open(out_path, "w").write(head + s + "\n")


if __name__ == "__main__":
    convert(sys.argv[1], sys.argv[2])
