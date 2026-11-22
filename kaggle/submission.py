"""Kaggle "PhysioNet - Digitization of ECG Images": offline CPU submission of ecgdig.

Hybrid when the attached dataset has models/row.onnx (and models/page.onnx): the row network and
the page-structure network inside the classical pipeline, as on our benchmarks (photo_norm is on by
default). onnxruntime is installed from the dataset's wheels/ when the image lacks it (no internet).

Runs as a Kaggle script kernel with the competition data and a dataset holding the ecgdig source
(kaggle/README.md). Reads test.csv (id, lead, fs, number_of_rows), digitizes every test image once
and writes submission.csv (the only name the competition accepts) with one value per expected row:
- lead II: the 10 s rhythm read by ecgdig, sampled at fs;
- other leads: 2.5 s starting where ecgdig found that lead printed (its first finite sample);
- unknown samples are filled from the nearest read sample, a lead not read is all zeros, and an
  image that fails or exceeds the time limit is all zeros (the metric needs a number per row).
"""
import glob
import os
import signal
import sys
import time
import zipfile
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

def _comp_dir():
    if os.environ.get("ECG_COMP_DIR"):
        return os.environ["ECG_COMP_DIR"]
    for d in ["/kaggle/input/physionet-ecg-image-digitization",
              "/kaggle/input/competitions/physionet-ecg-image-digitization"]:
        if os.path.exists(os.path.join(d, "test.csv")):
            return d
    hits = glob.glob("/kaggle/input/**/test.csv", recursive=True)
    return os.path.dirname(hits[0]) if hits else "/kaggle/input/physionet-ecg-image-digitization"


COMP = _comp_dir()
OUT = os.environ.get("ECG_OUT", "/kaggle/working/submission.csv")   # the competition requires this name
PER_IMAGE_S = int(os.environ.get("ECG_IMAGE_TIMEOUT", "300"))
# Kaggle stops a notebook at 9 h. The networks cost ~4x the classical reader on CPU, so when the time
# left cannot cover the remaining images, the rest are read classically (and past the end, left as 0).
BUDGET_S = float(os.environ.get("ECG_BUDGET_S", str(8.3 * 3600)))
CLASSIC_S = float(os.environ.get("ECG_CLASSIC_S", "40"))   # pessimistic classical seconds per image
HYBRID_S = float(os.environ.get("ECG_HYBRID_S", "150"))    # pessimistic hybrid seconds per image
T_START = time.time()
T_END = 0.0
N_WORKERS = 1
FS_ME = 500.0


def _find_ecgdig():
    """The ecgdig package from the attached dataset (folder or zip), or already importable."""
    try:
        import ecgdig  # noqa: F401
        return
    except ImportError:
        pass
    for root in ["/kaggle/input", os.path.dirname(os.path.abspath(__file__))]:
        hits = sorted(glob.glob(f"{root}/**/ecgdig/__init__.py", recursive=True))
        if hits:
            sys.path.insert(0, os.path.dirname(os.path.dirname(hits[0])))
            return
        for z in sorted(glob.glob(f"{root}/**/*.zip", recursive=True)):
            with zipfile.ZipFile(z) as zf:
                if any(n.endswith("ecgdig/__init__.py") for n in zf.namelist()):
                    zf.extractall("/kaggle/working/src")
                    sys.path.insert(0, os.path.dirname(os.path.dirname(
                        glob.glob("/kaggle/working/src/**/ecgdig/__init__.py", recursive=True)[0])))
                    return
    raise ImportError("ecgdig source not found in /kaggle/input")


def _unzip_dirs():
    """Kaggle stores the dataset's folders as <folder>.zip (kaggle datasets create -r zip): unpack
    models.zip and wheels.zip to /kaggle/working."""
    for name in ("models", "wheels"):
        for z in sorted(glob.glob(f"/kaggle/input/**/{name}.zip", recursive=True)):
            dst = f"/kaggle/working/{name}"
            if not os.path.isdir(dst):
                with zipfile.ZipFile(z) as zf:
                    zf.extractall(dst)


def _models():
    """Config fields for the networks found in the attached dataset (models/row.onnx, models/page.onnx)."""
    _unzip_dirs()
    kw = {}
    for name, key in (("row.onnx", "row_model"), ("page.onnx", "page_model")):
        hits = sorted(glob.glob(f"/kaggle/input/**/models/{name}", recursive=True)) + \
            sorted(glob.glob(f"/kaggle/working/models/**/{name}", recursive=True)) + \
            sorted(glob.glob(os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", name)))
        if hits:
            kw[key] = hits[0]
    if kw:
        try:
            import onnxruntime  # noqa: F401
        except ImportError:
            import subprocess
            whl = sorted(set(glob.glob("/kaggle/input/**/*.whl", recursive=True)
                             + glob.glob("/kaggle/working/wheels/**/*.whl", recursive=True)))
            tag = f"cp{sys.version_info.major}{sys.version_info.minor}"
            ort = [w for w in whl if os.path.basename(w).startswith("onnxruntime-") and f"-{tag}-" in w]
            print("python", tag, "wheels:", [os.path.basename(w) for w in whl], flush=True)
            if ort:
                links = []
                for d in sorted(set(os.path.dirname(w) for w in whl)):
                    links += ["--find-links", d]
                subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--no-index", *links, ort[0]],
                               check=False)
            try:
                import onnxruntime  # noqa: F401
            except ImportError:
                print(f"onnxruntime missing for {tag}: classical fallback", flush=True)
                return {}
    return kw


MODELS = {}


def _fill(v):
    """Unknown samples from the nearest known ones (linear inside, constant at the ends)."""
    m = np.isfinite(v)
    if not m.any():
        return np.zeros_like(v)
    i = np.arange(len(v))
    return np.interp(i, i[m], v[m])


def _lead_values(sig500, fs, n, long_lead):
    """n samples at fs from one ecgdig lead (500 Hz, 10 s, NaN where not printed)."""
    m = np.isfinite(sig500)
    if m.sum() < 2:
        return np.zeros(n)
    t_me = np.arange(len(sig500)) / FS_ME
    total = int(round(10 * fs))
    t = np.arange(total) / fs
    v = np.interp(t, t_me[m], sig500[m], left=np.nan, right=np.nan)
    holes = ~np.isfinite(np.interp(t, t_me, np.where(m, 0.0, np.nan)))
    v[holes] = np.nan
    if long_lead or n >= total:
        out = v[:n]
    else:
        fin = np.flatnonzero(np.isfinite(v))
        a = int(fin[0]) if len(fin) else 0
        a = min(a, max(0, total - n))
        out = v[a:a + n]
    if len(out) < n:
        out = np.concatenate([out, np.full(n - len(out), np.nan)])
    return _fill(out)


def _mode(left_s, n_left):
    """'hybrid' while the time left covers this image with the networks and the rest classically,
    'classical' while it covers the rest classically, else 'skip'."""
    if not MODELS:
        return "classical" if left_s > CLASSIC_S / 4 else "skip"
    per = 1.0 / max(1, N_WORKERS)
    if left_s > HYBRID_S + (n_left - 1) * CLASSIC_S * per:
        return "hybrid"
    return "classical" if left_s > CLASSIC_S / 4 else "skip"


def _one(job):
    global T_END, N_WORKERS
    image_id, leads, n_left, T_END, N_WORKERS = job   # leads: (lead, fs, n); n_left: images from this one on
    import warnings
    warnings.filterwarnings("ignore")

    def _alarm(*_):
        raise TimeoutError("timed out")
    res = {lead: np.zeros(n) for lead, fs, n in leads}
    t0 = time.time()
    left_s = T_END - t0 if T_END else 1e9
    mode = _mode(left_s, n_left)
    if mode == "skip":
        print(image_id, "skipped: out of time", flush=True)
        return image_id, res
    try:
        _find_ecgdig()
        from ecgdig import Config, digitize
        signal.signal(signal.SIGALRM, _alarm)
        signal.alarm(max(1, int(min(PER_IMAGE_S, left_s))))
        r = digitize(os.path.join(COMP, "test", f"{image_id}.png"),
                     Config(**MODELS) if mode == "hybrid" else Config())
        signal.alarm(0)
        names = list(r["leads"])
        for lead, fs, n in leads:
            if lead in names:
                res[lead] = _lead_values(np.asarray(r["signals"][names.index(lead)], float), fs, n,
                                         long_lead=(lead == "II"))
        status = r.get("layout", "") + ("" if mode == "hybrid" or not MODELS else " (classical: time)")
    except Exception as e:  # noqa: BLE001
        signal.alarm(0)
        status = f"failed: {type(e).__name__}: {str(e)[:80]}"
    print(image_id, status, f"{time.time() - t0:.1f}s", flush=True)
    return image_id, res


def main():
    global MODELS
    _find_ecgdig()
    MODELS = _models()
    print("models:", MODELS or "none (classical)", flush=True)
    if not MODELS:
        print("inputs:", sorted(glob.glob("/kaggle/input/**", recursive=True))[:40], flush=True)
    test = pd.read_csv(os.path.join(COMP, "test.csv"))
    groups = list(test.groupby("id", sort=False))
    jobs = []
    for k, (image_id, g) in enumerate(groups):
        jobs.append((str(image_id), [(str(l), float(f), int(n)) for l, f, n in
                                     zip(g["lead"], g["fs"], g["number_of_rows"])], len(groups) - k,
                     T_START + BUDGET_S, max(1, (os.cpu_count() or 1) // (2 if MODELS else 1))))
    print(len(jobs), "images,", os.cpu_count(), "CPUs", flush=True)
    ids, vals = [], []
    # each ONNX session uses 2 threads: half the CPUs as workers when networks are on
    nw = max(1, (os.cpu_count() or 1) // (2 if MODELS else 1))
    with ProcessPoolExecutor(max_workers=nw) as ex:
        for image_id, res in ex.map(_one, jobs, chunksize=1):
            for lead, v in res.items():
                v = np.nan_to_num(np.asarray(v, np.float64), nan=0.0, posinf=0.0, neginf=0.0)
                ids.extend(f"{image_id}_{i}_{lead}" for i in range(len(v)))
                vals.append(v)
    sub = pd.DataFrame({"id": ids, "value": np.concatenate(vals) if vals else np.zeros(0)})
    sub.to_csv(OUT, index=False, float_format="%.5f")
    out = OUT
    print("wrote", out, len(sub), "rows", flush=True)


if __name__ == "__main__":
    main()
