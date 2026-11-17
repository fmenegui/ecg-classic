"""Both metrics (official Kaggle, PMcardio) on locally kept PhysioNet/Kaggle images.

    PHYSIONET_ARTS=dirA,dirB python benchmarks/physionet/score_local.py out.json ['{"ink_model": "m.onnx"}']
"""
import warnings; warnings.filterwarnings("ignore")
import sys, os, glob, json
import numpy as np
from concurrent.futures import ProcessPoolExecutor
R = os.environ.get("ECGDIG_SRC", str(__import__("pathlib").Path(__file__).resolve().parents[2]))
sys.path.insert(0, R); sys.path.insert(0, R + "/benchmarks"); sys.path.insert(0, R + "/benchmarks/physionet")
ARTS = [d for d in os.environ.get("PHYSIONET_ARTS", "").split(",") if d]   # folders with raw/*.png, raw/<id>.csv
CFG = json.loads(sys.argv[2]) if len(sys.argv) > 2 else {}

def one(job):
    import eval_snr as E, metrics as MT
    from ecgdig import digitize, Config
    art, f = job
    rid, typ = os.path.basename(f)[:-4].split("-")
    G = E._load_gt(f"{art}/raw/{rid}.csv"); fs = G.shape[1] / 10
    rec = dict(id=rid, type=typ)
    try:
        r = digitize(f, Config(**CFG))
        S = np.full((12, r["signals"].shape[1]), np.nan)
        for k, nm in enumerate(r["leads"]):
            if nm in E.LEADS: S[E.LEADS.index(nm)] = r["signals"][k]
        P = E._to_gt_grid(S, fs, G.shape[1])
        kl, kp = MT.kaggle_image_snr(G, P, fs)
        rec.update(ok=True, snr=float(E._snr_record(G, P, fs)[0]), kaggle_lin=float(kl),
                   kaggle_db=float(10 * np.log10(max(kl, 1e-12))), pm=MT.pm_image_metrics(G, P, fs))
    except Exception as e:
        rec.update(ok=False, error=repr(e)[:200])
    return rec

jobs = [(a, f) for a in ARTS for f in sorted(glob.glob(a + "/raw/*.png"))]
with ProcessPoolExecutor(2) as ex:
    res = list(ex.map(one, jobs))
json.dump(res, open(sys.argv[1], "w"))
import eval_snr as E
print(len(res), "images; old convention (mean of dB):", round(np.mean([r.get("snr", 0) for r in res]), 2))
print(E.report_both(res))
