"""Download only what is needed from PM-ECG-ID (Zenodo 13617673), by HTTP ranges (the zip is 34 GB).

    python benchmarks/pmcardio/fetch.py --out data/pm --shard 3 --nshards 20   # images with Image ID % 20 == 3
    python benchmarks/pmcardio/fetch.py --out data/pm --per-scenario 2          # stratified sample
"""
from __future__ import annotations

import argparse
import os
import random
import time
from concurrent.futures import ThreadPoolExecutor

URL = "https://zenodo.org/records/13617673/files/pmcardio_ecg_image_database.zip?download=1"


def _open(retries=10):
    """The remote archive; Zenodo sometimes answers 500/502 for minutes, so back off."""
    from remotezip import RemoteZip
    for i in range(retries):
        try:
            return RemoteZip(URL, timeout=120)
        except Exception as e:  # noqa: BLE001
            if i == retries - 1:
                raise
            print("open failed:", str(e)[:120], flush=True)
            time.sleep(min(240, 10 * 2 ** i) * random.uniform(0.6, 1.0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--shard", type=int, default=None)
    ap.add_argument("--nshards", type=int, default=20)
    ap.add_argument("--per-scenario", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--threads", type=int, default=6)
    a = ap.parse_args()
    import numpy as np
    import pandas as pd
    os.makedirs(a.out, exist_ok=True)
    with _open() as z:
        for n in ["final_data/metadata.csv", "final_data/data/leads.npz", "final_data/data/rhythms.npz"]:
            if not os.path.exists(os.path.join(a.out, n)):
                z.extract(n, a.out)
    m = pd.read_csv(os.path.join(a.out, "final_data/metadata.csv"))
    m["scen"] = m["Image relative path"].str.split("/").str[0]
    if a.per_scenario:
        rng = np.random.default_rng(a.seed); keep = []
        for _, g in m.groupby("scen"):
            ids = rng.choice(sorted(g["Image ID"].unique()), a.per_scenario, replace=False)
            keep += list(g[g["Image ID"].isin(ids)].index)
        m = m.loc[keep]
    if a.shard is not None:
        # shard by (scenario, Image ID): the pages of one ECG stay together
        key = m["scen"].astype("category").cat.codes.astype(np.int64) * 1000 + m["Image ID"].astype(np.int64)
        m = m[key % a.nshards == a.shard]
    sel = list(m["Image relative path"])
    root = os.path.join(a.out, "visual_data")

    def get(chunk):
        z = None
        for p in chunk:
            dst = os.path.join(root, p)
            if os.path.exists(dst):
                continue
            for tent in range(8):
                try:
                    z = z or _open()
                    z.extract("final_data/visual_data/" + p, os.path.join(a.out, "_tmp"))
                    break
                except Exception as e:  # noqa: BLE001
                    # Zenodo answers 500/502 in bursts: reopen and back off
                    print("attempt", tent, p, str(e)[:120], flush=True)
                    z = None
                    time.sleep(min(120, 5 * 2 ** tent) * random.uniform(0.6, 1.0))
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            src = os.path.join(a.out, "_tmp/final_data/visual_data", p)
            if os.path.exists(src):
                os.replace(src, dst)

    if a.shard is not None:
        time.sleep(6 * a.shard)   # 20 shards start together: do not hit Zenodo all at once
    chunks = [sel[i::a.threads] for i in range(a.threads)]
    with ThreadPoolExecutor(a.threads) as ex:
        list(ex.map(get, chunks))
    pd.Series(sel, name="path").to_csv(os.path.join(a.out, "sel.csv"), index=False)
    print("images", len(sel))


if __name__ == "__main__":
    main()
