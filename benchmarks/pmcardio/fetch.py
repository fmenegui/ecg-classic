"""Baixa do PM-ECG-ID (Zenodo 13617673) só o necessário, por faixas HTTP (o zip tem 34 GB).

    python benchmarks/pmcardio/fetch.py --out data/pm --shard 3 --nshards 20   # imagens com Image ID % 20 == 3
    python benchmarks/pmcardio/fetch.py --out data/pm --per-scenario 2          # amostra estratificada
"""
from __future__ import annotations

import argparse
import os
from concurrent.futures import ThreadPoolExecutor

URL = "https://zenodo.org/records/13617673/files/pmcardio_ecg_image_database.zip?download=1"


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
    from remotezip import RemoteZip
    os.makedirs(a.out, exist_ok=True)
    with RemoteZip(URL) as z:
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
        # fatia por (cenário, Image ID): as páginas de um ECG ficam juntas
        key = m["scen"].astype("category").cat.codes * 1000 + m["Image ID"]
        m = m[key % a.nshards == a.shard]
    sel = list(m["Image relative path"])
    root = os.path.join(a.out, "visual_data")

    def get(chunk):
        with RemoteZip(URL) as z:
            for p in chunk:
                dst = os.path.join(root, p)
                if os.path.exists(dst):
                    continue
                for tent in range(5):
                    try:
                        z.extract("final_data/visual_data/" + p, os.path.join(a.out, "_tmp"))
                        break
                    except Exception as e:  # noqa: BLE001
                        print("tentativa", tent, p, e, flush=True)
                os.makedirs(os.path.dirname(dst), exist_ok=True)
                src = os.path.join(a.out, "_tmp/final_data/visual_data", p)
                if os.path.exists(src):
                    os.replace(src, dst)

    chunks = [sel[i::a.threads] for i in range(a.threads)]
    with ThreadPoolExecutor(a.threads) as ex:
        list(ex.map(get, chunks))
    pd.Series(sel, name="path").to_csv(os.path.join(a.out, "sel.csv"), index=False)
    print("imagens", len(sel))


if __name__ == "__main__":
    main()
