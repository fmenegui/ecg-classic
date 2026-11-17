"""Read single files out of the Kaggle competition archive by HTTP ranges (like PM-ECG-ID).

The whole "PhysioNet - Digitization of ECG Images" archive is ~86 GB; downloading it file by file
through the API runs into the request limit. Kaggle answers the "download all" request with a
redirect to a signed storage URL; if that URL serves byte ranges, remotezip reads the zip's
central directory and then only the bytes of the files we want.

    python benchmarks/physionet/kaggle_zip.py probe
    python benchmarks/physionet/kaggle_zip.py fetch --ids 123,456 --out DIR [--types 0001,0003]
    python benchmarks/physionet/kaggle_zip.py shard --part 3 --nparts 20 --out DIR

Needs KAGGLE_API_TOKEN (and KAGGLE_USERNAME for legacy keys).
"""
from __future__ import annotations

import argparse
import csv
import io
import os
import sys
import time
from pathlib import Path

import requests

COMP = os.environ.get("COMP", "physionet-ecg-image-digitization")
TYPES = ("0001", "0003", "0004", "0005", "0006", "0009", "0010", "0011", "0012")


def signed_url():
    tok = os.environ.get("KAGGLE_API_TOKEN", "").strip()
    url = f"https://www.kaggle.com/api/v1/competitions/data/download-all/{COMP}"
    tries = [dict(headers={"Authorization": f"Bearer {tok}"})]
    if os.environ.get("KAGGLE_USERNAME"):
        tries.append(dict(auth=(os.environ["KAGGLE_USERNAME"], tok)))
    last = None
    for kw in tries:
        r = requests.get(url, allow_redirects=False, timeout=60, **kw)
        last = r
        if r.status_code in (301, 302, 303, 307, 308) and r.headers.get("Location"):
            return r.headers["Location"]
        if r.status_code == 200 and r.headers.get("content-type", "").startswith("application/json"):
            j = r.json()
            for k in ("url", "downloadUrl", "signedUrl"):
                if j.get(k):
                    return j[k]
    raise RuntimeError(f"no signed URL: HTTP {last.status_code} {last.text[:300]}")


def open_zip(retries=3):
    from remotezip import RemoteZip
    for i in range(retries):
        try:
            return RemoteZip(signed_url(), timeout=120)
        except Exception as e:  # noqa: BLE001
            print("open failed:", e, flush=True)
            time.sleep(10 * (i + 1))
    raise RuntimeError("could not open the remote archive")


def fetch(z, ids, out: Path, types=TYPES):
    names = set(z.namelist())
    out.mkdir(parents=True, exist_ok=True)
    got = miss = 0
    for rid in ids:
        want = [f"train/{rid}/{rid}.csv"] + [f"train/{rid}/{rid}-{t}.png" for t in types]
        d = out / rid
        d.mkdir(exist_ok=True)
        for n in want:
            dst = d / Path(n).name
            if dst.exists() and dst.stat().st_size > 0:
                got += 1
                continue
            if n not in names:
                miss += 1
                continue
            dst.write_bytes(z.read(n))
            got += 1
    return got, miss


def train_ids(z):
    meta = list(csv.DictReader(io.StringIO(z.read("train.csv").decode())))
    return [r["id"] for r in meta], {r["id"]: r for r in meta}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["probe", "fetch", "shard"])
    ap.add_argument("--ids", default="")
    ap.add_argument("--out", type=Path, default=Path("kdata"))
    ap.add_argument("--part", type=int, default=0)
    ap.add_argument("--nparts", type=int, default=20)
    ap.add_argument("--types", default=",".join(TYPES))
    a = ap.parse_args()
    if a.cmd == "probe":
        u = signed_url()
        print("signed URL host:", u.split("/")[2], flush=True)
        h = requests.get(u, headers={"Range": "bytes=0-99"}, timeout=60)
        print("range request:", h.status_code, "content-range:", h.headers.get("Content-Range"),
              "accept-ranges:", h.headers.get("Accept-Ranges"), flush=True)
        t0 = time.time()
        z = open_zip()
        names = z.namelist()
        print(f"archive opened in {time.time() - t0:.1f}s: {len(names)} files", flush=True)
        ids, _ = train_ids(z)
        print(len(ids), "train records; first", ids[:3], flush=True)
        t0 = time.time()
        g, m = fetch(z, ids[:1], a.out)
        print(f"fetched {g} files ({m} missing) of record {ids[0]} in {time.time() - t0:.1f}s", flush=True)
        for p in sorted((a.out / ids[0]).iterdir()):
            print(" ", p.name, p.stat().st_size, flush=True)
        return
    z = open_zip()
    if a.cmd == "fetch":
        ids = [x for x in a.ids.split(",") if x]
    else:
        all_ids, _ = train_ids(z)
        ids = all_ids[a.part::a.nparts]
        (a.out).mkdir(parents=True, exist_ok=True)
        (a.out / "train.csv").write_bytes(z.read("train.csv"))
    t0 = time.time()
    g, m = fetch(z, ids, a.out, tuple(a.types.split(",")))
    print(f"{len(ids)} records: {g} files, {m} missing, {time.time() - t0:.0f}s", flush=True)
    (a.out / "ids.txt").write_text(",".join(ids))


if __name__ == "__main__":
    sys.exit(main())
