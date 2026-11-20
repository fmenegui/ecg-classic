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
import threading
import time
from pathlib import Path

import requests

COMP = os.environ.get("COMP", "physionet-ecg-image-digitization")
TYPES = ("0001", "0003", "0004", "0005", "0006", "0009", "0010", "0011", "0012")


def _complete(p: Path) -> bool:
    """False for a PNG cut short (written before downloads were atomic and then cancelled)."""
    if p.suffix.lower() != ".png":
        return True
    with open(p, "rb") as f:
        f.seek(max(0, p.stat().st_size - 12))
        return f.read().endswith(b"IEND\xaeB`\x82")


def signed_url():
    tok = os.environ.get("KAGGLE_API_TOKEN", "").strip()
    url = f"https://www.kaggle.com/api/v1/competitions/data/download-all/{COMP}"
    tries = [dict(headers={"Authorization": f"Bearer {tok}"})]
    if os.environ.get("KAGGLE_USERNAME"):
        tries.append(dict(auth=(os.environ["KAGGLE_USERNAME"], tok)))
    last = None
    # diagnosis without revealing the token: its kind and length, and the answer to each attempt
    print(f"token: length {len(tok)}, new-style KGAT {tok.startswith('KGAT_')}, "
          f"username set {bool(os.environ.get('KAGGLE_USERNAME'))}", flush=True)
    for kw in tries:
        r = requests.get(url, allow_redirects=False, timeout=60, **kw)
        print("  attempt", "bearer" if "headers" in kw else "basic", "-> HTTP", r.status_code, flush=True)
        last = r
        if r.status_code in (301, 302, 303, 307, 308) and r.headers.get("Location"):
            return r.headers["Location"]
        if r.status_code == 200 and r.headers.get("content-type", "").startswith("application/json"):
            j = r.json()
            for k in ("url", "downloadUrl", "signedUrl"):
                if j.get(k):
                    return j[k]
    raise RuntimeError(f"no signed URL: HTTP {last.status_code} {last.text[:300]}")


def open_zip(retries=8, part=0, n=1):
    """Kaggle limits the download-all endpoint (HTTP 429 when 20 runners ask at once): the shards
    start staggered and back off with jitter up to ~15 min in all. n > 1: that many handles on one
    signed URL (one API call), for parallel range reads."""
    import random
    from remotezip import RemoteZip
    time.sleep(12 * min(part, 20))
    for i in range(retries):
        try:
            u = signed_url()
            zs = [RemoteZip(u, timeout=120) for _ in range(n)]
            return zs[0] if n == 1 else zs
        except Exception as e:  # noqa: BLE001
            print("open failed:", e, flush=True)
            time.sleep(min(240, 15 * 2 ** i) * random.uniform(0.6, 1.0))
    raise RuntimeError("could not open the remote archive")


def _read(z, name, retries=6):
    """One member of the remote archive; a read cut off mid-transfer (IncompleteRead, timeout) is
    retried with back-off instead of failing the whole shard."""
    for i in range(retries):
        try:
            return z.read(name)
        except Exception as e:  # noqa: BLE001
            if i == retries - 1:
                raise
            print(f"read {name} failed ({type(e).__name__}), retrying", flush=True)
            time.sleep(5 * 2 ** i)


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
            if dst.exists() and dst.stat().st_size > 0 and _complete(dst):
                got += 1
                continue
            if n not in names:
                miss += 1
                continue
            # atomic: a cancelled download leaves no truncated file; unique per writer, since pods
            # sharing the volume may fetch the same record at once
            tmp = dst.with_name(f"{dst.name}.{os.getpid()}.{threading.get_ident()}.part")
            tmp.write_bytes(_read(z, n))
            tmp.replace(dst)
            got += 1
    return got, miss


def train_ids(z):
    meta = list(csv.DictReader(io.StringIO(_read(z, "train.csv").decode())))
    return [r["id"] for r in meta], {r["id"]: r for r in meta}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["probe", "fetch", "shard"])
    ap.add_argument("--ids", default="")
    ap.add_argument("--out", type=Path, default=Path("kdata"))
    ap.add_argument("--part", type=int, default=0)
    ap.add_argument("--nparts", type=int, default=20)
    ap.add_argument("--types", default=",".join(TYPES))
    ap.add_argument("--ids-file", default="", help="where to write the record ids (default OUT/ids.txt)")
    ap.add_argument("--threads", type=int, default=1, help="parallel range readers (shard)")
    ap.add_argument("--stride", type=int, default=1, help="every k-th record of the shard (development subset)")
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
        # what else is in the archive (public test images, csvs): names, sizes and the test table only,
        # nothing is written out (competition data stays off the public artifacts)
        other = [n for n in names if not n.startswith("train/")]
        print(len(other), "files outside train/:", flush=True)
        for n in other[:60]:
            info = z.getinfo(n)
            line = f"  {n} {info.file_size}"
            if n.lower().endswith(".png"):
                hdr = z.read(n)[:32]
                if hdr[12:16] == b"IHDR":
                    line += f" {int.from_bytes(hdr[16:20], 'big')}x{int.from_bytes(hdr[20:24], 'big')}"
            print(line, flush=True)
        for n in other:
            if n.endswith(".csv") and z.getinfo(n).file_size < 2_000_000:
                txt = z.read(n).decode(errors="replace").splitlines()
                print(f"--- {n}: {len(txt)} lines; head:", flush=True)
                for l in txt[:8]:
                    print("   ", l[:160], flush=True)
        tr = z.read("train.csv").decode().splitlines()
        print("--- train.csv head:", *tr[:4], sep="\n    ", flush=True)
        t0 = time.time()
        g, m = fetch(z, ids[:1], a.out)
        print(f"fetched {g} files ({m} missing) of record {ids[0]} in {time.time() - t0:.1f}s", flush=True)
        for p in sorted((a.out / ids[0]).iterdir()):
            print(" ", p.name, p.stat().st_size, flush=True)
        return
    z = open_zip(part=a.part)
    if a.cmd == "fetch":
        ids = [x for x in a.ids.split(",") if x]
    else:
        all_ids, _ = train_ids(z)
        ids = all_ids[a.part::a.nparts][::a.stride]
        (a.out).mkdir(parents=True, exist_ok=True)
        tmp = a.out / f"train.csv.{os.getpid()}"
        tmp.write_bytes(_read(z, "train.csv"))
        tmp.replace(a.out / "train.csv")   # atomic: pods sharing the volume may write it at once
    t0 = time.time()
    if a.threads <= 1:
        g, m = fetch(z, ids, a.out, tuple(a.types.split(",")))
    else:
        # one archive handle per thread (a RemoteZip is not thread-safe); each record is one
        # unit of work. A single stream did not fetch the 977 records in 4.5 h on a pod.
        from concurrent.futures import ThreadPoolExecutor
        zs = [z] + open_zip(n=a.threads - 1) if a.threads > 2 else [z, open_zip()]
        done = [0, 0, 0]
        lock = threading.Lock()

        def work(k):
            for rid in ids[k::a.threads]:
                g_, m_ = fetch(zs[k], [rid], a.out, tuple(a.types.split(",")))
                with lock:
                    done[0] += 1; done[1] += g_; done[2] += m_
                    if done[0] % 50 == 0:
                        print(f"{done[0]}/{len(ids)} records, {time.time() - t0:.0f}s", flush=True)

        with ThreadPoolExecutor(a.threads) as ex:
            list(ex.map(work, range(a.threads)))
        g, m = done[1], done[2]
    print(f"{len(ids)} records: {g} files, {m} missing, {time.time() - t0:.0f}s", flush=True)
    (Path(a.ids_file) if a.ids_file else a.out / "ids.txt").write_text(",".join(ids))


if __name__ == "__main__":
    sys.exit(main())
