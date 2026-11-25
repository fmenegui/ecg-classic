"""Build real row strips on the training pod while the GPU trains (streaming).

Reads Kaggle training pages straight out of the competition archive by HTTP ranges (one API
request for the signed URL, then storage ranges: no per-file API calls), labels them with
realstrips.page_labels in a process pool, and writes chunks that train_gpu.py --real-dir picks
up as they appear. The competition data stays on the pod: nothing is uploaded or published.

Split by record: HOLDOUT records (every id % 5 == 0, plus the 15 local and the evaluation
subsets) are never read here; they remain for evaluation only.

    python hybrid/realstrips_build.py --out /workspace/real --max-pages 3000 --workers 10
"""
from __future__ import annotations

import argparse
import io
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent)); sys.path.insert(0, str(HERE.parent / "benchmarks" / "physionet"))

# the 15 local evaluation records (same list as benchmarks/compare.py and merge_shards.py). Until
# Nov 20 this list stopped after 8 ids, so 5 of them (not id % 5 == 0) could become real strips
LOCAL15 = set(("1257259783,2587667307,3042142152,426433979,994226251,119693645,1449491391,1962423499,"
               "235672018,2841936136,3574249147,3646789790,3879757040,4050029009,644026427").split(","))


XUP = [int(os.environ.get("REALSTRIPS_XUP", "1"))]   # labels at the stretched columns (--xup)


def is_holdout(rid: str) -> bool:
    return int(rid) % 5 == 0 or rid in LOCAL15


def _work(args):
    rid, typ, png, csv_bytes, seed = args
    import cv2
    import realstrips as RS
    try:
        img = cv2.imdecode(np.frombuffer(png, np.uint8), cv2.IMREAD_COLOR)
        G = RS.load_gt(io.BytesIO(csv_bytes))
        rect, segs, info = RS.page_labels(img, G)
        if info["n_jobs"] == 0 or info["kept"] < 0.6 * info["n_jobs"]:
            return rid, typ, None, info
        X, Y = RS.strips(rect, segs, np.random.default_rng(seed), per_row=2, xup=XUP[0])
        if not len(X):
            return rid, typ, None, info
        # compact: JPEG strips (q 92) and the label as one height per column (NaN = none)
        jp = [cv2.imencode(".jpg", x, [cv2.IMWRITE_JPEG_QUALITY, 92])[1].tobytes() for x in X]
        S = Y.shape[1]
        rows = np.arange(S, dtype=np.float32)[:, None]
        w = Y.sum(1)
        h = np.where(w > 0.5, (Y * rows[None]).sum(1) / np.maximum(w, 1e-6), np.nan).astype(np.float32 if XUP[0] > 1 else np.float16)
        return rid, typ, (jp, h), info
    except Exception as e:  # noqa: BLE001
        return rid, typ, None, dict(error=f"{type(e).__name__}: {e}"[:200])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--max-pages", type=int, default=3000)
    ap.add_argument("--workers", type=int, default=max(2, (os.cpu_count() or 4) // 2))
    ap.add_argument("--chunk", type=int, default=400, help="strips per chunk file")
    ap.add_argument("--readers", type=int, default=8, help="threads reading pages from the archive")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--xup", type=int, default=1,
                    help="labels at the columns of strips stretched this many times (train_gpu.py --xup)")
    ap.add_argument("--wait-hours", type=float, default=2.0, help="keep retrying the archive this long")
    ap.add_argument("--cache", type=Path, default=None,
                    help="folder (a persistent volume) keeping finished strip sets by content key; OUT becomes "
                         "a link to the cached set (reused when complete, built in place otherwise)")
    a = ap.parse_args()
    XUP[0] = a.xup
    os.environ["REALSTRIPS_XUP"] = str(a.xup)   # the labelling workers
    import kaggle_zip as KZ
    if a.cache:
        import hashlib
        h = hashlib.sha1()
        here = Path(__file__).resolve().parent
        for f in [here / "realstrips.py", Path(__file__).resolve()]:
            h.update(f.read_bytes())
        h.update(f"{a.max_pages} {a.chunk} {a.seed}".encode() + (f" xup{a.xup}".encode() if a.xup > 1 else b""))
        cdir = a.cache / f"real-{h.hexdigest()[:12]}"
        cdir.mkdir(parents=True, exist_ok=True)
        if a.out.exists() or a.out.is_symlink():
            if a.out.is_symlink() or not any(a.out.iterdir()):
                a.out.unlink() if a.out.is_symlink() else a.out.rmdir()
        if not a.out.exists():
            a.out.parent.mkdir(parents=True, exist_ok=True)
            a.out.symlink_to(cdir)
        if (cdir / "DONE").exists():
            print(f"real strips from the cache: {cdir}", flush=True)
            return
        a.out = cdir
        for f in cdir.glob("chunk_*.npz"):   # a partial set of an interrupted run is rebuilt
            f.unlink()
    a.out.mkdir(parents=True, exist_ok=True)
    t_end = time.time() + 3600 * a.wait_hours
    while True:   # Kaggle may answer 429 for hours after heavy use: keep trying, training goes on meanwhile
        try:
            z = KZ.open_zip()
            break
        except Exception as e:  # noqa: BLE001
            if time.time() > t_end:
                raise
            print("archive not available yet:", e, "- retrying in 10 min", flush=True)
            time.sleep(600)
    names = set(z.namelist())
    ids = sorted({n.split("/")[1] for n in names if n.startswith("train/") and n.count("/") == 2})
    ids = [i for i in ids if i.isdigit() and not is_holdout(i)]
    rng = np.random.default_rng(a.seed)
    rng.shuffle(ids)
    jobs = []
    for rid in ids:
        for t in KZ.TYPES:
            if f"train/{rid}/{rid}-{t}.png" in names:
                jobs.append((rid, t))
    rng.shuffle(jobs)
    jobs = jobs[:a.max_pages]
    print(f"real strips: {len(ids)} training records (holdout excluded), {len(jobs)} pages", flush=True)
    buf_x, buf_h, k, done, kept = [], [], 0, 0, 0
    csv_cache = {}

    def flush():
        nonlocal buf_x, buf_h, k
        if not buf_x:
            return
        tmp = a.out / f".chunk_{k:04d}.npz"
        np.savez(tmp, jpg=np.array(buf_x, dtype=object), h=np.stack(buf_h))
        os.replace(tmp, a.out / f"chunk_{k:04d}.npz")          # atomic: the trainer never sees half a file
        print(f"real strips: chunk {k} ({len(buf_x)} strips), pages {done}/{len(jobs)}, kept {kept}", flush=True)
        buf_x, buf_h, k = [], [], k + 1

    # the pages are read by a.readers threads, each with its own handle on the archive (one
    # stream read about 26 pages a minute, the labelling processes waited for it)
    import threading
    from collections import deque
    from concurrent.futures import ThreadPoolExecutor
    zs = [z] + (KZ.open_zip(n=a.readers - 1) if a.readers > 2 else [KZ.open_zip()] if a.readers == 2 else [])
    tl = threading.local()
    nxt = [0]
    lock = threading.Lock()

    def read(job):
        rid, t = job
        if not hasattr(tl, "k"):
            with lock:
                tl.k = nxt[0] % len(zs); nxt[0] += 1
        for attempt in range(3):
            try:
                with lock:
                    c = csv_cache.get(rid)
                if c is None:
                    c = zs[tl.k].read(f"train/{rid}/{rid}.csv")
                    with lock:
                        csv_cache[rid] = c
                return rid, t, zs[tl.k].read(f"train/{rid}/{rid}-{t}.png"), c
            except Exception as e:  # noqa: BLE001  (signed URL expired, network): reopen this handle
                print("archive read failed, reopening:", e, flush=True)
                time.sleep(30)
                zs[tl.k] = KZ.open_zip()
        return None

    with ProcessPoolExecutor(a.workers) as ex, ThreadPoolExecutor(a.readers) as rd:
        pending = set()
        reads = deque()
        it = iter(jobs)
        t0 = time.time()

        def top_up():
            while len(reads) < 4 * a.workers:
                try:
                    reads.append(rd.submit(read, next(it)))
                except StopIteration:
                    return

        top_up()
        while True:
            while len(pending) < 2 * a.workers and reads:
                r = reads.popleft().result()
                top_up()
                if r is not None:
                    rid, t, png, c = r
                    pending.add(ex.submit(_work, (rid, t, png, c, int(rng.integers(1 << 30)))))
            if not pending:
                break
            fut = next(as_completed(pending))
            pending.discard(fut)
            rid, t, got, info = fut.result()
            done += 1
            if got is not None:
                kept += 1
                buf_x += got[0]; buf_h += list(got[1])
                if len(buf_x) >= a.chunk:
                    flush()
            if done % 50 == 0:
                print(f"real strips: {done} pages in {time.time() - t0:.0f} s, kept {kept}", flush=True)
    flush()
    (a.out / "DONE").write_text(str(done))


if __name__ == "__main__":
    main()
