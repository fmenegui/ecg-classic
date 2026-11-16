#!/usr/bin/env python3
"""Benchmark on the Kaggle dataset "PhysioNet - Digitization of ECG Images".

  inspect : list the competition files and download one example record, to check the
            format (names, columns, sampling rate, duration per lead).
  eval    : (once the format is checked) download N training records, digitize each image
            with ecgdig and measure the SNR against the true signal.

Authentication: KAGGLE_API_TOKEN variable (repository secret), CLI kaggle>=2.2.
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import os
import re
import subprocess
import sys
import zipfile
from collections import Counter
from pathlib import Path

COMP = os.environ.get("COMP", "physionet-ecg-image-digitization")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))


def kaggle(*args, check=True):
    r = subprocess.run(["kaggle", *args], capture_output=True, text=True)
    if check and r.returncode != 0:
        raise RuntimeError(f"kaggle {' '.join(args)}\n{r.stdout}\n{r.stderr}")
    return r.stdout


def list_files(max_pages=400):
    """All files (name, size) through the paginated CLI."""
    out, token = [], None
    for _ in range(max_pages):
        args = ["competitions", "files", COMP, "--page-size", "200", "-v"]
        if token:
            args += ["--page-token", token]
        txt = kaggle(*args)
        lines = [l for l in txt.splitlines() if l.strip()]
        token = None
        rows = []
        for l in lines:
            m = re.match(r"Next Page Token = (\S+)", l)
            if m:
                token = m.group(1)
            elif "," in l:
                rows.append(l)
        rd = csv.reader(io.StringIO("\n".join(rows)))
        hdr = None
        for r in rd:
            if hdr is None:
                hdr = r
                continue
            out.append(dict(zip(hdr, r)))
        if not token:
            break
    return out


def download(name, dest: Path):
    """Download one competition file; return the local path (unzips a .zip)."""
    dest.mkdir(parents=True, exist_ok=True)
    base = Path(name).name
    have = [p for p in dest.rglob(base) if p.is_file()]
    if have:  # already downloaded (workflow cache): no Kaggle request spent
        return have[0]
    before = {p for p in dest.rglob("*")}
    import time
    for tent in range(7):
        r = subprocess.run(["kaggle", "competitions", "download", COMP, "-f", name, "-p", str(dest)],
                           capture_output=True, text=True)
        log = (r.stdout + " | " + r.stderr).strip()  # stderr shows usage limit/credential problems
        if "429" not in log:
            break
        time.sleep(min(300, 15 * 2 ** tent))  # Kaggle request limit: wait and retry
    new = [p for p in dest.rglob("*") if p not in before and p.is_file()]
    for z in [p for p in new if p.suffix == ".zip"]:
        try:
            with zipfile.ZipFile(z) as zf:
                zf.extractall(dest)
            z.unlink()
        except zipfile.BadZipFile:
            pass
    base = Path(name).name
    hits = [p for p in dest.rglob("*") if p.is_file() and p.name == base]
    if not hits:
        raise FileNotFoundError(f"{name} did not appear in {dest}; CLI output: {log.strip()[:300]}")
    return hits[0]


def inspect(out: Path):
    out.mkdir(parents=True, exist_ok=True)
    files = list_files()
    (out / "files.json").write_text(json.dumps(files, indent=0))
    names = [f.get("name") or f.get("ref") or "" for f in files]
    L = ["# PhysioNet (Kaggle): inspection", "", f"{len(names)} files.", ""]
    top = Counter(n.split("/")[0] for n in names)
    ext = Counter(Path(n).suffix for n in names)
    L += ["Top-level directories: " + ", ".join(f"{k} ({v})" for k, v in top.most_common(20)), "",
          "Extensions: " + ", ".join(f"{k or '∅'} ({v})" for k, v in ext.most_common(20)), "",
          "First 60 names:", "", "```", *names[:60], "```", ""]
    # small top-level CSVs and one example record
    small = [f for f in files if "/" not in (f.get("name") or "") and (f.get("name") or "").endswith(".csv")]
    for f in small[:6]:
        try:
            p = download(f["name"], out / "sample")
            head = p.read_text(errors="replace").splitlines()[:6]
        except Exception as e:  # noqa: BLE001
            head = [repr(e)]
        L += [f"## {f['name']} ({f.get('size', '?')})", "", "```", *head, "```", ""]
    # first training record (all files with the same folder prefix)
    train = [n for n in names if n.startswith("train/")]
    if train:
        rec = "/".join(train[0].split("/")[:2])
        members = [n for n in names if n.startswith(rec + "/") or n.startswith(rec + "-") or n == rec]
        L += [f"## Example record `{rec}` ({len(members)} files)", "", "```", *members[:40], "```", ""]
        for n in members[:20]:
            try:
                p = download(n, out / "sample" / Path(n).parent)
            except Exception as e:  # noqa: BLE001
                L.append(f"failed {n}: {e}")
                continue
            if p.suffix == ".csv":
                txt = p.read_text(errors="replace").splitlines()
                (out / "sample_record.csv").write_text("\n".join(txt[:20]))
                L += [f"`{n}`: {len(txt)} lines", "", "```", *txt[:4], "```", ""]
            elif p.suffix.lower() in (".png", ".jpg", ".jpeg"):
                import cv2
                im = cv2.imread(str(p))
                L.append(f"`{n}`: image {None if im is None else im.shape}")
        L.append("")
    L += ["## Downloaded files", "", "```", *[str(p) for p in sorted((out / "sample").rglob("*"))][:80], "```"]
    (out / "report.md").write_text("\n".join(L))
    print("\n".join(L))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["inspect", "eval", "prefetch"])
    ap.add_argument("--out", type=Path, default=Path("results/physionet"))
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--jobs", type=int, default=2)
    ap.add_argument("--keep-raw", type=int, default=5, help="records whose raw images go into the artifact")
    ap.add_argument("--config", action="append", default=[])
    ap.add_argument("--ids", default="", help="specific records, comma-separated")
    a = ap.parse_args()
    if a.mode == "inspect":
        inspect(a.out)
    elif a.mode == "prefetch":  # download only (workflow cache), no evaluation
        import csv as _csv
        cache = a.out / "data"
        meta = download("train.csv", cache)
        ids = [x for x in a.ids.split(",") if x]
        rows = [r for r in _csv.DictReader(open(meta)) if r["id"] in ids]
        miss = 0
        for r in rows:
            rid = r["id"]
            for nm in [f"train/{rid}/{rid}.csv"] + [f"train/{rid}/{rid}-{t}.png" for t in
                                                     ("0001", "0003", "0004", "0005", "0006", "0009", "0010", "0011", "0012")]:
                try:
                    have = any(q.is_file() for q in (cache / rid).rglob(Path(nm).name)) if (cache / rid).exists() else False
                    download(nm, cache / rid)
                    if not have:
                        import time as _t
                        _t.sleep(3)  # slowly: the Kaggle request limit drops bursts
                except Exception as e:  # noqa: BLE001
                    miss += 1
                    print("failed", nm, str(e)[:120], flush=True)
        print("missing", miss)
    else:
        from eval_snr import evaluate  # written after inspecting the format
        evaluate(a)


if __name__ == "__main__":
    sys.path.insert(0, str(HERE))
    main()
