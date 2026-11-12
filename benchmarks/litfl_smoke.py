#!/usr/bin/env python3
"""Download LITFL images listed in the user's existing metadata and run ecgdig."""
import argparse, csv, json, time, traceback, urllib.request
from pathlib import Path
import numpy as np
from ecgdig import digitize, Config

CSV_URL="https://raw.githubusercontent.com/fmenegui/image-based-ecg-classification-emergency/main/data/external/litfl/metadata/litfl_source_labels.csv"

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--metadata",default=CSV_URL)
    p.add_argument("--limit",type=int,default=12)
    p.add_argument("--out",type=Path,default=Path("results/litfl"))
    a=p.parse_args()
    a.out.mkdir(parents=True,exist_ok=True)
    with urllib.request.urlopen(a.metadata,timeout=30) as f:
        rows=list(csv.DictReader(line.decode("utf-8-sig") for line in f))
    results=[]
    for i,row in enumerate(rows[:a.limit]):
        path=a.out/row["img_filename"]
        entry={"filename":row["img_filename"],"source_url":row["source_url"],"status":"failed"}
        start=time.monotonic()
        try:
            req=urllib.request.Request(row["source_url"],headers={"User-Agent":"ECGResearchBenchmark/1.0"})
            with urllib.request.urlopen(req,timeout=35) as f:
                content=f.read(20*1024*1024+1)
                if len(content)>20*1024*1024: raise ValueError("Image over 20 MB")
            path.write_bytes(content)
            out=digitize(str(path),Config())
            sig=np.asarray(out["signals"],dtype=float)
            np.savez_compressed(a.out/f"signal_{i:04d}.npz",signals=sig)
            entry.update(status="ok",shape=list(sig.shape),finite_fraction=float(np.isfinite(sig).mean()))
        except Exception as e:
            entry.update(error=str(e),traceback=traceback.format_exc(limit=2))
        entry["elapsed_s"]=round(time.monotonic()-start,2)
        results.append(entry)
        print(f"{i+1}/{min(len(rows),a.limit)} {entry['status']} {entry['filename']}",flush=True)
        (a.out/"results.json").write_text(json.dumps(results,indent=2))
    summary={"total":len(results),"successful":sum(r["status"]=="ok" for r in results),"failed":sum(r["status"]!="ok" for r in results),"note":"LITFL labels are diagnoses, not signal ground truth. No SNR is available."}
    (a.out/"summary.json").write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary,indent=2))
if __name__=="__main__":main()
