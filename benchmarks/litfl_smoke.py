#!/usr/bin/env python3
"""Download LITFL images listed in the user's existing metadata and run ecgdig."""
import argparse, csv, json, time, traceback, urllib.request
from pathlib import Path
import numpy as np
from ecgdig import digitize, Config

CSV_URL="https://raw.githubusercontent.com/fmenegui/image-based-ecg-classification-emergency/main/data/external/litfl/metadata/litfl_source_labels.csv"
FALLBACK = [
("img_0000.png","https://litfl.com/wp-content/uploads/2018/08/ECG-Right-Bundle-Branch-Block-RBBB-6.jpg"),
("img_0001.png","https://litfl.com/wp-content/uploads/2018/08/ECG-Right-Bundle-Branch-Block-RBBB-5.jpg"),
("img_0002.png","https://litfl.com/wp-content/uploads/2018/08/ECG-Right-Bundle-Branch-Block-RBBB-3.jpg"),
("img_0003.png","https://litfl.com/wp-content/uploads/2013/02/OMI-ECG-Proximal-LAD-RBBB-2048x1011.png"),
("img_0004.png","https://litfl.com/wp-content/uploads/2022/07/RBBB-NEW.jpeg"),
("img_0005.png","https://litfl.com/wp-content/uploads/2021/01/ECG-Left-Bundle-Branch-Block-LBBB-2-2.jpg"),
("img_0006.png","https://litfl.com/wp-content/uploads/2021/01/ECG-LBBB-AF-3.jpg"),
("img_0007.png","https://litfl.com/wp-content/uploads/2018/08/Left-bundle-branch-block-LBBB-4.jpg"),
("img_0008.png","https://litfl.com/wp-content/uploads/2022/07/LBBB.jpeg"),
("img_0009.png","https://litfl.com/wp-content/uploads/2018/08/ECG-Atrial-Fibrillation-coarse-flutter-waves.jpg"),
("img_0010.png","https://litfl.com/wp-content/uploads/2018/08/ECG-Atrial-Fibrillation-coarse-flutter-waves-2-2.jpg"),
]

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--metadata",default=CSV_URL)
    p.add_argument("--limit",type=int,default=12)
    p.add_argument("--out",type=Path,default=Path("results/litfl"))
    a=p.parse_args()
    a.out.mkdir(parents=True,exist_ok=True)
    try:
        with urllib.request.urlopen(a.metadata,timeout=30) as f:
            rows=list(csv.DictReader(line.decode("utf-8-sig") for line in f))
    except Exception as exc:
        print(f"Metadata unavailable ({exc}), using bundled public LITFL URLs",flush=True)
        rows=[{"img_filename":name,"source_url":url} for name,url in FALLBACK]
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
