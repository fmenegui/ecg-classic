#!/usr/bin/env python3
"""PM-ECG-ID pilot: stratified sample, execution and honest per-image diagnostics."""
import argparse, csv, json, time, traceback
from pathlib import Path
import numpy as np
from ecgdig import digitize, Config

def locate(root, relative):
    p = root / str(relative).replace("\\", "/")
    if p.exists(): return p
    hits = list(root.rglob(Path(relative).name))
    return hits[0] if len(hits) == 1 else None

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--dataset",required=True,type=Path,help="Unpacked PM-ECG-ID directory")
    ap.add_argument("--output",default="results/pmcardio",type=Path)
    ap.add_argument("--n",type=int,default=100)
    ap.add_argument("--seed",type=int,default=2026)
    args=ap.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    meta=list(args.dataset.rglob("metadata.csv"))
    if len(meta)!=1: raise RuntimeError(f"Expected one metadata.csv, found {len(meta)}")
    with meta[0].open(newline="",encoding="utf-8-sig") as f: rows=list(csv.DictReader(f))
    def field(row,term):
        return next((v for k,v in row.items() if term in k.strip().lower()),"")
    def group(row):
        return str(Path(field(row,"image relative path")).parent)
    groups={}
    for row in rows: groups.setdefault(group(row),[]).append(row)
    rng=np.random.default_rng(args.seed)
    for entries in groups.values(): rng.shuffle(entries)
    selected=[]
    while len(selected)<min(args.n,len(rows)) and any(groups.values()):
        for key in sorted(groups):
            if groups[key] and len(selected)<args.n: selected.append(groups[key].pop())
    with (args.output/"sample_manifest.csv").open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=rows[0].keys());w.writeheader();w.writerows(selected)
    results=[]
    for i,row in enumerate(selected,1):
        relative=field(row,"image relative path")
        path=locate(args.dataset,relative)
        result={"image":relative,"ecg_id":field(row,"ecg id"),"group":group(row),"status":"failed"}
        t=time.monotonic()
        try:
            if path is None: raise FileNotFoundError(relative)
            out=digitize(str(path),Config())
            sig=np.asarray(out["signals"],dtype=float)
            result.update(status="ok",shape=str(sig.shape),finite_fraction=float(np.isfinite(sig).mean()))
            np.savez_compressed(args.output/f"prediction_{i:04d}.npz",signals=sig)
        except Exception as exc:
            result.update(error=str(exc),traceback=traceback.format_exc(limit=3))
        result["elapsed_seconds"]=round(time.monotonic()-t,3)
        results.append(result)
        print(f"[{i}/{len(selected)}] {result['status']} {relative}",flush=True)
        with (args.output/"results.json").open("w") as f:json.dump(results,f,indent=2)
    summary={"requested":args.n,"selected":len(selected),"successes":sum(r["status"]=="ok" for r in results),"failures":sum(r["status"]!="ok" for r in results),"mean_runtime_s":float(np.mean([r["elapsed_seconds"] for r in results])) if results else None,"note":"No SNR/RMSE claimed: PMcardio leads.npz contains layout-dependent cutouts requiring verified lead/time mapping before valid ground-truth comparison."}
    (args.output/"summary.json").write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary,indent=2))
if __name__=="__main__":main()
