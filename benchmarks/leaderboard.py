"""The development leaderboard: every experiment on the same sets, one row each.

    python benchmarks/leaderboard.py [--out docs/leaderboard.md] [--base dev2-base]

Collects the artifacts of the experiments (experiment.yml, or any pod-eval / kaggle-full run whose
tag starts with "dev"):
  PM-ECG-ID  podeval-<tag>-<run>-0/pm_results.json   (300 ECGs, 5 per scenario)
  Kaggle     kfull-<tag>-merged/results.json          (60 records, 540 images)
and writes one table: PMcardio-style SNR, failed leads and the 60-scenario mean on PM-ECG-ID;
the official Kaggle metric on all images, the holdout and per type. Experiments are compared on
the images both have (coverage is reported). Descriptions come from docs/experiments.csv
(tag,description); unknown tags are listed with their tag only.
"""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import compare as C  # noqa: E402
import metrics as MT  # noqa: E402

REPO = "fmenegui/ecg-classic"


def _artifacts():
    out, page = [], 1
    while True:
        r = json.loads(subprocess.run(["gh", "api", f"repos/{REPO}/actions/artifacts?per_page=100&page={page}"],
                                      capture_output=True, text=True, check=True).stdout)
        a = r.get("artifacts", [])
        out += [x for x in a if not x.get("expired")]
        if len(a) < 100:
            return out
        page += 1


def _download(art, dest):
    dest.mkdir(parents=True, exist_ok=True)
    subprocess.run(["gh", "run", "download", str(art["workflow_run"]["id"]), "-R", REPO, "-n", art["name"],
                    "-D", str(dest)], capture_output=True, check=False)


def collect(work):
    exps = {}
    for a in _artifacts():
        n = a["name"]
        if n.startswith("kfull-dev") and n.endswith("-merged"):
            tag = n[len("kfull-"):-len("-merged")]
            e = exps.setdefault(tag, {})
            if a["created_at"] > e.get("k_t", ""):
                e.update(k=a, k_t=a["created_at"])
        elif n.startswith("podeval-dev"):
            tag = "-".join(n[len("podeval-"):].split("-")[:-2])
            e = exps.setdefault(tag, {})
            if a["created_at"] > e.get("p_t", ""):
                e.update(p=a, p_t=a["created_at"])
    for tag, e in exps.items():
        if "k" in e:
            _download(e["k"], work / tag / "k")
        if "p" in e:
            _download(e["p"], work / tag / "p")
    return sorted(exps)


def _pm(spec):
    f = list(Path(spec).rglob("pm_results.json")) if Path(spec).exists() else []
    return C._pm_load(str(f[0])) if f else {}


def _k(spec):
    f = list(Path(spec).rglob("results.json")) if Path(spec).exists() else []
    return C._k_load(str(f[0])) if f else {}


def table(work, tags, base, desc):
    PM = {t: _pm(work / t / "p") for t in tags}
    K = {t: _k(work / t / "k") for t in tags}
    pm_keys = set.intersection(*[set(v) for v in PM.values() if v]) if any(PM.values()) else set()
    k_keys = set.intersection(*[set(v) for v in K.values() if v]) if any(K.values()) else set()
    hold = lambda x: int(x[0]) % 5 == 0 or x[0] in C.LOCAL15  # noqa: E731
    types = sorted({x[1] for x in k_keys})
    rows = []
    for t in tags:
        r = dict(tag=t, desc=desc.get(t, ""))
        if PM[t] and pm_keys:
            ks = sorted(pm_keys)
            s = C._pm_summ([PM[t][x] for x in ks])
            sc = {}
            for x in ks:
                sc.setdefault(x[0], []).append(PM[t][x])
            r.update(pm=s[0], pm_fail=s[2], pm_kag=s[3],
                     pm60=float(np.mean([C._pm_summ(v)[0] for v in sc.values()])) if len(sc) >= 50 else float("nan"))
        if K[t] and k_keys:
            ks = sorted(k_keys)
            r["k_all"] = MT.kaggle_score([K[t][x]["kaggle_lin"] for x in ks])
            r["k_hold"] = MT.kaggle_score([K[t][x]["kaggle_lin"] for x in ks if hold(x)])
            r["k_type"] = {ty: MT.kaggle_score([K[t][x]["kaggle_lin"] for x in ks if x[1] == ty]) for ty in types}
        if "pm" in r or "k_all" in r:
            rows.append(r)
    rows.sort(key=lambda r: -(r.get("k_all", -99) + r.get("pm", -99)))
    n_ecg = len({(x[0], x[1]) for x in pm_keys})
    L = ["# Development leaderboard", "",
         f"PM-ECG-ID: {n_ecg} ECGs common to all experiments ({100 * n_ecg / C.PM_TOTAL:.1f}% of {C.PM_TOTAL}); "
         f"Kaggle: {len(k_keys)} images ({100 * len(k_keys) / C.K_TOTAL:.1f}% of {C.K_TOTAL}), holdout = id % 5 == 0 "
         "plus the 15 local records.", "",
         f"References: PMcardio {C.PMCARDIO_60:.2f} dB (60-subset mean of its paper); Kaggle leaders 22.6-22.9 dB "
         "(private leaderboard, test set: not the same images).", "",
         "| Experiment | PM SNR (dB) | PM 60-scenario mean | PM failed leads | Kaggle all (dB) | Kaggle holdout | "
         + " | ".join(types) + " | What |",
         "|---|---|---|---|---|---|" + "---|" * len(types) + "---|"]
    f = lambda v: f"{v:.2f}" if isinstance(v, float) and np.isfinite(v) else "-"  # noqa: E731
    for r in rows:
        L.append(f"| {'**' + r['tag'] + '**' if r['tag'] == base else r['tag']} | {f(r.get('pm'))} | {f(r.get('pm60'))} | "
                 f"{f(r.get('pm_fail'))}% | {f(r.get('k_all'))} | {f(r.get('k_hold'))} | "
                 + " | ".join(f(r.get("k_type", {}).get(ty)) for ty in types) + f" | {r['desc']} |")
    return "\n".join(L) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("docs/leaderboard.md"))
    ap.add_argument("--base", default="dev2-base")
    ap.add_argument("--tags", default="", help="comma separated (default: every dev* experiment found)")
    a = ap.parse_args()
    desc = {}
    p = Path("docs/experiments.csv")
    if p.exists():
        desc = {r["tag"]: r["description"] for r in csv.DictReader(open(p))}
    with tempfile.TemporaryDirectory() as td:
        work = Path(td)
        tags = collect(work)
        if a.tags:
            tags = [t for t in tags if t in a.tags.split(",")]
        tags = [t for t in tags if (work / t / "k").exists() or (work / t / "p").exists()]
        md = table(work, tags, a.base, desc)
    a.out.write_text(md)
    print(md)


if __name__ == "__main__":
    main()
