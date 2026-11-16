#!/usr/bin/env python3
"""LITFL benchmark: 102 real ECGs (scans/photos from various sources) without a true signal.

Measured against a hand-reviewed reference (reference.json):
  - printed layout, rhythm-strip names, speed, duration and gain;
and, with no reference at all, the internal consistency of the extracted signal:
  - coverage: fraction of the samples the page shows that were extracted;
  - Einthoven: |I - II + III| / RMS(I, II, III) on the grid measurements BEFORE fusion (3x4: the
    three limb leads are in the same column, at the same instant);
  - duplicate: NRMSE between the rhythm strip II and the grid II over the same stretch (only
    when the strip is simultaneous; in LITFL montages it is not always).

Usage:
  python benchmarks/litfl/run.py                      # downloads (cache in benchmarks/litfl/cache) and runs everything
  python benchmarks/litfl/run.py --limit 20 --jobs 2
  python benchmarks/litfl/run.py --compare results/litfl_main.json   # flags regressions
Output: results/litfl/results.json (per image + summary), report.md and report.html (+ img/)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
import urllib.request
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
sys.path.insert(0, str(HERE))

UA = {"User-Agent": "Mozilla/5.0 (ECG digitization research benchmark)"}


def fetch(entry, cache: Path):
    url = entry["url"]
    ext = os.path.splitext(url.split("?")[0])[1] or ".jpg"
    p = cache / f"{entry['file']}{ext}"
    if not p.exists():
        req = urllib.request.Request(url, headers=UA)
        with urllib.request.urlopen(req, timeout=40) as f:
            data = f.read(25 * 1024 * 1024)
        p.write_bytes(data)
    return p


def _family(name):
    if name is None:
        return None
    return name.replace("-cabrera", "")


def _expected_fraction(layout_rows, fs=500, n=5000):
    """Fraction of the time each lead appears on paper, from the layout."""
    from ecgdig.layouts import LEADS
    frac = np.zeros(len(LEADS))
    for row in layout_rows:
        k = len(row)
        for nm in row:
            base = nm.lstrip("-")
            if base in LEADS:
                frac[LEADS.index(base)] = min(1.0, frac[LEADS.index(base)] + 1.0 / k)
    return frac


def evaluate_one(args):
    key, entry, path, cfg_kw, img_dir = args
    from ecgdig import digitize, Config
    out = dict(id=key, ok=False)
    t0 = time.monotonic()
    try:
        r = digitize(str(path), Config(**cfg_kw), return_debug=True)
    except Exception as e:  # noqa: BLE001
        out.update(error=repr(e)[:300], trace=traceback.format_exc(limit=2)[-500:], seconds=time.monotonic() - t0)
        return out
    out["seconds"] = round(time.monotonic() - t0, 1)
    if img_dir is not None:  # rectified page with the extracted trace drawn on top
        import cv2
        d = r["debug_image"]
        sc = min(1.0, 1100 / d.shape[1])
        cv2.imwrite(str(Path(img_dir) / f"{key}.jpg"),
                    cv2.resize(d, None, fx=sc, fy=sc, interpolation=cv2.INTER_AREA),
                    [cv2.IMWRITE_JPEG_QUALITY, 72])
        out["image"] = f"img/{key}.jpg"
    rows = r["layout_rows"]
    rhythm = [row[0] for row in rows if len(row) == 1] if any(len(x) > 1 for x in rows) else []
    gains = sorted(set(float(g) for g in r["gains"].values()))
    li = r.get("layout_info") or {}
    out.update(ok=True, layout=r["layout"], rows=rows, rhythm=rhythm, speed=r["speed_mm_per_s"],
               speed_source=r["speed_source"], duration=r["duration_s"], gains=gains,
               confidence=float(li.get("confidence") or 0.0), warning=li.get("warning"),
               rotation=r.get("rotation"), pulses_left=r.get("pulses_found"), pulses_right=r.get("pulses_right"),
               gain_rows_inferred=r.get("gain_rows_inferred"))
    # ---- hits against the reference
    ref = entry
    chk = {}
    if ref.get("layout"):
        chk["layout"] = _family(r["layout"]) == ref["layout"]
    if ref.get("rhythm") is not None and ref.get("layout") and ref["layout"] != "rows":
        chk["rhythm"] = rhythm == ref["rhythm"]
        chk["rhythm_strips_ok"] = sum(a == b for a, b in zip(rhythm, ref["rhythm"]))
        chk["rhythm_strips"] = len(ref["rhythm"])
    if ref.get("speed"):
        chk["speed"] = abs(r["speed_mm_per_s"] - ref["speed"]) < 1e-6
    if ref.get("duration"):
        chk["duration"] = abs(r["duration_s"] - ref["duration"]) / ref["duration"] < 0.05
    if ref.get("gain"):
        chk["gain"] = gains == sorted(ref["gain"]) if len(ref["gain"]) > 1 else all(abs(g - ref["gain"][0]) < 1e-6 for g in gains)
    out["check"] = chk
    # ---- internal consistency (no reference)
    sig = r["signals"]
    exp = _expected_fraction(rows)
    got = np.isfinite(sig).mean(1)
    m = exp > 0
    out["coverage"] = float(np.clip(got[m] / exp[m], 0, 1).mean()) if m.any() else None
    ps = r.get("per_segment", {})

    def first(nm):
        lst = ps.get(nm) or []
        return lst[0] if lst else None
    I, II, III = first("I"), first("II"), first("III")
    if I is not None and II is not None and III is not None:
        mm = np.isfinite(I) & np.isfinite(II) & np.isfinite(III)
        if mm.sum() > 200:
            V = np.stack([I[mm], II[mm], III[mm]])
            V = V - np.median(V, axis=1, keepdims=True)
            scale = np.sqrt(np.mean(V ** 2))
            if scale > 0:
                e = V[0] - V[1] + V[2]
                # normalized by the amplitude of the three (II alone may be almost flat)
                out["einthoven_ratio"] = float(np.sqrt(np.mean(e ** 2)) / scale)
    lst = ps.get("II") or []
    if len(lst) >= 2:
        a, b = lst[0], lst[1]
        mm = np.isfinite(a) & np.isfinite(b)
        if mm.sum() > 200:
            na, nb = np.linalg.norm(a[mm]), np.linalg.norm(b[mm])
            out["dup_II_nrmse"] = float(np.linalg.norm(a[mm] - b[mm]) / max(na, nb, 1e-9))
    return out


def summarize(results, ref):
    ok = [x for x in results if x["ok"]]
    S = dict(images=len(results), ran=len(ok), errors=len(results) - len(ok))
    for k in ("layout", "rhythm", "speed", "duration", "gain"):
        v = [x["check"][k] for x in ok if k in x.get("check", {})]
        S[f"{k}_acc"] = round(float(np.mean(v)), 3) if v else None
        S[f"{k}_n"] = len(v)
    st = [(x["check"]["rhythm_strips_ok"], x["check"]["rhythm_strips"]) for x in ok if "rhythm_strips" in x.get("check", {})]
    S["rhythm_strip_acc"] = round(sum(a for a, _ in st) / max(1, sum(b for _, b in st)), 3)
    allok = [all(x["check"].get(k, True) for k in ("layout", "speed", "duration", "gain")) for x in ok]
    S["all_structural_ok"] = round(float(np.mean(allok)), 3) if allok else None
    for k in ("coverage", "einthoven_ratio", "dup_II_nrmse"):
        v = [x[k] for x in ok if x.get(k) is not None]
        S[f"{k}_median"] = round(float(np.median(v)), 3) if v else None
        S[f"{k}_n"] = len(v)
    S["seconds_mean"] = round(float(np.mean([x["seconds"] for x in results if "seconds" in x])), 1)
    return S


def report_md(S, results, ref, cmp=None):
    L = ["# LITFL benchmark", "",
         f"{S['ran']}/{S['images']} images processed ({S['errors']} errors), {S['seconds_mean']} s/image.", "",
         "| Metric | Value | n |", "|---|---|---|"]
    names = dict(layout_acc="Correct layout", rhythm_acc="Rhythm strips (all right)", rhythm_strip_acc="Rhythm strips (per strip)",
                 speed_acc="Speed", duration_acc="Duration", gain_acc="Gain", all_structural_ok="Layout+speed+duration+gain",
                 coverage_median="Coverage (median)", einthoven_ratio_median="Einthoven |I−II+III|/RMS(I,II,III) (median)",
                 dup_II_nrmse_median="Rhythm II vs grid II, NRMSE (median)")
    for k, lab in names.items():
        n = S.get(k.replace("_acc", "_n").replace("_median", "_n"), "")
        if k in ("rhythm_strip_acc", "all_structural_ok"):
            n = ""
        val = S.get(k)
        old = f" (before {cmp[k]})" if cmp and cmp.get(k) is not None and cmp.get(k) != val else ""
        L.append(f"| {lab} | {val}{old} | {n} |")
    L += ["", "## Errors per image", "", "| Image | Failed on | Detected | Reference |", "|---|---|---|---|"]
    for x in results:
        if not x["ok"]:
            L.append(f"| {x['id']} | exception | {x.get('error','')[:60]} | |")
            continue
        bad = [k for k, v in x["check"].items() if v is False]
        if bad:
            rf = ref[x["id"]]
            L.append(f"| {x['id']} | {', '.join(bad)} | {x['layout']} {'/'.join(x['rhythm'])} {x['speed']:g} mm/s "
                     f"{x['duration']:g} s {x['gains']} | {rf.get('layout')} {'/'.join(rf.get('rhythm') or [])} "
                     f"{rf.get('speed')} {rf.get('duration')} {rf.get('gain')} |")
    return "\n".join(L) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reference", type=Path, default=HERE / "reference.json")
    ap.add_argument("--cache", type=Path, default=HERE / "cache")
    ap.add_argument("--out", type=Path, default=Path("results/litfl"))
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2)))
    ap.add_argument("--compare", type=Path, default=None, help="results.json of a previous run")
    ap.add_argument("--no-html", action="store_true", help="do not write report.html or the images")
    ap.add_argument("--config", action="append", default=[], metavar="KEY=VALUE",
                    help="Config field (e.g. --config nonrigid=poly)")
    a = ap.parse_args()
    import ast
    cfg_kw = {}
    for kv in a.config:
        k, v = kv.split("=", 1)
        try:
            cfg_kw[k] = ast.literal_eval(v)
        except (ValueError, SyntaxError):
            cfg_kw[k] = v
    ref = json.loads(a.reference.read_text())["images"]
    a.cache.mkdir(parents=True, exist_ok=True)
    a.out.mkdir(parents=True, exist_ok=True)
    img_dir = None
    if not a.no_html:
        img_dir = a.out / "img"
        img_dir.mkdir(exist_ok=True)
    todo = []
    for key in sorted(ref):
        if ref[key].get("review") == "unavailable":
            continue
        try:
            todo.append((key, ref[key], fetch(ref[key], a.cache), cfg_kw, img_dir))
        except Exception as e:  # noqa: BLE001
            print(f"download failed {key}: {e}", flush=True)
        if a.limit and len(todo) >= a.limit:
            break
    results = []
    with ProcessPoolExecutor(max_workers=a.jobs) as ex:
        for res in ex.map(evaluate_one, todo):
            results.append(res)
            print(res["id"], "ok" if res["ok"] else "ERROR", {k: v for k, v in res.get("check", {}).items() if v is False}, flush=True)
    S = summarize(results, ref)
    cmp = json.loads(a.compare.read_text())["summary"] if a.compare and a.compare.exists() else None
    (a.out / "results.json").write_text(json.dumps(dict(summary=S, config=cfg_kw, results=results), indent=1, default=str))
    (a.out / "report.md").write_text(report_md(S, results, ref, cmp))
    if not a.no_html:
        from report_html import write_html
        write_html(a.out / "report.html", S, results, ref, cmp, cfg_kw)
    print(json.dumps(S, indent=1))


if __name__ == "__main__":
    main()
