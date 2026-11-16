"""Per-segment mode: each lead is located by its own printed label.

The main ecgdig model assumes "page rows": each row has a baseline and is split into
columns in time. Several generators and devices do not follow this: each column of leads
has its own baselines (a large-amplitude lead pushes the column down), and the rows of
different columns do not line up.

Here the page is read as a set of segments: for each readable label (I..V6), the baseline
is the trace just to its right; the segment runs until the next label in the same band (or
to the end of the trace). Time comes from the x position relative to the left margin of the
first-column labels. Each segment is tracked with the same Viterbi as the main mode, in a
band limited by the neighbouring segments that overlap in x.

The choice between this mode and the main one is physical (Einthoven/Goldberger on the
simultaneous stretches); it needs no ground truth.
"""
from __future__ import annotations

import numpy as np
import cv2

from . import ocr
from .extract import column_runs, viterbi_track, find_pulses

NAMES = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]


def page_words(B, D, R, min_score=0.5):
    """Readable lead labels on the whole page (small, isolated components)."""
    n, lab, st, _ = cv2.connectedComponentsWithStats(B.astype(np.uint8), connectivity=8)
    small = (st[:, cv2.CC_STAT_WIDTH] < 12 * R) & (st[:, cv2.CC_STAT_HEIGHT] < 7 * R)
    small[0] = False
    iso = small[lab]
    out = []
    for (x0, y0, x1, y1) in ocr.find_words(iso, R):
        crop = iso[y0:y1, x0:x1]
        d = ocr.read_label(crop)
        if not d:
            continue
        name = max(d, key=d.get)
        sc = d[name]
        if name not in NAMES or sc < min_score or (name == "I" and list(d) == ["I"] and sc < 0.7):
            continue
        out.append(dict(name=name, score=float(sc), x0=int(x0), y0=int(y0), x1=int(x1), y1=int(y1)))
    return out


def _baseline_near(B, w, R):
    """Trace baseline just right of the label: median ink height over 25 mm."""
    H, W = B.shape
    xa, xb = int(w["x1"] + 1 * R), min(W, int(w["x1"] + 25 * R))
    yc = 0.5 * (w["y0"] + w["y1"])
    ya, yb = max(0, int(yc - 10 * R)), min(H, int(yc + 10 * R))
    sub = B[ya:yb, xa:xb]
    if sub.size == 0:
        return None
    ys = []
    for j in range(sub.shape[1]):
        col = np.nonzero(sub[:, j])[0]
        if len(col):
            # ink closest to the label height (the neighbouring trace may enter the window)
            k = col[np.argmin(np.abs(col + ya - yc))]
            ys.append(k + ya)
    if len(ys) < 5 * R:
        return None
    return float(np.median(ys))


def segment_mode(B, D, R, speed=25.0, fs=500, gain=None, min_labels=8):
    """Read the page by segments. Returns dict(leads, signals, per_segment, duration) or None."""
    H, W = B.shape
    words = page_words(B, D, R)
    # one label per (name, position): close duplicates merge into one
    words.sort(key=lambda w: -w["score"])
    keep = []
    for w in words:
        if any(abs(w["x0"] - k["x0"]) < 5 * R and abs(w["y0"] - k["y0"]) < 3 * R for k in keep):
            continue
        keep.append(w)
    words = keep
    if len({w["name"] for w in words}) < min_labels:
        return None
    for w in words:
        w["base"] = _baseline_near(B, w, R)
    words = [w for w in words if w["base"] is not None]
    if len({w["name"] for w in words}) < min_labels:
        return None
    # gain from the pulses (modal height), else the default
    if gain is None:
        P = find_pulses(B, R)
        hs = [p["foot"] - p["top"] for p in P if 4 * R <= p["foot"] - p["top"] <= 25 * R]
        gain = 10.0
        if len(hs) >= 3:
            h = float(np.median(hs)) / R
            gain = min((5.0, 10.0, 20.0), key=lambda g: abs(g - h)) if min(abs(h - g) for g in (5, 10, 20)) < 1.5 else h
    # end of each segment: next label to the right in the same band (±4 mm), or end of trace
    x_org = min(w["x0"] for w in words)
    segs = []
    for w in words:
        nxt = [v["x0"] for v in words if v is not w and v["x0"] > w["x1"] + 10 * R and abs(v["base"] - w["base"]) < 4 * R]
        xe = min(nxt) - 1 * R if nxt else None
        if xe is None:
            # end of trace: last column with ink near the baseline
            a, b = max(0, int(w["base"] - 6 * R)), min(H, int(w["base"] + 6 * R))
            has = B[a:b, int(w["x1"]):].any(0)
            idx = np.nonzero(has)[0]
            if not len(idx):
                continue
            # cut at gaps > 8 mm
            gaps = np.nonzero(np.diff(idx) > 8 * R)[0]
            last = idx[gaps[0]] if len(gaps) else idx[-1]
            xe = int(w["x1"]) + int(last)
        xs = int(w["x0"])
        if xe - xs < 15 * R:
            continue
        segs.append(dict(name=w["name"], base=w["base"], xa=xs, xb=int(xe), score=w["score"]))
    if len({s["name"] for s in segs}) < min_labels:
        return None
    dur = max(s["xb"] for s in segs) - x_org
    duration = dur / (speed * R)
    N = int(round(duration * fs))
    if N < fs:
        return None
    sig = {n_: np.full(N, np.nan) for n_ in NAMES}
    per_seg = {n_: [] for n_ in NAMES}
    for s in segs:
        # band: half the distance to the neighbouring segments that overlap in x
        nb = [t["base"] for t in segs if t is not s and min(t["xb"], s["xb"]) - max(t["xa"], s["xa"]) > 5 * R]
        up = min([s["base"] - v for v in nb if v < s["base"]] + [25 * R])
        dn = min([v - s["base"] for v in nb if v > s["base"]] + [25 * R])
        ya, yb = max(0, int(s["base"] - 0.9 * up)), min(H, int(s["base"] + 0.9 * dn))
        xa, xb = max(0, s["xa"]), min(W, s["xb"])
        runs = column_runs(B, D, ya, yb, xa, xb)
        sel = viterbi_track(runs, float(s["base"]), R, others=[float(v) for v in nb])
        cen = np.array([runs[i][k][2] if k is not None else np.nan for i, k in enumerate(sel)], float)
        ok = np.isfinite(cen)
        if ok.sum() < 10 * R:
            continue
        xs_ = xa + np.arange(len(cen))
        base = float(np.nanmedian(cen))
        v = (base - cen) / (gain * R)
        t = (xs_ - x_org) / (speed * R)
        n = np.clip(np.round(t * fs).astype(int), 0, N - 1)
        tt = np.arange(n[ok].min(), n[ok].max() + 1)
        vv = np.interp(tt / fs, t[ok], v[ok])
        m = np.full(N, np.nan); m[tt] = vv
        per_seg[s["name"]].append(m)
        cur = sig[s["name"]]
        sig[s["name"]] = np.where(np.isnan(cur), m, np.where(np.isnan(m), cur, 0.5 * (cur + m)))
    leads = [n_ for n_ in NAMES if np.isfinite(sig[n_]).any()]
    if len(leads) < min_labels:
        return None
    return dict(leads=leads, signals=np.stack([sig[n_] for n_ in leads]).astype(np.float32),
                per_segment={k: v for k, v in per_seg.items() if v}, duration=duration, gain=gain,
                n_words=len(words))


def physics_residual(leads, signals):
    """Normalized residual of Einthoven (I+III=II) and Goldberger (aVR+aVL+aVF=0) on the
    simultaneous stretches. Lower = more consistent reading. None if nothing can be tested."""
    d = {n: s for n, s in zip(leads, signals)}
    res = []
    for combo, coef in ((("I", "III", "II"), (1, 1, -1)), (("aVR", "aVL", "aVF"), (1, 1, 1))):
        if all(c in d for c in combo):
            V = np.stack([d[c] for c in combo]).astype(float)
            m = np.isfinite(V).all(0)
            if m.sum() < 200:
                continue
            V = V[:, m] - np.median(V[:, m], 1, keepdims=True)
            r = np.tensordot(np.array(coef, float), V, 1)
            res.append(float(np.sqrt(np.mean(r ** 2)) / (np.sqrt(np.mean(V ** 2)) + 1e-9)))
    return float(np.mean(res)) if res else None
