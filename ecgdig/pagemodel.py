"""Page-structure network (hybrid/pagegen.py, hybrid/train_page.py): rows, lead changes and lead
names of a rectified page.

The network sees ecgdig's rectified page resized to 4 px/mm and returns three maps: where the
baselines of the printed leads run, where one lead ends and the next begins in a row, and which
lead each trace band shows. `decode` turns them into rows (baseline y and x extent), the lead
changes of every row and a name distribution for every segment. The classical pipeline then works
from those rows (extents, pulses, layout hypotheses, tracking), so the network only replaces the
step where the classical reader fails most: counting and placing the rows.
"""
from __future__ import annotations

import numpy as np

S_MM = 4.0
FG_THR = [0.5]     # lead-class probability (1 - background) that makes a ridge a lead
LEADS = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
_SESS = {}


def _session(path):
    if path not in _SESS:
        import onnxruntime as ort
        so = ort.SessionOptions()
        so.intra_op_num_threads = 2
        _SESS[path] = ort.InferenceSession(path, so, providers=["CPUExecutionProvider"])
    return _SESS[path]


def predict(rect_bgr: np.ndarray, R: float, path: str):
    """Maps at 4 px/mm: baseline (H, W), lead change (H, W), class probabilities (13, H, W)."""
    import cv2
    f = S_MM / R
    im = cv2.resize(rect_bgr, None, fx=f, fy=f, interpolation=cv2.INTER_AREA)
    h, w = im.shape[:2]
    H, W = (h + 31) // 32 * 32, (w + 31) // 32 * 32
    x = np.zeros((H, W, 3), np.float32)
    x[:h, :w] = im[..., ::-1].astype(np.float32) / 255.0
    x[h:, :] = x[:h].reshape(-1, 3).mean(0) if h < H else 0
    x[:, w:] = x[:, :w].reshape(-1, 3).mean(0) if w < W else 0
    out = _session(path).run(None, {"x": x.transpose(2, 0, 1)[None]})[0][0]
    out = out[:, :h, :w]
    sig = lambda z: 1.0 / (1.0 + np.exp(-z))
    lg = out[2:]
    e = np.exp(lg - lg.max(0, keepdims=True))
    return sig(out[0]), sig(out[1]), e / e.sum(0, keepdims=True)


def _peaks(prof, min_dist, height):
    from scipy.signal import find_peaks
    pk, _ = find_peaks(prof, distance=max(1, int(min_dist)), height=height)
    return pk


def decode(base, bnd, cls, min_len_mm=15.0, thr=0.4):
    """Rows from the maps (coordinates at 4 px/mm).

    Ridges are the connected parts of baseline > thr; pieces of one row (broken by a faint stretch
    or a lead change) are chained when they follow each other within 1.5 mm of height and 40 mm
    of gap, so a slightly tilted page still gives one row per printed row. Returns a list of dicts
    sorted by y: y (median baseline), x0, x1, changes (x of the lead changes), names (13-vector of
    mean class probabilities per segment), strength (mean ridge value)."""
    import cv2
    H, W = base.shape
    m = (base > thr).astype(np.uint8)
    n, lab, st, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    pieces = []
    for i in range(1, n):
        x, y, w, h, area = st[i]
        if w < 2 * S_MM:
            continue
        sub = np.where(lab[y:y + h, x:x + w] == i, base[y:y + h, x:x + w], 0.0)
        colmax = sub.max(0)
        ok = colmax > thr
        xs = np.flatnonzero(ok) + x
        ys = sub.argmax(0)[ok] + y
        if len(xs) < 2 * S_MM:
            continue
        pieces.append(dict(xs=xs, ys=ys.astype(np.float64), v=colmax[ok]))
    # chain pieces left to right
    pieces.sort(key=lambda p: p["xs"][0])
    chains = []
    for p in pieces:
        best, bd = None, None
        for c in chains:
            gap = p["xs"][0] - c["xs"][-1]
            if gap < -2 * S_MM or gap > 40 * S_MM:
                continue
            ye = float(np.median(c["ys"][-int(5 * S_MM):]))
            yb = float(np.median(p["ys"][:int(5 * S_MM)]))
            d = abs(ye - yb)
            if d <= 1.5 * S_MM and (bd is None or d < bd):
                best, bd = c, d
        if best is None:
            chains.append(dict(xs=p["xs"].copy(), ys=p["ys"].copy(), v=p["v"].copy()))
        else:
            keep = p["xs"] > best["xs"][-1]
            best["xs"] = np.r_[best["xs"], p["xs"][keep]]; best["ys"] = np.r_[best["ys"], p["ys"][keep]]
            best["v"] = np.r_[best["v"], p["v"][keep]]
    rows = []
    fgm = 1.0 - cls[0]
    for c in chains:
        if len(c["xs"]) < min_len_mm * S_MM:
            continue
        # a ridge is a lead only where the network also sees a trace band (lead class, not
        # background): grid frame lines, empty grid after the trace, text lines and the edge of
        # the paper can look like a baseline but carry no lead
        yi = np.clip(np.round(c["ys"]).astype(int), 0, H - 1)
        fg = np.zeros(len(yi))
        for dy in range(-int(1.5 * S_MM), int(1.5 * S_MM) + 1):
            fg = np.maximum(fg, fgm[np.clip(yi + dy, 0, H - 1), c["xs"]])
        on = np.flatnonzero(fg > FG_THR[0])
        if len(on) < min_len_mm * S_MM:
            continue
        gaps = np.flatnonzero(np.diff(c["xs"][on]) > 8 * S_MM)
        st_ = np.r_[0, gaps + 1]; en_ = np.r_[gaps, len(on) - 1]
        j = int(np.argmax(c["xs"][on][en_] - c["xs"][on][st_]))
        a_, b_ = on[st_[j]], on[en_[j]]
        if c["xs"][b_] - c["xs"][a_] < min_len_mm * S_MM:
            continue
        xs_, ys_ = c["xs"][a_:b_ + 1], c["ys"][a_:b_ + 1]
        rows.append(dict(y=float(np.median(ys_)), x0=int(xs_[0]), x1=int(xs_[-1]),
                         strength=float(c["v"][a_:b_ + 1].mean()), trace=float(fg[a_:b_ + 1].mean()),
                         xs=xs_, ys=ys_))
    rows.sort(key=lambda r: r["y"])
    for r in rows:
        prof_b = np.zeros(r["x1"] - r["x0"] + 1)
        yfit = np.interp(np.arange(r["x0"], r["x1"] + 1), r["xs"], r["ys"])
        for j, xx in enumerate(range(r["x0"], r["x1"] + 1)):
            yc = int(round(yfit[j]))
            a, b = max(0, yc - int(3 * S_MM)), min(H, yc + int(3 * S_MM) + 1)
            prof_b[j] = bnd[a:b, xx].max() if b > a else 0.0
        ch = _peaks(prof_b, 5 * S_MM, 0.4) + r["x0"]
        r["changes"] = [int(c) for c in ch if r["x0"] + 3 * S_MM < c < r["x1"] - 3 * S_MM]
        edges = [r["x0"]] + r["changes"] + [r["x1"]]
        names = []
        for x_a, x_b in zip(edges[:-1], edges[1:]):
            xx = np.arange(x_a, x_b + 1)
            yy = np.clip(np.round(yfit[xx - r["x0"]]).astype(int), 0, H - 1)
            acc = np.zeros(cls.shape[0])
            for dy in range(-int(2 * S_MM), int(2 * S_MM) + 1):
                acc += cls[:, np.clip(yy + dy, 0, H - 1), xx].sum(1)
            names.append(acc / max(acc.sum(), 1e-9))
        r["names"] = names
        del r["xs"], r["ys"]
    return rows


MIN_REL_LEN = [0.5]   # a row spans at least this share of the longest row


def page_structure(rect_bgr, R, path, B=None):
    """Rows in rectified-image pixels (R px/mm) and the column count most rows agree on.

    The network also marks things it was never shown as negatives (grid frame lines, the empty
    grid after the trace, text lines, QR codes, the paper edge). Two checks drop them: a row spans
    at least half the longest row (every lead row and rhythm strip of a page spans the plot), and,
    when the classical binary map B is given and has ink on some row, the row has ink within
    1 mm of its baseline in B."""
    base, bnd, cls = predict(rect_bgr, R, path)
    rows = decode(base, bnd, cls)
    if rows:
        L = max(r["x1"] - r["x0"] for r in rows)
        rows = [r for r in rows if r["x1"] - r["x0"] >= MIN_REL_LEN[0] * L]
    k = R / S_MM
    if B is not None and rows:
        h = int(round(1.0 * R))
        ink = []
        for r in rows:
            y = int(round(r["y"] * k)); x0, x1 = int(r["x0"] * k), int(r["x1"] * k)
            band = B[max(0, y - h):y + h + 1, x0:x1 + 1]
            ink.append(float(band.any(0).mean()) if band.size else 0.0)
        if max(ink) > 0.05:
            rows = [r for r, v in zip(rows, ink) if v > 0.005]
    out = []
    for r in rows:
        out.append(dict(y=r["y"] * k, x0=r["x0"] * k, x1=r["x1"] * k, strength=r["strength"],
                        changes=[c * k for c in r["changes"]], names=r["names"]))
    ncol = [len(r["changes"]) + 1 for r in out]
    multi = [c for c in ncol if c > 1]
    cols = int(np.bincount(multi).argmax()) if len(multi) >= 2 else None
    return out, cols


def orientation_score(rect_bgr, R, path):
    """How much the page looks like an upright ECG to the network: the length (mm) of the rows
    it finds, weighted by how sure it is of their lead names. A page on its side gives no long
    rows; upside down, the names (trained on upright pages) lose confidence."""
    base, bnd, cls = predict(rect_bgr, R, path)
    rows = decode(base, bnd, cls)
    if not rows:
        return 0.0
    L = max(r["x1"] - r["x0"] for r in rows)
    sc = 0.0
    for r in rows:
        if r["x1"] - r["x0"] < MIN_REL_LEN[0] * L:
            continue
        conf = float(np.mean([np.max(n[1:]) / max(n[1:].sum(), 1e-9) for n in r["names"]]))
        sc += (r["x1"] - r["x0"]) / S_MM * conf
    return sc
