"""
Layout engine by hypothesis and verification.

Input: the structure already detected (trace rows, pulses with 1 or 2 levels, time axis per
row, quick trace of each full row). Output: the most likely layout (printed names of each
segment of each row, gain of each segment) with a decomposed score and a confidence.

Hypotheses: library families (standard and Cabrera; 3x4, 6x2, 12x1) at the top of the page,
with the remaining rows as rhythm strips of any lead, and a "free" hypothesis (number of
segments per row from the boundary evidence, names from OCR).

Score (informal log-likelihood, independent terms):
  OCR       label read near the start of each segment
  boundary  marks/labels at the lead-switch positions (and their absence elsewhere)
  limb      instants with >= 3 simultaneous limb leads must fit the 2D Einthoven/Goldberger
            space (least-squares residual)
  duplicate the same lead in two strips at the same instant must coincide
"""
from __future__ import annotations

import warnings

import numpy as np
import cv2
from scipy.ndimage import median_filter

from .layouts import LIBRARY, LEAD_NAMES, LIMB_COEF, is_limb
from . import ocr

FAMILIES = ["3x4", "3x4-cabrera", "6x2", "6x2-cabrera", "12x1", "12x1-cabrera",
            "6x1-prec", "6x1-limb", "6x1-limb-cabrera"]
STRIPS3 = [True]
ROW_DROP = [True]    # hypotheses that leave out edge rows (frame, ruler)
DROP_COST = [4.0]
DROP_LIMB_GAIN = [0.25]
DROP_NEAR = [0.88]
if STRIPS3[0]:
    FAMILIES += ["3x1-limb", "3x1-aug", "3x1-v13", "3x1-v46"]
OCR_QRS_BARS = [True]
BND_HALF_MM = [1.0]
FRACS = {1: [], 2: [0.5], 3: [1 / 3, 2 / 3], 4: [0.25, 0.5, 0.75]}
W_OCR, W_BND, W_LIMB, W_DUP = 4.0, 1.5, 15.0, 3.0
_UNEXPLAINED = [True]
CABRERA_NO_EVIDENCE = [10.0]
LABEL_HI = [15.0]
BND_RELATIVE = [True]
BND_OTHER_ROWS = [True]
# the row's own QRS is not a boundary mark either: in simultaneous formats the QRS falls at
# the same position in every row and mimics a boundary. Off: it also masks separator marks
# that the tracker followed as steep strokes (6x2 page read as 3x4+3)
BND_OWN_QRS = [False]
BND_COHERENT = [True]
LIMB_LOG = [True]


def _canon(name):
    return "aVR" if name == "-aVR" else name


def _sign(name):
    return -1.0 if name == "-aVR" else 1.0



def _limb_ratio(V, A):
    """Relative Einthoven/Goldberger residual: V (n, T) measurements, A (n, 2) coefficients
    on (I, II). Least squares per instant, vectorized by pattern of valid samples."""
    ok = ~np.isnan(V)
    num = den = 0.0
    nt = 0
    keys = np.packbits(ok, axis=0)
    cols = {}
    for t in range(V.shape[1]):
        cols.setdefault(keys[:, t].tobytes(), []).append(t)
    for _, ts in cols.items():
        m = ok[:, ts[0]]
        if m.sum() < 3:
            continue
        Am = A[m]
        P = Am @ np.linalg.pinv(Am)
        X = V[m][:, ts]
        Rz = X - P @ X
        num += float((Rz ** 2).sum()); den += float((X ** 2).sum())
        nt += len(ts)
    if den <= 0:
        return np.nan, nt
    return float(np.sqrt(num / den)), nt

class Ctx:
    """Everything the engine needs, already measured by the pipeline."""

    def __init__(self, B, D, rows, pulses, R, x0_r, sc_r, speed, duration, qtr, base_r, levels):
        self.B, self.D, self.rows, self.pulses, self.R = B, D, list(rows), pulses, R
        self.x0_r, self.sc_r, self.speed, self.duration = x0_r, sc_r, speed, duration
        self.qtr, self.base_r, self.levels = qtr, base_r, levels  # levels[r] = (g1, g2|None)
        self.n = len(rows)
        self.t = np.arange(0.0, duration, 0.01)  # 100 Hz grid for scoring
        self.vals = [self._row_px(r) for r in range(self.n)]
        self.words = self._read_words()
        self.bnd = {r: {q: self._boundary_evidence(r, q) for q in sorted({f for v in FRACS.values() for f in v})}
                    for r in range(self.n)}
        if BND_RELATIVE[0]:
            # evidence relative to the row's own background: grid leakage, time marks or noise
            # give a "boundary" at any x; what counts is the position standing out
            allf = [f for v in FRACS.values() for f in v]
            bgq = [q for q in np.arange(0.08, 0.93, 0.05) if min(abs(q - f) for f in allf) > 0.03]
            for r in range(self.n):
                bg = float(np.median([self._boundary_evidence(r, q) for q in bgq]))
                if bg > 0.05:
                    self.bnd[r] = {q: float(np.clip((e - bg) / (1 - bg + 1e-6), 0, 1))
                                   for q, e in self.bnd[r].items()}
        if BND_COHERENT[0] and self.n >= 3:
            # a column boundary is the same in every row of the block; a QRS falling at the
            # position by chance shows up in one row only. Evidence seen in fewer than half
            # of the rows counts proportionally less
            qs = list(self.bnd[0].keys())
            for q in qs:
                c = np.mean([self.bnd[r][q] > 0.5 for r in range(self.n)])
                f = min(1.0, c / 0.5)
                for r in range(self.n):
                    self.bnd[r][q] *= f
        self._hp = {}

    # ------------------------------------------------------------------ signals
    def x_of(self, r, t):
        return self.x0_r[r] + self.sc_r[r] * np.asarray(t) * self.speed * self.R

    def _row_px(self, r):
        q = self.qtr[r]
        xs = q["x0"] + np.arange(len(q["cen"]))
        ok = ~np.isnan(q["cen"])
        if ok.sum() < 10:
            return np.full(len(self.t), np.nan)
        v = np.interp(self.x_of(r, self.t), xs[ok], q["cen"][ok], left=np.nan, right=np.nan)
        return self.base_r[r] - v  # px above the baseline

    def seg_gain(self, r, name):
        g1, g2 = self.levels[r]
        if g2 is not None and not is_limb(name):
            return g2
        return g1

    def seg_signal(self, r, k, c, name):
        """(time mask, signal in mV under the CANONICAL name, high-pass) of segment c of k."""
        n = len(self.t)
        i0, i1 = c * n // k, (c + 1) * n // k
        key = (r, k, c, name)
        if key not in self._hp:
            v = np.full(n, np.nan)
            seg = self.vals[r][i0:i1] / (self.seg_gain(r, name) * self.R)
            # 40 ms margin away from the boundaries (marks, lead switch)
            m = 4
            seg2 = seg.copy()
            if c > 0:
                seg2[:m] = np.nan
            if c < k - 1:
                seg2[-m:] = np.nan
            v[i0:i1] = seg2 * _sign(name)
            ok = ~np.isnan(v)
            if ok.sum() > 20:
                vi = np.interp(np.arange(n), np.nonzero(ok)[0], v[ok])
                hp = vi - median_filter(vi, size=61, mode="nearest")  # removes drift (~0.6 s)
                hp[~ok] = np.nan
            else:
                hp = v
            self._hp[key] = hp
        return self._hp[key]

    # --------------------------------------------------------------------- OCR
    def _trace_mask(self):
        H, W = self.B.shape
        m = np.zeros((H, W), np.uint8)
        for q in self.qtr:
            for i, (t, b) in enumerate(zip(q["top"], q["bot"])):
                if not np.isnan(t):
                    x = q["x0"] + i
                    if 0 <= x < W:
                        m[max(0, int(t) - 2):min(H, int(b) + 3), x] = 1
        for p in self.pulses:  # pulses are not text
            if p is not None:
                top = min(p["top"], p.get("top2", p["top"]))
                m[max(0, int(top) - 3):int(p["foot"]) + 4, max(0, p["x_start"] - 3):p["x_end"] + 4] = 1
        return cv2.dilate(m, np.ones((3, 3), np.uint8)) > 0

    def _read_words(self):
        R = self.R
        # text = small isolated components (not touching the trace) + ink off the trace path
        # (label touching a QRS).
        Bt = self.B
        from .extract import GRID_SEGS, clean_grid_segments
        if GRID_SEGS[0]:
            # pieces of monochrome grid stuck to the letters break up the isolated component
            Bt = clean_grid_segments(Bt, R)
        n_, lab, st, _ = cv2.connectedComponentsWithStats(Bt.astype(np.uint8), connectivity=8)
        small = (st[:, cv2.CC_STAT_WIDTH] < 25 * R) & (st[:, cv2.CC_STAT_HEIGHT] < 9 * R)
        small[0] = False
        tm = self._trace_mask()
        pulse_m = np.zeros_like(tm)
        for p in self.pulses:
            if p is not None:
                top = min(p["top"], p.get("top2", p["top"]))
                pulse_m[max(0, int(top) - 3):int(p["foot"]) + 4, max(0, p["x_start"] - 3):p["x_end"] + 4] = True
        iso = small[lab] & ~pulse_m
        txt = (iso | (Bt & ~tm)) & ~pulse_m
        if OCR_QRS_BARS[0]:
            # a long vertical line (dashed column separator, border) touching the label pushes
            # the word past the maximum height and it vanishes: vertical strokes > 6 mm are not letters
            lv = cv2.morphologyEx(txt.astype(np.uint8), cv2.MORPH_OPEN,
                                  np.ones((max(3, int(6 * R)), 1), np.uint8)) > 0
            if lv.any():
                txt = txt & ~cv2.dilate(lv.astype(np.uint8), np.ones((1, 3), np.uint8)).astype(bool)
        words = []
        for w in ocr.find_words(txt, R):
            x0, y0, x1, y1 = w
            crop = txt[y0:y1, x0:x1]
            d = ocr.read_label(crop)
            if not d:
                continue
            best = max(d.values())
            if best < 0.35:
                continue
            isolated = iso[y0:y1, x0:x1].sum() >= 0.8 * crop.sum()
            single = list(d) == ["I"]
            if single and not isolated:  # vertical bar touching the trace = QRS fragment
                continue
            cy = 0.5 * (y0 + y1)
            r = int(np.argmin([abs(cy - y) for y in self.rows]))
            dk = float(np.percentile(self.D[y0:y1, x0:x1][crop], 90))
            words.append(dict(box=w, row=r, x=x0, dist=d, best=best, h=y1 - y0, single=single, dark=dk))
        # height consensus: all labels of a page use the same font
        conf = [w for w in words if not w["single"] and w["best"] >= 0.55]
        conf_ok = len(conf) >= 3
        # loose bars ("I") much taller or shorter than the page labels are pulses or QRS
        multi = [w["h"] for w in words if not w["single"] and w["best"] >= 0.45]
        if len(multi) >= 2:
            hm = float(np.median(multi))
            words = [w for w in words if not w["single"] or 0.7 * hm <= w["h"] <= 1.35 * hm]
        hl = None
        if conf_ok:
            hl = float(np.median([w["h"] for w in conf]))
            dl = float(np.median([w["dark"] for w in conf]))
            # same font (height) and same ink (intensity): grid leakage and trace fragments
            # are lighter than the page's own printed text
            words = [w for w in words if 0.7 * hl <= w["h"] <= 1.35 * hl and w["dark"] >= 0.8 * dl]
        # a word belongs to the row whose band contains it (label above OR below the trace)
        # 2nd local pass: row without a label at its start -> 3x3 opening in the expected window
        # (removes thin grid slivers stuck to the letters)
        H, W = txt.shape
        for r in range(self.n):
            xs = float(self.x_of(r, 0.0))
            if any(w["row"] == r and xs - 25 * R <= w["x"] <= xs + 15 * R for w in words):
                continue
            y = self.rows[r]
            up = (y - self.rows[r - 1]) if r > 0 else 30 * R
            dn = (self.rows[r + 1] - y) if r + 1 < self.n else 30 * R
            ya, yb = max(0, int(y - 0.5 * up)), min(H, int(y + 0.5 * dn))
            xa, xb = max(0, int(xs - 25 * R)), min(W, int(xs + 15 * R))
            win = cv2.morphologyEx((txt[ya:yb, xa:xb]).astype(np.uint8), cv2.MORPH_OPEN,
                                   np.ones((3, 3), np.uint8))
            # grid remnants become dots: remove components much smaller than a letter
            hl_ = hl if conf_ok else 2.5 * R
            n2, l2, st2, _ = cv2.connectedComponentsWithStats(win, connectivity=8)
            keep = st2[:, cv2.CC_STAT_AREA] >= max(25, 0.06 * hl_ * hl_)
            keep[0] = False
            win = keep[l2]
            for w in ocr.find_words(win, R):
                x0, y0, x1, y1 = w
                d = ocr.read_label(win[y0:y1, x0:x1])
                if not d or max(d.values()) < 0.4 or list(d) == ["I"]:
                    continue
                box = (x0 + xa, y0 + ya, x1 + xa, y1 + ya)
                dk = float(np.percentile(self.D[box[1]:box[3], box[0]:box[2]][win[y0:y1, x0:x1]], 90))
                words.append(dict(box=box, row=r, x=box[0], dist=d, best=max(d.values()), h=y1 - y0,
                                  single=False, dark=dk))
        if OCR_QRS_BARS[0]:
            # loose bars read as "I" repeated along a row are thin QRS (isolated peak of a
            # broken trace), not labels: a label appears once per segment. With >= 3 in a
            # row, only those at the start of the row stay.
            for r in range(self.n):
                sg = [w for w in words if w["row"] == r and w.get("single")]
                if len(sg) >= 3:
                    xs = float(self.x_of(r, 0.0))
                    drop = {id(w) for w in sg if w["box"][0] > xs + 3 * R}
                    words = [w for w in words if id(w) not in drop]
        return words

    def label_near(self, r, x_mm_lo, x_mm_hi, x_ref_mm=None):
        """Word of row r whose left edge falls in [lo, hi] mm; among several, the one with the
        highest score weighted by closeness to x_ref (segment start)."""
        R = self.R
        out, best = None, -1.0
        for w in self.words:
            x0 = w["box"][0]
            if w["row"] != r or not (x_mm_lo * R <= x0 <= x_mm_hi * R):
                continue
            sc = w["best"]
            if w.get("single"):
                # loose bar read as "I": QRS fragment, pulse edge or a leftover bar of "II";
                # it loses to a multi-character reading next to it
                sc *= 0.7
            if x_ref_mm is not None:
                sc *= float(np.exp(-abs(x0 / R - x_ref_mm) / 8.0))
            if sc > best:
                out, best = w, sc
        return out

    # -------------------------------------------------------------- boundaries
    def _boundary_evidence(self, r, q):
        """[0,1]: vertical mark that is not trace, or a label just after the position."""
        R = self.R
        x = float(self.x_of(r, q * self.duration))
        H, W = self.B.shape
        y = self.base_r[r]
        e = 0.0
        w = self.label_near(r, (x - 1.5 * R) / R, (x + 8 * R) / R)
        if w is not None:
            e = max(e, min(1.0, w["best"] / 0.6))
        # vertical mark: ink off the trace in a column near x crossing the baseline
        tm = getattr(self, "_tm", None)
        if tm is None:
            tm = self._tm = self._trace_mask_thin()
        if BND_OTHER_ROWS[0]:
            # a tall QRS of a neighbouring row crosses this row: it is trace, not a boundary mark
            st = getattr(self, "_steep", None)
            if st is None:
                st = self._steep = [self._trace_mask_steep(k) for k in range(self.n)]
            tm = tm.copy()
            for k in range(self.n):
                if k != r or BND_OWN_QRS[0]:
                    tm |= st[k]
        a, b = max(0, int(y - 4 * R)), min(H, int(y + 4 * R))
        hw_ = BND_HALF_MM[0]
        xa, xb = max(0, int(x - hw_ * R)), min(W, int(x + hw_ * R) + 1)
        if xb > xa and b > a:
            sub = self.B[a:b, xa:xb] & ~tm[a:b, xa:xb]
            col = sub.sum(0).max() if sub.size else 0
            e = max(e, min(1.0, col / (4.0 * R)))
        return e

    def _trace_mask_steep(self, k):
        H, W = self.B.shape
        m = np.zeros((H, W), bool)
        q = self.qtr[k]
        for i, (t, b) in enumerate(zip(q["top"], q["bot"])):
            if not np.isnan(t) and b - t >= 3 * self.R:
                x = q["x0"] + i
                if 0 <= x < W:
                    m[max(0, int(t) - 2):min(H, int(b) + 3), max(0, x - 2):x + 3] = True
        return m

    def _trace_mask_thin(self):
        H, W = self.B.shape
        m = np.zeros((H, W), bool)
        for q in self.qtr:
            for i, (t, b) in enumerate(zip(q["top"], q["bot"])):
                if not np.isnan(t) and b - t < 3 * self.R:  # non-vertical stretches only
                    x = q["x0"] + i
                    if 0 <= x < W:
                        m[max(0, int(t) - 2):min(H, int(b) + 3), x] = True
        return m

    # --------------------------------------------------------------- scoring
    def score(self, spec):
        """spec = list of rows (each a list of printed names). Returns (total, parts)."""
        R = self.R
        parts = dict(ocr=0.0, bnd=0.0, limb=0.0, dup=0.0)
        segs = []
        for r, row in enumerate(spec):
            k = len(row)
            if k == 0:
                continue      # row outside the hypothesis (frame, ruler): not scored
            for c, name in enumerate(row):
                segs.append((r, k, c, name))
            # boundaries
            want = set(np.round(FRACS.get(k, []), 4))
            for q, e in self.bnd[r].items():
                if round(q, 4) in want:
                    parts["bnd"] += e - 0.3
                elif e > 0.3:
                    parts["bnd"] -= e - 0.3
        # OCR
        for (r, k, c, name) in segs:
            xs = float(self.x_of(r, c / k * self.duration)) / R
            lo = xs - (25 if c == 0 else 2.0)
            w = self.label_near(r, lo, xs + LABEL_HI[0], x_ref_mm=xs)
            if w is None:
                continue
            d = w["dist"]
            # cap per segment: a wrong reading (lost minus sign, swapped digit) costs at most
            # 1 unit; the physics (Goldberger, duplicates) decides the rest
            top = max(d, key=d.get)
            # I/II/III differ only by the number of bars: a bar touching the trace or the
            # pulse vanishes and "II" becomes "I". Confusion among them costs half
            cap = 0.5 if (top in ("I", "II", "III") and _canon(name) in ("I", "II", "III")) else 1.0
            if w.get("single") and top == "I" and _canon(name) in ("II", "III"):
                cap = 0.0  # a single bar does not contradict II/III: the others may have merged into the trace
            parts["ocr"] += max(-cap, (d.get(name, 0.0) - d[top]) / 0.15)
        # reliable label with no segment to explain it: the hypothesis ignores a printed lead
        # (e.g. 6x1 on a 6x2 page with "V1".."V6" in the middle of the row)
        if _UNEXPLAINED[0]:
            starts = {}
            for (r, k, c, name) in segs:
                xs = float(self.x_of(r, c / k * self.duration)) / R
                starts.setdefault(r, []).append((xs - (25 if c == 0 else 2.0), xs + LABEL_HI[0]))
            for w in self.words:
                if w.get("single") or w["best"] < 0.55:
                    continue
                x0 = w["box"][0] / R
                if not any(lo <= x0 <= hi for lo, hi in starts.get(w["row"], [])):
                    parts["ocr"] -= 1.0
        # limb leads
        sig = {}
        has_grid = any(len(row) > 1 for row in spec)
        rhythm = []
        for (r, k, c, name) in segs:
            v = self.seg_signal(r, k, c, name)
            if has_grid and k == 1:
                rhythm.append((v, name))
            else:
                sig.setdefault(_canon(name), []).append((v, name))
        # a rhythm strip only enters the physics if it is simultaneous with the grid copy:
        # some pages (montages, prints of another stretch) carry a rhythm from another
        # instant, and then neither Goldberger nor the duplicate say anything about the name
        for v, name in rhythm:
            cps = sig.get(_canon(name), [])
            if cps:
                with np.errstate(all="ignore"), warnings.catch_warnings():
                    warnings.simplefilter("ignore", RuntimeWarning)
                    a = np.nanmean(np.stack([x for x, _ in cps]), 0)
                m = ~np.isnan(a) & ~np.isnan(v)
                if m.sum() >= 50 and a[m].std() > 0 and v[m].std() > 0:
                    nr = np.linalg.norm(a[m] - v[m]) / max(np.linalg.norm(a[m]), np.linalg.norm(v[m]), 1e-9)
                    # nearly flat lead (< ~80 µV RMS): the NRMSE is only quantization noise
                    flat = min(np.std(a[m]), np.std(v[m])) < 0.08
                    if nr > 0.85 and not flat:
                        parts["dup"] -= 0.5
                        continue
            sig.setdefault(_canon(name), []).append((v, name))
        limb_names = [n for n in sig if is_limb(n)]
        if len(limb_names) >= 3:
            T = len(self.t)
            Vs, As = [], []
            for nme in limb_names:
                arr = np.nanmean(np.stack([s for s, _ in sig[nme]]), 0) if len(sig[nme]) > 1 else sig[nme][0][0]
                Vs.append(arr); As.append(LIMB_COEF[nme])
            V = np.stack(Vs); A = np.array(As)
            T = len(self.t)
            ratio, ntest = _limb_ratio(V[:, ::2], A)
            if STRIPS3[0] and not has_grid and len(spec) <= 3 and np.isfinite(ratio):
                # full strips: the physics must close in each quarter of the page. On a 3x4 read
                # as 3 strips the first quarter really is I/II/III and closes; the others (aVR, V1, V4...) do not
                Vq = V[:, ::2]; qn = Vq.shape[1] // 4
                rq = [_limb_ratio(Vq[:, i * qn:(i + 1) * qn], A) for i in range(4)]
                rq = [x for x, nt in rq if nt > 20 and np.isfinite(x)]
                if len(rq) >= 3:
                    ratio = max(ratio, max(rq))
            if ntest > 20 and np.isfinite(ratio):
                ref = 0.45
                if LIMB_LOG[0]:
                    # likelihood: a small Einthoven/Goldberger residual is almost impossible by
                    # chance with the wrong names; the evidence grows like a log.
                    # Floor of 0.2: below it the hypotheses already satisfy the physics and the
                    # difference is noise (Cabrera and standard with the same leads tie)
                    lr = float(np.clip(np.log(ref / max(ratio, 0.2)), -1.0, 1.5))
                    parts["limb"] = lr * min(1.0, ntest / (0.25 * T))
                else:
                    parts["limb"] = (ref - ratio) * min(1.0, ntest / (0.25 * T))
        # duplicates
        for nme, lst in sig.items():
            for i in range(len(lst)):
                for j in range(i + 1, len(lst)):
                    a, b = lst[i][0], lst[j][0]
                    m = ~np.isnan(a) & ~np.isnan(b)
                    if m.sum() < 50 or a[m].std() == 0 or b[m].std() == 0:
                        continue
                    if max(a[m].std(), b[m].std()) < 0.08:
                        continue  # two nearly flat copies: the NRMSE is only noise, not evidence
                    # normalized RMS error: sensitive to shape AND amplitude (neighbouring
                    # leads are highly correlated but have different amplitudes)
                    na, nb = np.linalg.norm(a[m]), np.linalg.norm(b[m])
                    nrmse = np.linalg.norm(a[m] - b[m]) / max(na, nb, 1e-9)
                    parts["dup"] += 0.5 - nrmse
        # rhythm strips of one exam do not repeat the same lead (no vendor format prints,
        # e.g., II twice as rhythm)
        rhythm = [_canon(row[0]) for row in spec if len(row) == 1]
        if any(len(row) > 1 for row in spec) and len(rhythm) > len(set(rhythm)):
            parts["rep"] = -(len(rhythm) - len(set(rhythm)))
        total = W_OCR * parts["ocr"] + W_BND * parts["bnd"] + W_LIMB * parts["limb"] + W_DUP * parts["dup"] \
            + 3.0 * parts.get("rep", 0.0)
        return total, parts

    # ------------------------------------------------------------- hypotheses
    def _best_rhythm_names(self, grid_spec, rest_rows):
        """Name each rhythm strip (full row, 1 lead) by the best joint score."""
        spec = [list(r) for r in grid_spec]
        rest_rows = list(rest_rows)
        # vendor convention: 1 rhythm strip = II (GE, Philips, Mortara, Schiller);
        # 2 = V1 and II; 3 = V1, II, V5. Moderate prior: a read label or the physics can win.
        canon = {1: [{"II"}], 2: [{"V1", "II"}, {"V1", "II"}], 3: [{"V1", "II", "V5"}] * 3}.get(
            len(rest_rows), [set()] * len(rest_rows))
        for j, r in enumerate(rest_rows):
            best, best_n = -np.inf, None
            for nm in LEAD_NAMES:
                sc, parts_ = self.score(spec + [[nm]])
                # the strip name comes from the label, from direct comparison with the same
                # lead in the grid and from the convention; Goldberger (whole page) does not
                # enter: with a non-simultaneous strip or a poor trace it pulls to III/aVF/V3 for no reason
                sc -= W_LIMB * parts_["limb"]
                sc += 1.5 if nm in canon[j] else 0.0
                # the score includes the grid; only the difference between names matters
                if sc > best:
                    best, best_n = sc, nm
            spec.append([best_n])
        return spec

    def candidates(self, a=0, b=None):
        """Layout hypotheses for rows [a, b); the others stay empty (dropped)."""
        n_all = self.n
        b = n_all if b is None else b
        pre, pos = [[] for _ in range(a)], [[] for _ in range(n_all - b)]
        n_eff = b - a
        out = []
        for fam in FAMILIES:
            grid = LIBRARY[fam]
            if len(grid) > n_eff:
                continue
            rest = n_eff - len(grid)
            if (fam.startswith("12x1") or fam.startswith("6x1") or fam.startswith("3x1")) and rest:
                continue
            if fam.startswith("3x1") and not STRIPS3[0]:
                continue
            if (a or pos) and (fam.startswith("3x1") or fam.startswith("6x1")):
                continue      # drop rows only to close a 12-lead page
            if fam in ("3x1-v13", "3x1-v46"):
                # precordial leads have no physics among themselves (nothing closes like
                # Einthoven): only with a V label read at the start of some row; without it the
                # hypothesis wins just by having no physics term to lose
                vs = 0
                for r in range(a, b):
                    xs = float(self.x_of(r, 0.0)) / self.R
                    w = self.label_near(r, xs - 25, xs + LABEL_HI[0])
                    if w and max(w["dist"], key=w["dist"].get).startswith("V"):
                        vs += 1
                if vs < 2:
                    continue
            if rest > 3:
                continue
            g_ = pre + [list(r) for r in grid]
            spec = self._best_rhythm_names(g_, range(a + len(grid), b)) if rest else g_
            name = fam if not rest else f"{fam.split('-')[0]}+{rest}" + ("-cabrera" if "cabrera" in fam else "")
            out.append((name, spec + pos))
        if a or pos:
            return out
        # free hypothesis: k per row from the boundaries, names from OCR
        free = []
        for r in range(self.n):
            bestk, bests = 1, -np.inf
            for k, fr in FRACS.items():
                sc = sum((e - 0.3) if any(abs(q - f) < 1e-3 for f in fr) else -(max(e, 0.3) - 0.3)
                         for q, e in self.bnd[r].items())
                if sc > bests:
                    bestk, bests = k, sc
            row = []
            for c in range(bestk):
                xs = float(self.x_of(r, c / bestk * self.duration)) / self.R
                w = self.label_near(r, xs - (25 if c == 0 else 2.0), xs + 15)
                row.append(max(w["dist"], key=w["dist"].get) if w else f"?{r}{c}")
            free.append(row)
        if all(not n.startswith("?") for row in free for n in row):
            out.append(("livre", free))
        return out

    def choose(self):
        cands = [(nm, sp, 0.0) for nm, sp in self.candidates()]
        if ROW_DROP[0] and not getattr(self, "col_hint", None) and self.n >= 3:
            # edge rows may not be trace (grid frame, ruler, text band): the hypothesis
            # without them competes with the others and pays a cost. The usual score decides
            # (labels, boundaries, Einthoven/Goldberger, duplicates)
            # reliable calibration pulse = agrees in height and column with at least one other
            okp = [p for p in self.pulses if p is not None]
            trusted = [False] * self.n
            if len(okp) >= 2:
                h0 = float(np.median([p["foot"] - p["top"] for p in okp])); x0_ = float(np.median([p["x_start"] for p in okp]))
                trusted = [p is not None and abs((p["foot"] - p["top"]) - h0) <= 0.15 * h0 and abs(p["x_start"] - x0_) <= 3 * self.R
                           for p in self.pulses]
            # only an edge row CLOSER to its neighbour than the step of the others is dropped:
            # the grid frame sticks to the trace block; a real rhythm strip sits one step (or
            # more) below the last row
            ys_ = np.asarray(self.rows, float)
            gp = np.diff(ys_)
            near_top = self.n >= 4 and gp[0] < DROP_NEAR[0] * float(np.median(gp[1:]))
            near_bot = self.n >= 4 and gp[-1] < DROP_NEAR[0] * float(np.median(gp[:-1]))
            for a, nb in ((1, 0), (0, 1), (1, 1)):
                drop = list(range(a)) + list(range(self.n - nb, self.n))
                if self.n - len(drop) < 3 or (a and not near_top) or (nb and not near_bot):
                    continue
                # and only when the calibration pulse points to it: every remaining row has a
                # pulse and the dropped one does not. (On real pages the rhythm strip is
                # sometimes closer than the step and still worsens the physics because it does
                # not match the grid lead well: without the pulse as witness, it was dropped)
                if any(self.pulses[r] is not None for r in drop) or \
                        any(self.pulses[r] is None for r in range(self.n) if r not in drop):
                    continue
                # fixed cost per dropped row, larger if it has a reliable calibration pulse.
                # There is no safe measure of "looks like trace": the frame inherits the QRS of
                # the neighbouring row and a flat lead is almost straight
                cost = sum(DROP_COST[0] + (3.0 if trusted[r] else 0.0) for r in drop)
                cands += [(nm, sp, cost) for nm, sp in self.candidates(a, self.n - nb)]
        scored = []
        for name, spec, cost in cands:
            sc, parts = self.score(spec)
            sc -= cost
            # number of columns already measured on the image (columns with their own heights):
            # layouts with that number of columns gain, the others lose
            kh = getattr(self, "col_hint", None)
            if kh:
                kmax = max(len(row) for row in spec)
                sc += 8.0 if kmax == kh else -8.0
            if name != "livre":
                sc += 3.0  # prior for vendor formats (the free hypothesis depends only on OCR)
                if "cabrera" in name:
                    sc -= 1.5  # Cabrera is regional (Scandinavia): needs some evidence
                    # with no "-aVR" read anywhere on the page, physics alone is not enough:
                    # Goldberger closes "almost" equally for several lead permutations
                    if not any(w["dist"].get("-aVR", 0.0) >= 0.5 for w in self.words):
                        sc -= CABRERA_NO_EVIDENCE[0]
            else:
                # in a grid each lead appears once (rhythm strips are 1-lead rows)
                grid_names = [nm for row in spec if len(row) > 1 for nm in row]
                sc -= 3.0 * (len(grid_names) - len(set(map(_canon, grid_names))))
            scored.append((sc, name, spec, parts, cost))
        scored.sort(key=lambda z: -z[0])
        if not scored:
            return None
        if any(z[4] > 0 for z in scored):
            # dropping a row removes its penalties (wrong label, misplaced boundary): only
            # worth it if the physics improves. With the frame read as a lead,
            # Einthoven/Goldberger does not close; without it, it does
            full_ = [z for z in scored if z[4] == 0]
            lim0 = full_[0][3]["limb"] if full_ else -1.0
            scored = [z for z in scored if z[4] == 0 or z[3]["limb"] >= lim0 + DROP_LIMB_GAIN[0]]
        best = scored[0]
        margin = best[0] - scored[1][0] if len(scored) > 1 else 10.0
        conf = float(1 / (1 + np.exp(-margin / 2.0)))
        warn = None
        n_lab = 0
        for r in range(self.n):  # labels readable at the start of some row
            xs = float(self.x_of(r, 0.0)) / self.R
            w = self.label_near(r, xs - 25, xs + 15)
            if w is not None and max(w["dist"].values()) >= 0.45:
                n_lab += 1
        if n_lab == 0 and len(scored) > 1:
            # with no readable label, only the physics decides; pages of 6 precordial strips
            # in order mimic the Cabrera fan (they close Goldberger "almost")
            sets = lambda sp: {_canon(n) for row in sp for n in row}
            shape = lambda sp: [len(row) for row in sp]
            alts = [z for z in scored[1:] if sets(z[2]) != sets(best[2])]
            alts = [z for z in alts if shape(z[2]) == shape(best[2])] or alts
            if alts:
                conf = min(conf, 0.55)
                warn = f"no label read; {best[1]} chosen by physiology only (alternative: {alts[0][1]})"
        return dict(name=best[1], spec=best[2], score=best[0], parts=best[3], confidence=conf,
                    warning=warn, ranking=[(round(float(z[0]), 2), z[1]) for z in scored[:6]],
                    dropped=[r for r, row in enumerate(best[2]) if not row])
