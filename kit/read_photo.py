"""Ground truth for a photo of a kit page (kit/make_kit.py).

    python kit/read_photo.py --gt kit_out/gt --out labels photo1.jpg photo2.jpg ...

Finds the ArUco markers, the page (marker id // 4) and the homography from page millimetres to
photo pixels (all detected marker corners, RANSAC). Writes, per photo:
- <stem>.json: page, homography (3x3, mm -> px), markers found, reprojection error (px), and every
  printed segment as a polyline in photo pixels (one point per sample: the trace position the
  printer drew, before paper bending);
- <stem>_flat.png: the page brought back to a flat frame at --ppmm px/mm (what a perfect
  rectification would give), for training and for checking.

The homography is exact for a flat page. For a bent or crumpled page it is only the plane through
the markers; the json says how many markers were found so curved photos can be told apart.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def detect(img: np.ndarray):
    """ArUco 4x4 markers: {id: (4, 2) corners in pixels}."""
    d = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_1000)
    p = cv2.aruco.DetectorParameters()
    p.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    det = cv2.aruco.ArucoDetector(d, p)
    g = img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    corners, ids, _ = det.detectMarkers(g)
    if ids is None:
        return {}
    return {int(i): c.reshape(4, 2).astype(np.float64) for i, c in zip(ids.ravel(), corners)}


def page_homography(found: dict, gt: dict):
    """Homography page mm -> photo px from the markers of this page; (H, rms px, n markers)."""
    src, dst = [], []
    for m in gt["markers"]:
        if m["id"] in found:
            src.append(np.asarray(m["corners_mm"], float))
            dst.append(found[m["id"]])
    if not src:
        return None, None, 0
    src, dst = np.concatenate(src), np.concatenate(dst)
    if len(src) == 4:   # one marker: exact but poorly conditioned far from it
        H = cv2.getPerspectiveTransform(src.astype(np.float32), dst.astype(np.float32))
    else:
        H, _ = cv2.findHomography(src, dst, cv2.RANSAC, 3.0)
    pr = cv2.perspectiveTransform(src.reshape(-1, 1, 2), H).reshape(-1, 2)
    return H, float(np.sqrt(np.mean(np.sum((pr - dst) ** 2, 1)))), len(src) // 4


def segment_polylines(gt: dict, sig: np.ndarray, H: np.ndarray):
    """Each printed segment as photo pixels, one point per sample."""
    out = []
    leads = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
    for s in gt["segments"]:
        a, b = s["sample0"], s["sample1"]
        t = np.arange(a, b) / gt["fs"]
        x = s["x0_mm"] + (t - t[0]) * gt["speed_mm_s"]
        y = s["baseline_mm"] - sig[leads.index(s["lead"]), a:b] * gt["gain_mm_mv"]
        px = cv2.perspectiveTransform(np.stack([x, y], 1).reshape(-1, 1, 2), H).reshape(-1, 2)
        out.append(dict(s, px=np.round(px, 2).tolist()))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt", type=Path, required=True, help="the kit's gt folder")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--ppmm", type=float, default=10.0)
    ap.add_argument("photos", nargs="+", type=Path)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    for ph in a.photos:
        img = cv2.imread(str(ph))
        if img is None:
            print(ph, "unreadable"); continue
        found = detect(img)
        pages = sorted({i // 4 for i in found}, key=lambda p: -sum(i // 4 == p for i in found))
        if not pages:
            print(ph, "no marker found"); continue
        p = pages[0]
        gj = a.gt / f"page_{p:04d}.json"
        if not gj.exists():
            print(ph, f"page {p} is not in {a.gt}"); continue
        gt = json.loads(gj.read_text())
        sig = np.load(a.gt / f"page_{p:04d}.npz")["signals"]
        H, rms, n = page_homography(found, gt)
        if H is None:
            print(ph, "no marker of page", p); continue
        segs = segment_polylines(gt, sig, H)
        W, Hh = gt["page_mm"]
        S = np.diag([a.ppmm, a.ppmm, 1.0])
        flat = cv2.warpPerspective(img, S @ np.linalg.inv(H), (int(W * a.ppmm), int(Hh * a.ppmm)),
                                   flags=cv2.INTER_CUBIC, borderValue=(255, 255, 255))
        cv2.imwrite(str(a.out / f"{ph.stem}_flat.png"), flat)
        (a.out / f"{ph.stem}.json").write_text(json.dumps(dict(
            photo=ph.name, page=p, ecg_id=gt["ecg_id"], layout=gt["plan"]["layout"], markers=n,
            reproj_rms_px=rms, H_mm_to_px=H.tolist(), segments=segs)))
        print(f"{ph.name}: page {p}, {n} of 4 markers, reprojection {rms:.2f} px")


if __name__ == "__main__":
    main()
