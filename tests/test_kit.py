"""Print kit: markers found in a warped page give the page and its homography."""
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "kit"))
import read_photo as RP  # noqa: E402


def test_markers_and_homography():
    ppmm, p = 8.0, 3
    W, H = int(297 * ppmm), int(210 * ppmm)
    page = np.full((H, W), 255, np.uint8)
    d = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_1000)
    side, off = 14.0, 7.0
    pos = [(off, off), (297 - off - side, off), (297 - off - side, 210 - off - side), (off, 210 - off - side)]
    markers = []
    for k, (mx, my) in enumerate(pos):
        m = cv2.aruco.generateImageMarker(d, 4 * p + k, int(side * ppmm))
        x0, y0 = int(round(mx * ppmm)), int(round(my * ppmm))
        page[y0:y0 + m.shape[0], x0:x0 + m.shape[1]] = m
        markers.append(dict(id=4 * p + k, corners_mm=[[mx, my], [mx + side, my], [mx + side, my + side], [mx, my + side]]))
    M = cv2.getPerspectiveTransform(np.float32([[0, 0], [W, 0], [W, H], [0, H]]),
                                    np.float32([[120, 80], [W + 60, 30], [W - 40, H + 140], [70, H + 50]]))
    photo = cv2.warpPerspective(page, M, (W + 200, H + 200), borderValue=100)
    found = RP.detect(photo)
    assert sorted(found) == [12, 13, 14, 15]
    Hm, rms, n = RP.page_homography(found, dict(markers=markers))
    assert n == 4 and rms < 1.5
    true = M @ np.diag([ppmm, ppmm, 1.0])
    pts = np.array([[30.0, 40.0], [150.0, 105.0], [270.0, 180.0]]).reshape(-1, 1, 2)
    err = np.abs(cv2.perspectiveTransform(pts, Hm) - cv2.perspectiveTransform(pts, true)).max()
    assert err < 0.25 * ppmm   # < 0.25 mm anywhere on the page
