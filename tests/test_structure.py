"""Page structure: columns with their own heights, phase unwrapping and word grouping."""
import numpy as np
import cv2

from ecgdig import colalign, ocr
from ecgdig.geometry import _unwrap_quality

R = 10.0


def _trace(B, y, xa, xb, rng, beat=180):
    """Synthetic trace: baseline with 1 px noise and a QRS every `beat` px."""
    for x in range(xa, xb):
        yy = int(round(y + rng.normal(0, 0.6)))
        B[yy - 1:yy + 2, x] = True
        if (x - xa) % beat == beat // 2:
            B[y - 60:y + 25, x - 1:x + 2] = True


def _page(cols_rows, W=3200, H=2200, seed=0, margin=15):
    rng = np.random.default_rng(seed)
    B = np.zeros((H, W), bool)
    K = len(cols_rows)
    edges = np.linspace(300, 2900, K + 1).astype(int)
    for c, rows in enumerate(cols_rows):
        for y in rows:
            _trace(B, y, edges[c] + margin, edges[c + 1] - margin, rng)
    return B, edges


def test_col_struct_acha_colunas_com_alturas_proprias():
    left = [400, 500, 650, 850, 1010, 1090]
    right = [450, 600, 750, 950, 1350, 1600]
    B, edges = _page([left, right])
    st = colalign.col_struct(B, R)
    assert st is not None
    cols, P = st
    assert len(cols) == 2 and P.shape == (2, 6)
    assert np.abs(P[0] - left).max() <= 6 and np.abs(P[1] - right).max() <= 6
    assert abs(cols[0][1] - edges[1]) <= 40           # boundary between the columns


def test_col_struct_ignora_pagina_comum():
    rows = [400, 650, 900, 1150]
    B, _ = _page([rows, rows, rows, rows])
    assert colalign.col_struct(B, R) is None


def test_col_align_realinha_sem_sobrepor():
    left = [400, 500, 650]
    right = [450, 700, 1000]
    B, _ = _page([left, left, right, right], margin=0)   # no blank gap between the columns
    D = B.astype(np.float32)
    rect = np.full(B.shape + (3,), 255, np.uint8)
    pap = np.ones(B.shape, np.uint8)
    colalign.STRUCT_FIRST[0] = True       # the tracks path (by default it comes after the equal widths)
    try:
        _, _, _, B2, info = colalign.col_align(rect, pap, D, B, R)
    finally:
        colalign.STRUCT_FIRST[0] = False
    assert info is not None and info["how"] == "trilhos"
    ref = np.array(info["ref"])
    assert np.diff(ref).min() >= 240                   # interval = the largest across columns
    # after realignment, every reference row has ink across the whole trace width
    for y in ref.astype(int):
        cov = B2[y - 4:y + 5, 400:2800].any(0).mean()
        assert cov > 0.9


def test_unwrap_usa_a_maior_regiao():
    T = 25.0
    h, w = 12, 16
    ii, jj = np.mgrid[0:h, 0:w]
    true = 3.0 * jj + 0.5 * ii                         # line position grows 3 px per block
    ph = true % T
    q = np.full((h, w), 0.5)
    q[:, 3] = 0.0                                      # a bad band separates an island of 3 columns
    q[5, 1] = 0.99                                     # the page's best block is on the island
    u = _unwrap_quality(ph, q, T, 0.2)
    big = np.zeros((h, w), bool); big[:, 4:] = True
    assert np.isfinite(u[big]).all()                   # the whole large region was unwrapped
    d = (u - true)[big]
    assert np.ptp(d) < 1e-6                            # up to a common multiple of T


def _find_words_ref(text_mask, R, min_h_mm=1.4, max_h_mm=8.0):
    """The original quadratic sweep."""
    n, lab, st, _ = cv2.connectedComponentsWithStats(text_mask.astype(np.uint8), connectivity=8)
    boxes = []
    for i in range(1, n):
        x, y, w, h, a = (int(v) for v in st[i])
        if a < 4 or h > max_h_mm * R or w > 25 * R:
            continue
        boxes.append([x, y, x + w, y + h])
    boxes.sort(key=lambda b: b[0])
    words, used = [], [False] * len(boxes)
    for i, b in enumerate(boxes):
        if used[i]:
            continue
        cur = list(b); used[i] = True
        changed = True
        while changed:
            changed = False
            for j, c in enumerate(boxes):
                if used[j]:
                    continue
                hh = max(cur[3] - cur[1], c[3] - c[1], 1)
                vo = min(cur[3], c[3]) - max(cur[1], c[1])
                gap = max(c[0] - cur[2], cur[0] - c[2])
                if gap < 0.7 * hh and vo > -0.1 * hh and vo > 0.25 * min(c[3] - c[1], cur[3] - cur[1]) - 1:
                    cur = [min(cur[0], c[0]), min(cur[1], c[1]), max(cur[2], c[2]), max(cur[3], c[3])]
                    used[j] = True; changed = True
        h = cur[3] - cur[1]
        if min_h_mm * R <= h <= max_h_mm * R:
            words.append(tuple(cur))
    return words


def test_find_words_igual_a_varredura_completa():
    rng = np.random.default_rng(5)
    m = np.zeros((400, 900), bool)
    for _ in range(260):                               # loose small boxes of various sizes
        x, y = int(rng.integers(0, 860)), int(rng.integers(0, 360))
        w, h = int(rng.integers(3, 30)), int(rng.integers(6, 40))
        m[y:y + h, x:x + w] = True
    assert ocr.find_words(m, R) == _find_words_ref(m, R)
