"""Os atalhos de velocidade têm de dar o mesmo resultado dos cálculos de referência."""
import warnings

import numpy as np
import pytest

from ecgdig.geometry import _nanmedian_axis, _nanmedian3x3
from ecgdig.extract import _median5_cells, column_runs, viterbi_track


def test_nanmedian_axis_igual_ao_numpy():
    rng = np.random.default_rng(0)
    a = rng.random((37, 50)).astype(np.float32)
    a[rng.random(a.shape) < 0.3] = np.nan
    a[:, 7] = np.nan                         # coluna toda NaN
    a[:36, 9] = np.nan                       # só um valor
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        ref = np.nanmedian(a, axis=0)
    got = _nanmedian_axis(a, 0)
    assert got.dtype == a.dtype
    np.testing.assert_array_equal(np.isnan(ref), np.isnan(got))
    np.testing.assert_array_equal(ref[~np.isnan(ref)], got[~np.isnan(got)])


def test_nanmedian3x3_igual_ao_generic_filter():
    from scipy.ndimage import generic_filter
    rng = np.random.default_rng(1)
    f = rng.normal(size=(12, 17))
    f[rng.random(f.shape) < 0.25] = np.nan
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        ref = generic_filter(f, np.nanmedian, size=3, mode="nearest")
    got = _nanmedian3x3(f)
    np.testing.assert_array_equal(np.isnan(ref), np.isnan(got))
    np.testing.assert_array_equal(ref[~np.isnan(ref)], got[~np.isnan(got)])


def test_median5_cells_igual_ao_scipy():
    from scipy.ndimage import median_filter
    rng = np.random.default_rng(2)
    cells = rng.random((9, 11, 6, 6)).astype(np.float32)
    ref = median_filter(cells, size=(5, 5, 1, 1), mode="nearest")
    np.testing.assert_array_equal(ref, _median5_cells(cells))


def _column_runs_ref(B, D, ya, yb, xa, xb):
    """A versão antiga, coluna a coluna."""
    out = []
    sub, subD = B[ya:yb, xa:xb], D[ya:yb, xa:xb]
    yy = np.arange(ya, yb, dtype=np.float64)
    for j in range(sub.shape[1]):
        d = np.diff(np.concatenate([[0], sub[:, j].astype(np.int8), [0]]))
        s, e = np.nonzero(d == 1)[0], np.nonzero(d == -1)[0] - 1
        if len(s) > 1:
            keep = np.r_[True, (s[1:] - e[:-1] - 1) > 3]
            grp = np.cumsum(keep) - 1
            s = np.array([s[grp == g].min() for g in range(grp.max() + 1)])
            e = np.array([e[grp == g].max() for g in range(grp.max() + 1)])
        col = []
        for a, b in zip(s, e):
            w = subD[a:b + 1, j].astype(np.float64)
            m = w.sum() + 1e-9
            col.append((a + ya, b + ya, float((w * yy[a:b + 1]).sum() / m), float(m)))
        if len(col) > 1:
            big = [c for c in col if c[3] >= 0.6]
            col = big or col
        out.append(col)
    return out


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_column_runs_igual_a_referencia(seed):
    rng = np.random.default_rng(seed)
    D = rng.random((80, 60)).astype(np.float32)
    D[rng.random(D.shape) < 0.5] *= 0.1
    B = D > 0.45
    ref = _column_runs_ref(B, D, 5, 70, 3, 55)
    got = column_runs(B, D, 5, 70, 3, 55)
    assert len(ref) == len(got)
    for cr, cg in zip(ref, got):
        assert [(int(a), int(b)) for a, b, _, _ in cr] == [(int(a), int(b)) for a, b, _, _ in cg]
        assert all(isinstance(a, np.integer) and isinstance(b, np.integer) for a, b, _, _ in cg)
        np.testing.assert_allclose([c[2:] for c in cr], [c[2:] for c in cg], rtol=1e-12, atol=1e-12)
    # o Viterbi dá o mesmo caminho com os segmentos de uma ou de outra versão
    assert viterbi_track(ref, 40.0, 10.0, lw=3.0, w_edge=0.3) == viterbi_track(got, 40.0, 10.0, lw=3.0, w_edge=0.3)


def test_column_runs_faixa_alem_da_imagem():
    B = np.zeros((20, 10), bool)
    B[15:, 4] = True
    D = B.astype(np.float32)
    got = column_runs(B, D, 10, 40, 0, 10)      # yb passa do fim: a faixa é cortada
    (top, bot, cen, mass), = got[4]
    assert (top, bot) == (15, 19) and cen == pytest.approx(17.0) and mass == pytest.approx(5.0)
    assert all(not c for i, c in enumerate(got) if i != 4)
