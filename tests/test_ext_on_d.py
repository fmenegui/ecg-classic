"""Row starts extended over a chain of trace pieces holding beats."""
import numpy as np

from ecgdig.digitize import _extend_ext_on_d


def _page(beats=(320, 420), gap_at=None, tall=True):
    B = np.zeros((200, 3000), bool)
    B[100, 500:2500] = True                      # the row's long stretch
    for x in range(300, 500, 40):                 # faint start broken into pieces
        B[100, x:x + 15] = True
    for x in beats:                               # QRS: 6 mm tall
        B[70:131 if tall else 85, x:x + 4] = True
    if gap_at is not None:
        B[:, gap_at:gap_at + 150] = False         # 15 mm without ink
    return B


def test_chain_with_two_beats_is_accepted():
    out = _extend_ext_on_d(_page(), [100], [(500, 2500)], [None], 40, 10.0)
    assert out[0] == (300, 2500)


def test_row_with_pulse_keeps_its_start():
    out = _extend_ext_on_d(_page(), [100], [(500, 2500)], [{"x_end": 480}], 40, 10.0)
    assert out[0] == (500, 2500)


def test_one_beat_or_flat_pieces_are_not_enough():
    assert _extend_ext_on_d(_page(beats=(420,)), [100], [(500, 2500)], [None], 40, 10.0)[0] == (500, 2500)
    assert _extend_ext_on_d(_page(tall=False), [100], [(500, 2500)], [None], 40, 10.0)[0] == (500, 2500)


def test_grid_lines_5mm_apart_are_not_beats():
    out = _extend_ext_on_d(_page(beats=(400, 450)), [100], [(500, 2500)], [None], 40, 10.0)
    assert out[0] == (500, 2500)


def test_long_gap_stops_the_chain():
    out = _extend_ext_on_d(_page(gap_at=340), [100], [(500, 2500)], [None], 40, 10.0)
    assert out[0][0] >= 340


def test_rows_must_move_together():
    B = np.zeros((400, 3000), bool)
    for y in (100, 300):
        B[y, 500:2500] = True
    for x in range(300, 500, 40):
        B[100, x:x + 15] = True
    for x in (320, 420):
        B[70:131, x:x + 4] = True
    # only the first of two rows has the faint start: not a page start, the rows stay
    out = _extend_ext_on_d(B, [100, 300], [(500, 2500), (500, 2500)], [None, None], 40, 10.0)
    assert out == [(500, 2500), (500, 2500)]
