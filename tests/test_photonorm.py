"""photonorm.tone_map: a darkened or washed-out page comes back to the usual range; a normal page
is left alone and a white page with faint ink is only stretched."""
import numpy as np

from ecgdig.photonorm import tone_map


def page(paper=235, ink=20, grid=180):
    im = np.full((400, 600, 3), paper, np.uint8)
    im[::20] = grid; im[:, ::20] = grid
    im[200:203, 50:550] = ink
    return im


def test_untouched():
    im = page()
    out, info = tone_map(im)
    assert not info["applied"] and out is im
    faint = page(paper=255, ink=200, grid=230)                 # faint ink on a white digital page
    out, info = tone_map(faint)
    assert info["applied"] and info["gamma"] == 1.0 and out[201, 300].mean() < 100   # stretched only


def test_dark_and_washed_out_restored():
    base = page().astype(np.float32)
    dark = (base * 0.2).astype(np.uint8)                        # like brightness -160
    washed = (255 * (page(paper=200).astype(np.float32) / 255) ** 0.125).astype(np.uint8)   # gamma 0.125
    for im in (dark, washed):
        out, info = tone_map(im)
        assert info["applied"]
        paper = np.percentile(out.mean(axis=2), 75)
        ink = out[201, 300].mean()
        assert paper > 200 and ink < 80 and paper - ink > 150


def test_faint_grey_pen_on_red_grid():
    im = np.full((400, 600, 3), 255, np.uint8)
    im[::20] = (16, 80, 240); im[:, ::20] = (16, 80, 240)        # saturated orange-red grid (BGR)
    im[200:203, 50:550] = 208                                    # faint grey pen
    out, info = tone_map(im)
    assert not info["applied"]                                   # off by default
    out, info = tone_map(im, white_max_channel=True)
    assert info["applied"] and info["channel"] == "max"
    assert out[201, 300].max() < 60                              # the pen is dark now
    assert out[0, 300, 2] > 100                                  # the grid keeps its red channel
