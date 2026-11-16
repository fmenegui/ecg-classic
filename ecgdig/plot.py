"""Redraw a digitized ECG on standard grid paper (25 mm/s, 10 mm/mV)."""
from __future__ import annotations

import numpy as np

_GRID_MINOR = "#f6c9c9"
_GRID_MAJOR = "#e88f8f"


def _grid(ax, w_mm, h_mm):
    for x in np.arange(0, w_mm + 0.01, 1):
        ax.axvline(x, color=_GRID_MAJOR if x % 5 == 0 else _GRID_MINOR,
                   lw=0.6 if x % 5 == 0 else 0.3, zorder=0)
    for y in np.arange(0, h_mm + 0.01, 1):
        ax.axhline(y, color=_GRID_MAJOR if y % 5 == 0 else _GRID_MINOR,
                   lw=0.6 if y % 5 == 0 else 0.3, zorder=0)
    ax.set_xlim(0, w_mm); ax.set_ylim(0, h_mm)
    ax.set_aspect("equal"); ax.axis("off")


def plot_ecg(ax, signals, leads, fs=500, layout="3x4+1", rhythm=None, title=None, speed=25.0):
    """Draw on `ax` in the requested layout (library name or list of rows, like
    `res["layout_rows"]`). "-aVR" is drawn inverted. `speed` in mm/s."""
    from .digitize import LAYOUTS
    spec = layout if isinstance(layout, (list, tuple)) else LAYOUTS.get(layout, [[l] for l in leads])
    n = signals.shape[1]
    dur = n / fs
    pre = 10.0  # mm on the left (pulse)
    w_mm = pre + dur * speed + 2
    row_h = 25.0
    h_mm = row_h * len(spec) + 6
    _grid(ax, w_mm, h_mm)
    idx = {l: i for i, l in enumerate(leads)}
    for r, row in enumerate(spec):
        y0 = h_mm - row_h * (r + 0.7)
        # 1 mV pulse
        ax.plot([2, 3, 3, 8, 8, 9], [y0, y0, y0 + 10, y0 + 10, y0, y0], color="k", lw=0.9)
        k = len(row)
        for c, lead in enumerate(row):
            sgn = -1.0 if lead.startswith("-") else 1.0
            base = lead.lstrip("-")
            if base not in idx:
                continue
            n0, n1 = c * n // k, (c + 1) * n // k
            t = np.arange(n0, n1) / fs
            v = sgn * signals[idx[base], n0:n1]
            ax.plot(pre + t * speed, y0 + v * 10, color="k", lw=0.8)
            ax.text(pre + n0 / fs * speed + 1, y0 + 6, lead, fontsize=7, color="#222")
    if title:
        ax.set_title(title, fontsize=10, loc="left")
