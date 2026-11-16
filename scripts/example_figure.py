"""docs/example.png: a synthetic photo (rotation, perspective, shadow, JPEG) with exact ground truth.

    python scripts/example_figure.py [seed]
"""
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from ecgdig import digitize, Config, LEADS  # noqa: E402
from ecgdig.synth import synth_signals, render  # noqa: E402
from ecgdig.evaluate import snr_db  # noqa: E402

seed = int(sys.argv[1]) if len(sys.argv) > 1 else 7
rng = np.random.default_rng(seed)
sig = synth_signals(rng)
img, shown, _ = render(sig, rng, degrade="medium")
r = digitize(img, Config(), return_debug=True)
pred = r["signals"]


def panel(ax, lead, t0, t1):
    i = LEADS.index(lead)
    t = np.arange(sig.shape[1]) / 500.0
    m = shown[i] & (t >= t0) & (t <= t1)
    g = np.where(m, sig[i], np.nan)
    p = np.where(m, pred[i][:sig.shape[1]], np.nan)
    ax.plot(t, g, color="#1b1b1b", lw=1.6, label="ground truth")
    ax.plot(t, p, color="#e4572e", lw=1.0, label="digitized")
    ax.set_xlim(t0, t1); ax.set_ylabel("mV"); ax.grid(alpha=0.3)
    ax.set_title(f"{lead} · SNR {snr_db(g, p):.1f} dB", fontsize=10, loc="left")


fig = plt.figure(figsize=(12, 8.6), dpi=110)
ax = fig.add_axes([0.03, 0.52, 0.45, 0.45]); ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)); ax.axis("off")
ax.set_title("Input (synthetic photo: rotation, perspective, shadow, JPEG)", fontsize=10)
ax = fig.add_axes([0.52, 0.52, 0.45, 0.45]); ax.imshow(cv2.cvtColor(r["debug_image"], cv2.COLOR_BGR2RGB)); ax.axis("off")
ax.set_title("Rectified + extracted traces (blue)", fontsize=10)
ax = fig.add_axes([0.06, 0.28, 0.91, 0.19]); panel(ax, "II", 0, 10); ax.legend(loc="upper right", fontsize=9)
ax = fig.add_axes([0.06, 0.05, 0.91, 0.19]); panel(ax, "V4", 7.5, 10); ax.set_xlabel("s")
out = ROOT / "docs" / "example.png"
fig.savefig(out); plt.close(fig)
print("written", out, r["layout"])
