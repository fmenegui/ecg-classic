"""Layout library and lead-name helpers.

A layout is a list of rows; each row is a list of leads that split the displayed duration
equally (e.g. 3x4 -> 4 leads of 2.5 s per row at 25 mm/s). "-aVR" is the inverted lead of
the Cabrera system.
"""
from __future__ import annotations

import numpy as np

LEADS = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
LEAD_NAMES = LEADS + ["-aVR"]                # names that may be printed on a page
LIMB = {"I", "II", "III", "aVR", "aVL", "aVF", "-aVR"}

# each limb lead as a combination of (I, II)  [Einthoven/Goldberger]
LIMB_COEF = {
    "I": (1.0, 0.0), "II": (0.0, 1.0), "III": (-1.0, 1.0),
    "aVR": (-0.5, -0.5), "aVL": (1.0, -0.5), "aVF": (-0.5, 1.0), "-aVR": (0.5, 0.5),
}

_STD_3x4 = [["I", "aVR", "V1", "V4"], ["II", "aVL", "V2", "V5"], ["III", "aVF", "V3", "V6"]]
_CAB_3x4 = [["aVL", "II", "V1", "V4"], ["I", "aVF", "V2", "V5"], ["-aVR", "III", "V3", "V6"]]
_STD_6x2 = [["I", "V1"], ["II", "V2"], ["III", "V3"], ["aVR", "V4"], ["aVL", "V5"], ["aVF", "V6"]]
_CAB_6x2 = [["aVL", "V1"], ["I", "V2"], ["-aVR", "V3"], ["II", "V4"], ["aVF", "V5"], ["III", "V6"]]
_CAB_12 = ["aVL", "I", "-aVR", "II", "aVF", "III", "V1", "V2", "V3", "V4", "V5", "V6"]

LIBRARY = {
    "3x4": _STD_3x4,
    "3x4+1": _STD_3x4 + [["II"]],
    "3x4+2": _STD_3x4 + [["II"], ["V1"]],
    "3x4+3": _STD_3x4 + [["V1"], ["II"], ["V5"]],
    "6x2": _STD_6x2,
    "6x2+1": _STD_6x2 + [["II"]],
    "12x1": [[l] for l in LEADS],
    "3x4-cabrera": _CAB_3x4,
    "3x4+1-cabrera": _CAB_3x4 + [["II"]],
    "3x4+3-cabrera": _CAB_3x4 + [["V1"], ["II"], ["V5"]],
    "6x2-cabrera": _CAB_6x2,
    "6x2+1-cabrera": _CAB_6x2 + [["II"]],
    "12x1-cabrera": [[l] for l in _CAB_12],
    # pages of 6 full-width strips (e.g. Schiller/Mortara print limb and precordial leads on separate sheets)
    "6x1-prec": [[f"V{i}"] for i in range(1, 7)],
    "6x1-limb": [[l] for l in LEADS[:6]],
    "6x1-limb-cabrera": [[l] for l in _CAB_12[:6]],
    # 3-channel devices: three 10 s strips per sheet, one lead group per sheet
    "3x1-limb": [["I"], ["II"], ["III"]],
    "3x1-aug": [["aVR"], ["aVL"], ["aVF"]],
    "3x1-v13": [["V1"], ["V2"], ["V3"]],
    "3x1-v46": [["V4"], ["V5"], ["V6"]],
    # 12 leads on two sheets of 3 rows x 2 columns (5 s each): limb leads, then precordials
    "3x2-limb": [["I", "aVR"], ["II", "aVL"], ["III", "aVF"]],
    "3x2-prec": [["V1", "V4"], ["V2", "V5"], ["V3", "V6"]],
}


def lead_signal(sig12: np.ndarray, name: str) -> np.ndarray:
    """Signal (N,) of a printed name, from the 12 standard leads."""
    if name == "-aVR":
        return -sig12[LEADS.index("aVR")]
    return sig12[LEADS.index(name)]


def to_standard(name: str, v: np.ndarray):
    """Map a measurement of a printed name to (standard lead, value)."""
    if name == "-aVR":
        return "aVR", -v
    return name, v


def is_limb(name: str) -> bool:
    return name in LIMB
