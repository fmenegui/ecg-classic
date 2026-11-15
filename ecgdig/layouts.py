"""Biblioteca de layouts e utilidades de nomes de derivações.

Um layout é uma lista de linhas; cada linha é uma lista de derivações que dividem
igualmente a duração exibida (ex.: 3x4 -> 4 derivações de 2.5 s por linha a 25 mm/s).
"-aVR" é a derivação invertida do sistema de Cabrera.
"""
from __future__ import annotations

import numpy as np

LEADS = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
LEAD_NAMES = LEADS + ["-aVR"]                # nomes que podem aparecer impressos
LIMB = {"I", "II", "III", "aVR", "aVL", "aVF", "-aVR"}

# cada derivação de membros como combinação de (I, II)  [Einthoven/Goldberger]
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
    # páginas de 6 tiras de largura total (ex.: Schiller/Mortara imprimem membros e precordiais em folhas separadas)
    "6x1-prec": [[f"V{i}"] for i in range(1, 7)],
    "6x1-limb": [[l] for l in LEADS[:6]],
    "6x1-limb-cabrera": [[l] for l in _CAB_12[:6]],
    # aparelhos de 3 canais: 3 tiras de 10 s por folha, um grupo de derivações em cada folha
    "3x1-limb": [["I"], ["II"], ["III"]],
    "3x1-aug": [["aVR"], ["aVL"], ["aVF"]],
    "3x1-v13": [["V1"], ["V2"], ["V3"]],
    "3x1-v46": [["V4"], ["V5"], ["V6"]],
}


def lead_signal(sig12: np.ndarray, name: str) -> np.ndarray:
    """Sinal (N,) de um nome impresso a partir das 12 derivações padrão."""
    if name == "-aVR":
        return -sig12[LEADS.index("aVR")]
    return sig12[LEADS.index(name)]


def to_standard(name: str, v: np.ndarray):
    """Converte medida de um nome impresso para (derivação padrão, valor)."""
    if name == "-aVR":
        return "aVR", -v
    return name, v


def is_limb(name: str) -> bool:
    return name in LIMB
