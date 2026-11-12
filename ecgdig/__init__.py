"""ecgdig — digitalizador clássico de ECG em papel (visão computacional, sem deep learning)."""
from .digitize import digitize, Config, LEADS, LAYOUTS  # noqa: F401

__all__ = ["digitize", "Config", "LEADS", "LAYOUTS"]
