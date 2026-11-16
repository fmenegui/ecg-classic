"""ecgdig: classical digitizer for paper ECGs (computer vision, no deep learning)."""
from .digitize import digitize, Config, LEADS, LAYOUTS  # noqa: F401

__all__ = ["digitize", "Config", "LEADS", "LAYOUTS"]
