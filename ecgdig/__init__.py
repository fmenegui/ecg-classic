"""ecgdig: classical digitizer for paper ECGs (computer vision, no deep learning)."""
import ast as _ast
import importlib as _importlib
import os as _os

from .digitize import digitize, Config, LEADS, LAYOUTS  # noqa: F401

__all__ = ["digitize", "Config", "LEADS", "LAYOUTS"]


def apply_flags(spec: str) -> None:
    """Ablation switches "module.FLAG=value;..." (1-element lists in ecgdig.*), to compare variants
    on a whole benchmark without changing the code. ECGDIG_FLAGS sets them at import."""
    for it in [x.strip() for x in (spec or "").split(";") if x.strip()]:
        mod, rest = it.split(".", 1)
        var, val = rest.split("=", 1)
        getattr(_importlib.import_module(__name__ + "." + mod), var)[0] = _ast.literal_eval(val)


apply_flags(_os.environ.get("ECGDIG_FLAGS", ""))
