"""ECGDIG_FLAGS switches module flags at import (every evaluator and pod gets the same mechanism)."""
import os
import subprocess
import sys


def test_env_flags_applied_at_import():
    env = dict(os.environ, ECGDIG_FLAGS="digitize.EXT_ON_D=True; geometry.H_SELECT_MM=0.2")
    out = subprocess.run([sys.executable, "-c", "import sys, ecgdig; "
                          "print(sys.modules['ecgdig.digitize'].EXT_ON_D[0], sys.modules['ecgdig.geometry'].H_SELECT_MM[0])"],
                         env=env, capture_output=True, text=True, check=True,
                         cwd=os.path.join(os.path.dirname(__file__), ".."))
    assert out.stdout.split() == ["True", "0.2"]
