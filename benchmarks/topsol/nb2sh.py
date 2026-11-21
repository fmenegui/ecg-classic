"""A Kaggle notebook made of shell cells (%env, !commands) -> a bash script for a pod.

    python benchmarks/topsol/nb2sh.py NOTEBOOK.ipynb OUT.sh

/kaggle/input -> $TS_INPUT, /kaggle/working -> $TS_WORK; %env X=Y -> export X=Y; !cmd -> cmd;
%%capture and other magics dropped. The 4th place solution (tomoon33) is written this way.
"""
import json
import re
import shlex
import sys


def convert(nb):
    out = ["set -x"]
    for c in nb["cells"]:
        if c["cell_type"] != "code":
            continue
        for line in "".join(c["source"]).splitlines():
            s = line.rstrip()
            if s.startswith("%%") or s.startswith("%") and not s.startswith("%env"):
                continue
            if s.startswith("%env"):
                k, _, v = s[4:].strip().partition("=")
                v = v.strip()
                if len(v) >= 2 and v[0] == v[-1] and v[0] in "\"'":
                    v = v[1:-1]
                s = f"export {k.strip()}={shlex.quote(v)}"
            elif s.lstrip().startswith("!"):
                s = s.lstrip()[1:]
            s = s.replace("/kaggle/input", "$TS_INPUT").replace("/kaggle/working", "$TS_WORK")
            out.append(s)
    return "\n".join(out) + "\n"


if __name__ == "__main__":
    open(sys.argv[2], "w").write(convert(json.load(open(sys.argv[1]))))
