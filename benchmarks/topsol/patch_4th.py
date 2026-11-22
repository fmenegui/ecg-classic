"""4th place solution: an image it cannot read (no grid found) aborts the whole submission
("Missing prediction for submission row"; Nov 22: 20 of 300 probe pages). Write NaN for that image
instead, so the other images are scored and that one counts as a failure.

    python benchmarks/topsol/patch_4th.py PATH/run_ecg_digitization_inference.py
"""
import sys

OLD = '''        if values is None:
            raise RuntimeError(
                "Missing prediction for submission row. "'''
NEW = '''        if values is None:
            print("MISSING", base_id, lead, flush=True)
            values = np.full(n_rows, np.nan, dtype=np.float32)
        if values is None:
            raise RuntimeError(
                "Missing prediction for submission row. "'''

for p in sys.argv[1:]:
    s = open(p).read()
    if OLD in s:
        open(p, "w").write(s.replace(OLD, NEW))
        print("patched", p)
    else:
        print("patch target not found", p)
