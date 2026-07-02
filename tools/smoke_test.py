"""Fast smoke test for CI: execute every lesson notebook end-to-end with the
training/sampling sizes shrunk so it runs in seconds instead of minutes.

It catches runtime errors, broken cells, and shape/logic bugs — not model quality
(that's what the notebooks themselves demonstrate). The two math oracles
(`d3pm_reference_check.py`, `sedd_reference_check.py`) validate correctness; this
validates that the notebooks *run*.

Run: python tools/smoke_test.py    ->  exits 0 if all notebooks execute, else 1.
"""
import glob
import os
import re
import sys
import time

import matplotlib
matplotlib.use("Agg")
import nbformat

# Shrink only the TRAINING step counts (the bottleneck). We deliberately do NOT
# touch T / n / sweep sizes — those are coupled to other hardcoded shapes
# (e.g. a label tensor sized to n, or a fixed t=40 that assumes T=128), and an
# untrained model still samples fast. This keeps the smoke test robust.
SUBS = [
    (re.compile(r"STEPS,\s*BATCH\s*=\s*\d+\s*,\s*(\d+)"), r"STEPS, BATCH = 3, \1"),
    (re.compile(r"\bSTEPS\s*=\s*\d{3,}"), "STEPS = 3"),
    (re.compile(r"\bsteps\s*=\s*\d{3,}"), "steps=3"),
    (re.compile(r"range\(1,\s*\d{3,}\s*\)"), "range(1, 4)"),
]


def shrink(src):
    for pat, repl in SUBS:
        src = pat.sub(repl, src)
    return src


def run_notebook(path):
    nb = nbformat.read(path, as_version=4)
    g = {"__name__": "__main__"}
    for i, cell in enumerate(nb.cells):
        if cell.cell_type != "code":
            continue
        exec(compile(shrink(cell.source), f"{path}#cell{i}", "exec"), g)


def main():
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(here)
    notebooks = sorted(glob.glob("0*_*.ipynb"))
    if not notebooks:
        print("no notebooks found"); return 1
    ok = True
    for path in notebooks:
        t0 = time.time()
        try:
            run_notebook(path)
            print(f"[OK]   {path}  ({time.time()-t0:.0f}s)")
        except Exception as e:  # noqa: BLE001 - CI wants the traceback
            import traceback
            print(f"[FAIL] {path}: {type(e).__name__}: {e}")
            traceback.print_exc()
            ok = False
    print("=== smoke test:", "PASS" if ok else "FAIL", "===")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
