"""Lets the scripts in this folder import the project's code in src/qon, and
keeps the long runs gentle on a laptop:

* each worker process uses ONE processor core (numerical libraries would
  otherwise each grab every core they can find);
* the run is marked "low priority", so anything else you do on the computer
  goes first.
"""
import os
import sys
from pathlib import Path

for var in ("VECLIB_MAXIMUM_THREADS", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(var, "1")
try:
    os.nice(10)
except OSError:
    pass
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
