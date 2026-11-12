"""
STEP 5 - PART 2 (the extension): can a quantum KERNEL beat classical
machine learning at estimating p(fill)?

    .venv/bin/python scripts/05_run_part2_kernels.py            (about 1 hour on 3 cores)
    .venv/bin/python scripts/05_run_part2_kernels.py --cores 8  (faster, hotter)

The inputs are always the classical features; the learning algorithm is what
changes.  See src/qon/kernel_backtest.py for the list of methods.  The
expensive steps are saved as they finish, so the run can be stopped and
restarted.  (To redo the planned run from scratch, delete
results/tables/part2_predictions.parquet first.)
"""
import argparse
import warnings

import _path  # noqa: F401

from qon import config as C
from qon.kernel_backtest import run_part2, run_part2_followup

warnings.filterwarnings("ignore")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cores", type=int, default=C.N_CORES)
    args = ap.parse_args()
    C.ensure_dirs()
    if not (C.TABLES / "part2_predictions.parquet").exists():
        run_part2(n_jobs=args.cores)
    # Follow-up added after the planned results were seen: the same kernels
    # with kernel ridge regression instead of an SVM (see notebook 4, section 7).
    run_part2_followup(n_jobs=args.cores)
    print("Part 2 finished.")
