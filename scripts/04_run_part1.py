"""
STEP 4 - PART 1: "Is it quantum, or is it noise (or the clock)?"

    .venv/bin/python scripts/04_run_part1.py            (about 1 hour on 3 cores)
    .venv/bin/python scripts/04_run_part1.py --cores 6  (faster, hotter)

Trains the same four models on every feature set and saves every single
prediction, so that all tables and figures can be rebuilt without retraining.
The run is RESUMABLE: every finished backtest is saved immediately, and a
restart skips whatever is already done.

  main run        sets A, B, C, E, F at blinding 0 / 1 / 4 days
  drift check     the two drift sets at blinding 0 and 4
  ablation        B_shots (perfect chip, finite shots only), blinding 0
  small window    A, B, C, E, F with an HSBC-sized training window (1000)
  small angles    alpha = 0.1 circuit: B, C and the two drift sets, blinding 0
"""
import argparse
import warnings

import _path  # noqa: F401

from qon import config as C
from qon.backtest import run_grid
from qon.feature_sets import load_feature_sets

warnings.filterwarnings("ignore")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cores", type=int, default=C.N_CORES)
    args = ap.parse_args()
    C.ensure_dirs()
    fs = load_feature_sets()
    sets, events = fs["sets"], fs["events"]

    jobs = ([(s, b) for b in C.BLINDING_DAYS for s in ["A", "B", "C", "E", "F"]]
            + [(s, b) for b in (0, 4) for s in ["C_drift_chrono", "C_drift_random"]]
            + [("B_shots", 0)])
    run_grid(sets, events, jobs, tag="main", n_jobs=args.cores).to_parquet(
        C.TABLES / "part1_predictions.parquet", index=False)

    jobs = [(s, 0) for s in ["A", "B", "C", "E", "F"]]
    run_grid(sets, events, jobs, tag="train1000", train_size=C.TRAIN_SIZE_SMALL,
             n_jobs=args.cores).to_parquet(C.TABLES / "part1_predictions_train1000.parquet", index=False)

    small = load_feature_sets(alpha=0.1)["sets"]
    jobs = [(s, 0) for s in ["B", "C", "C_drift_chrono", "C_drift_random"]]
    run_grid(small, events, jobs, tag="alpha0.1", n_jobs=args.cores).to_parquet(
        C.TABLES / "part1_predictions_alpha0.1.parquet", index=False)
    print("Part 1 finished.")
