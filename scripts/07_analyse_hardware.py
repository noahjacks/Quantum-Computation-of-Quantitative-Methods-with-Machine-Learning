"""
STEP 7 - Analyse the real-hardware results (feature set D).

    .venv/bin/python scripts/07_analyse_hardware.py              (after step 6, mode run)
    .venv/bin/python scripts/07_analyse_hardware.py --rehearsal  (no hardware needed)

--rehearsal builds a stand-in results file from the noisy simulator and runs
exactly the same analysis on it.  It exists to prove the analysis works
before any quantum time is spent; its numbers say nothing about real
hardware and every output is labelled REHEARSAL.

See src/qon/hardware_analysis.py for what is computed and how to read it.
"""
import argparse
import json
import warnings

import _path  # noqa: F401
import pandas as pd

from qon import config as C
from qon.feature_sets import HARDWARE_FILE, load_feature_sets
from qon.hardware_analysis import REHEARSAL_FILE, analyse, make_rehearsal_file

warnings.filterwarnings("ignore")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--rehearsal", action="store_true")
    ap.add_argument("--events", type=int, default=600, help="rehearsal only: number of events")
    args = ap.parse_args()
    fs = load_feature_sets()
    if args.rehearsal:
        print("REHEARSAL: simulated stand-in data, NOT real hardware.")
        results = make_rehearsal_file(fs["sets"], fs["events"], n_events=args.events)
        tag = "REHEARSAL"
    else:
        if not HARDWARE_FILE.exists():
            raise SystemExit("No hardware results yet. Run scripts/06_hardware_run.py --mode run "
                             "first, or use --rehearsal.")
        results = pd.read_parquet(HARDWARE_FILE)
        tag = "hardware"
    out = analyse(results, fs["sets"], fs["events"])
    out["table"].to_csv(C.TABLES / f"partD_{tag}_auc.csv", index=False)
    summary = {k: out[k] for k in ("n_events", "n_test_events", "noise", "drift")}
    (C.TABLES / f"partD_{tag}_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"saved results/tables/partD_{tag}_auc.csv and partD_{tag}_summary.json")
