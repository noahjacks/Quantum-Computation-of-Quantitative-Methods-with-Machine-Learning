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
SAVED_RUN = C.ROOT / "results" / "hardware_run" / "hardware_features.parquet"

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
        # the run writes to data/processed (not kept in git); a copy of the
        # 14 Nov 2026 run is kept in results/hardware_run so that anyone who
        # downloads the project can repeat this analysis
        source = HARDWARE_FILE if HARDWARE_FILE.exists() else SAVED_RUN
        if not source.exists():
            raise SystemExit("No hardware results yet. Run scripts/06_hardware_run.py --mode run "
                             "first, or use --rehearsal.")
        print(f"hardware results: {source.relative_to(C.ROOT)}")
        results = pd.read_parquet(source)
        tag = "hardware"
        record_file = source.with_suffix(".json")
        if record_file.exists():
            record = json.loads(record_file.read_text())
            if not record.get("complete", True):
                print(f"NOTE: the hardware run is not complete ({record.get('n_circuits_done')} of "
                      f"{record.get('n_circuits')} circuits). Analysing what has been measured; "
                      "the events are a random subset of the plan.")
    out = analyse(results, fs["sets"], fs["events"])
    out["table"].to_csv(C.TABLES / f"partD_{tag}_auc.csv", index=False)
    out["differences"].to_csv(C.TABLES / f"partD_{tag}_differences.csv", index=False)
    summary = {k: out[k] for k in ("n_events", "n_test_events", "noise", "drift")}
    (C.TABLES / f"partD_{tag}_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"saved results/tables/partD_{tag}_auc.csv, partD_{tag}_differences.csv "
          f"and partD_{tag}_summary.json")
