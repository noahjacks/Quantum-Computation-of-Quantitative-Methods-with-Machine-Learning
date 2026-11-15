"""
STEP 8 - Combine two hardware runs and analyse them together.

    .venv/bin/python scripts/08_combine_runs.py

Run 1: results/hardware_run/hardware_features.parquet       (14 Nov 2026, 17:30 UTC)
Run 2: results/hardware_run/run2_hardware_features.parquet  (14 Nov 2026, 23:43 UTC,
       a teammate's account, same chip, same settings, different random events)

Events measured in both runs are kept from run 1 only in the main analysis,
and are used separately to check that the two runs agree.  This step was
added after both runs; it is not part of the pre-registered plan.
"""
import json
import warnings

import _path  # noqa: F401
import numpy as np
import pandas as pd

from qon import config as C
from qon.feature_sets import load_feature_sets
from qon.hardware_analysis import _decodable, analyse, decodability_test
from qon.noise import fit_classical_surrogate

warnings.filterwarnings("ignore")
RUNS = C.ROOT / "results" / "hardware_run"

if __name__ == "__main__":
    fs = load_feature_sets()
    r1 = pd.read_parquet(RUNS / "hardware_features.parquet").assign(run=1)
    r2 = pd.read_parquet(RUNS / "run2_hardware_features.parquet").assign(run=2)
    names = [c for c in r1.columns if c[0] in "XYZ" and c[1:].isdigit()]
    e1, e2 = r1[~r1["is_sentinel"]], r2[~r2["is_sentinel"]]
    both = sorted(set(e1["row"]) & set(e2["row"]))
    print(f"run 1: {len(e1)} events, run 2: {len(e2)} events, measured in both: {len(both)}")

    extra = {}
    if both:
        a = e1.set_index("row").loc[both, names].to_numpy()
        b = e2.set_index("row").loc[both, names].to_numpy()
        extra["events_in_both_runs"] = len(both)
        extra["agreement_between_runs_correlation"] = float(np.corrcoef(a.ravel(), b.ravel())[0, 1])
        extra["median_abs_difference_between_runs"] = float(np.median(np.abs(a - b)))
    r2 = r2[r2["is_sentinel"] | ~r2["row"].isin(both)].copy()
    r2["position"] += int(r1["position"].max()) + 1          # run 2 came after run 1
    combined = pd.concat([r1, r2], ignore_index=True)
    combined.to_parquet(RUNS / "combined_hardware_features.parquet")

    out = analyse(combined, fs["sets"], fs["events"], verbose=False)
    # each run repeated its OWN check circuit, so the repeated-circuit numbers
    # cannot be pooled across runs; they are in each run's own record instead
    for key in ("n_sentinels", "median_observed_std", "median_shot_noise_std", "excess_ratio",
                "median_shift_first_vs_second_half"):
        out["drift"].pop(key, None)
    print(out["table"].pivot(index="set", columns="model", values="auc").round(3).to_string())
    print(out["differences"].round(3).to_string(index=False))
    print("Drift:", {k: round(v, 4) for k, v in out["drift"].items()})

    # which RUN did an event come from, judged from the noise alone?
    ev = combined[~combined["is_sentinel"]].sort_values("row")
    rows = ev["row"].to_numpy()
    B, D = fs["sets"]["B"][rows], ev[names].to_numpy()
    sur = fit_classical_surrogate(B, D)
    resid = D - (sur.intercept + sur.slope * B)
    which = decodability_test(resid, (ev["run"].to_numpy() == 2).astype(int))
    extra["noise_reveals_which_run_auc"] = which["auc"]
    extra["which_run_auc_if_shuffled_975"] = which["shuffled_975"]
    extra["which_run_p_value"] = which["p_value"]
    day = fs["events"]["day"].to_numpy()[rows]
    extra["correlation_run_vs_date"] = float(np.corrcoef(ev["run"].to_numpy(), day)[0, 1])
    print("Two runs compared:", {k: round(v, 4) if isinstance(v, float) else v for k, v in extra.items()})

    out["table"].to_csv(C.TABLES / "partD_combined_auc.csv", index=False)
    out["differences"].to_csv(C.TABLES / "partD_combined_differences.csv", index=False)
    summary = {k: out[k] for k in ("n_events", "n_test_events", "noise", "drift")}
    summary["two_runs"] = extra
    (C.TABLES / "partD_combined_summary.json").write_text(json.dumps(summary, indent=2))
    print("saved results/tables/partD_combined_auc.csv, partD_combined_differences.csv "
          "and partD_combined_summary.json")
