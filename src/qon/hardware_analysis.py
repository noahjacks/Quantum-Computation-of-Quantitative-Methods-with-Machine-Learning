"""
hardware_analysis.py  --  WHAT TO DO WITH THE REAL-HARDWARE RESULTS (SET D)
==========================================================================

Once scripts/06_hardware_run.py has produced measurements from a real IBM
chip, this module answers the three questions of the project pitch:

1. IS IT THE NOISE?   Build set E by shrinking and jittering the perfect-chip
   features (B) with ordinary random numbers until they statistically match
   the hardware features (D).  Train the same models on D and on E.
       E matches D  ->  the hardware effect is reproducible classically.
       D beats E    ->  the hardware adds something our classical copy lacks.

2. IS IT THE CLOCK?   The run order was random, so drift cannot line up with
   the dates.  Three numbers document it: the repeated "sentinel" circuits
   show how much the chip moved during the run; a classifier tests whether
   the noise reveals WHEN IN THE RUN a circuit was executed (i.e. whether
   there was drift at all); and the correlation between run order and event
   date confirms that the randomisation worked (it should be ~0).

3. DOES HARDWARE HELP AT ALL?   Compare D with A (classical), B (perfect
   chip) and C (noisy simulator) on exactly the same events.

Because a hardware budget of minutes only covers a few hundred events, the
comparison here is a smaller backtest than Part 1: all models are trained on
the earlier part of the hardware sample and tested on the later part, and
the score is one pooled AUC with a bootstrap interval.  With a few hundred
events only LARGE effects are detectable - which is fine, because HSBC's
claimed effect (AUC 0.63 -> 0.75-0.97) is enormous.

`make_rehearsal_file` produces a stand-in results file from the noisy
simulator so that this whole analysis can be tested before any quantum time
is spent.  Rehearsal output is always labelled as such.

ADDED AFTER THE REAL RUN (14 Nov 2026; recorded in PREREGISTRATION.md
section 10).  The planned outputs above are unchanged.  Three things were
added to state the result precisely:
  * `paired_differences`  the difference between D and every other set with
    its own 95% interval (the planned table only gave each set's interval);
  * `decodability_test`   whether "the noise reveals the run position" could
    be luck, by repeating the test with shuffled labels;
  * a warning check: the chip's distortion depends on the inputs, and our
    inputs contain the date, so a naive "does the noise reveal the DATE?"
    test says yes even though the run order was random.  That is not the
    clock; it is reported so that nobody mistakes it for one.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from . import config as C
from .backtest import run_one
from .circuit import output_names
from .hardware import build_run_plan, sentinel_drift
from .noise import fit_classical_surrogate

REHEARSAL_FILE = C.DATA_PROCESSED / "hardware_features_REHEARSAL.parquet"


def make_rehearsal_file(sets: dict, events: pd.DataFrame, n_events: int = 600,
                        shots: int = C.SHOTS, seed: int = 0) -> pd.DataFrame:
    """A fake 'hardware' results file built from the simulated drifting chip
    (random run order).  For testing the analysis only."""
    rng = np.random.default_rng(seed)
    plan = build_run_plan(events, n_events)
    names = output_names(sets["B"].shape[1] // 3)
    source = sets["C_drift_random"]
    sentinel_row = int(plan.loc[~plan["is_sentinel"], "row"].iloc[0])
    rows = np.where(plan["row"].to_numpy() >= 0, plan["row"].to_numpy(), sentinel_row)
    values = source[rows].copy()
    is_s = plan["is_sentinel"].to_numpy()
    # sentinels: the same circuit measured again -> fresh shot noise each time
    sd = np.sqrt(np.clip(1.0 - values[is_s] ** 2, 0.0, None) / shots)
    values[is_s] += sd * rng.standard_normal(values[is_s].shape)
    out = pd.concat([plan.reset_index(drop=True), pd.DataFrame(values, columns=names)], axis=1)
    out["shots"] = shots
    out["job_id"] = "REHEARSAL (simulated, not hardware)"
    out.to_parquet(REHEARSAL_FILE, index=False)
    return out


def pooled_auc(y: np.ndarray, p: np.ndarray, day: np.ndarray, n_boot: int = 1000,
               seed: int = C.BOOTSTRAP_SEED) -> tuple[float, float, float]:
    """One AUC over all test events, with a 95% interval from re-drawing
    whole test days (events of one day are not independent)."""
    rng = np.random.default_rng(seed)
    days = np.unique(day)
    groups = [np.where(day == d)[0] for d in days]
    boots = []
    for _ in range(n_boot):
        idx = np.concatenate([groups[g] for g in rng.integers(0, len(days), len(days))])
        if 0 < y[idx].sum() < len(idx):
            boots.append(roc_auc_score(y[idx], p[idx]))
    return float(roc_auc_score(y, p)), float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975))


def paired_differences(y: np.ndarray, day: np.ndarray, preds: dict, reference: str = "D",
                       n_boot: int = 2000, seed: int = C.BOOTSTRAP_SEED) -> pd.DataFrame:
    """AUC of `reference` minus AUC of every other set, on the same test
    events, with a 95% interval.

    Whole test days are re-drawn, and the SAME re-drawn days are used for
    both sets.  The interval therefore describes the difference itself, which
    is much tighter than comparing two separate intervals by eye.  A
    difference is only "real" if its interval excludes zero.

    preds : set name -> predictions for the test events (same order as y)."""
    rng = np.random.default_rng(seed)
    days = np.unique(day)
    groups = [np.where(day == d)[0] for d in days]
    boots = {k: [] for k in preds}
    for _ in range(n_boot):
        idx = np.concatenate([groups[g] for g in rng.integers(0, len(days), len(days))])
        if 0 < y[idx].sum() < len(idx):
            for k, p in preds.items():
                boots[k].append(roc_auc_score(y[idx], p[idx]))
    boots = {k: np.asarray(v) for k, v in boots.items()}
    base = roc_auc_score(y, preds[reference])
    rows = []
    for k in preds:
        if k == reference:
            continue
        d = boots[reference] - boots[k]
        lo, hi = float(np.quantile(d, 0.025)), float(np.quantile(d, 0.975))
        rows.append({"comparison": f"{reference} minus {k}",
                     "difference": float(base - roc_auc_score(y, preds[k])),
                     "ci_low": lo, "ci_high": hi, "significant": bool(lo > 0 or hi < 0)})
    return pd.DataFrame(rows)


def _decodable(features: np.ndarray, labels: np.ndarray) -> float:
    """Can a simple model tell the two groups in `labels` apart from
    `features`?  Returns an AUC on held-out events (0.5 = no)."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import KFold, cross_val_predict
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    model = make_pipeline(StandardScaler(), LogisticRegression(C=0.01, max_iter=500))
    p = cross_val_predict(model, features, labels, cv=KFold(5, shuffle=True, random_state=0),
                          method="predict_proba")[:, 1]
    return float(roc_auc_score(labels, p))


def decodability_test(features: np.ndarray, labels: np.ndarray, n_shuffles: int = 300,
                      seed: int = 0) -> dict:
    """`_decodable`, plus what the same test gives when the labels are
    shuffled (so there is nothing to find).  p_value = how often shuffled
    labels do at least as well as the real ones."""
    rng = np.random.default_rng(seed)
    observed = _decodable(features, labels)
    shuffled = np.array([_decodable(features, rng.permutation(labels)) for _ in range(n_shuffles)])
    return {"auc": observed, "shuffled_mean": float(shuffled.mean()),
            "shuffled_975": float(np.quantile(shuffled, 0.975)),
            "p_value": float((1 + (shuffled >= observed).sum()) / (1 + n_shuffles))}


def analyse(results: pd.DataFrame, sets: dict, events: pd.DataFrame,
            train_fraction: float = 0.5, verbose: bool = True) -> dict:
    """Run the full set-D analysis on a hardware (or rehearsal) results file."""
    names = [c for c in results.columns if c[0] in "XYZ" and c[1:].isdigit()]
    shots = int(results["shots"].iloc[0])
    ev = results[~results["is_sentinel"]].sort_values("row")
    rows = ev["row"].to_numpy()
    D = ev[names].to_numpy()
    B = sets["B"][rows]
    y = events["filled"].to_numpy()[rows]
    day = events["day"].to_numpy()[rows]

    # ---- 1. how does the hardware distort the perfect-chip values? ----------
    n_fit = int(len(rows) * train_fraction)                # earliest events only
    surrogate = fit_classical_surrogate(B[:n_fit], D[:n_fit])
    E = surrogate.apply(B, seed=C.NOISE_SEED + 7)
    noise = {"median_shrink": float(np.median(surrogate.slope)),
             "median_jitter": float(np.median(surrogate.sigma)),
             "shot_noise_alone": float(np.median(np.sqrt(np.clip(1 - D.mean(0) ** 2, 0, None) / shots))),
             "correlation_with_perfect_chip": float(np.corrcoef(D.ravel(), B.ravel())[0, 1])}

    # ---- 2. the same models on every feature set, same events ---------------
    feature_sets = {"A": sets["A"][rows], "B": B, "C": sets["C"][rows], "D": D,
                    "E": E, "F": sets["F"][rows]}
    train_size = n_fit
    counts_before = np.array([(day < d).sum() for d in np.unique(day)])
    test_days = np.unique(day)[counts_before >= train_size]
    table = []
    preds = {}
    average = {}                                             # set -> mean of the four models
    for name, X in feature_sets.items():
        res = run_one(X, y, day, test_days, blinding=0, train_size=train_size,
                      retune_every=10 ** 6, label=name)      # tune once, on the first window
        preds[name] = res
        for model, g in res.groupby("model"):
            auc, lo, hi = pooled_auc(y[g["row"].to_numpy()], g["p"].to_numpy(), g["day"].to_numpy())
            table.append({"set": name, "model": model, "auc": auc, "ci_low": lo, "ci_high": hi})
        g = res.groupby("row")["p"].mean()                   # average of the four models
        average[name] = g
        d_of = res.drop_duplicates("row").set_index("row")["day"]
        auc, lo, hi = pooled_auc(y[g.index.to_numpy()], g.to_numpy(), d_of.loc[g.index].to_numpy())
        table.append({"set": name, "model": "All", "auc": auc, "ci_low": lo, "ci_high": hi})
    table = pd.DataFrame(table)

    # ---- 2b. is D really different from the others?  (added after the run) ----
    test_rows = average["A"].index.to_numpy()
    assert all((average[k].index.to_numpy() == test_rows).all() for k in average)
    differences = paired_differences(y[test_rows], day[test_rows],
                                     {k: average[k].to_numpy() for k in average})

    # ---- 3. drift -------------------------------------------------------------
    drift = sentinel_drift(results, shots)
    # (a) the randomisation worked: run order is unrelated to the events' dates
    position = ev["position"].to_numpy()
    drift["correlation_run_order_vs_date"] = float(np.corrcoef(position, day)[0, 1])
    # (b) did the chip drift measurably?  Ask a classifier to tell, from the
    #     noise alone, whether an event ran in the first or second half of the
    #     run.  0.5 = no detectable drift; well above 0.5 = the chip moved.
    #     Because of (a), such drift still cannot reveal the DATE of an event.
    resid = D - (surrogate.intercept + surrogate.slope * B)
    late_run = (position > np.median(position)).astype(int)
    position_test = decodability_test(resid, late_run)
    drift["noise_reveals_run_position_auc"] = position_test["auc"]
    #     (added after the run) could that be luck?  The same test on shuffled labels:
    drift["run_position_auc_if_shuffled_975"] = position_test["shuffled_975"]
    drift["run_position_p_value"] = position_test["p_value"]
    # (c) (added after the run) A WARNING, NOT A CLOCK.  What is left of the
    #     hardware values after the best straight-line fit to the perfect values
    #     still depends on the inputs, and our inputs contain the date.  So the
    #     leftovers "reveal" early versus late DATES even though the run order
    #     was random.  The perfect-chip values do the same, far more strongly,
    #     with no hardware involved at all.
    late_date = (day > np.median(day)).astype(int)
    drift["leftover_reveals_date_auc"] = _decodable(resid, late_date)
    drift["perfect_chip_values_reveal_date_auc"] = _decodable(B, late_date)

    out = {"n_events": int(len(rows)), "n_test_events": int(len(preds["A"]) // 4),
           "noise": noise, "table": table, "differences": differences, "drift": drift}
    if verbose:
        wide = table.pivot(index="set", columns="model", values="auc").round(3)
        print(f"{len(rows)} hardware events, {out['n_test_events']} of them used for testing")
        print("How the chip distorts the perfect values:", {k: round(v, 4) for k, v in noise.items()})
        print(wide[["All", "LR", "RF", "XGB", "NN"]].to_string())
        print("Is D different from the others? (difference in AUC of the model average, 95% interval)")
        for r in differences.itertuples():
            print(f"   {r.comparison}: {r.difference:+.3f}  ({r.ci_low:+.3f} to {r.ci_high:+.3f})"
                  f"{'  <- real difference' if r.significant else ''}")
        print("Drift:", {k: (round(v, 4) if isinstance(v, float) else v) for k, v in drift.items()})
    return out
