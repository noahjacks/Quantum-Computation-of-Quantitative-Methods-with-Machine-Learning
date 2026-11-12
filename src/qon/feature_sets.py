"""
feature_sets.py  --  BUILD THE SIX COMPETING SETS OF MODEL INPUTS
=================================================================

This is the heart of Part 1.  Every set describes the SAME quote requests;
only the way the description is computed differs.  The same models are then
trained on each set, so any difference in score is caused by the inputs.

    A  the classical features, just scaled               "the baseline to beat"
    B  A pushed through the circuit on a PERFECT
       quantum computer (exact simulation)               "quantum, zero noise"
    C  the same circuit on a NOISY simulated chip
       (gate errors, readout errors, 4096 shots)         "does noise help?"
    D  the same circuit on REAL IBM hardware             "HSBC's condition"
       (produced by scripts/06_hardware_run.py; this
       file picks it up automatically if it exists)
    E  B, shrunk and jittered with ordinary random
       numbers so that it statistically matches the
       noisy set (D if present, otherwise C)             "is it just classical noise?"
    F  random classical non-linear features, same size   "is it just more features?"

Plus the drift check ("is it the clock?"): the noisy chip with error rates
that wander over time, once with the events sent in TIME ORDER (what a naive
pipeline does) and once in RANDOM order (our protocol):

    C_drift_chrono , C_drift_random

and one ablation, B_shots = B plus shot noise only.

Events from the calibration period are included as rows too (they are needed
as training history for the first test days) but are flagged, and nothing is
ever scored on them.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from . import config as C
from .circuit import make_spec, output_names
from .dataset import FEATURE_COLUMNS, load_events
from .noise import (batch_of_events, drifting_gate_errors, fit_classical_surrogate,
                    make_noise_params, measure, random_fourier_features)
from .preprocessing import AngleScaler
from .simulator import simulate_noiseless, simulate_noisy

SETS_FILE = C.DATA_PROCESSED / "feature_sets.npz"
EVENTS_ALL_FILE = C.DATA_PROCESSED / "events_all.parquet"
HARDWARE_FILE = C.DATA_PROCESSED / "hardware_features.parquet"   # written by the hardware script


def combine_events(events: dict | None = None) -> pd.DataFrame:
    """Calibration events followed by main events, in time order.
    `row` is the position in this table; `day` counts trading days from the
    start of the calibration period."""
    ev = events if events is not None else load_events()
    cal = ev["calibration"].copy()
    main = ev["main"].copy()
    cal["period"] = "calibration"
    main["period"] = "main"
    cal["day"] = cal["day_index"]
    main["day"] = cal["day_index"].max() + 1 + main["day_index"]
    out = pd.concat([cal, main], ignore_index=True)
    out.insert(0, "row", np.arange(len(out)))
    return out


def build_feature_sets(alpha: float = C.ALPHA, n_jobs: int = 8, verbose: bool = True) -> dict:
    """Compute every feature set for every event and save them to disk."""
    events = combine_events()
    is_cal = (events["period"] == "calibration").to_numpy()
    X_raw = events[FEATURE_COLUMNS].to_numpy(dtype=float)
    n = len(events)

    # ---- scaling (fitted on the calibration period only) -------------------
    macro = load_events()["macro_history"]
    history = {FEATURE_COLUMNS.index(col): macro[col].to_numpy() for col in macro.columns}
    scaler = AngleScaler.fit(X_raw[is_cal], history=history)
    Z = scaler.zscore(X_raw)                     # what classical models see
    angles = scaler.angles(X_raw)                # what the circuit sees

    # ---- the circuit --------------------------------------------------------
    spec = make_spec(n_features=X_raw.shape[1], alpha=alpha)
    phi = spec.gate_angles(angles)
    n_pairs = spec.n_qubits - 1
    sets: dict[str, np.ndarray] = {"A": Z}

    if verbose:
        print(f"Simulating {n:,} circuits on {spec.n_qubits} qubits (alpha={alpha}) ...", flush=True)
    B = simulate_noiseless(spec, phi)
    sets["B"] = B

    # ---- the simulated chip -------------------------------------------------
    params = make_noise_params(spec)
    rng = np.random.default_rng(C.NOISE_SEED + 1)
    random_order = rng.permutation(n)            # our protocol: random run order
    chrono_order = np.arange(n)                  # naive: oldest event first
    batch_random = batch_of_events(random_order)
    batch_chrono = batch_of_events(chrono_order)

    exact_noisy = simulate_noisy(spec, phi, params.lam_slot, n_jobs=n_jobs)
    sets["C"] = measure(exact_noisy, params, batch_random, seed=C.NOISE_SEED + 2)

    # ablation: perfect gates and readout, finite shots only
    shots_only = make_noise_params(spec, scale=0.0)
    sets["B_shots"] = measure(B, shots_only, batch_random, seed=C.NOISE_SEED + 3, trex=False)

    # ---- drift: same wandering chip, two different run orders ---------------
    for name, batch in (("C_drift_chrono", batch_chrono), ("C_drift_random", batch_random)):
        lam = drifting_gate_errors(params, batch, n_pairs)
        exact = simulate_noisy(spec, phi, lam, n_jobs=n_jobs)
        sets[name] = measure(exact, params, batch, seed=C.NOISE_SEED + 4)
        if verbose:
            print(f"  built {name}", flush=True)

    # ---- F: random classical features ---------------------------------------
    sets["F"] = random_fourier_features(Z, Z[is_cal], spec.n_outputs, seed=C.NOISE_SEED + 5)

    # ---- D (real hardware, if the team has run it) and E ---------------------
    info = {"alpha": alpha, "n_events": n, "n_qubits": spec.n_qubits,
            "n_inputs": int(X_raw.shape[1]), "n_outputs": spec.n_outputs,
            "has_hardware": False, "E_matched_to": "C"}
    surrogate = fit_classical_surrogate(B[is_cal], sets["C"][is_cal])
    sets["E"] = surrogate.apply(B, seed=C.NOISE_SEED + 6)
    info["median_shrink_C_vs_B"] = float(np.median(surrogate.slope))
    info["median_jitter_C_vs_B"] = float(np.median(surrogate.sigma))
    info["median_signal_std_B"] = float(np.median(B.std(axis=0)))

    np.savez_compressed(SETS_FILE if alpha == C.ALPHA else _alt_file(alpha),
                        **sets, batch_random=batch_random, batch_chrono=batch_chrono,
                        random_order=random_order, phi=phi.astype(np.float32),
                        output_names=np.array(output_names(spec.n_qubits)))
    if alpha == C.ALPHA:
        events.drop(columns=FEATURE_COLUMNS).to_parquet(EVENTS_ALL_FILE, index=False)
        (C.DATA_PROCESSED / "feature_sets_meta.json").write_text(json.dumps(info, indent=2))
    if verbose:
        print(json.dumps(info, indent=2))
        for k, v in sets.items():
            print(f"  set {k:<15} shape {v.shape}")
    return {"sets": sets, "events": events, "info": info}


def _alt_file(alpha: float):
    return C.DATA_PROCESSED / f"feature_sets_alpha{alpha:g}.npz"


def load_feature_sets(alpha: float = C.ALPHA) -> dict:
    """Load what build_feature_sets() saved: {"sets": {...}, "events": table}."""
    path = SETS_FILE if alpha == C.ALPHA else _alt_file(alpha)
    with np.load(path, allow_pickle=False) as data:
        keys = [k for k in data.files
                if k not in ("batch_random", "batch_chrono", "random_order", "phi", "output_names")]
        sets = {k: data[k] for k in keys}
        extra = {k: data[k] for k in ("batch_random", "batch_chrono", "random_order")}
    events = pd.read_parquet(EVENTS_ALL_FILE)
    return {"sets": sets, "events": events, **extra}


if __name__ == "__main__":
    build_feature_sets()
