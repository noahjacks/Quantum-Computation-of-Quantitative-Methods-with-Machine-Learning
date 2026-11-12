"""
kernel_backtest.py  --  PART 2: THE LEARNING ALGORITHM IS THE VARIABLE
======================================================================

Every method below is given the SAME information - the 98 classical inputs
of each quote request - and the SAME rolling train/test protocol as Part 1.
Only the algorithm that turns inputs into p(fill) differs:

  classical machine learning (the four models of Part 1)
      LR, RF, XGB, NN

  classical kernel machines
      SVM-linear        similarity = dot product
      SVM-RBF           similarity = Gaussian of the distance

  quantum kernel machines
      SVM-PQK           projected quantum kernel, perfect quantum computer
      SVM-PQK-noisy     the same on the noisy simulated chip (4096 shots)
      SVM-FQK           fidelity quantum kernel, perfect quantum computer,
                        with the bandwidth alpha chosen on calibration data
      SVM-FQK-shots     the same, each overlap estimated from 4096 shots

"Advantage" would mean: a quantum-kernel row scores clearly above every
classical row.

Because kernel machines compare all pairs of events, Part 2 uses an evenly
thinned sample (60 events per day) and a training window of 800 events.
The four classical models are re-run on exactly the same sample so that the
comparison is like for like.
"""
from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from . import config as C
from .backtest import check_no_overlap, run_one, training_rows
from .circuit import make_spec
from .kernels import (C_GRID, GAMMA_MULTIPLIERS, build_mps, concentration,
                      fidelity_kernel_band, fidelity_kernel_cross, geometric_difference,
                      kernel_target_alignment, make_folds, ridge_scores, squared_distances,
                      svm_fit_predict, take, tune_ridge, tune_svm)
from .preprocessing import AngleScaler

DISTANCE_METHODS = {"SVM-RBF": "A", "SVM-PQK": "B", "SVM-PQK-noisy": "C"}


def kernel_subsample(events: pd.DataFrame) -> np.ndarray:
    """Rows (in time order) of the thinned sample: the same number of events
    from every day, chosen at random with a fixed seed."""
    rng = np.random.default_rng(C.KERNEL_SEED)
    keep = []
    for _, rows in events.groupby("day")["row"]:
        rows = rows.to_numpy()
        keep.append(np.sort(rng.choice(rows, size=min(C.KERNEL_EVENTS_PER_DAY, len(rows)),
                                       replace=False)))
    return np.concatenate(keep)


def scaled_angles() -> np.ndarray:
    """The circuit's input angles (before multiplying by alpha) for all events."""
    from .dataset import FEATURE_COLUMNS, load_events
    from .feature_sets import combine_events
    ev = load_events()
    table = combine_events(ev)
    X = table[FEATURE_COLUMNS].to_numpy(dtype=float)
    is_cal = (table["period"] == "calibration").to_numpy()
    history = {FEATURE_COLUMNS.index(c): ev["macro_history"][c].to_numpy()
               for c in ev["macro_history"].columns}
    return AngleScaler.fit(X[is_cal], history=history).angles(X)


def _median_gamma_from_d2(D2: np.ndarray) -> float:
    return 1.0 / float(np.median(D2[np.triu_indices(len(D2), k=1)]))


# ---------------------------------------------------------------------------
# step 1: choose the fidelity kernel's bandwidth on calibration data only
# ---------------------------------------------------------------------------
def screen_fqk_alpha(angles_sub: np.ndarray, y_sub: np.ndarray, day_sub: np.ndarray,
                     is_cal_sub: np.ndarray, alphas=C.FQK_ALPHAS, n_jobs: int = 8,
                     verbose: bool = True) -> pd.DataFrame:
    """For each alpha: train an SVM on 800 calibration events, validate on the
    last 10 calibration days (with a 2-day gap in between).  No event of the
    test window is touched."""
    last_cal_day = int(day_sub[is_cal_sub].max())
    val = np.where(is_cal_sub & (day_sub > last_cal_day - 10))[0]
    train = np.where(is_cal_sub & (day_sub <= last_cal_day - 12))[0][-C.KERNEL_TRAIN_SIZE:]
    folds = make_folds(len(train), C.CV_FOLDS, C.KERNEL_SEED)
    rows = []
    for alpha in alphas:
        t0 = time.time()
        spec = make_spec(angles_sub.shape[1], alpha=alpha)
        mps = build_mps(spec, spec.gate_angles(angles_sub[np.r_[train, val]]))
        tr, va = take(mps, slice(0, len(train))), take(mps, slice(len(train), None))
        K_train = fidelity_kernel_cross(tr, tr, n_jobs=n_jobs).astype(float)
        K_val = fidelity_kernel_cross(va, tr, n_jobs=n_jobs).astype(float)
        _, c_best, cv_auc = tune_svm({"fqk": K_train}, y_sub[train], folds)
        p = svm_fit_predict(K_train, y_sub[train], K_val, c_best)
        stats = concentration(K_train)
        rows.append({"alpha": alpha, "validation_auc": roc_auc_score(y_sub[val], p),
                     "cv_auc": cv_auc, "best_C": c_best,
                     "median_similarity": stats["median"], "p99_similarity": stats["p99"],
                     "alignment": kernel_target_alignment(K_train, y_sub[train])})
        if verbose:
            print(f"  alpha={alpha:<5} validation AUC {rows[-1]['validation_auc']:.3f} "
                  f"median similarity {stats['median']:.2e}  ({time.time() - t0:.0f}s)", flush=True)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# step 2: the rolling backtest for all kernel machines
# ---------------------------------------------------------------------------
def run_kernel_methods(D2: dict[str, np.ndarray], K_fixed: dict[str, np.ndarray],
                       Z_sub: np.ndarray, y: np.ndarray, day: np.ndarray,
                       test_days: np.ndarray, blinding: int) -> pd.DataFrame:
    """D2[name]     full squared-distance matrix of the thinned sample for
                    each distance-based kernel (RBF, PQK, PQK-noisy)
       K_fixed[name] ready-made kernel matrices (fidelity kernels)
       Returns one row per (test event, method) with p(fill)."""
    out = []
    best: dict[str, tuple] = {}
    for k, d in enumerate(test_days):
        train = training_rows(day, int(d), blinding, C.KERNEL_TRAIN_SIZE)
        check_no_overlap(day, train, int(d), blinding)
        test = np.where(day == d)[0]
        ytr = y[train]
        retune = k % C.RETUNE_EVERY == 0
        folds = make_folds(len(train), C.CV_FOLDS, C.KERNEL_SEED + k)
        preds = {}

        # -- linear kernel on the classical inputs
        K_lin = Z_sub[train] @ Z_sub[train].T / Z_sub.shape[1]
        if retune:
            best["SVM-linear"] = tune_svm({"linear": K_lin}, ytr, folds)[:2]
        preds["SVM-linear"] = svm_fit_predict(K_lin, ytr, Z_sub[test] @ Z_sub[train].T / Z_sub.shape[1],
                                              best["SVM-linear"][1])

        # -- Gaussian kernels on distances (classical RBF and the projected quantum kernels)
        for name, mat in D2.items():
            d_tr, d_te = mat[np.ix_(train, train)], mat[np.ix_(test, train)]
            g0 = _median_gamma_from_d2(d_tr)
            if retune:
                best[name] = tune_svm({m: np.exp(-m * g0 * d_tr) for m in GAMMA_MULTIPLIERS},
                                      ytr, folds)[:2]
            mult, c_value = best[name]
            preds[name] = svm_fit_predict(np.exp(-mult * g0 * d_tr), ytr,
                                          np.exp(-mult * g0 * d_te), c_value)

        # -- ready-made kernels (fidelity quantum kernels)
        for name, K in K_fixed.items():
            k_tr = K[np.ix_(train, train)].astype(float)
            k_te = K[np.ix_(test, train)].astype(float)
            if retune:
                best[name] = tune_svm({"fixed": k_tr}, ytr, folds)[:2]
            preds[name] = svm_fit_predict(k_tr, ytr, k_te, best[name][1])

        for name, p in preds.items():
            out.append(pd.DataFrame({"pos": test, "day": int(d), "model": name, "p": p}))
    res = pd.concat(out, ignore_index=True)
    res["blinding"] = blinding
    return res


def run_part2(n_jobs: int = C.N_CORES, verbose: bool = True) -> dict:
    from joblib import Parallel, delayed
    from .feature_sets import load_feature_sets

    fs = load_feature_sets()
    events, sets = fs["events"], fs["sets"]
    sub = kernel_subsample(events)                       # global rows of the thinned sample
    ev = events.iloc[sub].reset_index(drop=True)
    y, day = ev["filled"].to_numpy(), ev["day"].to_numpy()
    is_cal = (ev["period"] == "calibration").to_numpy()
    test_days = np.unique(day[~is_cal])
    angles = scaled_angles()[sub]
    Z = sets["A"][sub]
    if verbose:
        print(f"Part 2 sample: {len(sub):,} events ({is_cal.sum():,} calibration, "
              f"{(~is_cal).sum():,} test window), {len(test_days)} test days", flush=True)

    # ---- 1. bandwidth of the fidelity kernel, chosen on calibration data ----
    # (each expensive step is saved, so a stopped run picks up where it left off)
    screen_file = C.TABLES / "part2_fqk_alpha_screen.csv"
    if screen_file.exists():
        screen = pd.read_csv(screen_file)
    else:
        screen = screen_fqk_alpha(angles, y, day, is_cal, n_jobs=n_jobs, verbose=verbose)
        screen.to_csv(screen_file, index=False)
    alpha_star = float(screen.sort_values("validation_auc", ascending=False)["alpha"].iloc[0])

    # ---- 2. kernel ingredients for the whole thinned sample -----------------
    width = C.KERNEL_TRAIN_SIZE + C.KERNEL_EVENTS_PER_DAY * (max(C.KERNEL_BLINDING_DAYS) + 2)
    kernel_file = C.DATA_PROCESSED / f"fqk_kernel_alpha{alpha_star:g}.npy"
    if kernel_file.exists():
        K_fqk = np.load(kernel_file)
    else:
        t0 = time.time()
        spec = make_spec(angles.shape[1], alpha=alpha_star)
        mps = build_mps(spec, spec.gate_angles(angles))
        K_fqk = fidelity_kernel_band(mps, width=width, n_jobs=n_jobs)
        del mps
        np.save(kernel_file, K_fqk)
        if verbose:
            print(f"  fidelity kernel (alpha={alpha_star}) for {len(sub):,} events: "
                  f"{(time.time() - t0) / 60:.1f} min", flush=True)
    assert K_fqk.shape == (len(sub), len(sub))
    # what a real device would return: each overlap estimated from SHOTS runs
    rng = np.random.default_rng(C.KERNEL_SEED + 1)
    noisy = rng.binomial(C.SHOTS, np.clip(np.triu(K_fqk, 1), 0, 1)) / C.SHOTS
    K_fqk_shots = (noisy + noisy.T + np.eye(len(sub))).astype(np.float32)
    K_fixed = {"SVM-FQK": K_fqk, "SVM-FQK-shots": K_fqk_shots}
    D2 = {name: squared_distances(sets[s][sub], sets[s][sub]).astype(np.float32)
          for name, s in DISTANCE_METHODS.items()}

    # ---- 3. rolling backtests ------------------------------------------------
    t0 = time.time()
    kernel_res = Parallel(n_jobs=min(n_jobs, len(C.KERNEL_BLINDING_DAYS)))(
        delayed(run_kernel_methods)(D2, K_fixed, Z, y, day, test_days, b)
        for b in C.KERNEL_BLINDING_DAYS)
    ml_res = Parallel(n_jobs=min(n_jobs, len(C.KERNEL_BLINDING_DAYS)))(
        delayed(run_one)(Z, y, day, test_days, b, C.KERNEL_TRAIN_SIZE)
        for b in C.KERNEL_BLINDING_DAYS)
    ml = pd.concat(ml_res, ignore_index=True).rename(columns={"row": "pos"}).drop(columns="set")
    res = pd.concat([*kernel_res, ml], ignore_index=True)
    res["row"] = sub[res["pos"].to_numpy()]
    res["filled"] = y[res["pos"].to_numpy()]
    res = res.drop(columns="pos")
    res.to_parquet(C.TABLES / "part2_predictions.parquet", index=False)
    if verbose:
        print(f"  backtests: {(time.time() - t0) / 60:.1f} min", flush=True)

    # ---- 4. diagnostics on the last pre-window training set ------------------
    train = training_rows(day, int(test_days[0]), 0, C.KERNEL_TRAIN_SIZE)
    assert is_cal[train].all()
    kernels = {"SVM-FQK": K_fqk[np.ix_(train, train)].astype(float)}
    for name, mat in D2.items():
        d_tr = mat[np.ix_(train, train)].astype(float)
        kernels[name] = np.exp(-_median_gamma_from_d2(d_tr) * d_tr)
    spec1 = make_spec(angles.shape[1], alpha=1.0)
    m1 = build_mps(spec1, spec1.gate_angles(angles[train]))
    kernels["SVM-FQK (alpha=1)"] = fidelity_kernel_cross(m1, m1, n_jobs=n_jobs).astype(float)
    diag = []
    for name, K in kernels.items():
        row = {"kernel": name, **concentration(K),
               "alignment": kernel_target_alignment(K, y[train])}
        if name != "SVM-RBF":
            row["geometric_difference_vs_RBF"] = geometric_difference(kernels["SVM-RBF"], K)
        diag.append(row)
    diag = pd.DataFrame(diag)
    diag.to_csv(C.TABLES / "part2_kernel_diagnostics.csv", index=False)
    meta = {"alpha_star": alpha_star, "n_events": int(len(sub)), "train_size": C.KERNEL_TRAIN_SIZE,
            "sqrt_n_train": float(np.sqrt(len(train))), "band_width": int(width)}
    (C.TABLES / "part2_meta.json").write_text(json.dumps(meta, indent=2))
    if verbose:
        print(screen.round(4).to_string(index=False))
        print(diag.round(4).to_string(index=False))
    return {"predictions": res, "screen": screen, "diagnostics": diag, "meta": meta}


# ---------------------------------------------------------------------------
# FOLLOW-UP (added after the planned results were seen - see notebook 4)
# ---------------------------------------------------------------------------
def run_kernel_ridge(D2: dict[str, np.ndarray], K_fixed: dict[str, np.ndarray],
                     Z_sub: np.ndarray, y: np.ndarray, day: np.ndarray,
                     test_days: np.ndarray, blinding: int) -> pd.DataFrame:
    """The same rolling backtest and the same kernels as run_kernel_methods,
    with kernel ridge regression in place of the SVM."""
    out = []
    best: dict[str, tuple] = {}
    for k, d in enumerate(test_days):
        train = training_rows(day, int(d), blinding, C.KERNEL_TRAIN_SIZE)
        check_no_overlap(day, train, int(d), blinding)
        test = np.where(day == d)[0]
        ytr = y[train].astype(float)
        retune = k % C.RETUNE_EVERY == 0
        folds = make_folds(len(train), C.CV_FOLDS, C.KERNEL_SEED + k)
        variants = {"KRR-linear": ({"linear": Z_sub[train] @ Z_sub[train].T / Z_sub.shape[1]},
                                   {"linear": Z_sub[test] @ Z_sub[train].T / Z_sub.shape[1]})}
        for name, mat in D2.items():
            d_tr, d_te = mat[np.ix_(train, train)].astype(float), mat[np.ix_(test, train)].astype(float)
            g0 = _median_gamma_from_d2(d_tr)
            variants[name.replace("SVM", "KRR")] = (
                {m: np.exp(-m * g0 * d_tr) for m in GAMMA_MULTIPLIERS},
                {m: np.exp(-m * g0 * d_te) for m in GAMMA_MULTIPLIERS})
        for name, K in K_fixed.items():
            variants[name.replace("SVM", "KRR")] = ({"fixed": K[np.ix_(train, train)].astype(float)},
                                                    {"fixed": K[np.ix_(test, train)].astype(float)})
        for name, (k_train, k_test) in variants.items():
            if retune:
                best[name] = tune_ridge(k_train, ytr, folds)[:2]
            label, lam = best[name]
            p = np.clip(ridge_scores(k_train[label], ytr, k_test[label], lam), 0.01, 0.99)
            out.append(pd.DataFrame({"pos": test, "day": int(d), "model": name, "p": p}))
    res = pd.concat(out, ignore_index=True)
    res["blinding"] = blinding
    return res


def run_part2_followup(n_jobs: int = C.N_CORES, verbose: bool = True) -> pd.DataFrame:
    """Kernel ridge regression with every kernel of Part 2.  Re-uses the saved
    fidelity kernel, so it takes a minute or two."""
    from joblib import Parallel, delayed
    from .feature_sets import load_feature_sets

    fs = load_feature_sets()
    events, sets = fs["events"], fs["sets"]
    sub = kernel_subsample(events)
    ev = events.iloc[sub].reset_index(drop=True)
    y, day = ev["filled"].to_numpy(), ev["day"].to_numpy()
    test_days = np.unique(day[(ev["period"] == "main").to_numpy()])
    meta = json.loads((C.TABLES / "part2_meta.json").read_text())
    K_fqk = np.load(C.DATA_PROCESSED / f"fqk_kernel_alpha{meta['alpha_star']:g}.npy")
    rng = np.random.default_rng(C.KERNEL_SEED + 1)        # the same noisy kernel as the main run
    noisy = rng.binomial(C.SHOTS, np.clip(np.triu(K_fqk, 1), 0, 1)) / C.SHOTS
    K_fixed = {"SVM-FQK": K_fqk, "SVM-FQK-shots": (noisy + noisy.T + np.eye(len(sub))).astype(np.float32)}
    D2 = {name: squared_distances(sets[s][sub], sets[s][sub]).astype(np.float32)
          for name, s in DISTANCE_METHODS.items()}
    parts = Parallel(n_jobs=min(n_jobs, len(C.KERNEL_BLINDING_DAYS)))(
        delayed(run_kernel_ridge)(D2, K_fixed, sets["A"][sub], y, day, test_days, b)
        for b in C.KERNEL_BLINDING_DAYS)
    res = pd.concat(parts, ignore_index=True)
    res["row"] = sub[res["pos"].to_numpy()]
    res["filled"] = y[res["pos"].to_numpy()]
    res = res.drop(columns="pos")
    res.to_parquet(C.TABLES / "part2_followup_predictions.parquet", index=False)
    if verbose:
        print(f"follow-up: {len(res):,} predictions saved", flush=True)
    return res
