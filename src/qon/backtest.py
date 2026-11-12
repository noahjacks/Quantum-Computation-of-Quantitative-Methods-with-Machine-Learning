"""
backtest.py  --  TRAIN ON THE PAST, TEST ON THE FUTURE, REPEAT
==============================================================

A "backtest" replays history the way a real trading desk would have lived it.
For every test day d (the 68 days of the window):

    1. Take the most recent TRAIN_SIZE requests that happened BEFORE day d.
    2. (Every 10 test days) choose the model's settings by cross-validation
       inside that training set.
    3. Train the model and predict p(fill) for every request of day d.

Then score the predictions.  This is HSBC's protocol (their Section 3.3).

"Blinding" (HSBC's term): in practice the newest outcomes may not be usable
yet, so some time is left out between the last training request and the test
request.  Our data is daily, so blinding is counted in whole trading days:

    blinding 0 : train on everything up to yesterday (day d-1)
    blinding 1 : train on everything up to day d-2
    blinding 4 : train on everything up to day d-5

HSBC's headline gains were at their shortest blinding and shrank as blinding
grew - one of the clues that made critics suspect an artefact.

For the first test days the training requests come from the calibration
period (the 40 days before the window).  Nothing is ever scored there.

Guarantee: `check_no_overlap` asserts for every single fit that the newest
training day is strictly older than the test day minus the blinding.
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.base import clone
from sklearn.model_selection import GridSearchCV, KFold

from . import config as C
from .models import MODEL_NAMES, make_model


def training_rows(day: np.ndarray, test_day: int, blinding: int, train_size: int) -> np.ndarray:
    """Rows of the `train_size` most recent events that are allowed as
    training data for `test_day` (rows are stored in time order)."""
    allowed = np.where(day <= test_day - 1 - blinding)[0]
    return allowed[-train_size:]


def check_no_overlap(day: np.ndarray, train: np.ndarray, test_day: int, blinding: int) -> None:
    assert len(train) > 0, "no training data"
    assert day[train].max() <= test_day - 1 - blinding < test_day, "training data too recent"


def run_one(X: np.ndarray, y: np.ndarray, day: np.ndarray, test_days: np.ndarray,
            blinding: int, train_size: int = C.TRAIN_SIZE,
            models: tuple[str, ...] = MODEL_NAMES, retune_every: int = C.RETUNE_EVERY,
            label: str = "") -> pd.DataFrame:
    """Rolling backtest of all models on one feature set and one blinding.

    X, y, day cover ALL events (calibration + main) in time order.
    Returns one row per (test event, model) with the predicted p(fill)."""
    out = []
    best_params: dict[str, dict] = {}
    cv = KFold(n_splits=C.CV_FOLDS, shuffle=True, random_state=C.MODEL_SEED)
    for k, d in enumerate(test_days):
        train = training_rows(day, int(d), blinding, train_size)
        check_no_overlap(day, train, int(d), blinding)
        test = np.where(day == d)[0]
        Xtr, ytr, Xte = X[train], y[train], X[test]
        for name in models:
            est, grid = make_model(name)
            if k % retune_every == 0:
                search = GridSearchCV(est, grid, scoring="roc_auc", cv=cv, n_jobs=1, refit=True)
                search.fit(Xtr, ytr)
                best_params[name] = search.best_params_
                fitted = search.best_estimator_
            else:
                fitted = clone(est).set_params(**best_params[name]).fit(Xtr, ytr)
            out.append(pd.DataFrame({"row": test, "day": int(d), "model": name,
                                     "p": fitted.predict_proba(Xte)[:, 1]}))
    res = pd.concat(out, ignore_index=True)
    res["blinding"] = blinding
    res["set"] = label
    return res


def _run_and_save(path, X, y, day, test_days, blinding, train_size, models, name):
    res = run_one(X, y, day, test_days, blinding, train_size, models, C.RETUNE_EVERY, name)
    tmp = path.with_suffix(".tmp")
    res.to_parquet(tmp, index=False)
    tmp.rename(path)                       # only complete files get the final name
    return path


def run_grid(sets: dict[str, np.ndarray], events: pd.DataFrame, jobs: list[tuple[str, int]],
             tag: str, train_size: int = C.TRAIN_SIZE, models: tuple[str, ...] = MODEL_NAMES,
             n_jobs: int = 3, verbose: bool = True) -> pd.DataFrame:
    """Run many (feature set, blinding) backtests, a few at a time.

    jobs: list of (set name, blinding).  Each finished backtest is saved
    straight away under results/tables/parts/, and anything already saved is
    skipped, so the run can be stopped and restarted without losing work."""
    parts = C.TABLES / "parts"
    parts.mkdir(parents=True, exist_ok=True)
    y = events["filled"].to_numpy()
    day = events["day"].to_numpy()
    test_days = np.unique(day[(events["period"] == "main").to_numpy()])
    path_of = {(name, b): parts / f"{tag}__{name}__blind{b}.parquet" for name, b in jobs}
    todo = [(name, b) for name, b in jobs if not path_of[(name, b)].exists()]
    if verbose:
        print(f"{tag}: {len(jobs)} backtests, {len(jobs) - len(todo)} already done, "
              f"{len(todo)} to run on {n_jobs} cores", flush=True)
    t0 = time.time()
    Parallel(n_jobs=n_jobs, verbose=10 if verbose else 0)(
        delayed(_run_and_save)(path_of[(name, b)], sets[name], y, day, test_days, b,
                               train_size, models, name)
        for name, b in todo)
    res = pd.concat([pd.read_parquet(path_of[j]) for j in jobs], ignore_index=True)
    res = res.merge(events[["row", "filled"]], on="row", how="left")
    if verbose:
        print(f"{tag}: finished, {len(res):,} predictions, "
              f"{(time.time() - t0) / 60:.1f} min", flush=True)
    return res
