"""
stats.py  --  SCORING, AND HOW SURE WE ARE
==========================================

The score: AUC
--------------
Every model outputs p(fill) for every test request.  The AUC ("area under the
ROC curve") is the probability that a randomly chosen FILLED request received
a higher p(fill) than a randomly chosen UNFILLED one.

    AUC = 0.5   the model is guessing
    AUC = 1.0   the model is always right

It is the score HSBC reports.  Their "34% improvement" is the ratio of two
AUCs (e.g. 0.97 / 0.72 - 1).  We report the same ratio, and also the plain
difference in AUC, which is easier to interpret.

We compute one AUC per test DAY and then average over the days ("mean daily
AUC").  Scoring within a day stops a model from looking clever just by
knowing which days had many fills.

Error bars: the bootstrap
-------------------------
Our 68 test days are one draw from all the days that could have happened.  To
see how much a number would wobble with a different draw, we re-draw the 68
days at random with replacement 2000 times and recompute the number each
time.  The middle 95% of those values is the "95% confidence interval".
When two feature sets are compared we re-draw the SAME days for both (a
"paired" comparison), which removes most of the day-to-day noise.
If a confidence interval for a difference contains 0, we cannot claim that
there is a difference.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from . import config as C


def daily_auc(y: np.ndarray, p: np.ndarray, day: np.ndarray) -> pd.Series:
    """AUC of each test day (days with only fills or only non-fills are skipped)."""
    out = {}
    for d in np.unique(day):
        m = day == d
        if 0 < y[m].sum() < m.sum():
            out[int(d)] = roc_auc_score(y[m], p[m])
    return pd.Series(out, dtype=float)


def brier_score(y: np.ndarray, p: np.ndarray) -> float:
    """Mean squared error of the predicted probabilities (lower is better).
    Unlike AUC it also punishes probabilities that are too high or too low."""
    return float(np.mean((np.asarray(p, dtype=float) - np.asarray(y, dtype=float)) ** 2))


def bootstrap_mean(values: np.ndarray, n_boot: int = C.N_BOOTSTRAP,
                   seed: int = C.BOOTSTRAP_SEED) -> tuple[float, float, float]:
    """Mean of `values` (one per day) with a 95% bootstrap confidence interval."""
    v = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(v), size=(n_boot, len(v)))
    means = v[idx].mean(axis=1)
    return float(v.mean()), float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def summarize(auc_by_day: pd.DataFrame, baseline: str) -> pd.DataFrame:
    """auc_by_day: one row per test day, one column per feature set / model.
    Returns mean AUC, its 95% interval, and the paired difference to
    `baseline` with its 95% interval and HSBC-style relative gain."""
    rows = []
    base = auc_by_day[baseline].to_numpy()
    for col in auc_by_day.columns:
        v = auc_by_day[col].to_numpy()
        ok = np.isfinite(v) & np.isfinite(base)
        m, lo, hi = bootstrap_mean(v[ok])
        dm, dlo, dhi = bootstrap_mean(v[ok] - base[ok])
        rows.append({"name": col, "mean_auc": m, "ci_low": lo, "ci_high": hi,
                     "median_auc": float(np.median(v[ok])),
                     "diff_vs_" + baseline: dm, "diff_ci_low": dlo, "diff_ci_high": dhi,
                     "relative_gain_%": 100.0 * (m / base[ok].mean() - 1.0),
                     "significant": bool(dlo > 0 or dhi < 0) and col != baseline,
                     "n_days": int(ok.sum())})
    return pd.DataFrame(rows).set_index("name")
