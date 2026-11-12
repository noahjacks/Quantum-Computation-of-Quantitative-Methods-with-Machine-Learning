"""
preprocessing.py  --  PUT EVERY FEATURE ON THE SAME SCALE
=========================================================

Why this is needed
------------------
The 98 inputs have wildly different units: a price is around 100, a yield is
around 0.05, a trade count can be 5,000.  Both the classical models and the
quantum circuit need them on a common scale.

For the quantum circuit a feature becomes a ROTATION ANGLE of a gate, so it
must live in a sensible range.  We follow HSBC's recipe (their Section 3.2):

    1. clip extreme values                  (guards against outliers)
    2. z = (x - mean) / standard deviation  (the usual "z-score")
    3. angle = 2*pi * tanh(z / 3)           (squashes into (-2pi, +2pi))

The one thing we do more carefully than the paper: the clip limits, mean and
standard deviation are measured on the CALIBRATION period only (the 40 days
before the test window).  The paper measured them on the whole sample, which
lets a little information from the future leak into the scaling.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import config as C


@dataclass
class AngleScaler:
    lo: np.ndarray
    hi: np.ndarray
    mean: np.ndarray
    std: np.ndarray

    @classmethod
    def fit(cls, X_calibration: np.ndarray,
            history: dict[int, np.ndarray] | None = None) -> "AngleScaler":
        """Measure clip limits, mean and std of every column on calibration
        events.  `history` optionally maps a column index to a longer series
        of past values to use instead (used for the market-wide features,
        which hardly move within the 40-day calibration period)."""
        X = np.asarray(X_calibration, dtype=float)
        columns = [X[:, j] for j in range(X.shape[1])]
        for j, values in (history or {}).items():
            values = np.asarray(values, dtype=float)
            columns[j] = values[np.isfinite(values)]       # skip days with no data
        lo = np.array([np.quantile(c, C.WINSOR_QUANTILES[0]) for c in columns])
        hi = np.array([np.quantile(c, C.WINSOR_QUANTILES[1]) for c in columns])
        clipped = [np.clip(c, l, h) for c, l, h in zip(columns, lo, hi)]
        mean = np.array([c.mean() for c in clipped])
        std = np.array([c.std() for c in clipped])
        std = np.where(std > 0, std, 1.0)        # a constant feature stays at z = 0
        return cls(lo=lo, hi=hi, mean=mean, std=std)

    def zscore(self, X: np.ndarray) -> np.ndarray:
        """Steps 1-2.  This is what the classical models receive (set A)."""
        return (np.clip(np.asarray(X, dtype=float), self.lo, self.hi) - self.mean) / self.std

    def angles(self, X: np.ndarray) -> np.ndarray:
        """Steps 1-3.  This is what the quantum circuit receives."""
        return 2.0 * np.pi * np.tanh(self.zscore(X) / C.TANH_DIVISOR)
