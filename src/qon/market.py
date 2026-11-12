"""
market.py  --  MARKET-WIDE DATA (THE SAME FOR EVERY BOND ON A GIVEN DAY)
========================================================================

Plain-English background
------------------------
Some information is about the whole market rather than one bond: the level of
interest rates, how nervous the stock market is (VIX), how much extra yield
investors demand for holding corporate bonds ("credit spread"), inflation,
unemployment ...  This module loads those series from the free sources and
lines them up with the bond trading calendar.

It also builds a few "factors".  A factor is simply the daily return of some
broad portfolio, e.g. "all corporate bonds" or "the stock market".  A bond's
*beta* to a factor measures how strongly the bond's return moves when the
factor moves; betas are among the 216 features.

Two kinds of series are stored:

* ``level[name][row]``  the value at the close of trading day `row`
                         (e.g. the 10-year Treasury yield).
* ``cum[name][row]``    a cumulative-growth index: the value of 1 unit
                         invested at the start.  The return between two days
                         r1 < r2 is  cum[r2] / cum[r1] - 1.

Publication lags: monthly statistics (inflation, unemployment ...) are only
published weeks after the month ends.  To avoid "knowing the future", each
monthly number only becomes visible after a conservative delay.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import config as C
from .panel import Panel

TSY_TENORS = (2, 5, 10, 20, 30)          # Treasury maturities we track (years)
MONTHLY_LAG_DAYS = 50                    # monthly data visible 50 days after month start
EBP_LAG_DAYS = 60


def _read_fred(series: str) -> pd.Series:
    df = pd.read_csv(C.DATA_RAW / "fred" / f"{series}.csv", parse_dates=["observation_date"])
    s = pd.to_numeric(df[series], errors="coerce")
    s.index = pd.DatetimeIndex(df["observation_date"])
    return s


def _read_french() -> pd.DataFrame:
    """Daily stock-market factors (percent) from Kenneth French's library."""
    path = C.DATA_RAW / "french" / "F-F_Research_Data_Factors_daily.csv"
    rows = []
    with open(path, "r", encoding="latin-1") as fh:
        for line in fh:
            parts = [p.strip() for p in line.split(",")]
            if len(parts) == 5 and len(parts[0]) == 8 and parts[0].isdigit():
                rows.append(parts)
    df = pd.DataFrame(rows, columns=["date", "MKT", "SMB", "HML", "RF"])
    df["date"] = pd.to_datetime(df["date"], format="%Y%m%d")
    return df.set_index("date").astype(float) / 100.0      # percent -> decimal


def par_bond_duration(yield_pct: np.ndarray, maturity: float) -> np.ndarray:
    """Modified duration of a Treasury trading at face value ("par") with
    semi-annual coupons:  D = (1 - (1 + y/2)^(-2m)) / y."""
    y = np.maximum(np.asarray(yield_pct, dtype=float) / 100.0, 1e-6)
    return (1.0 - (1.0 + y / 2.0) ** (-2.0 * maturity)) / y


@dataclass
class Market:
    dates: np.ndarray                 # (T,) the bond trading calendar
    level: dict[str, np.ndarray]      # name -> (T,) value at the close of each day
    cum: dict[str, np.ndarray]        # name -> (T,) cumulative growth index
    illiq: np.ndarray                 # (T,) market-wide illiquidity of each day

    # -- helpers used by the feature code ---------------------------------
    def window_ret(self, name: str, boundaries: np.ndarray) -> np.ndarray:
        """Return of factor `name` over each window.  `boundaries` is a
        decreasing array of rows b[0] > b[1] > ...; element k of the result
        is the return from the close of row b[k+1] to the close of row b[k]."""
        c = self.cum[name][boundaries]
        return c[:-1] / c[1:] - 1.0

    def window_diff(self, name: str, boundaries: np.ndarray) -> np.ndarray:
        """Change of a level series over each window (same convention)."""
        v = self.level[name][boundaries]
        return v[:-1] - v[1:]

    def window_mean_illiq(self, boundaries: np.ndarray) -> np.ndarray:
        """Average market illiquidity inside each window (b[k+1], b[k]]."""
        return np.array([np.nanmean(self.illiq[boundaries[k + 1] + 1: boundaries[k] + 1])
                         for k in range(len(boundaries) - 1)])


def build_market(panel: Panel) -> Market:
    idx = pd.DatetimeIndex(panel.dates)
    level: dict[str, np.ndarray] = {}
    cum: dict[str, np.ndarray] = {}

    def align(series: pd.Series) -> np.ndarray:
        """Carry the last published value forward onto the bond calendar."""
        s = series.dropna()
        s = s[~s.index.duplicated(keep="last")].sort_index()
        return s.reindex(s.index.union(idx)).ffill().reindex(idx).to_numpy(dtype=float)

    # ---- daily levels from FRED -------------------------------------------
    for name, series in {"DGS10": "DGS10", "T10Y2Y": "T10Y2Y", "DTB3": "DTB3",
                         "IG_OAS": "BAMLC0A0CM", "HY_OAS": "BAMLH0A0HYM2",
                         "VIX": "VIXCLS", "USD": "DTWEXBGS", "OIL": "DCOILWTICO"}.items():
        level[name] = align(_read_fred(series))
    # Economic-policy-uncertainty index: very jumpy day to day, so use the
    # average of the last 30 calendar days.
    epu = _read_fred("USEPUINDXD").dropna()
    level["EPU30"] = align(epu.rolling("30D").mean())

    # ---- monthly / quarterly statistics with a publication delay ----------
    def lagged(series: pd.Series, lag_days: int) -> np.ndarray:
        s = series.dropna()
        s.index = s.index + pd.Timedelta(days=lag_days)     # date it becomes visible
        return align(s)

    indpro = _read_fred("INDPRO")
    cpi = _read_fred("CPIAUCSL")
    unrate = _read_fred("UNRATE")
    level["INDPRO_YOY"] = lagged(indpro / indpro.shift(12) - 1.0, MONTHLY_LAG_DAYS)
    level["CPI_YOY"] = lagged(cpi / cpi.shift(12) - 1.0, MONTHLY_LAG_DAYS)
    level["UNRATE_CHG3"] = lagged(unrate - unrate.shift(3), MONTHLY_LAG_DAYS)
    level["SLOOS"] = lagged(_read_fred("DRTSCILM"), MONTHLY_LAG_DAYS)
    ebp = pd.read_csv(C.DATA_RAW / "ebp" / "ebp_csv.csv", parse_dates=["date"]).set_index("date")["ebp"]
    level["EBP"] = lagged(ebp, EBP_LAG_DAYS)

    # ---- stock-market factors (Kenneth French) ----------------------------
    ff = _read_french()
    ff = ff[ff.index >= idx[0] - pd.Timedelta(days=10)]
    for name in ("MKT", "SMB", "HML", "RF"):
        # Build the growth index on the stock-market calendar (which has a few
        # more days than the bond calendar), then read it off on bond days.
        cum[name] = align((1.0 + ff[name]).cumprod())

    # ---- Treasury returns, approximated from constant-maturity yields ------
    # daily return ~ carry (yield * time)  -  duration * change in yield
    for tenor in TSY_TENORS:
        y = _read_fred(f"DGS{tenor}").dropna()
        y = y[y.index >= idx[0] - pd.Timedelta(days=10)]
        dur = pd.Series(par_bond_duration(y.values, tenor), index=y.index)
        dt_years = y.index.to_series().diff().dt.days / 365.0
        ret = (y.shift(1) / 100.0) * dt_years - dur.shift(1) * (y.diff() / 100.0)
        cum[f"TSY{tenor}"] = align((1.0 + ret.fillna(0.0)).cumprod())
        level[f"TSYDUR{tenor}"] = align(dur)

    # ---- the corporate-bond market factor, built from the panel itself -----
    # Value-weighted average of all bonds' day-to-day returns.  Weights are
    # yesterday's market value (size x price).  Single-bond returns are capped
    # at +-25% so that one bad data point cannot move the whole index.
    w = panel.wide
    T = panel.n_days
    mv = w["amt"][:-1].astype(np.float64) * w["dirty"][:-1]
    r = panel.daily_ret[1:].astype(np.float64)
    ok = np.isfinite(r) & np.isfinite(mv) & (mv > 0)
    r_c = np.clip(np.where(ok, r, 0.0), -0.25, 0.25)
    wgt = np.where(ok, mv, 0.0)
    bond_mkt = np.zeros(T)
    bond_mkt[1:] = (r_c * wgt).sum(axis=1) / wgt.sum(axis=1)
    cum["BONDMKT"] = np.cumprod(1.0 + bond_mkt)
    # Average duration of the bonds in that index (value-weighted).  Used by
    # the simulated dealer to scale the market move to each bond (target.py).
    mv_today = w["amt"].astype(np.float64) * w["dirty"]
    ok_d = np.isfinite(mv_today) & np.isfinite(w["mod_dur"])
    level["MKTDUR"] = (np.where(ok_d, mv_today * w["mod_dur"], 0.0).sum(axis=1)
                       / np.where(ok_d, mv_today, 0.0).sum(axis=1))

    # ---- market-wide illiquidity -------------------------------------------
    # For each bond-day: |return| per $ million traded (the "Amihud" measure:
    # a big price move on little volume = illiquid).  Market value = the
    # median across bonds.
    with np.errstate(divide="ignore", invalid="ignore"):
        amihud = np.abs(panel.daily_ret) / np.where(w["dvol"] > 0, w["dvol"], np.nan)
    illiq = np.full(T, np.nan)
    illiq[1:] = np.nanmedian(amihud[1:], axis=1)

    # ---- safety checks: no gaps where the data is actually used -------------
    # Series used over the full 3-year look-back must be complete everywhere;
    # series only read "as of yesterday" must be complete over the calibration
    # period and the test window (the ICE credit-spread indices on FRED only
    # start in November 2023, which is early enough).
    recent = C.CALIBRATION_DAYS + 80
    for name, arr in cum.items():
        if not np.isfinite(arr).all():
            raise ValueError(f"Factor index {name} has gaps")
    for name in ("VIX", "DGS10", "OIL", "MKTDUR", *[f"TSYDUR{t}" for t in TSY_TENORS]):
        if not np.isfinite(level[name]).all():
            raise ValueError(f"Market series {name} has gaps")
    for name, arr in level.items():
        if not np.isfinite(arr[-recent:]).all():
            raise ValueError(f"Market series {name} has gaps near the test window")
    return Market(dates=panel.dates, level=level, cum=cum, illiq=illiq)
