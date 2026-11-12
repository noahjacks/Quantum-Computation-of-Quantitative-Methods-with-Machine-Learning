"""
panel.py  --  LOAD THE REAL BOND DATA INTO "DAY x BOND" TABLES
==============================================================

Plain-English background
------------------------
A *corporate bond* is a loan that a company takes from investors.  Bonds are
not traded on an exchange like shares; an investor phones or messages a
*dealer* (a bank) and they agree a price.  In the US every such trade must be
reported to a public system called TRACE.  The Open Source Bond Asset Pricing
(OSBAP) project cleans TRACE and publishes one row per bond per day with the
day's prices, yield, trading volume and so on.  That file is our raw material.

What this module does
---------------------
It reads the rows for the dates we need and reshapes each column into a
"wide" table with one ROW PER TRADING DAY and one COLUMN PER BOND.  A cell is
empty (NaN) if the bond did not trade that day - most bonds do not trade every
day, which is exactly why predicting fills is hard.

All later code works on these wide tables because "look only at rows up to
yesterday" then becomes a simple slice, which makes it easy to guarantee that
no feature ever peeks into the future.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import pyarrow.dataset as ds

from . import config as C

# Columns of the OSBAP file we use -> short name of the wide table.
_WIDE_COLUMNS = {
    "pr": "pr",                      # clean price, % of face value
    "prfull": "dirty",               # price including accrued interest
    "acclast": "accrued",            # interest accrued since the last coupon
    "accpmt": "coupons_paid",        # all coupons paid since issue
    "ytm": "ytm",                    # yield to maturity (decimal, 0.05 = 5%)
    "credit_spread": "spread",       # yield above a same-maturity Treasury
    "mod_dur": "mod_dur",            # modified duration (years)
    "convexity": "convexity",
    "bond_maturity": "ttm",          # years until the bond repays
    "bond_age": "age",               # years since the bond was issued
    "bond_amt_outstanding": "amt",   # size of the bond, $ thousands
    "qvolume": "qvol",               # face value traded that day, $ millions
    "dvolume": "dvol",               # dollars traded that day, $ millions
    "trade_count": "ntrades",        # number of trades that day
    "prc_bid": "bid",                # avg price customers SOLD at (dealer bid)
    "prc_ask": "ask",                # avg price customers BOUGHT at (dealer ask)
    "prc_hi": "hi",                  # highest trade price of the day
    "prc_lo": "lo",                  # lowest trade price of the day
}


@dataclass
class Panel:
    """All bond data as (n_days, n_bonds) arrays plus per-bond constants."""

    dates: np.ndarray                 # (T,) trading days, datetime64[D]
    bonds: np.ndarray                 # (N,) 9-character CUSIP bond identifiers
    wide: dict[str, np.ndarray]       # name -> float32 array of shape (T, N)
    sector: np.ndarray                # (N,) Fama-French 17 industry code 1..17
    is_144a: np.ndarray               # (N,) 1.0 if a private "Rule 144A" bond
    issuer: np.ndarray                # (N,) integer code of the issuing company
    traded: np.ndarray = field(init=False)          # (T, N) bool
    last_trade_row: np.ndarray = field(init=False)  # (T, N) int16
    value: np.ndarray = field(init=False)           # (T, N) price + all income
    daily_ret: np.ndarray = field(init=False)       # (T, N) day-to-day return

    def __post_init__(self) -> None:
        w = self.wide
        T, N = w["pr"].shape
        self.traded = np.isfinite(w["pr"])
        # last_trade_row[r, i] = most recent row <= r in which bond i traded
        # (-1 if it has not traded yet).  This is the workhorse for all
        # "as of yesterday" look-ups.
        rows = np.arange(T, dtype=np.int16)[:, None]
        self.last_trade_row = np.maximum.accumulate(
            np.where(self.traded, rows, np.int16(-1)), axis=0).astype(np.int16)
        # "value" = clean price + accrued interest + every coupon paid so far.
        # The change in this number between two dates is exactly what an
        # investor earned, so returns are differences of `value`.
        self.value = (w["pr"] + w["accrued"] + w["coupons_paid"]).astype(np.float32)
        # Day-to-day total return, defined only if the bond traded on BOTH
        # of two consecutive trading days.
        self.daily_ret = np.full((T, N), np.nan, dtype=np.float32)
        self.daily_ret[1:] = (self.value[1:] - self.value[:-1]) / w["dirty"][:-1]

    # -- convenience -------------------------------------------------------
    @property
    def n_days(self) -> int:
        return len(self.dates)

    @property
    def n_bonds(self) -> int:
        return len(self.bonds)

    def row_of(self, date) -> int:
        """Row index of a calendar date (must be a trading day in the panel)."""
        d = np.datetime64(pd.Timestamp(date), "D")
        r = int(np.searchsorted(self.dates, d))
        if r >= len(self.dates) or self.dates[r] != d:
            raise KeyError(f"{date} is not a trading day in the panel")
        return r

    def asof_rows(self, row: int, limit: int) -> np.ndarray:
        """For every bond: the most recent row <= `row` with a trade, or -1 if
        that trade is more than `limit` trading days old (or never happened)."""
        idx = self.last_trade_row[row].astype(np.int64)
        return np.where(idx >= row - limit, idx, -1)

    def gather(self, name_or_array, rows: np.ndarray) -> np.ndarray:
        """Pick, for every bond i, the value in row rows[i] (NaN where -1)."""
        mat = self.wide[name_or_array] if isinstance(name_or_array, str) else name_or_array
        cols = np.arange(mat.shape[1])
        out = mat[np.clip(rows, 0, None), cols].astype(np.float64)
        out[rows < 0] = np.nan
        return out


def load_panel(start: str = C.PANEL_START, end: str = C.WINDOW_END,
               path=C.OSBAP_PARQUET, verbose: bool = True) -> Panel:
    """Read the OSBAP file for [start, end] and build the wide tables."""
    columns = ["cusip_id", "trd_exctn_dt", "db_type", "ff17num", *_WIDE_COLUMNS]
    dataset = ds.dataset(str(path))
    flt = ((ds.field("trd_exctn_dt") >= pd.Timestamp(start)) &
           (ds.field("trd_exctn_dt") <= pd.Timestamp(end)))
    df = dataset.to_table(columns=columns, filter=flt).to_pandas()
    df["cusip_id"] = df["cusip_id"].astype(str)
    n_raw = len(df)
    # The public file contains a few exact duplicate rows; keep one of each.
    df = df.drop_duplicates(["cusip_id", "trd_exctn_dt"], keep="first")

    # --- trading calendar ---------------------------------------------------
    # On US bond-market holidays that are not stock-market holidays (Columbus
    # Day, Veterans Day) a few hundred trades are still reported.  Those thin
    # days would distort every "consecutive trading day" calculation, so a day
    # only counts as a trading day if it has at least 30% of the usual number
    # of rows.
    per_day = df.groupby("trd_exctn_dt").size()
    usual = per_day.rolling(21, min_periods=1, center=True).median()
    good_days = per_day.index[per_day >= 0.30 * usual]
    dropped_days = [d for d in per_day.index if d not in set(good_days)]
    df = df[df["trd_exctn_dt"].isin(good_days)]

    dates = np.sort(df["trd_exctn_dt"].unique()).astype("datetime64[D]")
    bonds = np.sort(df["cusip_id"].unique())
    row = np.searchsorted(dates, df["trd_exctn_dt"].values.astype("datetime64[D]"))
    col = np.searchsorted(bonds, df["cusip_id"].values)
    T, N = len(dates), len(bonds)

    wide: dict[str, np.ndarray] = {}
    for src, name in _WIDE_COLUMNS.items():
        mat = np.full((T, N), np.nan, dtype=np.float32)
        mat[row, col] = df[src].to_numpy(dtype=np.float32, na_value=np.nan)
        wide[name] = mat

    # --- per-bond constants ---------------------------------------------------
    last = df.sort_values("trd_exctn_dt").groupby("cusip_id").last().reindex(bonds)
    sector = last["ff17num"].to_numpy(dtype=np.int64)
    is_144a = (last["db_type"].to_numpy() == 3).astype(np.float64)
    # The first 6 characters of a CUSIP identify the issuing company.
    issuer_str = np.array([b[:6] for b in bonds])
    _, issuer = np.unique(issuer_str, return_inverse=True)

    panel = Panel(dates=dates, bonds=bonds, wide=wide, sector=sector,
                  is_144a=is_144a, issuer=issuer.astype(np.int64))
    if verbose:
        print(f"Panel: {n_raw:,} rows read, {T} trading days "
              f"({dates[0]} .. {dates[-1]}), {N:,} bonds; "
              f"{len(dropped_days)} thin holiday sessions dropped: "
              f"{[str(pd.Timestamp(d).date()) for d in dropped_days]}")
    return panel
