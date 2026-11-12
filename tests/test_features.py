"""
CHECK: the features mean what their definitions say.

The pipeline computes features with fast array code on big "day x bond"
tables.  Here a handful of features are recomputed the slow, obvious way -
bond by bond, straight from the raw file with pandas - and compared.
"""
import numpy as np
import pandas as pd
import pyarrow.dataset as ds
import pytest

from qon import config as C
from qon.dataset import _allocate
from qon.features import _days360, _masked_slope, _ols, estimate_coupons, features_for_day


@pytest.fixture(scope="module")
def case(panel, market):
    t = panel.row_of(C.WINDOW_START) + 5
    c0 = panel.row_of(C.WINDOW_START) - C.CALIBRATION_DAYS
    coupon = estimate_coupons(panel, last_row=c0 - 1)
    res = features_for_day(panel, market, t, coupon)
    rng = np.random.default_rng(3)
    bonds = rng.choice(np.where(res["eligible"])[0], 25, replace=False)
    raw = ds.dataset(str(C.OSBAP_PARQUET)).to_table(
        filter=ds.field("cusip_id").isin(list(panel.bonds[bonds]))).to_pandas()
    raw["cusip_id"] = raw["cusip_id"].astype(str)
    raw = raw.drop_duplicates(["cusip_id", "trd_exctn_dt"])
    raw = raw[raw["trd_exctn_dt"].isin(pd.DatetimeIndex(panel.dates))]   # our trading calendar
    return {"t": t, "res": res, "bonds": bonds, "raw": raw, "coupon": coupon}


def _history(case, panel, i):
    """All rows of bond i strictly before the event day, as a pandas table
    indexed by trading-day number."""
    df = case["raw"][case["raw"]["cusip_id"] == panel.bonds[i]].copy()
    df["row"] = np.searchsorted(panel.dates, df["trd_exctn_dt"].values.astype("datetime64[D]"))
    return df[df["row"] < case["t"]].set_index("row").sort_index()


def _asof(df, row, limit=C.ASOF_LIMIT_MONTHLY):
    sub = df.loc[(df.index <= row) & (df.index >= row - limit)]
    return sub.iloc[-1]


def test_selected_features_match_a_slow_recomputation(case, panel, market):
    t, f = case["t"], case["res"]["features"]
    a = t - 1
    for i in case["bonds"]:
        h = _history(case, panel, i)
        last = h.iloc[-1]
        gap = (panel.dates[t] - np.datetime64(last["trd_exctn_dt"], "D")).astype(int)
        assert np.isclose(f[31][i], last["pr"])                                   # price
        assert np.isclose(f[132][i], gap)                                         # days since last trade
        assert np.isclose(f[2][i], last["bond_maturity"] - gap / 365.25)          # time to maturity
        assert np.isclose(f[19][i], last["ytm"]) and np.isclose(f[21][i], last["credit_spread"])
        assert np.isclose(f[45][i], last["mod_dur"] * last["credit_spread"])      # DTS

        def month_ret(k):                       # return of month k, value = pr + accall
            end, start = _asof(h, a - 21 * (k - 1)), _asof(h, a - 21 * k)
            return (end["pr"] + end["accall"] - start["pr"] - start["accall"]) / start["prfull"]
        assert np.isclose(f[67][i], month_ret(1), atol=1e-6)                      # 1m reversal
        mom6 = np.prod([1 + month_ret(k) for k in range(2, 8)]) - 1
        assert np.isclose(f[69][i], mom6, atol=1e-6)                              # 6m momentum
        rets12 = np.array([month_ret(k) for k in range(1, 13)])
        assert np.isclose(f[85][i], rets12.std(ddof=1), atol=1e-6)                # 12m volatility
        assert np.isclose(f[79][i], rets12.max(), atol=1e-6)                      # max monthly return

        year = h.loc[h.index > a - 252]
        assert np.isclose(f[81][i], last["pr"] / year["pr"].max())                # 52-week high ratio
        month = h.loc[h.index > a - 21]
        assert np.isclose(f[122][i], np.log1p(month["qvolume"].sum()), atol=1e-5)  # log volume
        assert np.isclose(f[124][i], month["trade_count"].sum())                  # number of trades
        assert np.isclose(f[125][i], 1 - len(month) / 21)                         # zero-trading days
        quarter = h.loc[h.index > a - 63]
        both = quarter.dropna(subset=["prc_bid", "prc_ask"])
        eff = ((both["prc_ask"] - both["prc_bid"]) / ((both["prc_ask"] + both["prc_bid"]) / 2)).mean()
        assert np.isclose(f[120][i], eff, atol=1e-6)                              # effective spread
        # day-to-day returns exist only between consecutive trading days
        v = h["pr"] + h["accall"]
        consecutive = np.diff(h.index.to_numpy()) == 1
        dret = (np.diff(v.to_numpy()) / h["prfull"].to_numpy()[:-1])[consecutive]
        drow = h.index.to_numpy()[1:][consecutive]
        assert np.isclose(f[94][i], dret[drow > a - 63].std(ddof=1), atol=1e-6)   # daily-return vol


def test_market_wide_features_are_yesterdays_values(case, panel):
    t, f = case["t"], case["res"]["features"]
    yesterday = pd.Timestamp(panel.dates[t - 1])
    for num, series in ((197, "DGS10"), (203, "VIXCLS"), (199, "DTB3")):
        s = pd.read_csv(C.DATA_RAW / "fred" / f"{series}.csv", parse_dates=["observation_date"])
        s = s.set_index("observation_date")[series].dropna()
        assert np.isclose(f[num][0], s[s.index <= yesterday].iloc[-1])


def test_coupon_recovered_from_accrued_interest_matches_coupon_payments(case, panel):
    """Independent check: a semi-annual bond pays coupon/2 twice a year, so
    2 x (the usual jump in 'coupons paid so far') must equal our estimate."""
    checked = agree = 0
    for i in case["bonds"]:
        h = _history(case, panel, i)
        jumps = np.diff(h["accpmt"].to_numpy())
        jumps = jumps[jumps > 1e-4]
        if len(jumps) >= 3:
            checked += 1
            agree += abs(2 * np.median(jumps) - case["coupon"][i]) < 0.01
    assert checked >= 15 and agree / checked >= 0.9


def test_helpers():
    # 30/360 day count
    d1 = np.array(["2024-01-31", "2024-02-28", "2023-12-15"], dtype="datetime64[D]")
    d2 = np.array(["2024-02-29", "2024-03-01", "2024-06-15"], dtype="datetime64[D]")
    assert list(_days360(d1, d2)) == [29, 3, 180]
    # least squares against numpy
    rng = np.random.default_rng(0)
    x = rng.normal(size=36)
    Y = 0.3 + 1.7 * x[:, None] + rng.normal(size=(36, 4))
    slopes, resid = _ols(Y, x[:, None])
    for j in range(4):
        assert np.isclose(slopes[0, j], np.polyfit(x, Y[:, j], 1)[0])
    assert np.allclose(resid.mean(axis=0), 0, atol=1e-12)
    Y[5, 2] = np.nan
    b = _masked_slope(Y, x, min_obs=10)
    keep = np.arange(36) != 5
    assert np.isclose(b[2], np.polyfit(x[keep], Y[keep, 2], 1)[0])
    assert np.isclose(b[0], slopes[0, 0])
    # fair split of events across days
    assert list(_allocate(10, np.array([100, 100, 100]))) == [4, 3, 3] or \
        sorted(_allocate(10, np.array([100, 100, 100]))) == [3, 3, 4]
    assert list(_allocate(10, np.array([2, 100, 100]))) == [2, 4, 4]
    assert _allocate(16000, np.full(68, 7000)).sum() == 16000
