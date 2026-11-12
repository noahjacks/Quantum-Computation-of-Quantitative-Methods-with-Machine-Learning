"""
CHECK: no feature can see the future.

This is the most important test of the data pipeline.  A model that is
accidentally shown tomorrow's prices looks brilliant in a backtest and is
worthless in reality ("look-ahead bias").

Method: compute all features for the requests of trading day t.  Then DESTROY
every piece of data from day t onwards (bond data and market data are
overwritten with NaN) and compute the features again.  If even one number
changes, some feature was using information it could not have had.
"""
import copy

import numpy as np
import pytest

from qon import config as C
from qon.features import BUILT_FEATURE_NUMBERS, estimate_coupons, features_for_day
from qon.panel import Panel


def _same(a, b):
    return np.array_equal(np.asarray(a, dtype=float), np.asarray(b, dtype=float), equal_nan=True)


@pytest.mark.parametrize("offset", [0, 23, 67])
def test_features_do_not_change_when_the_future_is_destroyed(panel, market, offset):
    t = panel.row_of(C.WINDOW_START) + offset
    c0 = panel.row_of(C.WINDOW_START) - C.CALIBRATION_DAYS
    coupon = estimate_coupons(panel, last_row=c0 - 1)
    before = features_for_day(panel, market, t, coupon)

    saved = {name: arr[t:].copy() for name, arr in panel.wide.items()}
    try:
        for arr in panel.wide.values():
            arr[t:] = np.nan                                  # destroy the future
        blind_panel = Panel(dates=panel.dates, bonds=panel.bonds, wide=panel.wide,
                            sector=panel.sector, is_144a=panel.is_144a, issuer=panel.issuer)
        blind_market = copy.deepcopy(market)
        for series in (*blind_market.level.values(), *blind_market.cum.values(), blind_market.illiq):
            series[t:] = np.nan
        blind_coupon = estimate_coupons(blind_panel, last_row=c0 - 1)
        after = features_for_day(blind_panel, blind_market, t, blind_coupon)
    finally:
        for name, arr in panel.wide.items():                  # put the data back
            arr[t:] = saved[name]

    assert _same(coupon, blind_coupon)
    assert before["eligible"].sum() > 3000
    assert np.array_equal(before["eligible"], after["eligible"])
    for key in ("last_price", "half_spread", "duration", "ref_rows"):
        assert _same(before[key], after[key]), key
    for num in BUILT_FEATURE_NUMBERS:
        assert _same(before["features"][num], after["features"][num]), f"feature {num} looks ahead"


def test_monthly_statistics_respect_their_publication_delay(market):
    """Inflation for month m is published in the middle of month m+1.  Our
    value on any day must come from a month that started >= 50 days earlier."""
    import pandas as pd
    cpi = pd.read_csv(C.DATA_RAW / "fred" / "CPIAUCSL.csv", parse_dates=["observation_date"])
    cpi = cpi.set_index("observation_date")["CPIAUCSL"]
    yoy = (cpi / cpi.shift(12) - 1.0).dropna()
    for row in (-1, -30, -200):
        day = pd.Timestamp(market.dates[row])
        visible = yoy[yoy.index + pd.Timedelta(days=50) <= day]
        assert np.isclose(market.level["CPI_YOY"][row], visible.iloc[-1])
        assert (day - visible.index[-1]).days >= 50
