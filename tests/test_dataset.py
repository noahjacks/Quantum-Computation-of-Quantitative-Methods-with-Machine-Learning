"""
CHECK: the finished event tables are what we say they are.

* the right number of rows, spread evenly over the days, no missing values;
* every "filled" label can be re-derived from first principles straight from
  the raw price data, without using any of the pipeline's own label code.
"""
import numpy as np
import pandas as pd

from qon import config as C
from qon.dataset import FEATURE_COLUMNS
from qon.target import calibrate_kappa, replay_quotes


def test_tables_have_the_promised_shape(events):
    main, cal, meta = events["main"], events["calibration"], events["meta"]
    assert len(main) == C.N_EVENTS and len(cal) == C.N_CALIBRATION_EVENTS
    assert len(FEATURE_COLUMNS) == 98 and FEATURE_COLUMNS[0] == "f000_rfq_side"
    for tbl in (main, cal):
        assert np.isfinite(tbl[FEATURE_COLUMNS].to_numpy()).all()
        assert set(tbl["filled"].unique()) == {0, 1}
        assert set(tbl["side"].unique()) == {-1, 1}
        assert tbl["date"].is_monotonic_increasing
    per_day = main.groupby("date").size()
    assert per_day.max() - per_day.min() <= 1                 # evenly spread
    assert str(main["date"].min().date()) == C.WINDOW_START
    assert str(main["date"].max().date()) == C.WINDOW_END
    assert cal["date"].max() < main["date"].min()             # calibration is strictly earlier
    assert abs(meta["fill_rate_main"] - C.TARGET_FILL_RATE) < 0.03
    # the same (bond, day, side) never appears twice
    assert not main.duplicated(["bond", "date", "side"]).any()


def test_labels_can_be_rederived_from_raw_prices(events, panel):
    """filled = 1 exactly when our quote beats the real customer price."""
    main, kappa = events["main"], events["meta"]["kappa"]
    sample = main.sample(1500, random_state=0)
    col = pd.Series(np.arange(panel.n_bonds), index=panel.bonds)
    for _, e in sample.iterrows():
        t = panel.row_of(e["date"])
        i = col[e["bond"]]
        markup = kappa * e["half_spread"]
        if e["side"] == 1:        # customer buys: we offer; real buyers paid `ask`
            expected = e["ref_price"] * (1 + markup) <= float(panel.wide["ask"][t, i])
        else:                     # customer sells: we bid; real sellers got `bid`
            expected = e["ref_price"] * (1 - markup) >= float(panel.wide["bid"][t, i])
        assert int(expected) == e["filled"]


def test_reference_price_only_uses_the_last_trade_and_the_market_move(events, panel, market):
    """ref = last traded price x (1 + market move since then x duration ratio)."""
    main = events["main"].sample(300, random_state=1)
    col = pd.Series(np.arange(panel.n_bonds), index=panel.bonds)
    cum = market.cum["BONDMKT"]
    for _, e in main.iterrows():
        t, i = panel.row_of(e["date"]), col[e["bond"]]
        r = int(panel.last_trade_row[t - 1, i])               # last trade strictly before t
        assert t - r <= C.MAX_STALE_DAYS_REF
        last_price = float(panel.wide["pr"][r, i])
        duration = float(panel.wide["mod_dur"][r, i])
        expected = last_price * (1 + (cum[t] / cum[r] - 1) * duration / market.level["MKTDUR"][t - 1])
        assert np.isclose(e["ref_price"], expected, rtol=1e-9)
        assert np.isclose(e["f031_price"], last_price)


def test_replay_rule_on_hand_made_numbers():
    class Fake:
        wide = {"ask": np.array([[101.0, 100.4, np.nan]], dtype=np.float32),
                "bid": np.array([[99.0, 99.9, 99.5]], dtype=np.float32)}
    ref = np.array([100.0, 100.0, 100.0])
    half = np.array([0.005, 0.005, 0.005])                    # 0.5% half-spread
    buy, sell = replay_quotes(Fake, 0, ref, half, kappa=1.0)
    # our ask = 100.5 ; our bid = 99.5
    assert buy[0] == 1.0      # customers paid 101.0 -> our 100.5 was better: filled
    assert buy[1] == 0.0      # customers paid 100.4 -> our 100.5 was worse: not filled
    assert np.isnan(buy[2])   # nobody bought today: no request
    assert sell[0] == 1.0     # customers got 99.0 -> our 99.5 was better: filled
    assert sell[1] == 0.0     # customers got 99.9 -> our 99.5 was worse
    assert sell[2] == 1.0     # exactly equal counts as filled


def test_kappa_calibration_hits_the_target_rate():
    k = np.random.default_rng(0).normal(size=100_000)
    kappa = calibrate_kappa(k, 0.37)
    assert abs((k >= kappa).mean() - 0.37) < 1e-3
