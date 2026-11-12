"""
target.py  --  THE THING WE PREDICT: WAS THE QUOTE "FILLED"?
===========================================================

Plain-English background
------------------------
When an investor wants to buy or sell a bond they send a *request for quote*
(RFQ) to several dealers at once.  Each dealer answers with a price.  The
investor takes the best one; that dealer's quote is *filled* and everybody
else's is not.  A dealer never sees the competitors' quotes, so "what is the
probability my quote gets filled?" - written p(fill) - is valuable to know:
quote too timidly and you never trade, too aggressively and you give money
away.  HSBC's paper is about predicting p(fill) from the state of the market.

The problem: nobody publishes RFQs.  HSBC's 1.7 million are private.

Our stand-in: "replayed quotes"
-------------------------------
Public data (TRACE) records every trade that actually happened, including the
average price customers paid to BUY a bond from dealers on a day (`ask`) and
the average price they received when SELLING to dealers (`bid`).  Each of
those is the outcome of real quote requests: it is the price of the winning
quote.  So we replay history with one extra, simulated dealer:

1. Every (bond, day, side) where customers really traded is one quote request.
   side = +1: the customer wants to BUY  (dealers quote an "ask"/offer price)
   side = -1: the customer wants to SELL (dealers quote a "bid" price)

2. Our dealer uses a deliberately simple rule:
       reference price = the bond's last traded price before today, brought
                         up to date for how far the whole corporate-bond
                         market has moved since that trade (see below)
       mark-up         = KAPPA x (the bond's usual half bid-ask spread)
       ask quote       = reference x (1 + mark-up)   (customer buys)
       bid quote       = reference x (1 - mark-up)   (customer sells)

   Why bring the reference up to date?  Bond prices all move together when
   interest rates move, and bonds that repay far in the future ("long
   duration") move the most.  Every dealer sees the current market level on
   their screen, so a real dealer never quotes off a stale price.  If ours
   did, "was the quote filled?" would mostly mean "did the market go up or
   down today?", which nobody can predict and which has nothing to do with
   quoting.  So:
       reference = last price x (1 + market move since last trade
                                     x bond duration / market duration)

3. Our quote is FILLED if it would have beaten the price the customer really
   got:   ask quote <= real price paid       (customer buys)
          bid quote >= real price received   (customer sells)

So the bonds, prices and market conditions are all real; only the dealer's
quoting rule is ours.  KAPPA is set once, on the calibration period, so that
about 37% of quotes are filled (the fill rate HSBC reports) and then frozen.

Honest limitations (also listed in the README)
----------------------------------------------
* We only see requests that ended in a trade.  Real RFQs where the customer
  walked away are invisible in public data.
* The public file is daily, so "the price the customer got" is that day's
  volume-weighted average for the bond and side, not a single trade.
* A real dealer quotes far more cleverly than our rule.
"""
from __future__ import annotations

import numpy as np

from . import config as C
from .market import Market
from .panel import Panel


def updated_reference(market: Market, t: int, last_price: np.ndarray,
                      ref_rows: np.ndarray, duration: np.ndarray) -> np.ndarray:
    """The dealer's reference price on day `t` for every bond: the last traded
    price (from row ref_rows[i] < t), moved by the corporate-bond market's
    return since then, scaled by the bond's duration relative to the market's.

    The market return up to day t is something every dealer can see live; it
    is used ONLY to form the dealer's quote, never as a model feature."""
    cum = market.cum["BONDMKT"]
    market_move = cum[t] / cum[np.clip(ref_rows, 0, None)] - 1.0
    sensitivity = duration / market.level["MKTDUR"][t - 1]
    ref = last_price * (1.0 + market_move * sensitivity)
    return np.where(ref_rows >= 0, ref, np.nan)


def replay_quotes(panel: Panel, t: int, ref_price: np.ndarray,
                  half_spread: np.ndarray, kappa: float):
    """Outcome of our dealer's quotes on trading day `t` for every bond.

    Returns two float arrays of length n_bonds (NaN = no such request today):
        fill_buy   1.0 / 0.0 for "customer buys" requests
        fill_sell  1.0 / 0.0 for "customer sells" requests
    """
    real_ask = panel.wide["ask"][t].astype(np.float64)   # customers bought at
    real_bid = panel.wide["bid"][t].astype(np.float64)   # customers sold at
    markup = kappa * half_spread
    our_ask = ref_price * (1.0 + markup)
    our_bid = ref_price * (1.0 - markup)
    fill_buy = np.where(np.isfinite(real_ask) & np.isfinite(our_ask),
                        (our_ask <= real_ask).astype(float), np.nan)
    fill_sell = np.where(np.isfinite(real_bid) & np.isfinite(our_bid),
                         (our_bid >= real_bid).astype(float), np.nan)
    return fill_buy, fill_sell


def required_markup_multiple(panel: Panel, t: int, ref_price: np.ndarray,
                             half_spread: np.ndarray):
    """The largest KAPPA at which each request would still be filled.
    (A request is filled exactly when KAPPA <= this number.)  Used only to
    calibrate KAPPA on the calibration period."""
    real_ask = panel.wide["ask"][t].astype(np.float64)
    real_bid = panel.wide["bid"][t].astype(np.float64)
    with np.errstate(invalid="ignore", divide="ignore"):
        k_buy = (real_ask / ref_price - 1.0) / half_spread
        k_sell = (1.0 - real_bid / ref_price) / half_spread
    return k_buy, k_sell


def calibrate_kappa(k_values: np.ndarray, target_rate: float = C.TARGET_FILL_RATE) -> float:
    """KAPPA such that a fraction `target_rate` of calibration requests fill."""
    k = np.asarray(k_values, dtype=float)
    k = k[np.isfinite(k)]
    return float(np.quantile(k, 1.0 - target_rate))
