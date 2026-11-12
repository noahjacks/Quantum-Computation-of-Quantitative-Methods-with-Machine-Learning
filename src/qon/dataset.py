"""
dataset.py  --  ASSEMBLE THE FINAL TABLE OF EVENTS
==================================================

One *event* = one replayed quote request = one row of the final table:

    event_id | date | bond | side | 97 market features ... | filled (0/1)

Two tables are produced:

* the CALIBRATION set: requests from the 40 trading days before the window.
  Used only to fix choices (feature scaling, the dealer's KAPPA, kernel
  bandwidths).  Never used to score anything.
* the MAIN set: 16,000 requests from HSBC's window (24 Jul - 29 Oct 2024),
  about the same number every day, exactly like HSBC's sample.

The order of events inside a day is random (the public data has no clock
times).  That order is stored in `event_id` and is the order in which the
"quantum computer" processes the events in the drift experiment.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from . import config as C
from .features import (BUILT_FEATURE_NUMBERS, MACRO_FEATURE_NUMBERS, estimate_coupons,
                       features_for_day, macro_values)
from .feature_catalog import column_name
from .market import Market, build_market
from .panel import Panel, load_panel
from .target import (calibrate_kappa, replay_quotes, required_markup_multiple,
                     updated_reference)

FEATURE_COLUMNS = ["f000_rfq_side"] + [column_name(n) for n in BUILT_FEATURE_NUMBERS]


def _allocate(n_total: int, capacity: np.ndarray) -> np.ndarray:
    """Split n_total events as evenly as possible across days, never asking a
    day for more events than it has."""
    n_days = len(capacity)
    alloc = np.zeros(n_days, dtype=int)
    remaining = n_total
    open_days = np.ones(n_days, dtype=bool)
    while remaining > 0 and open_days.any():
        share = remaining // open_days.sum()
        if share == 0:                       # hand out the last few one by one
            for d in np.where(open_days)[0][:remaining]:
                alloc[d] += 1
            break
        give = np.minimum(share, capacity - alloc) * open_days
        alloc += give
        remaining -= int(give.sum())
        open_days &= alloc < capacity
    return alloc


def build_events(panel: Panel | None = None, market: Market | None = None,
                 verbose: bool = True) -> dict:
    """Run the whole data pipeline and return both event tables."""
    C.ensure_dirs()
    P = panel if panel is not None else load_panel(verbose=verbose)
    M = market if market is not None else build_market(P)
    w0, w1 = P.row_of(C.WINDOW_START), P.row_of(C.WINDOW_END)
    c0 = w0 - C.CALIBRATION_DAYS
    rng = np.random.default_rng(C.EVENT_SEED)

    # Coupons are fixed properties of a bond; estimate them once from data
    # strictly before the first event day.
    coupon = estimate_coupons(P, last_row=c0 - 1)

    # ---- pass 1: features and candidate requests for every day -------------
    # For each day we keep only what is needed later: the feature rows of the
    # eligible bonds, plus the reference price and mark-up used by the dealer.
    days = {}
    k_calibration = []
    for t in range(c0, w1 + 1):
        res = features_for_day(P, M, t, coupon)
        ref_price = updated_reference(M, t, res["last_price"], res["ref_rows"], res["duration"])
        k_buy, k_sell = required_markup_multiple(P, t, ref_price, res["half_spread"])
        elig = res["eligible"]
        cand_buy = np.where(elig & np.isfinite(k_buy))[0]
        cand_sell = np.where(elig & np.isfinite(k_sell))[0]
        elig_idx = np.where(elig)[0]
        X = np.column_stack([res["features"][n][elig_idx] for n in BUILT_FEATURE_NUMBERS])
        days[t] = {"elig_idx": elig_idx, "X": X, "cand_buy": cand_buy, "cand_sell": cand_sell,
                   "ref_price": ref_price, "half_spread": res["half_spread"]}
        if t < w0:
            k_calibration.append(k_buy[cand_buy])
            k_calibration.append(k_sell[cand_sell])
        if verbose and (t - c0) % 20 == 0:
            print(f"  features for {P.dates[t]}: {elig.sum():,} eligible bonds, "
                  f"{len(cand_buy) + len(cand_sell):,} candidate requests", flush=True)

    # ---- the dealer's aggressiveness, fixed on the calibration period ------
    kappa = calibrate_kappa(np.concatenate(k_calibration))

    # ---- pass 2: sample events and attach labels ---------------------------
    def sample(first: int, last: int, n_total: int) -> pd.DataFrame:
        rows = list(range(first, last + 1))
        capacity = np.array([len(days[t]["cand_buy"]) + len(days[t]["cand_sell"]) for t in rows])
        alloc = _allocate(n_total, capacity)
        frames = []
        for t, n_t in zip(rows, alloc):
            d = days[t]
            bond = np.concatenate([d["cand_buy"], d["cand_sell"]])
            side = np.concatenate([np.ones(len(d["cand_buy"])), -np.ones(len(d["cand_sell"]))])
            pick = rng.permutation(len(bond))[:n_t]          # random order within the day
            bond, side = bond[pick], side[pick]
            fill_buy, fill_sell = replay_quotes(P, t, d["ref_price"], d["half_spread"], kappa)
            y = np.where(side > 0, fill_buy[bond], fill_sell[bond])
            pos = np.searchsorted(d["elig_idx"], bond)       # row of each bond in d["X"]
            assert (d["elig_idx"][pos] == bond).all()
            data = {"date": np.repeat(P.dates[t], n_t), "day_index": t - first,
                    "bond": P.bonds[bond], "side": side.astype(int),
                    "f000_rfq_side": side}
            for j, num in enumerate(BUILT_FEATURE_NUMBERS):
                data[column_name(num)] = d["X"][pos, j]
            data["ref_price"] = d["ref_price"][bond]
            data["half_spread"] = d["half_spread"][bond]
            data["filled"] = y.astype(int)
            frames.append(pd.DataFrame(data))
        out = pd.concat(frames, ignore_index=True)
        out.insert(0, "event_id", np.arange(len(out)))
        return out

    calib = sample(c0, w0 - 1, C.N_CALIBRATION_EVENTS)
    main = sample(w0, w1, C.N_EVENTS)
    for name, tbl in (("calibration", calib), ("main", main)):
        if not np.isfinite(tbl[FEATURE_COLUMNS].to_numpy()).all():
            raise ValueError(f"{name} table contains missing feature values")

    # Market-wide features (rates, VIX ...) barely move within 40 days, so
    # their "typical range" for scaling is measured over a longer stretch of
    # the past: one value per trading day over the ~3.5 years of history we
    # load, up to the day before the window.  (The two credit-spread indices
    # are only available on FRED from November 2023, so their range comes
    # from those ~8 months; earlier days are simply empty for them.)
    macro_hist = pd.DataFrame(
        [{column_name(n): v for n, v in macro_values(M, t - 1).items()}
         for t in range(2, w0)])
    assert list(macro_hist.columns) == [column_name(n) for n in MACRO_FEATURE_NUMBERS]
    macro_hist.to_parquet(C.DATA_PROCESSED / "macro_history.parquet", index=False)

    meta = {"kappa": kappa, "n_main": len(main), "n_calibration": len(calib),
            "n_macro_history_days": len(macro_hist),
            "window": [str(P.dates[w0]), str(P.dates[w1])],
            "calibration": [str(P.dates[c0]), str(P.dates[w0 - 1])],
            "n_window_days": int(w1 - w0 + 1),
            "fill_rate_main": float(main["filled"].mean()),
            "fill_rate_calibration": float(calib["filled"].mean()),
            "n_features": len(FEATURE_COLUMNS)}
    calib.to_parquet(C.DATA_PROCESSED / "events_calibration.parquet", index=False)
    main.to_parquet(C.DATA_PROCESSED / "events_main.parquet", index=False)
    (C.DATA_PROCESSED / "events_meta.json").write_text(json.dumps(meta, indent=2))
    if verbose:
        print(json.dumps(meta, indent=2))
    return {"calibration": calib, "main": main, "macro_history": macro_hist, "meta": meta}


def load_events() -> dict:
    """Load the event tables written by build_events()."""
    return {"calibration": pd.read_parquet(C.DATA_PROCESSED / "events_calibration.parquet"),
            "main": pd.read_parquet(C.DATA_PROCESSED / "events_main.parquet"),
            "macro_history": pd.read_parquet(C.DATA_PROCESSED / "macro_history.parquet"),
            "meta": json.loads((C.DATA_PROCESSED / "events_meta.json").read_text())}


if __name__ == "__main__":
    build_events()
