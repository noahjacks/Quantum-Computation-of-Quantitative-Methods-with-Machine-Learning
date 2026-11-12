"""
features.py  --  TURN RAW BOND DATA INTO THE MODEL'S INPUTS ("FEATURES")
=======================================================================

Plain-English background
------------------------
A *feature* is one number that describes the situation just before a quote
request arrives: how long until the bond repays, how risky the market thinks
it is, how often it has traded lately, how nervous markets are today, etc.
The team's list (docs/Features_Qiskit.csv) has 216 of them.  97 can be built
from free public data; they are built here.  docs/FEATURE_COVERAGE.md lists
every one of the 216 and says either how it was built or why it could not be.

THE GOLDEN RULE: NO PEEKING INTO THE FUTURE
-------------------------------------------
A quote request arrives on trading day `t`.  Every feature for that request is
computed from data up to and including day `t-1` ("yesterday's close") and
nothing later.  In the code `a = t - 1` is the last row that may be touched.
tests/test_no_lookahead.py proves this by destroying all data from day `t`
onwards and checking that every feature stays exactly the same.

Time conventions
----------------
* a "month" is 21 trading days, a "week" 5, a "year" 252;
* month k (k = 1 is the most recent) runs from row  a - 21k  to  a - 21(k-1);
* monthly return r_k = (value at the end - value at the start) / price at
  the start, where "value" = price + accrued interest + coupons received.
  Because bonds do not trade every day, "value at a date" means the value at
  the bond's last trade on or before that date (at most 10 trading days old).

Function names like `f067` do not exist; instead each feature is stored under
its number from the team's list, e.g. out[67] is "Short-term reversal (1m)".
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from . import config as C
from .market import Market, TSY_TENORS
from .panel import Panel

K = C.N_MONTHS_HISTORY        # 36 monthly returns
N_WEEKS = 151                 # ~3 years of weekly returns (for the tail beta)

# Earliest row for which features can be computed (enough history behind it).
MIN_EVENT_ROW = 1 + max(C.MONTH * (K + 1), C.WEEK * N_WEEKS) + C.ASOF_LIMIT_MONTHLY


# ---------------------------------------------------------------------------
# small numerical helpers
# ---------------------------------------------------------------------------
def _ols(Y: np.ndarray, X: np.ndarray):
    """Ordinary least squares of every column of Y (n, N) on the columns of
    X (n, k) plus a constant.  Returns slopes (k, N) and residuals (n, N)."""
    A = np.column_stack([np.ones(len(X)), X])
    coef = np.linalg.pinv(A) @ Y                  # (k+1, N); NaN columns stay NaN
    return coef[1:], Y - A @ coef


def _masked_slope(Y: np.ndarray, x: np.ndarray, min_obs: int) -> np.ndarray:
    """Slope of each column of Y (n, N) on x (n,), ignoring NaN entries."""
    v = np.isfinite(Y)
    n = v.sum(axis=0)
    Y0 = np.where(v, Y, 0.0)
    xv = x[:, None] * v
    with np.errstate(invalid="ignore", divide="ignore"):
        mx = xv.sum(axis=0) / n
        my = Y0.sum(axis=0) / n
        sxy = ((x[:, None] - mx) * (Y0 - my) * v).sum(axis=0)
        sxx = (((x[:, None] - mx) ** 2) * v).sum(axis=0)
        beta = sxy / sxx
    beta[n < min_obs] = np.nan
    return beta


def _days360(d1: np.ndarray, d2: np.ndarray) -> np.ndarray:
    """Number of days between two dates on the bond market's "30/360"
    calendar (every month counts as 30 days)."""
    def ymd(d):
        y = d.astype("datetime64[Y]").astype(int) + 1970
        m = d.astype("datetime64[M]").astype(int) % 12 + 1
        day = (d - d.astype("datetime64[M]")).astype(int) + 1
        return y, m, np.minimum(day, 30)
    y1, m1, dd1 = ymd(d1)
    y2, m2, dd2 = ymd(d2)
    return (y2 - y1) * 360 + (m2 - m1) * 30 + (dd2 - dd1)


def estimate_coupons(panel: Panel, last_row: int) -> np.ndarray:
    """Feature 1 (coupon rate, % of face value per year) is not in the public
    file, but it can be recovered exactly from the accrued interest: between
    two dates inside one coupon period, accrued interest grows by
    coupon x (days / 360).  Interest accrues up to the SETTLEMENT date, which
    was 2 business days after the trade until 27 May 2024 and 1 business day
    after that.  Uses rows <= last_row only."""
    traded = panel.traded[: last_row + 1]
    rows, cols = np.nonzero(traded)
    order = np.lexsort((rows, cols))
    rows, cols = rows[order], cols[order]
    trade_date = panel.dates[rows]
    settle = np.where(trade_date < np.datetime64("2024-05-28"),
                      np.busday_offset(trade_date, 2, roll="forward"),
                      np.busday_offset(trade_date, 1, roll="forward"))
    acc = panel.wide["accrued"][rows, cols].astype(np.float64)
    paid = panel.wide["coupons_paid"][rows, cols].astype(np.float64)
    same_bond = cols[1:] == cols[:-1]
    same_period = np.abs(paid[1:] - paid[:-1]) < 1e-6
    d360 = _days360(settle[:-1], settle[1:])
    good = same_bond & same_period & (d360 > 0)
    est = 360.0 * (acc[1:] - acc[:-1])[good] / d360[good]
    coupon = pd.Series(est).groupby(cols[1:][good]).median()
    out = np.full(panel.n_bonds, np.nan)
    out[coupon.index.to_numpy()] = coupon.to_numpy()
    return out


def macro_values(M: Market, a: int) -> dict[int, float]:
    """The 14 market-wide features as they were known at the close of row `a`
    (they are the same for every bond)."""
    L = M.level
    return {197: L["DGS10"][a], 198: L["T10Y2Y"][a], 199: L["DTB3"][a],
            200: L["IG_OAS"][a], 201: L["HY_OAS"][a],
            202: L["HY_OAS"][a] - L["IG_OAS"][a], 203: L["VIX"][a],
            207: L["EBP"][a], 208: L["INDPRO_YOY"][a], 209: L["CPI_YOY"][a],
            210: L["UNRATE_CHG3"][a], 211: L["EPU30"][a], 214: L["SLOOS"][a],
            216: L["USD"][a]}


MACRO_FEATURE_NUMBERS = (197, 198, 199, 200, 201, 202, 203, 207, 208, 209, 210, 211, 214, 216)


# ---------------------------------------------------------------------------
# the main routine
# ---------------------------------------------------------------------------
def features_for_day(P: Panel, M: Market, t: int, coupon: np.ndarray) -> dict:
    """Compute every feature for every bond for quote requests on trading
    day `t`, using data up to day t-1 only.

    Returns a dict with
        "features"    {feature number: array of length n_bonds}
        "eligible"    bool array: bond has every feature and is liquid enough
        "last_price"  the bond's last traded price before day t
        "ref_rows"    the row (day) of that last trade
        "duration"    the bond's duration at that trade
        "half_spread" the dealer's usual mark-up for this bond (see target.py)
    """
    if t < MIN_EVENT_ROW:
        raise ValueError("not enough history before this day")
    a = t - 1                                   # last row we may look at
    w = P.wide
    N = P.n_bonds
    f: dict[int, np.ndarray] = {}

    with warnings.catch_warnings(), np.errstate(invalid="ignore", divide="ignore"):
        warnings.simplefilter("ignore", category=RuntimeWarning)

        # ================= reference trade (the bond's last trade) =========
        ref_rows = P.asof_rows(a, C.MAX_STALE_DAYS_REF - 1)
        has_ref = ref_rows >= 0
        price = P.gather("pr", ref_rows)
        ytm = P.gather("ytm", ref_rows)
        spread = P.gather("spread", ref_rows)
        dur = P.gather("mod_dur", ref_rows)
        cvx = P.gather("convexity", ref_rows)
        amt = P.gather("amt", ref_rows)
        gap_days = np.where(has_ref,
                            (P.dates[t] - P.dates[np.clip(ref_rows, 0, None)]).astype(float),
                            np.nan)             # calendar days since last trade
        ttm = P.gather("ttm", ref_rows) - gap_days / 365.25
        age = P.gather("age", ref_rows) + gap_days / 365.25

        # ================= monthly returns ==================================
        b = a - C.MONTH * np.arange(K + 2)       # month boundaries b[0] = a
        rows_b = np.stack([P.asof_rows(int(r), C.ASOF_LIMIT_MONTHLY) for r in b[: K + 1]])
        Vb = np.stack([P.gather(P.value, rows_b[k]) for k in range(K + 1)])
        Db = np.stack([P.gather("dirty", rows_b[k]) for k in range(K + 1)])
        Sb = np.stack([P.gather("spread", rows_b[k]) for k in range(13)])
        r = (Vb[:-1] - Vb[1:]) / Db[1:]          # (K, N); r[0] = most recent month
        full_history = (rows_b >= 0).all(axis=0)

        def comp(first: int, last: int) -> np.ndarray:
            """Compounded return over months first..last (1 = most recent)."""
            return np.prod(1.0 + r[first - 1: last], axis=0) - 1.0

        # ================= market factors over the same 36 windows =========
        bK = b[: K + 1]
        rf = M.window_ret("RF", bK)
        mkt_b = M.window_ret("BONDMKT", bK)       # corporate-bond market return
        ltg = M.window_ret("TSY20", bK)           # long-term government bond
        term = ltg - rf                           # TERM factor
        deff = mkt_b - ltg                        # DEF (default) factor
        mkt_b_ex = mkt_b - rf
        ff3 = np.column_stack([M.window_ret(n, bK) for n in ("MKT", "SMB", "HML")])
        d_vix = M.window_diff("VIX", bK)
        d_y10 = M.window_diff("DGS10", bK)
        oil = M.level["OIL"][bK]
        oil_ret = oil[:-1] / oil[1:] - 1.0
        liq = M.window_mean_illiq(b)              # K+1 window averages
        d_liq = liq[:-1] - liq[1:]                # innovation in market illiquidity
        tsy = np.column_stack([M.window_ret(f"TSY{n}", bK) for n in TSY_TENORS])  # (K, 5)
        tsy_dur_now = np.array([M.level[f"TSYDUR{n}"][a] for n in TSY_TENORS])
        # Treasury whose duration is closest to the bond's ("duration-matched")
        match = np.abs(np.where(np.isfinite(dur), dur, 0.0)[:, None] - tsy_dur_now[None, :]).argmin(axis=1)
        tsy_m = tsy[:, match]                     # (K, N) matched Treasury return
        r_ex = r - rf[:, None]                    # bond return above cash
        r_da = r - tsy_m                          # "duration-adjusted" return

        # ================= daily data over recent windows ==================
        s21 = slice(a - C.MONTH + 1, a + 1)
        s63 = slice(a - 3 * C.MONTH + 1, a + 1)
        s252 = slice(a - C.YEAR + 1, a + 1)
        rd63 = P.daily_ret[s63].astype(np.float64)
        n_ret63 = np.isfinite(rd63).sum(axis=0)
        amihud_d = np.abs(rd63) / np.where(w["dvol"][s63] > 0, w["dvol"][s63], np.nan)
        amihud = np.nanmean(amihud_d, axis=0)
        bid63 = w["bid"][s63].astype(np.float64)
        ask63 = w["ask"][s63].astype(np.float64)
        two_sided = np.isfinite(bid63) & np.isfinite(ask63)
        n_two_sided = two_sided.sum(axis=0)
        eff_spread = np.nanmean(np.where(two_sided, (ask63 - bid63) / ((ask63 + bid63) / 2.0), np.nan), axis=0)
        vol21 = np.nansum(w["qvol"][s21].astype(np.float64), axis=0)
        ntr21 = np.nansum(w["ntrades"][s21].astype(np.float64), axis=0)
        mom6 = comp(2, 7)                          # the standard "momentum" signal

        # ---------------- 1. Bond contract / static -----------------------
        f[1] = coupon.copy()                                  # Coupon rate
        f[2] = ttm                                            # Time to maturity
        f[3] = age                                            # Bond age
        f[4] = age / (age + ttm)                              # Age / original maturity
        f[5] = np.log(amt)                                    # Amount outstanding (log)
        f[16] = P.is_144a.copy()                              # Rule 144A flag
        f[18] = P.sector.astype(float)                        # Industry sector

        # ---------------- 2. Yield, spread & valuation --------------------
        f[19] = ytm                                           # Yield to maturity
        f[21] = spread                                        # Credit spread
        f[24] = spread - Sb[1]                                # Spread change 1m
        f[25] = spread - Sb[3]                                # Spread change 3m
        sector_median = np.full(18, np.nan)
        for s in range(1, 18):
            sel = (P.sector == s) & np.isfinite(spread)
            if sel.any():
                sector_median[s] = np.median(spread[sel])
        f[27] = spread - sector_median[P.sector]              # Spread-to-sector-median
        f[28] = spread / dur                                  # Spread per unit duration
        f[30] = 100.0 / price                                 # Bond book-to-market
        f[31] = price                                         # Price (% of par)
        f[32] = np.where(np.isfinite(price), (price < 80.0).astype(float), np.nan)  # Distress flag
        f[33] = ytm - M.level["DTB3"][a] / 100.0              # Carry
        # Value: what is left of the spread after explaining it with the
        # bond's duration and industry (ratings are not in the free data).
        ok = np.isfinite(spread) & np.isfinite(dur)
        X = np.zeros((N, 18))
        X[np.arange(N), P.sector - 1] = 1.0
        X[:, 17] = np.where(ok, dur, 0.0)
        coef = np.linalg.lstsq(X[ok], spread[ok], rcond=None)[0]
        f[35] = np.where(ok, spread - X @ coef, np.nan)       # Value (fitted-spread residual)

        # ---------------- 3. Interest-rate risk ---------------------------
        f[41] = dur                                           # Modified duration
        f[44] = cvx                                           # Convexity
        f[45] = dur * spread                                  # DTS (duration x spread)
        f[48] = _ols(r_ex, term[:, None])[0][0]               # Term-structure beta
        slopes_by_tenor = np.stack([_ols(r, tsy[:, [j]])[0][0] for j in range(len(TSY_TENORS))])
        f[49] = slopes_by_tenor[match, np.arange(N)]          # Treasury return beta
        f[50] = r_da[0]                                       # Duration-adjusted return
        f[51] = np.where(np.isfinite(ttm), 1.0 + (ttm >= 5.0) + (ttm >= 10.0), np.nan)  # Maturity bucket

        # ---------------- 5. Bond momentum & reversal ----------------------
        f[67] = r[0]                                          # Short-term reversal (1m)
        f[68] = comp(2, 4)                                    # Momentum 3m (skip 1)
        f[69] = mom6                                          # Momentum 6m (skip 1)
        f[70] = comp(2, 12)                                   # Momentum 12m (skip 1)
        f[71] = comp(13, 36)                                  # Long-term reversal 36m
        f[72] = comp(7, 12)                                   # Intermediate momentum
        f[73] = Sb[1] - Sb[6]                                 # Spread momentum
        f[74] = np.prod(1.0 + r_da[1:7], axis=0) - 1.0        # Excess-return momentum
        _, resid = _ols(r_ex, np.column_stack([deff, term]))  # 2-factor bond model
        idio_vol = resid.std(axis=0, ddof=3)
        f[75] = resid[1:7].sum(axis=0) / idio_vol             # Idiosyncratic momentum
        ok6 = np.isfinite(mom6)
        sec_sum = np.bincount(P.sector[ok6], weights=mom6[ok6], minlength=18)
        sec_cnt = np.bincount(P.sector[ok6], minlength=18)
        own = np.where(ok6, mom6, 0.0)
        ind_mom = (sec_sum[P.sector] - own) / (sec_cnt[P.sector] - ok6)
        f[77] = ind_mom                                       # Industry bond momentum
        n_iss = int(P.issuer.max()) + 1
        iss_sum = np.bincount(P.issuer[ok6], weights=mom6[ok6], minlength=n_iss)
        iss_cnt = np.bincount(P.issuer[ok6], minlength=n_iss)
        others = iss_cnt[P.issuer] - ok6
        iss_mom = (iss_sum[P.issuer] - own) / others
        f[78] = np.where(others > 0, iss_mom, ind_mom)        # Issuer bond momentum
        f[79] = r[:12].max(axis=0)                            # Max monthly return (12m)
        f[80] = r[:12].min(axis=0)                            # Min monthly return (12m)
        hi252 = np.nanmax(w["pr"][s252].astype(np.float64), axis=0)
        f[81] = price / hi252                                 # 52-week high ratio
        f[82] = np.where(np.isfinite(r[:12]).all(axis=0), (r[:12] > 0).mean(axis=0), np.nan)  # Return consistency
        rows_w = P.asof_rows(a - C.WEEK, C.ASOF_LIMIT_MONTHLY)
        f[83] = (Vb[0] - P.gather(P.value, rows_w)) / P.gather("dirty", rows_w)  # Weekly reversal
        ok_am = np.isfinite(amihud)
        illiq_rank = np.full(N, np.nan)
        illiq_rank[ok_am] = (rankdata(amihud[ok_am]) - 1.0) / max(ok_am.sum() - 1, 1)
        f[84] = mom6 * illiq_rank                             # Momentum x illiquidity

        # ---------------- 6. Bond risk & return distribution ---------------
        f[85] = r[:12].std(axis=0, ddof=1)                    # Total volatility (12m)
        f[86] = idio_vol                                      # Idiosyncratic volatility
        f[87] = np.sqrt((np.minimum(r, 0.0) ** 2).mean(axis=0))   # Downside volatility
        var5 = np.quantile(r, 0.05, axis=0)
        f[88] = var5                                          # Value-at-Risk 5%
        below = r <= var5
        f[89] = np.where(below, r, 0.0).sum(axis=0) / below.sum(axis=0)  # Expected shortfall 5%
        m = r - r.mean(axis=0)
        s2 = (m ** 2).mean(axis=0)
        f[90] = (m ** 3).mean(axis=0) / s2 ** 1.5             # Return skewness
        f[91] = (m ** 4).mean(axis=0) / s2 ** 2 - 3.0         # Return kurtosis
        mm = mkt_b - mkt_b.mean()
        f[92] = (m * (mm ** 2)[:, None]).mean(axis=0) / (np.sqrt(s2) * (mm ** 2).mean())  # Co-skewness
        f[93] = (Sb[:12] - Sb[1:13]).std(axis=0, ddof=1)      # Spread volatility
        f[94] = np.nanstd(rd63, axis=0, ddof=1)               # Daily-return volatility
        p252 = w["pr"][s252].astype(np.float64)
        f[95] = np.nanmax(1.0 - p252 / np.fmax.accumulate(p252, axis=0), axis=0)  # Max drawdown (12m)
        f[96] = r_da[:12].std(axis=0, ddof=1)                 # Duration-adj. volatility
        rd252 = P.daily_ret[s252].astype(np.float64)
        dev = np.abs(rd252 - np.nanmean(rd252, axis=0))
        is_jump = dev > 3.0 * np.nanstd(rd252, axis=0, ddof=1)
        f[97] = is_jump.sum(axis=0) / np.isfinite(rd252).sum(axis=0)  # Jump risk
        f[98] = (m[1:] * m[:-1]).sum(axis=0) / (m ** 2).sum(axis=0)   # Return autocorrelation
        roll_vol = np.stack([r[j: j + 12].std(axis=0, ddof=1) for j in range(K - 11)])
        f[99] = roll_vol.std(axis=0, ddof=1)                  # Volatility of volatility
        wb = a - C.WEEK * np.arange(N_WEEKS + 1)
        rows_wb = [P.asof_rows(int(x), C.ASOF_LIMIT_MONTHLY) for x in wb]
        Vw = np.stack([P.gather(P.value, x) for x in rows_wb])
        Dw = np.stack([P.gather("dirty", x) for x in rows_wb])
        rw = (Vw[:-1] - Vw[1:]) / Dw[1:]                      # weekly bond returns
        mw = M.window_ret("BONDMKT", wb)                      # weekly market returns
        tail = mw <= np.quantile(mw, 0.10)                    # the worst 10% of weeks
        f[100] = _masked_slope(rw[tail], mw[tail], min_obs=10)  # Tail beta
        f[102] = np.sqrt((np.maximum(r, 0.0) ** 2).mean(axis=0))  # Upside volatility

        # ---------------- 7. Bond factor betas -----------------------------
        f[103] = _ols(r_ex, mkt_b_ex[:, None])[0][0]          # Bond market beta
        two = _ols(r_ex, np.column_stack([deff, term]))[0]
        f[104] = two[0]                                       # DEF beta
        f[105] = two[1]                                       # TERM beta
        three = _ols(r_ex, ff3)[0]
        f[106], f[107], f[108] = three[0], three[1], three[2]  # Stock mkt / SMB / HML beta
        down = mkt_b_ex < np.median(mkt_b_ex)
        f[109] = _ols(r_ex[down], mkt_b_ex[down, None])[0][0]  # Downside bond beta
        f[114] = _ols(r_ex, d_vix[:, None])[0][0]             # VIX beta
        f[115] = _ols(r_ex, d_y10[:, None])[0][0]             # Rate-level beta
        f[116] = _ols(r_ex, oil_ret[:, None])[0][0]           # Oil/commodity beta

        # ---------------- 8. Liquidity & microstructure --------------------
        f[117] = amihud                                       # Amihud illiquidity
        x1, x0 = rd63[1:], rd63[:-1]
        v = np.isfinite(x1) & np.isfinite(x0)
        n_pairs = v.sum(axis=0)
        sx1 = np.where(v, x1, 0.0)
        sx0 = np.where(v, x0, 0.0)
        cov = ((sx1 * sx0).sum(axis=0) - sx1.sum(axis=0) * sx0.sum(axis=0) / n_pairs) / (n_pairs - 1)
        f[118] = np.where(n_pairs >= 8, 2.0 * np.sqrt(np.maximum(-cov, 0.0)), np.nan)  # Roll measure
        f[120] = eff_spread                                   # Effective spread (TRACE)
        f[122] = np.log1p(vol21)                              # Trading volume (log)
        f[123] = vol21 / (amt / 1000.0)                       # Turnover
        f[124] = ntr21                                        # Number of trades
        f[125] = 1.0 - P.traded[s21].mean(axis=0)             # Zero-trading days %
        f[126] = np.nanmean((w["hi"][s63].astype(np.float64) - w["lo"][s63]) / w["pr"][s63], axis=0)  # Price dispersion
        f[127] = vol21 / ntr21                                # Avg trade size
        f[132] = gap_days                                     # Days since last trade
        f[133] = np.nan_to_num(w["qvol"][s63].astype(np.float64)).std(axis=0, ddof=1)  # Volume volatility
        f[136] = _ols(r_ex, d_liq[:, None])[0][0]             # Liquidity beta

        # ---------------- 10. Ownership, flows & supply ---------------------
        rows21 = P.asof_rows(a, C.MONTH)
        age21 = P.gather("age", rows21)
        amt21 = P.gather("amt", rows21)
        is_new = np.isfinite(age21) & (age21 < 1.0) & np.isfinite(amt21)
        new_amt = np.bincount(P.issuer[is_new], weights=amt21[is_new], minlength=n_iss)
        f[192] = np.log1p(new_amt[P.issuer])                  # New issuance by issuer
        active = P.traded[s252].any(axis=0)
        f[195] = np.bincount(P.issuer[active], minlength=n_iss)[P.issuer].astype(float)  # Bonds outstanding per issuer

        # ---------------- 11. Macro & market-level --------------------------
        for num, value in macro_values(M, a).items():
            f[num] = np.full(N, float(value))

        # ================= who is eligible? ================================
        stacked = np.stack([f[k] for k in sorted(f)])
        eligible = (has_ref & full_history
                    & (n_ret63 >= C.MIN_DAILY_RETURNS_63D)
                    & (n_two_sided >= C.MIN_TWO_SIDED_DAYS_63D)
                    & np.isfinite(stacked).all(axis=0))
        half_spread = np.maximum(eff_spread / 2.0, C.MIN_HALF_SPREAD)

    return {"features": f, "eligible": eligible, "last_price": price,
            "half_spread": half_spread, "ref_rows": ref_rows, "duration": dur}


BUILT_FEATURE_NUMBERS = (
    1, 2, 3, 4, 5, 16, 18,
    19, 21, 24, 25, 27, 28, 30, 31, 32, 33, 35,
    41, 44, 45, 48, 49, 50, 51,
    67, 68, 69, 70, 71, 72, 73, 74, 75, 77, 78, 79, 80, 81, 82, 83, 84,
    85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 102,
    103, 104, 105, 106, 107, 108, 109, 114, 115, 116,
    117, 118, 120, 122, 123, 124, 125, 126, 127, 132, 133, 136,
    192, 195,
    197, 198, 199, 200, 201, 202, 203, 207, 208, 209, 210, 211, 214, 216,
)
