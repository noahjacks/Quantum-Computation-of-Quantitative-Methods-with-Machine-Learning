"""
feature_catalog.py  --  WHICH OF THE 216 FEATURES WE BUILT, AND HOW
==================================================================

The team's list (docs/Features_Qiskit.csv) has 216 features.  This file is the
single record of what happened to each one:

* BUILT[n] = (short name, how we computed it from free data)
* NOT_BUILT_REASON explains, group by group, why a feature is impossible
  without paid data.

`write_coverage_report()` turns this into docs/FEATURE_COVERAGE.md so that
anyone can check the claim "we used every feature that is free".
"""
from __future__ import annotations

import pandas as pd

from . import config as C

# number -> (column slug, how it is computed)
BUILT: dict[int, tuple[str, str]] = {
    1: ("coupon_rate", "Recovered exactly from how fast accrued interest grows between two trades (the public file omits the coupon)."),
    2: ("time_to_maturity", "Years to maturity at the last trade, rolled forward to the event day."),
    3: ("bond_age", "Years since issue at the last trade, rolled forward to the event day."),
    4: ("age_over_original_maturity", "age / (age + time to maturity)."),
    5: ("log_amount_outstanding", "log of face value outstanding ($ thousands)."),
    16: ("rule_144a_flag", "1 if the bond is reported in the TRACE Rule 144A database."),
    18: ("industry_sector", "Fama-French 17-industry code of the issuer (1-17)."),
    19: ("yield_to_maturity", "Yield at the last trade (decimal)."),
    21: ("credit_spread", "Yield minus the Treasury yield of the same maturity, at the last trade."),
    24: ("spread_change_1m", "Credit spread now minus 21 trading days ago."),
    25: ("spread_change_3m", "Credit spread now minus 63 trading days ago."),
    27: ("spread_to_sector_median", "Spread minus the median spread of all bonds in the same industry that day. (The list says sector/rating cell; ratings are not free, so industry only.)"),
    28: ("spread_per_duration", "Credit spread / modified duration."),
    30: ("bond_book_to_market", "100 / clean price (face value over market price)."),
    31: ("price", "Clean price at the last trade, % of face value."),
    32: ("distress_flag", "1 if price < 80."),
    33: ("carry", "Yield to maturity minus the 3-month Treasury bill rate."),
    35: ("value_residual", "Residual of a same-day regression of spread on duration and industry dummies across all bonds. (Rating omitted: not free.)"),
    41: ("modified_duration", "Modified duration at the last trade."),
    44: ("convexity", "Convexity at the last trade."),
    45: ("dts", "Duration x spread."),
    48: ("term_structure_beta", "Slope of the bond's 36 monthly excess returns on the TERM factor (20y Treasury return minus T-bill return)."),
    49: ("treasury_return_beta", "Slope of the bond's 36 monthly returns on the return of the Treasury whose duration is closest to the bond's."),
    50: ("duration_adjusted_return", "Last month's bond return minus the duration-matched Treasury return."),
    51: ("maturity_bucket", "1 = under 5 years, 2 = 5-10 years, 3 = over 10 years."),
    67: ("reversal_1m", "Bond total return over the last month (21 trading days)."),
    68: ("momentum_3m", "Compounded return over months 2-4 (skipping the most recent month)."),
    69: ("momentum_6m", "Compounded return over months 2-7."),
    70: ("momentum_12m", "Compounded return over months 2-12."),
    71: ("long_term_reversal_36m", "Compounded return over months 13-36."),
    72: ("intermediate_momentum", "Compounded return over months 7-12."),
    73: ("spread_momentum", "Spread 1 month ago minus spread 6 months ago."),
    74: ("excess_return_momentum", "Momentum (months 2-7) of duration-adjusted returns."),
    75: ("idiosyncratic_momentum", "Sum of months 2-7 residuals from a DEF+TERM factor regression, divided by the residual volatility."),
    77: ("industry_bond_momentum", "Average 6-month momentum of the other bonds in the same industry."),
    78: ("issuer_bond_momentum", "Average 6-month momentum of the issuer's other bonds (industry momentum if it has no other bond)."),
    79: ("max_monthly_return_12m", "Largest of the last 12 monthly returns."),
    80: ("min_monthly_return_12m", "Smallest of the last 12 monthly returns."),
    81: ("high_52w_ratio", "Last price / highest price of the last 252 trading days."),
    82: ("return_consistency", "Fraction of the last 12 monthly returns that were positive."),
    83: ("weekly_reversal", "Total return over the last 5 trading days."),
    84: ("momentum_x_illiquidity", "6-month momentum x the bond's illiquidity rank (0 = most liquid, 1 = least)."),
    85: ("total_volatility_12m", "Standard deviation of the last 12 monthly returns."),
    86: ("idiosyncratic_volatility", "Std of the residuals of a 36-month DEF+TERM factor regression."),
    87: ("downside_volatility", "Root-mean-square of the negative monthly returns (36 months)."),
    88: ("value_at_risk_5pct", "5th percentile of the last 36 monthly returns."),
    89: ("expected_shortfall_5pct", "Average of the monthly returns at or below the 5% VaR."),
    90: ("return_skewness", "Skewness of the last 36 monthly returns."),
    91: ("return_kurtosis", "Excess kurtosis of the last 36 monthly returns."),
    92: ("co_skewness", "Co-skewness of the bond with the corporate-bond market (36 months)."),
    93: ("spread_volatility", "Std of the last 12 monthly changes in credit spread."),
    94: ("daily_return_volatility", "Std of day-to-day returns over the last 63 trading days."),
    95: ("max_drawdown_12m", "Largest peak-to-trough fall in price over the last 252 trading days."),
    96: ("duration_adjusted_volatility", "Std of the last 12 duration-adjusted monthly returns."),
    97: ("jump_risk", "Share of day-to-day returns in the last year more than 3 standard deviations from their mean."),
    98: ("return_autocorrelation", "First-order autocorrelation of the 36 monthly returns."),
    99: ("volatility_of_volatility", "Std of the rolling 12-month volatility over the last 25 months."),
    100: ("tail_beta", "Slope of the bond's weekly return on the bond market's, using only the worst 10% of weeks in 3 years."),
    102: ("upside_volatility", "Root-mean-square of the positive monthly returns (36 months)."),
    103: ("bond_market_beta", "Slope of 36 monthly excess returns on the corporate-bond market (value-weighted, built from the panel)."),
    104: ("def_beta", "DEF slope in a joint DEF+TERM regression (DEF = corporate market minus long Treasury return)."),
    105: ("term_beta", "TERM slope in the same joint DEF+TERM regression."),
    106: ("stock_market_beta", "Market slope in a Fama-French 3-factor regression (Kenneth French's daily factors)."),
    107: ("smb_beta", "SMB (size) slope in the same 3-factor regression."),
    108: ("hml_beta", "HML (value) slope in the same 3-factor regression."),
    109: ("downside_bond_beta", "Bond-market beta using only the months when the market return was below its median."),
    114: ("vix_beta", "Slope of 36 monthly excess returns on the monthly change in the VIX."),
    115: ("rate_level_beta", "Slope on the monthly change in the 10-year Treasury yield."),
    116: ("oil_beta", "Slope on the monthly return of WTI crude oil."),
    117: ("amihud_illiquidity", "Average of |daily return| / $ millions traded over the last 63 trading days."),
    118: ("roll_measure", "2 x sqrt(-covariance of consecutive daily returns), 0 if the covariance is positive (63 days)."),
    120: ("effective_spread", "Average of (customer buy price - customer sell price) / mid on days with both, last 63 days."),
    122: ("log_trading_volume", "log(1 + face value traded in the last 21 trading days, $ millions)."),
    123: ("turnover", "Face value traded in the last 21 days / amount outstanding."),
    124: ("number_of_trades", "Number of trades in the last 21 trading days."),
    125: ("zero_trading_days", "Share of the last 21 trading days with no trade."),
    126: ("price_dispersion", "Average (day's high - low) / price over the last 63 days. (Range-based: single trades are not public.)"),
    127: ("avg_trade_size", "Face value traded / number of trades over the last 21 days ($ millions)."),
    132: ("days_since_last_trade", "Calendar days between the bond's last trade and the event day."),
    133: ("volume_volatility", "Std of daily face value traded over the last 63 days (0 on no-trade days)."),
    136: ("liquidity_beta", "Slope of 36 monthly excess returns on the monthly change in market-wide illiquidity (median Amihud across bonds)."),
    192: ("new_issuance_by_issuer", "log(1 + face value of the issuer's bonds that are under 1 year old)."),
    195: ("bonds_per_issuer", "Number of the issuer's bonds that traded in the last 252 trading days."),
    197: ("treasury_10y_yield", "FRED DGS10, previous close."),
    198: ("curve_slope_2s10s", "FRED T10Y2Y, previous close."),
    199: ("tbill_3m_rate", "FRED DTB3, previous close."),
    200: ("ig_index_oas", "FRED BAMLC0A0CM (ICE BofA US Corporate OAS), previous close."),
    201: ("hy_index_oas", "FRED BAMLH0A0HYM2 (ICE BofA US High Yield OAS), previous close."),
    202: ("hy_minus_ig_spread", "High-yield OAS minus investment-grade OAS."),
    203: ("vix", "FRED VIXCLS, previous close."),
    207: ("excess_bond_premium", "Gilchrist-Zakrajsek EBP from the Federal Reserve, with a 60-day publication delay."),
    208: ("industrial_production_growth", "Year-on-year growth of FRED INDPRO, with a 50-day publication delay."),
    209: ("cpi_inflation", "Year-on-year growth of FRED CPIAUCSL, with a 50-day publication delay."),
    210: ("unemployment_change", "3-month change in FRED UNRATE, with a 50-day publication delay."),
    211: ("economic_policy_uncertainty", "30-day average of the daily Baker-Bloom-Davis index (FRED USEPUINDXD)."),
    214: ("senior_loan_officer_survey", "FRED DRTSCILM (net % of banks tightening), with a 50-day publication delay."),
    216: ("dollar_index", "FRED DTWEXBGS, previous close."),
}

# Why the rest could not be built.  Specific numbers first, then whole groups.
NOT_BUILT_SPECIFIC: dict[int, str] = {
    6: "Original issue size is in Mergent FISD (paid).",
    7: "Call terms are in Mergent FISD (paid).", 8: "Call terms are in Mergent FISD (paid).",
    9: "Put terms are in Mergent FISD (paid).", 10: "Convertibility is in Mergent FISD (paid).",
    11: "Seniority is in Mergent FISD (paid).", 12: "Security/collateral is in Mergent FISD (paid).",
    13: "Covenants are in Mergent FISD (paid).", 14: "Coupon type is in Mergent FISD (paid).",
    15: "Coupon type is in Mergent FISD (paid).",
    17: "All bonds in the free US panel are US-dollar bonds, so this would be a constant.",
    20: "Needs the call schedule (Mergent FISD, paid).",
    22: "Needs the bond's exact cash-flow schedule (Mergent FISD, paid).",
    23: "Needs call terms and an interest-rate model.",
    26: "Needs credit ratings (licensed; removed from the public file).",
    29: "Needs a default-probability model built on paid issuer data.",
    34: "Needs a clean spread curve per issuer; not reliable from daily averages alone.",
    36: "Needs CDS quotes (Markit, paid).", 37: "Needs CDS quotes (Markit, paid).",
    38: "Needs CDS quotes (Markit, paid).",
    39: "Needs issuer equity and balance-sheet data linked to the bond (CRSP/Compustat, paid).",
    40: "Needs a clean yield curve per issuer; not reliable from daily averages alone.",
    42: "For fixed-rate bonds this equals modified duration (feature 41), so it would be an exact duplicate.",
    43: "Needs call terms and an interest-rate model.",
    46: "Needs the exact cash-flow schedule (Mergent FISD, paid).",
    47: "Needs the exact cash-flow schedule (Mergent FISD, paid).",
    52: "Needs call terms (Mergent FISD, paid).",
    76: "Needs credit ratings (licensed).",
    101: "Its definition (beta on VIX changes) is word-for-word the same as feature 114, so it would be an exact duplicate.",
    110: "Needs the published downside-risk factor series; not bundled with the free panel.",
    111: "Needs the published liquidity-risk factor series; not bundled with the free panel.",
    112: "The Jurado-Ludvigson-Ng index is only distributed as a spreadsheet on the authors' site (no stable link).",
    113: "CPI is monthly and released with a lag, which cannot be lined up with our rolling 21-day windows.",
    119: "Needs dealer quotes (Bloomberg/MarketAxess, paid).",
    121: "Needs individual trades (the free file has daily summaries only).",
    128: "Needs individual trade sizes (the free file has daily summaries only).",
    129: "Needs individual trade sizes (the free file has daily summaries only).",
    130: "Needs the dealer/customer flag of each trade (Enhanced TRACE, paid).",
    131: "Needs dealer identities (academic TRACE, paid).",
    134: "Vendor product (MarketAxess/Bloomberg, paid).",
    135: "Needs index membership lists (Bloomberg/ICE, paid).",
    137: "Needs buy and sell volumes separately (the free file only has total volume).",
    138: "Needs signed individual trades (Enhanced TRACE, paid).",
    204: "The MOVE index is an ICE product and is not on FRED.",
    205: "Needs a fitted forward-rate model estimated on a long history; out of scope.",
    206: "FRED-MD refused automated downloads when this project was built.",
    212: "The Jurado-Ludvigson-Ng index is only distributed as a spreadsheet on the authors' site (no stable link).",
    213: "Published in Moody's / S&P reports (paid).",
    215: "ICI flow data is not available as a stable free download.",
}
NOT_BUILT_GROUP: dict[str, str] = {
    "Credit rating": "Credit ratings are licensed by S&P and Moody's and are removed from the public file.",
    "Issuer fundamentals": "Needs company accounts linked to each bond (Compustat + a bond-company link, paid).",
    "Equity-market signals": "Needs the issuer's share price linked to each bond (CRSP + a bond-company link, paid).",
    "Ownership, flows & supply": "Needs fund and insurer holdings (eMAXX / NAIC, paid).",
}


def column_name(number: int) -> str:
    """Column name used in the data tables, e.g. 'f067_reversal_1m'."""
    return f"f{number:03d}_{BUILT[number][0]}"


def load_feature_list() -> pd.DataFrame:
    """The team's original list of 216 features."""
    df = pd.read_csv(C.FEATURE_LIST_CSV)
    return df.rename(columns={"#": "number"})


def coverage_table() -> pd.DataFrame:
    """One row per feature of the team's list: built or not, and why."""
    df = load_feature_list()
    status, column, note = [], [], []
    for _, row in df.iterrows():
        n = int(row["number"])
        if n in BUILT:
            status.append("BUILT")
            column.append(column_name(n))
            note.append(BUILT[n][1])
        else:
            status.append("not free")
            column.append("")
            note.append(NOT_BUILT_SPECIFIC.get(n) or NOT_BUILT_GROUP.get(row["Category"])
                        or "Needs paid data.")
    df["status"], df["column"], df["how / why not"] = status, column, note
    return df


def write_coverage_report() -> pd.DataFrame:
    df = coverage_table()
    n_built = int((df["status"] == "BUILT").sum())
    lines = ["# Feature coverage: what happened to each of the 216 features",
             "",
             f"**{n_built} of {len(df)} features are built from free public data.** "
             "The rest need paid or licensed data. One extra input, the side of the "
             "quote request (customer buys = +1, customer sells = -1), is added because "
             "HSBC's events also include the request's own characteristics.",
             "",
             "| Group | In the list | Built |", "|---|---|---|"]
    grp = df.groupby("Category", sort=False)["status"].agg(["size", lambda s: (s == "BUILT").sum()])
    for cat, (size, built) in grp.iterrows():
        lines.append(f"| {cat} | {size} | {built} |")
    lines += ["", "| # | Group | Feature | Status | Column | How it was built / why it could not be |",
              "|---|---|---|---|---|---|"]
    for _, r in df.iterrows():
        lines.append(f"| {r['number']} | {r['Category']} | {r['Feature']} | {r['status']} | "
                     f"`{r['column']}` | {r['how / why not']} |".replace("| `` |", "| |"))
    (C.ROOT / "docs" / "FEATURE_COVERAGE.md").write_text("\n".join(lines) + "\n")
    return df
