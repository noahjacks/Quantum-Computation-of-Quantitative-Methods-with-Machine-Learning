# Feature coverage: what happened to each of the 216 features

**97 of 216 features are built from free public data.** The rest need paid or licensed data. One extra input, the side of the quote request (customer buys = +1, customer sells = -1), is added because HSBC's events also include the request's own characteristics.

| Group | In the list | Built |
|---|---|---|
| Bond contract / static | 18 | 7 |
| Yield, spread & valuation | 22 | 11 |
| Interest-rate risk | 12 | 7 |
| Credit rating | 14 | 0 |
| Bond momentum & reversal | 18 | 17 |
| Bond risk & return distribution | 18 | 17 |
| Bond factor betas | 14 | 10 |
| Liquidity & microstructure | 22 | 12 |
| Issuer fundamentals | 24 | 0 |
| Equity-market signals | 22 | 0 |
| Ownership, flows & supply | 12 | 2 |
| Macro & market-level | 20 | 14 |

| # | Group | Feature | Status | Column | How it was built / why it could not be |
|---|---|---|---|---|---|
| 1 | Bond contract / static | Coupon rate | BUILT | `f001_coupon_rate` | Recovered exactly from how fast accrued interest grows between two trades (the public file omits the coupon). |
| 2 | Bond contract / static | Time to maturity | BUILT | `f002_time_to_maturity` | Years to maturity at the last trade, rolled forward to the event day. |
| 3 | Bond contract / static | Bond age | BUILT | `f003_bond_age` | Years since issue at the last trade, rolled forward to the event day. |
| 4 | Bond contract / static | Age / original maturity | BUILT | `f004_age_over_original_maturity` | age / (age + time to maturity). |
| 5 | Bond contract / static | Amount outstanding (log) | BUILT | `f005_log_amount_outstanding` | log of face value outstanding ($ thousands). |
| 6 | Bond contract / static | Offering amount (log) | not free | | Original issue size is in Mergent FISD (paid). |
| 7 | Bond contract / static | Callable flag | not free | | Call terms are in Mergent FISD (paid). |
| 8 | Bond contract / static | Make-whole call flag | not free | | Call terms are in Mergent FISD (paid). |
| 9 | Bond contract / static | Putable flag | not free | | Put terms are in Mergent FISD (paid). |
| 10 | Bond contract / static | Convertible flag | not free | | Convertibility is in Mergent FISD (paid). |
| 11 | Bond contract / static | Seniority (senior/sub) | not free | | Seniority is in Mergent FISD (paid). |
| 12 | Bond contract / static | Secured flag | not free | | Security/collateral is in Mergent FISD (paid). |
| 13 | Bond contract / static | Covenant count | not free | | Covenants are in Mergent FISD (paid). |
| 14 | Bond contract / static | Floating-rate flag | not free | | Coupon type is in Mergent FISD (paid). |
| 15 | Bond contract / static | Zero/step coupon flag | not free | | Coupon type is in Mergent FISD (paid). |
| 16 | Bond contract / static | Rule 144A flag | BUILT | `f016_rule_144a_flag` | 1 if the bond is reported in the TRACE Rule 144A database. |
| 17 | Bond contract / static | Currency / domicile | not free | | All bonds in the free US panel are US-dollar bonds, so this would be a constant. |
| 18 | Bond contract / static | Industry sector | BUILT | `f018_industry_sector` | Fama-French 17-industry code of the issuer (1-17). |
| 19 | Yield, spread & valuation | Yield to maturity | BUILT | `f019_yield_to_maturity` | Yield at the last trade (decimal). |
| 20 | Yield, spread & valuation | Yield to worst | not free | | Needs the call schedule (Mergent FISD, paid). |
| 21 | Yield, spread & valuation | Credit spread (vs Treasury) | BUILT | `f021_credit_spread` | Yield minus the Treasury yield of the same maturity, at the last trade. |
| 22 | Yield, spread & valuation | Z-spread | not free | | Needs the bond's exact cash-flow schedule (Mergent FISD, paid). |
| 23 | Yield, spread & valuation | Option-adjusted spread | not free | | Needs call terms and an interest-rate model. |
| 24 | Yield, spread & valuation | Spread change 1m | BUILT | `f024_spread_change_1m` | Credit spread now minus 21 trading days ago. |
| 25 | Yield, spread & valuation | Spread change 3m | BUILT | `f025_spread_change_3m` | Credit spread now minus 63 trading days ago. |
| 26 | Yield, spread & valuation | Spread-to-rating-median | not free | | Needs credit ratings (licensed; removed from the public file). |
| 27 | Yield, spread & valuation | Spread-to-sector-median | BUILT | `f027_spread_to_sector_median` | Spread minus the median spread of all bonds in the same industry that day. (The list says sector/rating cell; ratings are not free, so industry only.) |
| 28 | Yield, spread & valuation | Spread per unit duration | BUILT | `f028_spread_per_duration` | Credit spread / modified duration. |
| 29 | Yield, spread & valuation | Spread per unit default prob | not free | | Needs a default-probability model built on paid issuer data. |
| 30 | Yield, spread & valuation | Bond book-to-market | BUILT | `f030_bond_book_to_market` | 100 / clean price (face value over market price). |
| 31 | Yield, spread & valuation | Price (% of par) | BUILT | `f031_price` | Clean price at the last trade, % of face value. |
| 32 | Yield, spread & valuation | Distress flag (price<80) | BUILT | `f032_distress_flag` | 1 if price < 80. |
| 33 | Yield, spread & valuation | Carry | BUILT | `f033_carry` | Yield to maturity minus the 3-month Treasury bill rate. |
| 34 | Yield, spread & valuation | Roll-down | not free | | Needs a clean spread curve per issuer; not reliable from daily averages alone. |
| 35 | Yield, spread & valuation | Value (fitted-spread residual) | BUILT | `f035_value_residual` | Residual of a same-day regression of spread on duration and industry dummies across all bonds. (Rating omitted: not free.) |
| 36 | Yield, spread & valuation | CDS-bond basis | not free | | Needs CDS quotes (Markit, paid). |
| 37 | Yield, spread & valuation | CDS spread (5y) | not free | | Needs CDS quotes (Markit, paid). |
| 38 | Yield, spread & valuation | CDS spread slope | not free | | Needs CDS quotes (Markit, paid). |
| 39 | Yield, spread & valuation | Model-implied spread gap | not free | | Needs issuer equity and balance-sheet data linked to the bond (CRSP/Compustat, paid). |
| 40 | Yield, spread & valuation | Yield vs issuer curve residual | not free | | Needs a clean yield curve per issuer; not reliable from daily averages alone. |
| 41 | Interest-rate risk | Modified duration | BUILT | `f041_modified_duration` | Modified duration at the last trade. |
| 42 | Interest-rate risk | Spread duration | not free | | For fixed-rate bonds this equals modified duration (feature 41), so it would be an exact duplicate. |
| 43 | Interest-rate risk | Effective duration | not free | | Needs call terms and an interest-rate model. |
| 44 | Interest-rate risk | Convexity | BUILT | `f044_convexity` | Convexity at the last trade. |
| 45 | Interest-rate risk | DTS (duration x spread) | BUILT | `f045_dts` | Duration x spread. |
| 46 | Interest-rate risk | Key-rate duration 2y | not free | | Needs the exact cash-flow schedule (Mergent FISD, paid). |
| 47 | Interest-rate risk | Key-rate duration 10y | not free | | Needs the exact cash-flow schedule (Mergent FISD, paid). |
| 48 | Interest-rate risk | Term-structure beta | BUILT | `f048_term_structure_beta` | Slope of the bond's 36 monthly excess returns on the TERM factor (20y Treasury return minus T-bill return). |
| 49 | Interest-rate risk | Treasury return beta | BUILT | `f049_treasury_return_beta` | Slope of the bond's 36 monthly returns on the return of the Treasury whose duration is closest to the bond's. |
| 50 | Interest-rate risk | Duration-adjusted return | BUILT | `f050_duration_adjusted_return` | Last month's bond return minus the duration-matched Treasury return. |
| 51 | Interest-rate risk | Maturity bucket | BUILT | `f051_maturity_bucket` | 1 = under 5 years, 2 = 5-10 years, 3 = over 10 years. |
| 52 | Interest-rate risk | Negative convexity flag | not free | | Needs call terms (Mergent FISD, paid). |
| 53 | Credit rating | Composite rating (numeric) | not free | | Credit ratings are licensed by S&P and Moody's and are removed from the public file. |
| 54 | Credit rating | IG/HY flag | not free | | Credit ratings are licensed by S&P and Moody's and are removed from the public file. |
| 55 | Credit rating | Fallen-angel flag | not free | | Credit ratings are licensed by S&P and Moody's and are removed from the public file. |
| 56 | Credit rating | Rising-star flag | not free | | Credit ratings are licensed by S&P and Moody's and are removed from the public file. |
| 57 | Credit rating | Rating change 3m | not free | | Credit ratings are licensed by S&P and Moody's and are removed from the public file. |
| 58 | Credit rating | Rating change 12m | not free | | Credit ratings are licensed by S&P and Moody's and are removed from the public file. |
| 59 | Credit rating | Rating momentum x spread | not free | | Credit ratings are licensed by S&P and Moody's and are removed from the public file. |
| 60 | Credit rating | Agency disagreement | not free | | Credit ratings are licensed by S&P and Moody's and are removed from the public file. |
| 61 | Credit rating | Split-rated flag | not free | | Credit ratings are licensed by S&P and Moody's and are removed from the public file. |
| 62 | Credit rating | Outlook negative | not free | | Credit ratings are licensed by S&P and Moody's and are removed from the public file. |
| 63 | Credit rating | Outlook positive | not free | | Credit ratings are licensed by S&P and Moody's and are removed from the public file. |
| 64 | Credit rating | Issuer vs issue rating gap | not free | | Credit ratings are licensed by S&P and Moody's and are removed from the public file. |
| 65 | Credit rating | Market-implied rating gap | not free | | Credit ratings are licensed by S&P and Moody's and are removed from the public file. |
| 66 | Credit rating | Months since last rating action | not free | | Credit ratings are licensed by S&P and Moody's and are removed from the public file. |
| 67 | Bond momentum & reversal | Short-term reversal (1m) | BUILT | `f067_reversal_1m` | Bond total return over the last month (21 trading days). |
| 68 | Bond momentum & reversal | Momentum 3m (skip 1) | BUILT | `f068_momentum_3m` | Compounded return over months 2-4 (skipping the most recent month). |
| 69 | Bond momentum & reversal | Momentum 6m (skip 1) | BUILT | `f069_momentum_6m` | Compounded return over months 2-7. |
| 70 | Bond momentum & reversal | Momentum 12m (skip 1) | BUILT | `f070_momentum_12m` | Compounded return over months 2-12. |
| 71 | Bond momentum & reversal | Long-term reversal 36m | BUILT | `f071_long_term_reversal_36m` | Compounded return over months 13-36. |
| 72 | Bond momentum & reversal | Intermediate momentum | BUILT | `f072_intermediate_momentum` | Compounded return over months 7-12. |
| 73 | Bond momentum & reversal | Spread momentum | BUILT | `f073_spread_momentum` | Spread 1 month ago minus spread 6 months ago. |
| 74 | Bond momentum & reversal | Excess-return momentum | BUILT | `f074_excess_return_momentum` | Momentum (months 2-7) of duration-adjusted returns. |
| 75 | Bond momentum & reversal | Idiosyncratic momentum | BUILT | `f075_idiosyncratic_momentum` | Sum of months 2-7 residuals from a DEF+TERM factor regression, divided by the residual volatility. |
| 76 | Bond momentum & reversal | Rating-adjusted momentum | not free | | Needs credit ratings (licensed). |
| 77 | Bond momentum & reversal | Industry bond momentum | BUILT | `f077_industry_bond_momentum` | Average 6-month momentum of the other bonds in the same industry. |
| 78 | Bond momentum & reversal | Issuer bond momentum | BUILT | `f078_issuer_bond_momentum` | Average 6-month momentum of the issuer's other bonds (industry momentum if it has no other bond). |
| 79 | Bond momentum & reversal | Max monthly return (12m) | BUILT | `f079_max_monthly_return_12m` | Largest of the last 12 monthly returns. |
| 80 | Bond momentum & reversal | Min monthly return (12m) | BUILT | `f080_min_monthly_return_12m` | Smallest of the last 12 monthly returns. |
| 81 | Bond momentum & reversal | 52-week high ratio | BUILT | `f081_high_52w_ratio` | Last price / highest price of the last 252 trading days. |
| 82 | Bond momentum & reversal | Return consistency | BUILT | `f082_return_consistency` | Fraction of the last 12 monthly returns that were positive. |
| 83 | Bond momentum & reversal | Weekly reversal | BUILT | `f083_weekly_reversal` | Total return over the last 5 trading days. |
| 84 | Bond momentum & reversal | Momentum x illiquidity | BUILT | `f084_momentum_x_illiquidity` | 6-month momentum x the bond's illiquidity rank (0 = most liquid, 1 = least). |
| 85 | Bond risk & return distribution | Total volatility (12m) | BUILT | `f085_total_volatility_12m` | Standard deviation of the last 12 monthly returns. |
| 86 | Bond risk & return distribution | Idiosyncratic volatility | BUILT | `f086_idiosyncratic_volatility` | Std of the residuals of a 36-month DEF+TERM factor regression. |
| 87 | Bond risk & return distribution | Downside volatility | BUILT | `f087_downside_volatility` | Root-mean-square of the negative monthly returns (36 months). |
| 88 | Bond risk & return distribution | Value-at-Risk 5% | BUILT | `f088_value_at_risk_5pct` | 5th percentile of the last 36 monthly returns. |
| 89 | Bond risk & return distribution | Expected shortfall 5% | BUILT | `f089_expected_shortfall_5pct` | Average of the monthly returns at or below the 5% VaR. |
| 90 | Bond risk & return distribution | Return skewness | BUILT | `f090_return_skewness` | Skewness of the last 36 monthly returns. |
| 91 | Bond risk & return distribution | Return kurtosis | BUILT | `f091_return_kurtosis` | Excess kurtosis of the last 36 monthly returns. |
| 92 | Bond risk & return distribution | Co-skewness | BUILT | `f092_co_skewness` | Co-skewness of the bond with the corporate-bond market (36 months). |
| 93 | Bond risk & return distribution | Spread volatility | BUILT | `f093_spread_volatility` | Std of the last 12 monthly changes in credit spread. |
| 94 | Bond risk & return distribution | Daily-return volatility | BUILT | `f094_daily_return_volatility` | Std of day-to-day returns over the last 63 trading days. |
| 95 | Bond risk & return distribution | Max drawdown (12m) | BUILT | `f095_max_drawdown_12m` | Largest peak-to-trough fall in price over the last 252 trading days. |
| 96 | Bond risk & return distribution | Duration-adj. volatility | BUILT | `f096_duration_adjusted_volatility` | Std of the last 12 duration-adjusted monthly returns. |
| 97 | Bond risk & return distribution | Jump risk | BUILT | `f097_jump_risk` | Share of day-to-day returns in the last year more than 3 standard deviations from their mean. |
| 98 | Bond risk & return distribution | Return autocorrelation | BUILT | `f098_return_autocorrelation` | First-order autocorrelation of the 36 monthly returns. |
| 99 | Bond risk & return distribution | Volatility of volatility | BUILT | `f099_volatility_of_volatility` | Std of the rolling 12-month volatility over the last 25 months. |
| 100 | Bond risk & return distribution | Tail beta | BUILT | `f100_tail_beta` | Slope of the bond's weekly return on the bond market's, using only the worst 10% of weeks in 3 years. |
| 101 | Bond risk & return distribution | Variance risk exposure | not free | | Its definition (beta on VIX changes) is word-for-word the same as feature 114, so it would be an exact duplicate. |
| 102 | Bond risk & return distribution | Upside volatility | BUILT | `f102_upside_volatility` | Root-mean-square of the positive monthly returns (36 months). |
| 103 | Bond factor betas | Bond market beta | BUILT | `f103_bond_market_beta` | Slope of 36 monthly excess returns on the corporate-bond market (value-weighted, built from the panel). |
| 104 | Bond factor betas | DEF beta | BUILT | `f104_def_beta` | DEF slope in a joint DEF+TERM regression (DEF = corporate market minus long Treasury return). |
| 105 | Bond factor betas | TERM beta | BUILT | `f105_term_beta` | TERM slope in the same joint DEF+TERM regression. |
| 106 | Bond factor betas | Stock market beta | BUILT | `f106_stock_market_beta` | Market slope in a Fama-French 3-factor regression (Kenneth French's daily factors). |
| 107 | Bond factor betas | SMB beta | BUILT | `f107_smb_beta` | SMB (size) slope in the same 3-factor regression. |
| 108 | Bond factor betas | HML beta | BUILT | `f108_hml_beta` | HML (value) slope in the same 3-factor regression. |
| 109 | Bond factor betas | Downside bond beta | BUILT | `f109_downside_bond_beta` | Bond-market beta using only the months when the market return was below its median. |
| 110 | Bond factor betas | Credit-risk factor beta | not free | | Needs the published downside-risk factor series; not bundled with the free panel. |
| 111 | Bond factor betas | Liquidity factor beta | not free | | Needs the published liquidity-risk factor series; not bundled with the free panel. |
| 112 | Bond factor betas | Macro uncertainty beta | not free | | The Jurado-Ludvigson-Ng index is only distributed as a spreadsheet on the authors' site (no stable link). |
| 113 | Bond factor betas | Inflation beta | not free | | CPI is monthly and released with a lag, which cannot be lined up with our rolling 21-day windows. |
| 114 | Bond factor betas | VIX beta | BUILT | `f114_vix_beta` | Slope of 36 monthly excess returns on the monthly change in the VIX. |
| 115 | Bond factor betas | Rate-level beta | BUILT | `f115_rate_level_beta` | Slope on the monthly change in the 10-year Treasury yield. |
| 116 | Bond factor betas | Oil/commodity beta | BUILT | `f116_oil_beta` | Slope on the monthly return of WTI crude oil. |
| 117 | Liquidity & microstructure | Amihud illiquidity | BUILT | `f117_amihud_illiquidity` | Average of |daily return| / $ millions traded over the last 63 trading days. |
| 118 | Liquidity & microstructure | Roll measure | BUILT | `f118_roll_measure` | 2 x sqrt(-covariance of consecutive daily returns), 0 if the covariance is positive (63 days). |
| 119 | Liquidity & microstructure | Bid-ask spread (quoted) | not free | | Needs dealer quotes (Bloomberg/MarketAxess, paid). |
| 120 | Liquidity & microstructure | Effective spread (TRACE) | BUILT | `f120_effective_spread` | Average of (customer buy price - customer sell price) / mid on days with both, last 63 days. |
| 121 | Liquidity & microstructure | Imputed round-trip cost | not free | | Needs individual trades (the free file has daily summaries only). |
| 122 | Liquidity & microstructure | Trading volume (log) | BUILT | `f122_log_trading_volume` | log(1 + face value traded in the last 21 trading days, $ millions). |
| 123 | Liquidity & microstructure | Turnover | BUILT | `f123_turnover` | Face value traded in the last 21 days / amount outstanding. |
| 124 | Liquidity & microstructure | Number of trades | BUILT | `f124_number_of_trades` | Number of trades in the last 21 trading days. |
| 125 | Liquidity & microstructure | Zero-trading days % | BUILT | `f125_zero_trading_days` | Share of the last 21 trading days with no trade. |
| 126 | Liquidity & microstructure | Price dispersion | BUILT | `f126_price_dispersion` | Average (day's high - low) / price over the last 63 days. (Range-based: single trades are not public.) |
| 127 | Liquidity & microstructure | Avg trade size | BUILT | `f127_avg_trade_size` | Face value traded / number of trades over the last 21 days ($ millions). |
| 128 | Liquidity & microstructure | Institutional trade share | not free | | Needs individual trade sizes (the free file has daily summaries only). |
| 129 | Liquidity & microstructure | Retail trade share | not free | | Needs individual trade sizes (the free file has daily summaries only). |
| 130 | Liquidity & microstructure | Interdealer share | not free | | Needs the dealer/customer flag of each trade (Enhanced TRACE, paid). |
| 131 | Liquidity & microstructure | Dealer inventory change | not free | | Needs dealer identities (academic TRACE, paid). |
| 132 | Liquidity & microstructure | Days since last trade | BUILT | `f132_days_since_last_trade` | Calendar days between the bond's last trade and the event day. |
| 133 | Liquidity & microstructure | Volume volatility | BUILT | `f133_volume_volatility` | Std of daily face value traded over the last 63 days (0 on no-trade days). |
| 134 | Liquidity & microstructure | Liquidity score (vendor) | not free | | Vendor product (MarketAxess/Bloomberg, paid). |
| 135 | Liquidity & microstructure | Index inclusion flag | not free | | Needs index membership lists (Bloomberg/ICE, paid). |
| 136 | Liquidity & microstructure | Liquidity beta | BUILT | `f136_liquidity_beta` | Slope of 36 monthly excess returns on the monthly change in market-wide illiquidity (median Amihud across bonds). |
| 137 | Liquidity & microstructure | Trade imbalance | not free | | Needs buy and sell volumes separately (the free file only has total volume). |
| 138 | Liquidity & microstructure | Price impact (Kyle λ) | not free | | Needs signed individual trades (Enhanced TRACE, paid). |
| 139 | Issuer fundamentals | Book leverage | not free | | Needs company accounts linked to each bond (Compustat + a bond-company link, paid). |
| 140 | Issuer fundamentals | Market leverage | not free | | Needs company accounts linked to each bond (Compustat + a bond-company link, paid). |
| 141 | Issuer fundamentals | Net debt / EBITDA | not free | | Needs company accounts linked to each bond (Compustat + a bond-company link, paid). |
| 142 | Issuer fundamentals | Interest coverage | not free | | Needs company accounts linked to each bond (Compustat + a bond-company link, paid). |
| 143 | Issuer fundamentals | Cash / assets | not free | | Needs company accounts linked to each bond (Compustat + a bond-company link, paid). |
| 144 | Issuer fundamentals | Current ratio | not free | | Needs company accounts linked to each bond (Compustat + a bond-company link, paid). |
| 145 | Issuer fundamentals | Short-term debt share | not free | | Needs company accounts linked to each bond (Compustat + a bond-company link, paid). |
| 146 | Issuer fundamentals | Debt maturity wall | not free | | Needs company accounts linked to each bond (Compustat + a bond-company link, paid). |
| 147 | Issuer fundamentals | ROA | not free | | Needs company accounts linked to each bond (Compustat + a bond-company link, paid). |
| 148 | Issuer fundamentals | Operating profitability | not free | | Needs company accounts linked to each bond (Compustat + a bond-company link, paid). |
| 149 | Issuer fundamentals | Gross profitability | not free | | Needs company accounts linked to each bond (Compustat + a bond-company link, paid). |
| 150 | Issuer fundamentals | Earnings volatility | not free | | Needs company accounts linked to each bond (Compustat + a bond-company link, paid). |
| 151 | Issuer fundamentals | Asset growth | not free | | Needs company accounts linked to each bond (Compustat + a bond-company link, paid). |
| 152 | Issuer fundamentals | Net debt issuance | not free | | Needs company accounts linked to each bond (Compustat + a bond-company link, paid). |
| 153 | Issuer fundamentals | Capex / assets | not free | | Needs company accounts linked to each bond (Compustat + a bond-company link, paid). |
| 154 | Issuer fundamentals | Accruals | not free | | Needs company accounts linked to each bond (Compustat + a bond-company link, paid). |
| 155 | Issuer fundamentals | Free cash flow / debt | not free | | Needs company accounts linked to each bond (Compustat + a bond-company link, paid). |
| 156 | Issuer fundamentals | Altman Z-score | not free | | Needs company accounts linked to each bond (Compustat + a bond-company link, paid). |
| 157 | Issuer fundamentals | Ohlson O-score | not free | | Needs company accounts linked to each bond (Compustat + a bond-company link, paid). |
| 158 | Issuer fundamentals | Merton distance-to-default | not free | | Needs company accounts linked to each bond (Compustat + a bond-company link, paid). |
| 159 | Issuer fundamentals | Campbell-Hilscher-Szilagyi PD | not free | | Needs company accounts linked to each bond (Compustat + a bond-company link, paid). |
| 160 | Issuer fundamentals | Firm size (log assets) | not free | | Needs company accounts linked to each bond (Compustat + a bond-company link, paid). |
| 161 | Issuer fundamentals | Tangibility | not free | | Needs company accounts linked to each bond (Compustat + a bond-company link, paid). |
| 162 | Issuer fundamentals | Sales growth | not free | | Needs company accounts linked to each bond (Compustat + a bond-company link, paid). |
| 163 | Equity-market signals | Equity market cap (log) | not free | | Needs the issuer's share price linked to each bond (CRSP + a bond-company link, paid). |
| 164 | Equity-market signals | Equity momentum 12-1 | not free | | Needs the issuer's share price linked to each bond (CRSP + a bond-company link, paid). |
| 165 | Equity-market signals | Equity 1m return | not free | | Needs the issuer's share price linked to each bond (CRSP + a bond-company link, paid). |
| 166 | Equity-market signals | Equity 6m return | not free | | Needs the issuer's share price linked to each bond (CRSP + a bond-company link, paid). |
| 167 | Equity-market signals | Equity volatility | not free | | Needs the issuer's share price linked to each bond (CRSP + a bond-company link, paid). |
| 168 | Equity-market signals | Equity idiosyncratic vol | not free | | Needs the issuer's share price linked to each bond (CRSP + a bond-company link, paid). |
| 169 | Equity-market signals | Implied volatility (ATM) | not free | | Needs the issuer's share price linked to each bond (CRSP + a bond-company link, paid). |
| 170 | Equity-market signals | Implied vol skew | not free | | Needs the issuer's share price linked to each bond (CRSP + a bond-company link, paid). |
| 171 | Equity-market signals | Equity book-to-market | not free | | Needs the issuer's share price linked to each bond (CRSP + a bond-company link, paid). |
| 172 | Equity-market signals | Earnings-to-price | not free | | Needs the issuer's share price linked to each bond (CRSP + a bond-company link, paid). |
| 173 | Equity-market signals | Analyst forecast dispersion | not free | | Needs the issuer's share price linked to each bond (CRSP + a bond-company link, paid). |
| 174 | Equity-market signals | Analyst revisions | not free | | Needs the issuer's share price linked to each bond (CRSP + a bond-company link, paid). |
| 175 | Equity-market signals | Earnings surprise (SUE) | not free | | Needs the issuer's share price linked to each bond (CRSP + a bond-company link, paid). |
| 176 | Equity-market signals | Earnings announcement return | not free | | Needs the issuer's share price linked to each bond (CRSP + a bond-company link, paid). |
| 177 | Equity-market signals | Short interest ratio | not free | | Needs the issuer's share price linked to each bond (CRSP + a bond-company link, paid). |
| 178 | Equity-market signals | Equity lending fee | not free | | Needs the issuer's share price linked to each bond (CRSP + a bond-company link, paid). |
| 179 | Equity-market signals | Equity beta | not free | | Needs the issuer's share price linked to each bond (CRSP + a bond-company link, paid). |
| 180 | Equity-market signals | Stock turnover | not free | | Needs the issuer's share price linked to each bond (CRSP + a bond-company link, paid). |
| 181 | Equity-market signals | Equity drawdown | not free | | Needs the issuer's share price linked to each bond (CRSP + a bond-company link, paid). |
| 182 | Equity-market signals | Stock illiquidity (Amihud) | not free | | Needs the issuer's share price linked to each bond (CRSP + a bond-company link, paid). |
| 183 | Equity-market signals | Insider net buying | not free | | Needs the issuer's share price linked to each bond (CRSP + a bond-company link, paid). |
| 184 | Equity-market signals | Equity-bond return gap | not free | | Needs the issuer's share price linked to each bond (CRSP + a bond-company link, paid). |
| 185 | Ownership, flows & supply | Mutual fund ownership % | not free | | Needs fund and insurer holdings (eMAXX / NAIC, paid). |
| 186 | Ownership, flows & supply | Insurance ownership % | not free | | Needs fund and insurer holdings (eMAXX / NAIC, paid). |
| 187 | Ownership, flows & supply | Ownership concentration (HHI) | not free | | Needs fund and insurer holdings (eMAXX / NAIC, paid). |
| 188 | Ownership, flows & supply | Fund flow pressure | not free | | Needs fund and insurer holdings (eMAXX / NAIC, paid). |
| 189 | Ownership, flows & supply | ETF ownership % | not free | | Needs fund and insurer holdings (eMAXX / NAIC, paid). |
| 190 | Ownership, flows & supply | ETF flow exposure | not free | | Needs fund and insurer holdings (eMAXX / NAIC, paid). |
| 191 | Ownership, flows & supply | Fire-sale pressure | not free | | Needs fund and insurer holdings (eMAXX / NAIC, paid). |
| 192 | Ownership, flows & supply | New issuance by issuer | BUILT | `f192_new_issuance_by_issuer` | log(1 + face value of the issuer's bonds that are under 1 year old). |
| 193 | Ownership, flows & supply | New-issue concession | not free | | Needs fund and insurer holdings (eMAXX / NAIC, paid). |
| 194 | Ownership, flows & supply | Fed eligibility flag | not free | | Needs fund and insurer holdings (eMAXX / NAIC, paid). |
| 195 | Ownership, flows & supply | Bonds outstanding per issuer | BUILT | `f195_bonds_per_issuer` | Number of the issuer's bonds that traded in the last 252 trading days. |
| 196 | Ownership, flows & supply | Upcoming coupon/maturity cash | not free | | Needs fund and insurer holdings (eMAXX / NAIC, paid). |
| 197 | Macro & market-level | 10y Treasury yield | BUILT | `f197_treasury_10y_yield` | FRED DGS10, previous close. |
| 198 | Macro & market-level | 2s10s slope | BUILT | `f198_curve_slope_2s10s` | FRED T10Y2Y, previous close. |
| 199 | Macro & market-level | 3m T-bill rate | BUILT | `f199_tbill_3m_rate` | FRED DTB3, previous close. |
| 200 | Macro & market-level | IG index OAS | BUILT | `f200_ig_index_oas` | FRED BAMLC0A0CM (ICE BofA US Corporate OAS), previous close. |
| 201 | Macro & market-level | HY index OAS | BUILT | `f201_hy_index_oas` | FRED BAMLH0A0HYM2 (ICE BofA US High Yield OAS), previous close. |
| 202 | Macro & market-level | HY − IG spread | BUILT | `f202_hy_minus_ig_spread` | High-yield OAS minus investment-grade OAS. |
| 203 | Macro & market-level | VIX | BUILT | `f203_vix` | FRED VIXCLS, previous close. |
| 204 | Macro & market-level | MOVE index | not free | | The MOVE index is an ICE product and is not on FRED. |
| 205 | Macro & market-level | Cochrane-Piazzesi factor | not free | | Needs a fitted forward-rate model estimated on a long history; out of scope. |
| 206 | Macro & market-level | Ludvigson-Ng macro factor | not free | | FRED-MD refused automated downloads when this project was built. |
| 207 | Macro & market-level | Excess bond premium (GZ) | BUILT | `f207_excess_bond_premium` | Gilchrist-Zakrajsek EBP from the Federal Reserve, with a 60-day publication delay. |
| 208 | Macro & market-level | Industrial production growth | BUILT | `f208_industrial_production_growth` | Year-on-year growth of FRED INDPRO, with a 50-day publication delay. |
| 209 | Macro & market-level | CPI inflation | BUILT | `f209_cpi_inflation` | Year-on-year growth of FRED CPIAUCSL, with a 50-day publication delay. |
| 210 | Macro & market-level | Unemployment change | BUILT | `f210_unemployment_change` | 3-month change in FRED UNRATE, with a 50-day publication delay. |
| 211 | Macro & market-level | Economic policy uncertainty | BUILT | `f211_economic_policy_uncertainty` | 30-day average of the daily Baker-Bloom-Davis index (FRED USEPUINDXD). |
| 212 | Macro & market-level | JLN macro uncertainty | not free | | The Jurado-Ludvigson-Ng index is only distributed as a spreadsheet on the authors' site (no stable link). |
| 213 | Macro & market-level | Default rate (trailing 12m) | not free | | Published in Moody's / S&P reports (paid). |
| 214 | Macro & market-level | Senior loan officer survey | BUILT | `f214_senior_loan_officer_survey` | FRED DRTSCILM (net % of banks tightening), with a 50-day publication delay. |
| 215 | Macro & market-level | Bond fund aggregate flows | not free | | ICI flow data is not available as a stable free download. |
| 216 | Macro & market-level | Dollar index | BUILT | `f216_dollar_index` | FRED DTWEXBGS, previous close. |
