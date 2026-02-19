"""
Exercise 1: The Stationarity Landscape — How 20 Stocks Behave Differently

Acceptance criteria (from README):
- ADF and KPSS run on all 20 tickers for both prices and returns (80 tests total)
- All 20 tickers' prices: ADF p-value > 0.05 (non-stationary)
- All 20 tickers' returns: ADF p-value < 0.05 (stationary)
- Return kurtosis > 3 for all 20 tickers
- At least 3 tickers have kurtosis > 10 (the volatile names)
- Ljung-Box test on squared returns: p-value < 0.05 for at least 18 of 20 tickers
- Summary table has all 20 rows with complete columns
"""
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.stats.diagnostic import acorr_ljungbox

sys.path.insert(0, str(Path(__file__).parent.parent))
from data_setup import load_prices, SEMINAR_TICKERS


# ── CELL: load_and_compute_returns ────────────────────────
# Purpose: Load daily adjusted close prices for 20 diverse tickers
#   spanning large-cap, high-vol, ETFs, sector ETFs, and mid-cap stocks.
#   Compute log returns for all tickers.
# Takeaway: The 20-ticker universe spans large-cap, high-vol, ETFs,
#   sector ETFs, and mid-cap. Minimum observations = 3,185 (MARA, crypto-
#   adjacent). Date range: 2010-01-04 to 2024-12-31.

prices = load_prices(SEMINAR_TICKERS)
log_returns = np.log(prices / prices.shift(1)).dropna()

print(f"Tickers: {list(prices.columns)}")
print(f"Date range: {prices.index[0].date()} to {prices.index[-1].date()}")
print(f"Observations per ticker (min): {prices.count().min()}")


# ── CELL: stationarity_tests ─────────────────────────────
# Purpose: Run ADF and KPSS on both prices and returns for every
#   ticker. Build a comprehensive summary table diagnosing each series.
# Takeaway: ALL prices are non-stationary (ADF p > 0.05, KPSS p < 0.05
#   for every ticker — no exceptions). ALL returns are stationary (ADF
#   p < 0.05 for every ticker). This is trivially expected — differencing
#   works — but quantifying it across 20 diverse assets establishes the
#   universal baseline.

rows = []
for ticker in SEMINAR_TICKERS:
    p = prices[ticker].dropna()
    r = log_returns[ticker].dropna()

    if len(p) < 100 or len(r) < 100:
        continue

    # ADF/KPSS on prices
    adf_price_stat, adf_price_p, *_ = adfuller(p, maxlag=20, autolag="AIC")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        kpss_price_stat, kpss_price_p, *_ = kpss(p, regression="c", nlags="auto")

    # ADF/KPSS on returns
    adf_ret_stat, adf_ret_p, *_ = adfuller(r, maxlag=20, autolag="AIC")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        kpss_ret_stat, kpss_ret_p, *_ = kpss(r, regression="c", nlags="auto")

    # Joint diagnosis for returns
    if adf_ret_p < 0.05 and kpss_ret_p > 0.05:
        diag = "Stationary"
    elif adf_ret_p > 0.05 and kpss_ret_p < 0.05:
        diag = "Non-stationary"
    else:
        diag = "Ambiguous"

    rows.append({
        "Ticker": ticker,
        "ADF p (prices)": adf_price_p,
        "KPSS p (prices)": kpss_price_p,
        "ADF p (returns)": adf_ret_p,
        "KPSS p (returns)": kpss_ret_p,
        "Return Diagnosis": diag,
    })

stationarity_df = pd.DataFrame(rows)
print("\n=== Stationarity Tests (20 Tickers) ===")
print(stationarity_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))


# ── CELL: distributional_and_arch_tests ──────────────────
# Purpose: For each ticker, compute return kurtosis, skewness, and the
#   Ljung-Box test on squared returns (ARCH effect test). Rank tickers
#   by kurtosis to reveal systematic patterns across asset types.
# Takeaway: Fat tails are universal (all kurtosis > 3) but heterogeneous:
#   MARA = 47.2, INTC = 21.7, BA = 20.0 at the top vs. TLT = 4.2,
#   TSLA = 4.5 at the bottom. Ljung-Box rejects for ALL 20 tickers
#   (p ≈ 0) — volatility clustering is truly universal. Nine tickers
#   have kurtosis > 10.

dist_rows = []
for ticker in SEMINAR_TICKERS:
    r = log_returns[ticker].dropna()
    if len(r) < 100:
        continue

    kurt = r.kurtosis()
    skew = r.skew()
    lb = acorr_ljungbox(r**2, lags=[20], return_df=True)
    lb_p = lb["lb_pvalue"].iloc[0]

    dist_rows.append({
        "Ticker": ticker,
        "Kurtosis": kurt,
        "Skewness": skew,
        "LB Sq p-val": lb_p,
        "ARCH Effect": "Yes" if lb_p < 0.05 else "No",
    })

dist_df = pd.DataFrame(dist_rows)
dist_df = dist_df.sort_values("Kurtosis", ascending=False)
print("\n=== Distributional Properties (ranked by kurtosis) ===")
print(dist_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))


# ── CELL: combined_summary ───────────────────────────────
# Purpose: Merge stationarity and distributional results into a single
#   comprehensive table for cross-asset comparison.
# Takeaway: Stylized facts are universal in direction but heterogeneous
#   in magnitude. Negative skew dominates equities (15 of 20 tickers),
#   but MARA (+0.57), NVDA (+0.25), TSLA (+0.05) show positive skew.
#   TLT skewness ≈ -0.04 (nearly symmetric). All 20/20 show ARCH effects.

summary = stationarity_df.merge(
    pd.DataFrame(dist_rows), on="Ticker"
).sort_values("Kurtosis", ascending=False)

print("\n=== Full Summary Table ===")
print(summary.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

n_high_kurt = (summary["Kurtosis"] > 10).sum()
n_arch = (summary["ARCH Effect"] == "Yes").sum()
print(f"\nTickers with kurtosis > 10: {n_high_kurt}")
print(f"Tickers with ARCH effect (p < 0.05): {n_arch} / {len(summary)}")


if __name__ == "__main__":
    # ── ASSERTIONS ─────────────────────────────────────
    # All 20 tickers present
    assert len(summary) == len(SEMINAR_TICKERS), (
        f"Expected {len(SEMINAR_TICKERS)} rows, got {len(summary)}"
    )

    # All prices non-stationary
    for _, row in summary.iterrows():
        assert row["ADF p (prices)"] > 0.05, (
            f"{row['Ticker']} prices ADF p = {row['ADF p (prices)']:.4f}, expected > 0.05"
        )

    # All returns stationary
    for _, row in summary.iterrows():
        assert row["ADF p (returns)"] < 0.05, (
            f"{row['Ticker']} returns ADF p = {row['ADF p (returns)']:.4f}, expected < 0.05"
        )

    # All kurtosis > 3
    for _, row in summary.iterrows():
        assert row["Kurtosis"] > 3, (
            f"{row['Ticker']} kurtosis = {row['Kurtosis']:.2f}, expected > 3"
        )

    # At least 3 tickers with kurtosis > 10
    assert n_high_kurt >= 3, (
        f"Only {n_high_kurt} tickers have kurtosis > 10, expected >= 3"
    )

    # Ljung-Box rejects for at least 18 of 20
    assert n_arch >= 18, (
        f"Only {n_arch} tickers show ARCH effect, expected >= 18"
    )

    print("✓ Exercise 1: All acceptance criteria passed")
