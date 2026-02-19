"""
Exercise 3: Data Quality Gauntlet — How Clean Is "Clean" Data?

Acceptance criteria (from README):
- All 20 tickers downloaded with >= 2000 rows each
- Completeness ratio computed for all tickers; at least one ticker has ratio < 100%
- Stale price detection finds >= 1 instance across the 20-ticker set
- Outlier return detection flags >= 5 dates across all tickers (March 2020 COVID crash should be among them)
- OHLC consistency check runs on all rows; violation count reported per ticker
- Impact assessment shows return difference between raw and cleaned data for at least the worst-case ticker
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yfinance as yf
from pandas.tseries.holiday import USFederalHolidayCalendar
from pandas.tseries.offsets import CustomBusinessDay

sys.path.insert(0, str(Path(__file__).parent.parent))
from data_setup import CACHE_DIR

# ── CELL: define_data_universe ──────────────────────────
# Purpose: Select 20 diverse tickers spanning large-cap, mid-cap,
#   small-cap, and ETFs. Diversity matters — data quality issues are
#   more frequent in less liquid names and during corporate events.
# Takeaway: Even "clean" data from a reputable source (yfinance) has
#   quality issues. Testing at scale and across market caps reveals them.

TICKERS = [
    # Large-cap liquid
    "AAPL", "MSFT", "JPM", "TSLA", "SPY", "QQQ", "IWM", "XLE",
    # Large-cap with history
    "GE", "META", "AMZN", "GOOG", "BRK-B", "V",
    # Healthcare/pharma
    "UNH", "PFE",
    # Industrials
    "INTC", "BA", "DIS", "NFLX"
]

START = "2010-01-01"
END = "2025-01-01"

print(f"Analyzing data quality for {len(TICKERS)} tickers")


# ── CELL: download_ohlcv_data ───────────────────────────
# Purpose: Download 10 years of daily OHLCV data for all 20 tickers.
# Takeaway: yfinance handles bulk downloads but silently drops rows
#   for tickers with data issues, suspensions, or delistings. The
#   shape of the result tells you which tickers have problems.

cache_file = CACHE_DIR / "data_quality_test.parquet"
if cache_file.exists():
    print("Loading cached data...")
    data = pd.read_parquet(cache_file)
else:
    print("Downloading 20 tickers...")
    data = yf.download(TICKERS, start=START, end=END, auto_adjust=False, progress=False)
    data.to_parquet(cache_file)

print(f"Data shape: {data.shape}")
print(f"Date range: {data.index[0].date()} to {data.index[-1].date()}")


# ── CELL: audit_missing_data ─────────────────────────────
# Purpose: For each ticker, compute completeness ratio: actual rows
#   vs. expected trading days (using NYSE calendar). Identify gaps.
# Takeaway: Large-cap liquid stocks are ~100% complete. Less liquid
#   names have missing days around corporate actions, suspensions, etc.

us_bd = CustomBusinessDay(calendar=USFederalHolidayCalendar())
expected_dates = pd.date_range(start=START, end=END, freq=us_bd)
n_expected = len(expected_dates)

close = data["Close"]
completeness = {}

print("\nCompleteness Audit:")
print(f"Expected trading days (NYSE calendar): {n_expected}")

for ticker in TICKERS:
    n_actual = close[ticker].dropna().shape[0]
    ratio = n_actual / n_expected
    completeness[ticker] = {
        "actual": n_actual,
        "expected": n_expected,
        "ratio": ratio
    }
    if ratio < 0.99:
        print(f"  {ticker}: {n_actual}/{n_expected} = {ratio:.2%} (INCOMPLETE)")

# Find tickers with long gaps (> 5 consecutive trading days missing)
print("\nLong gaps (>5 consecutive trading days):")
for ticker in TICKERS:
    ticker_data = close[ticker].reindex(expected_dates)
    missing_mask = ticker_data.isna()

    # Find runs of missing data
    missing_runs = missing_mask.astype(int).groupby((missing_mask != missing_mask.shift()).cumsum()).sum()
    long_gaps = missing_runs[missing_runs > 5]

    if len(long_gaps) > 0:
        print(f"  {ticker}: {len(long_gaps)} gaps, max {long_gaps.max()} days")


# ── CELL: detect_stale_prices ───────────────────────────
# Purpose: Find consecutive days with identical Close prices (to 4 decimals).
# Takeaway: Stale prices occur in illiquid names, during trading halts,
#   or from data vendor errors. They corrupt return calculations (zero
#   return when there should be movement) and signal data unreliability.

print("\nStale Price Detection:")
stale_counts = {}

for ticker in TICKERS:
    prices = close[ticker].dropna()
    # Round to 4 decimals to avoid floating-point noise
    rounded = prices.round(4)
    stale_mask = rounded == rounded.shift(1)
    n_stale = stale_mask.sum()
    stale_counts[ticker] = n_stale

    if n_stale > 10:
        print(f"  {ticker}: {n_stale} days with stale prices")

total_stale = sum(stale_counts.values())
print(f"Total stale price days across all tickers: {total_stale}")


# ── CELL: detect_outlier_returns ────────────────────────
# Purpose: Flag daily returns exceeding ±15%. Determine whether each
#   is genuine (COVID crash, earnings, FDA approval) or a data artifact.
# Takeaway: March 2020 will show many legitimate outliers (COVID crash).
#   Other outliers may be split errors, bad data, or genuine events.
#   The classifier separates signal from noise.

outlier_threshold = 0.15
outliers = []

for ticker in TICKERS:
    returns = close[ticker].pct_change()
    outlier_mask = returns.abs() > outlier_threshold
    outlier_dates = returns[outlier_mask].dropna()

    for date, ret in outlier_dates.items():
        outliers.append({
            "ticker": ticker,
            "date": date,
            "return": ret,
            "genuine": None  # Will classify below
        })

outliers_df = pd.DataFrame(outliers)

# Simple classifier: March 2020 outliers are genuine (COVID crash)
# Outliers near known split dates are artifacts
# Everything else is flagged for manual review
covid_start, covid_end = pd.Timestamp("2020-03-01"), pd.Timestamp("2020-04-01")

def classify_outlier(row):
    if covid_start <= row["date"] <= covid_end:
        return "genuine (COVID crash)"
    elif row["date"].month == 3 and row["date"].year == 2020:
        return "genuine (COVID crash)"
    elif abs(row["return"]) > 0.5:
        return "artifact (likely split)"
    else:
        return "review"

outliers_df["classification"] = outliers_df.apply(classify_outlier, axis=1)

print(f"\nOutlier Return Detection (>{outlier_threshold:.0%}):")
print(f"Total outliers flagged: {len(outliers_df)}")
print(outliers_df["classification"].value_counts())

print(f"\nCOVID crash examples (March 2020):")
covid_outliers = outliers_df[outliers_df["classification"].str.contains("COVID")]
print(covid_outliers[["ticker", "date", "return"]].head(10).to_string(index=False))


# ── CELL: check_ohlc_consistency ────────────────────────
# Purpose: Verify that High >= max(Open, Close) and Low <= min(Open, Close).
#   Also check Volume > 0. Count violations per ticker.
# Takeaway: OHLC violations are rare in major stocks but occur more
#   frequently in: (1) low-volume stocks, (2) around corporate actions,
#   (3) with bad data vendors. Zero violations = high quality signal.

print("\nOHLC Consistency Checks:")
open_p = data["Open"]
high_p = data["High"]
low_p = data["Low"]
close_p = data["Close"]
volume = data["Volume"]

ohlc_violations = {}

for ticker in TICKERS:
    o = open_p[ticker].dropna()
    h = high_p[ticker].dropna()
    l = low_p[ticker].dropna()
    c = close_p[ticker].dropna()
    v = volume[ticker].dropna()

    # Align all series
    idx = o.index.intersection(h.index).intersection(l.index).intersection(c.index)
    o, h, l, c, v = o[idx], h[idx], l[idx], c[idx], v[idx]

    # Check violations
    high_violations = (h < o) | (h < c)
    low_violations = (l > o) | (l > c)
    volume_violations = v <= 0

    n_violations = high_violations.sum() + low_violations.sum() + volume_violations.sum()
    ohlc_violations[ticker] = n_violations

    if n_violations > 0:
        print(f"  {ticker}: {n_violations} OHLC violations")

total_violations = sum(ohlc_violations.values())
print(f"Total OHLC violations across all tickers: {total_violations}")


# ── CELL: impact_assessment ─────────────────────────────
# Purpose: Take a simple buy-and-hold strategy and compute returns
#   using raw data vs. cleaned data (forward-filled, outliers capped).
#   Measure the impact on the worst-case ticker.
# Takeaway: For large-cap liquid stocks, data quality issues have
#   minimal impact (<0.5% annualized). For names with gaps or bad data,
#   the impact can be material (>2% annualized error).

print("\nImpact Assessment: Buy-and-Hold Returns (Raw vs. Cleaned)")

impact_results = []

for ticker in TICKERS:
    raw_prices = close[ticker].dropna()

    # Cleaned version: forward-fill gaps (limit 5 days), cap outlier returns at ±30%
    cleaned_prices = close[ticker].ffill(limit=5).dropna()
    cleaned_returns = cleaned_prices.pct_change().dropna()
    cleaned_returns_capped = cleaned_returns.clip(lower=-0.30, upper=0.30)

    # Reconstruct price series from capped returns
    # Start from first price, apply capped returns
    cleaned_prices_reconstructed = cleaned_prices.iloc[0] * (1 + cleaned_returns_capped).cumprod()
    # Add back the first price (which has no return)
    cleaned_prices_reconstructed = pd.concat([
        pd.Series([cleaned_prices.iloc[0]], index=[cleaned_prices.index[0]]),
        cleaned_prices_reconstructed
    ])

    # Compute annualized returns
    n_years = (raw_prices.index[-1] - raw_prices.index[0]).days / 365.25

    if n_years > 0 and len(raw_prices) > 1 and len(cleaned_prices_reconstructed) > 1:
        raw_total = (raw_prices.iloc[-1] / raw_prices.iloc[0]) - 1
        cleaned_total = (cleaned_prices_reconstructed.iloc[-1] / cleaned_prices_reconstructed.iloc[0]) - 1

        raw_annualized = (1 + raw_total) ** (1 / n_years) - 1
        cleaned_annualized = (1 + cleaned_total) ** (1 / n_years) - 1

        impact = abs(raw_annualized - cleaned_annualized)

        impact_results.append({
            "ticker": ticker,
            "raw_ann": raw_annualized,
            "cleaned_ann": cleaned_annualized,
            "impact": impact
        })

impact_df = pd.DataFrame(impact_results).sort_values("impact", ascending=False)

print(impact_df.head(10).to_string(index=False))

worst_case = impact_df.iloc[0]
print(f"\nWorst-case ticker: {worst_case['ticker']}")
print(f"  Raw annualized return: {worst_case['raw_ann']:.2%}")
print(f"  Cleaned annualized return: {worst_case['cleaned_ann']:.2%}")
print(f"  Impact: {worst_case['impact']:.2%}")


if __name__ == "__main__":
    # ── ASSERTIONS ─────────────────────────────────────

    # Verify all 20 tickers downloaded with >= 2000 rows each
    for ticker in TICKERS:
        n_rows = completeness[ticker]["actual"]
        assert n_rows >= 2000, (
            f"{ticker} should have >= 2000 rows, got {n_rows}"
        )

    # Verify at least one ticker has completeness < 100%
    incomplete_tickers = [t for t, m in completeness.items() if m["ratio"] < 1.0]
    assert len(incomplete_tickers) >= 1, (
        f"Expected at least 1 ticker with <100% completeness, got {len(incomplete_tickers)}"
    )

    # Verify stale prices detected
    assert total_stale >= 1, (
        f"Expected >= 1 stale price instance, got {total_stale}"
    )

    # Verify outliers flagged (should include COVID crash)
    assert len(outliers_df) >= 5, (
        f"Expected >= 5 outlier returns, got {len(outliers_df)}"
    )

    covid_count = outliers_df["classification"].str.contains("COVID").sum()
    assert covid_count > 0, (
        f"Expected COVID crash outliers in March 2020, found {covid_count}"
    )

    # OHLC consistency check ran on all tickers
    assert len(ohlc_violations) == len(TICKERS), (
        f"OHLC check should run on all {len(TICKERS)} tickers"
    )

    # Impact assessment computed for worst-case ticker
    assert len(impact_df) > 0, "Impact assessment should produce results"
    assert worst_case["impact"] >= 0, "Impact should be non-negative"

    print("\n✓ Exercise 3: All acceptance criteria passed")
