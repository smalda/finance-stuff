"""
Week 4 Data Setup
Downloads and caches all datasets needed by lecture, seminar, and homework code.
Run this file first: python data_setup.py

Builds a cross-sectional feature matrix of ~150 S&P 500 stocks, 2010-2023 monthly.
Features: size, book-to-market, profitability, investment, momentum (from Week 3).
Target: next-month stock return.

Since Week 3's actual output is not available, we build the feature matrix from
scratch using yfinance price data for momentum and volatility, and approximate
the remaining characteristics from price-derived proxies. Simulated features
are clearly marked. The educational value is in the ML pipeline, not data quality.
"""
import time
import warnings

import numpy as np
import pandas as pd
import yfinance as yf
from pathlib import Path

warnings.filterwarnings("ignore", category=FutureWarning)

CACHE_DIR = Path(__file__).parent / ".cache"
CACHE_DIR.mkdir(exist_ok=True)

# ── Week-wide parameters ──────────────────────────────────
# ~150 tickers that have been consistently in the S&P 500 since 2010.
# Intentionally limited for manageable download times.
TICKERS = [
    "AAPL", "ABBV", "ABT", "ACN", "ADBE", "ADP", "AIG", "AMAT", "AMD", "AMGN",
    "AMZN", "AVGO", "AXP", "BA", "BAC", "BDX", "BIIB", "BK", "BKNG", "BLK",
    "BMY", "BRK-B", "BSX", "C", "CAT", "CB", "CI", "CL", "CMCSA", "CME",
    "COF", "COP", "COST", "CRM", "CSCO", "CVS", "CVX", "D", "DE", "DHR",
    "DIS", "DOW", "DUK", "ECL", "EL", "EMR", "EXC", "F", "FDX", "GD",
    "GE", "GILD", "GM", "GOOG", "GS", "HD", "HON", "IBM", "INTC", "INTU",
    "ISRG", "ITW", "JNJ", "JPM", "KHC", "KO", "LIN", "LLY", "LMT", "LOW",
    "MA", "MCD", "MDLZ", "MDT", "MET", "META", "MMC", "MMM", "MO", "MRK",
    "MS", "MSFT", "NEE", "NFLX", "NKE", "NOC", "NOW", "NSC", "NVDA", "ORCL",
    "PEP", "PFE", "PG", "PGR", "PM", "PNC", "PYPL", "QCOM", "RTX", "SBUX",
    "SCHW", "SHW", "SLB", "SO", "SPG", "SPGI", "SYK", "T", "TFC", "TGT",
    "TMO", "TMUS", "TXN", "UNH", "UNP", "UPS", "USB", "V", "VZ", "WBA",
    "WFC", "WM", "WMT", "XOM", "ZTS",
]

START = "2009-01-01"  # Start earlier to compute 12-month momentum from 2010
END = "2024-01-01"
FEATURE_START = "2010-01-01"  # Feature matrix starts here

# SPY for market context in seminar exercises
SPY_TICKER = "SPY"


def _download_with_retry(tickers, start, end, max_retries=3):
    """Download price data from yfinance with retries."""
    for attempt in range(max_retries):
        try:
            data = yf.download(
                tickers, start=start, end=end, auto_adjust=True,
                group_by="ticker", threads=True
            )
            if data is not None and len(data) > 0:
                return data
        except Exception as e:
            print(f"  Attempt {attempt + 1}/{max_retries} failed: {e}")
            time.sleep(2 ** attempt)
    raise RuntimeError(f"Failed to download after {max_retries} retries")


def load_price_data():
    """Download or load cached daily adjusted close prices for the universe."""
    cache_file = CACHE_DIR / "daily_close.parquet"
    if cache_file.exists():
        return pd.read_parquet(cache_file)

    print("Downloading daily price data from yfinance...")
    raw = _download_with_retry(TICKERS, START, END)

    # Extract adjusted Close for each ticker
    close_dict = {}
    for ticker in TICKERS:
        try:
            if ticker in raw.columns.get_level_values(0):
                series = raw[ticker]["Close"].dropna()
                if len(series) > 252:  # Need at least 1 year
                    close_dict[ticker] = series
        except (KeyError, TypeError):
            pass

    close = pd.DataFrame(close_dict)
    close.index = pd.to_datetime(close.index)
    close.to_parquet(cache_file)
    print(f"  Saved {close.shape[1]} tickers, {len(close)} days")
    return close


def load_spy_data():
    """Download or load cached SPY daily close."""
    cache_file = CACHE_DIR / "spy_close.parquet"
    if cache_file.exists():
        return pd.read_parquet(cache_file)

    print("Downloading SPY data...")
    spy = yf.download(SPY_TICKER, start=START, end=END, auto_adjust=True)
    spy_close = spy[["Close"]].copy()
    spy_close.columns = ["SPY"]
    spy_close.to_parquet(cache_file)
    return spy_close


def build_monthly_returns(close):
    """Compute monthly stock returns from daily close prices."""
    monthly_close = close.resample("ME").last()
    monthly_returns = monthly_close.pct_change()
    return monthly_returns


def build_feature_matrix():
    """
    Build the cross-sectional feature matrix: stock-month panel with
    5 characteristics and next-month return target.

    Features (all computed cross-sectionally, lagged by 1 month):
    - size: log market cap proxy (log of 12-month average price * arbitrary
            scaling; rank-order is what matters, not absolute level)
    - book_to_market: simulated from inverse price-to-book proxy
            (lower price relative to long-term mean suggests higher B/M)
    - profitability: simulated from 12-month price momentum quality
            (stocks with steady, positive returns are "profitable")
    - investment: simulated from asset growth proxy
            (recent price appreciation as a proxy for firm expansion)
    - momentum: standard 12-1 month momentum (cumulative return from
            month t-12 to t-1, skipping the most recent month)

    NOTE: size, book_to_market, profitability, and investment are SIMULATED
    proxies, not true fundamentals. They preserve the cross-sectional rank
    structure needed for ML demonstrations. The educational value is in the
    ML pipeline, not the data quality.
    """
    cache_file = CACHE_DIR / "feature_matrix.parquet"
    if cache_file.exists():
        return pd.read_parquet(cache_file)

    print("Building feature matrix...")
    close = load_price_data()

    # Monthly close prices
    monthly_close = close.resample("ME").last()
    monthly_returns = monthly_close.pct_change()

    # Compute features month by month
    records = []
    months = monthly_close.index[monthly_close.index >= FEATURE_START]

    for i, date in enumerate(months):
        # Need at least 13 months of history for momentum (12-1)
        hist_start_idx = monthly_close.index.get_loc(date) - 13
        if hist_start_idx < 0:
            continue

        # Get the historical window
        hist = monthly_close.iloc[hist_start_idx:monthly_close.index.get_loc(date) + 1]
        hist_ret = monthly_returns.iloc[hist_start_idx:monthly_returns.index.get_loc(date) + 1]

        for ticker in monthly_close.columns:
            price_now = monthly_close.loc[date, ticker]
            if pd.isna(price_now) or price_now <= 0:
                continue

            # Check we have enough history
            ticker_hist = hist[ticker].dropna()
            if len(ticker_hist) < 12:
                continue

            # --- Feature 1: Size (log price proxy) ---
            avg_price_12m = ticker_hist[-12:].mean()
            if pd.isna(avg_price_12m) or avg_price_12m <= 0:
                continue
            size = np.log(avg_price_12m)

            # --- Feature 2: Book-to-market proxy ---
            # Use inverse of recent price appreciation relative to long-run
            long_run_price = ticker_hist.mean()
            book_to_market = long_run_price / price_now if price_now > 0 else np.nan

            # --- Feature 3: Profitability proxy ---
            # Use consistency of positive monthly returns
            recent_rets = hist_ret[ticker].dropna()[-12:]
            if len(recent_rets) < 6:
                continue
            profitability = recent_rets[recent_rets > 0].sum() / (recent_rets.abs().sum() + 1e-10)

            # --- Feature 4: Investment proxy ---
            # Use recent 6-month price change (proxy for firm expansion)
            if len(ticker_hist) >= 7:
                investment = (ticker_hist.iloc[-1] / ticker_hist.iloc[-7]) - 1
            else:
                investment = np.nan

            # --- Feature 5: Momentum (12-1 month) ---
            # Standard: cumulative return from t-12 to t-1, skip most recent month
            if len(ticker_hist) >= 13:
                mom_start = ticker_hist.iloc[-13]
                mom_end = ticker_hist.iloc[-2]  # Skip most recent month
                momentum = (mom_end / mom_start) - 1 if mom_start > 0 else np.nan
            else:
                momentum = np.nan

            # --- Target: next-month return ---
            next_idx = monthly_close.index.get_loc(date) + 1
            if next_idx >= len(monthly_close):
                continue
            next_ret = monthly_returns.iloc[next_idx][ticker]
            if pd.isna(next_ret):
                continue

            records.append({
                "date": date,
                "ticker": ticker,
                "size": size,
                "book_to_market": book_to_market,
                "profitability": profitability,
                "investment": investment,
                "momentum": momentum,
                "next_month_return": next_ret,
            })

    feature_matrix = pd.DataFrame(records)

    # Drop rows with any NaN in features
    feature_cols = ["size", "book_to_market", "profitability", "investment", "momentum"]
    feature_matrix = feature_matrix.dropna(subset=feature_cols + ["next_month_return"])

    # Cross-sectional rank normalization within each month (0 to 1)
    # This handles the time-varying scale problem
    for col in feature_cols:
        feature_matrix[col] = feature_matrix.groupby("date")[col].rank(pct=True)

    feature_matrix = feature_matrix.reset_index(drop=True)
    feature_matrix.to_parquet(cache_file, index=False)

    n_stocks = feature_matrix["ticker"].nunique()
    n_months = feature_matrix["date"].nunique()
    print(f"  Feature matrix: {len(feature_matrix)} rows, "
          f"{n_stocks} stocks, {n_months} months")
    return feature_matrix


def load_feature_matrix():
    """Load the cached feature matrix, building it if necessary."""
    return build_feature_matrix()


# ── Shared constants for all scripts ──────────────────────
FEATURE_COLS = ["size", "book_to_market", "profitability", "investment", "momentum"]
TARGET_COL = "next_month_return"
RANDOM_SEED = 42


if __name__ == "__main__":
    # Download and cache all data
    close = load_price_data()
    print(f"Price data: {close.shape[1]} tickers, {len(close)} days, "
          f"{close.index[0].date()} to {close.index[-1].date()}")

    spy = load_spy_data()
    print(f"SPY data: {len(spy)} days")

    fm = load_feature_matrix()
    n_stocks = fm["ticker"].nunique()
    n_months = fm["date"].nunique()
    print(f"Feature matrix: {len(fm)} rows, {n_stocks} stocks, {n_months} months")
    print(f"Date range: {fm['date'].min().date()} to {fm['date'].max().date()}")
    print(f"Features: {FEATURE_COLS}")
    print(f"Target mean: {fm[TARGET_COL].mean():.4f}, std: {fm[TARGET_COL].std():.4f}")

    # Assertions
    assert n_stocks >= 80, f"Expected >= 80 stocks, got {n_stocks}"
    assert n_months >= 80, f"Expected >= 80 months, got {n_months}"
    assert len(fm) >= 5000, f"Expected >= 5000 rows, got {len(fm)}"
    for col in FEATURE_COLS:
        assert col in fm.columns, f"Missing feature column: {col}"
        assert fm[col].isna().sum() == 0, f"NaNs in {col}"
    assert TARGET_COL in fm.columns, "Missing target column"

    print("\nAll data_setup assertions passed")
