"""
Week 2 Data Setup — Financial Time Series & Volatility
Downloads and caches all datasets needed by lecture, seminar, and homework code.
Run this file first: python data_setup.py
"""
import yfinance as yf
import pandas as pd
from pathlib import Path

CACHE_DIR = Path(__file__).parent / ".cache"
CACHE_DIR.mkdir(exist_ok=True)
CACHE_FILE = CACHE_DIR / "all_prices.parquet"

# ── Week-wide parameters ──────────────────────────────────
START = "2010-01-01"
END = "2025-01-01"

# Lecture uses primarily SPY
LECTURE_TICKERS = ["SPY"]

# Seminar uses a diverse 20-ticker universe
SEMINAR_TICKERS = [
    "AAPL", "MSFT", "JPM", "JNJ",      # large-cap
    "TSLA", "NVDA", "MARA",              # high-volatility
    "SPY", "QQQ", "IWM", "TLT", "GLD",  # ETFs
    "XLE", "XLF", "XLK",                 # sector ETFs
    "GE", "INTC", "BA", "PFE", "DIS",   # mid-cap / special
]

# Seminar ex2 / ex3 / ex4 and homework subsets
GARCH_TICKERS = ["SPY", "AAPL", "JPM", "TSLA", "TLT"]
FRACDIFF_TICKERS = ["SPY", "AAPL", "TSLA", "TLT", "GLD"]

# Homework uses 10 diverse tickers
HW_TICKERS = [
    "SPY", "QQQ", "TLT", "GLD",          # ETFs
    "AAPL", "MSFT", "JPM", "TSLA",       # equities
    "XLE", "BA",                           # sector / special
]

ALL_TICKERS = sorted(set(
    LECTURE_TICKERS + SEMINAR_TICKERS + GARCH_TICKERS
    + FRACDIFF_TICKERS + HW_TICKERS
))


def _ensure_cache():
    """Download all tickers once and cache. Returns the full DataFrame."""
    if CACHE_FILE.exists():
        return pd.read_parquet(CACHE_FILE)

    raw = yf.download(ALL_TICKERS, start=START, end=END, auto_adjust=True, progress=False)
    if isinstance(raw.columns, pd.MultiIndex):
        prices = raw["Close"]
    else:
        prices = raw[["Close"]].rename(columns={"Close": ALL_TICKERS[0]})
    prices.to_parquet(CACHE_FILE)
    return prices


def load_prices(tickers=None):
    """Load adjusted close prices for requested tickers from the single cache."""
    df = _ensure_cache()
    if tickers is None:
        return df
    available = [t for t in tickers if t in df.columns]
    return df[available]


def load_all():
    """Download all tickers used across all notebooks."""
    return _ensure_cache()


if __name__ == "__main__":
    df = load_all()
    print(f"✓ Data: {len(df)} rows, {df.shape[1]} tickers")
    print(f"  Date range: {df.index[0].date()} to {df.index[-1].date()}")
    print(f"  Tickers: {list(df.columns)}")
    missing = df.isna().sum()
    if missing.any():
        print(f"  Missing values:\n{missing[missing > 0]}")
