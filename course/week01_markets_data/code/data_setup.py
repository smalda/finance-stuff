"""
Week 1 Data Setup
Downloads and caches all datasets needed by lecture, seminar, and homework code.
Run this file first: python data_setup.py
"""
import yfinance as yf
import pandas as pd
from pathlib import Path

CACHE_DIR = Path(__file__).parent / ".cache"
CACHE_DIR.mkdir(exist_ok=True)

# ── Week-wide parameters ──────────────────────────────────
TICKERS = ["AAPL", "TSLA", "JPM", "SPY", "GE"]
START = "2010-01-01"
END = "2025-01-01"


def load_or_download():
    """Download OHLCV data (auto_adjust=False) if not cached, else load from Parquet."""
    cache_file = CACHE_DIR / "ohlcv_raw.parquet"
    if cache_file.exists():
        return pd.read_parquet(cache_file)

    raw = yf.download(TICKERS, start=START, end=END, auto_adjust=False)
    raw.to_parquet(cache_file)
    return raw


def load_adjusted():
    """Download OHLCV data (auto_adjust=True) if not cached, else load from Parquet."""
    cache_file = CACHE_DIR / "ohlcv_adjusted.parquet"
    if cache_file.exists():
        return pd.read_parquet(cache_file)

    adj = yf.download(TICKERS, start=START, end=END, auto_adjust=True)
    adj.to_parquet(cache_file)
    return adj


if __name__ == "__main__":
    df = load_or_download()
    print(f"Raw data: {len(df)} rows, {df.index[0].date()} to {df.index[-1].date()}")
    adj = load_adjusted()
    print(f"Adjusted data: {len(adj)} rows, {adj.index[0].date()} to {adj.index[-1].date()}")
    print("data_setup: All downloads complete")
