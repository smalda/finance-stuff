"""Week 3 — Shared data downloads and caching for Factor Models & Cross-Sectional Analysis.

Downloads equity prices, fundamental data, sector info, and official factor returns.
All downstream section files import from this module — no other file calls yf.download()
or getfactormodels directly.
"""
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf
from tqdm import tqdm

# ── Constants ────────────────────────────────────────────────────────────
CACHE_DIR = Path(__file__).parent / ".cache"
CACHE_DIR.mkdir(exist_ok=True)

START = "2014-01-01"
END = "2024-12-31"
FUND_START = "2021-01-01"  # approximate start of yfinance fundamental window

# 200 S&P 500 tickers — diverse across sectors and market caps
TICKERS = [
    # Technology
    "AAPL", "MSFT", "NVDA", "AVGO", "AMD", "CSCO", "ADBE", "CRM", "INTC", "TXN",
    "QCOM", "AMAT", "LRCX", "MU", "NOW", "INTU", "SNPS", "CDNS", "KLAC", "MCHP",
    # Communication Services
    "META", "GOOG", "NFLX", "DIS", "CMCSA", "T", "VZ", "TMUS", "EA", "TTWO",
    # Consumer Discretionary
    "AMZN", "TSLA", "HD", "MCD", "NKE", "LOW", "SBUX", "TJX", "BKNG", "ORLY",
    "MAR", "YUM", "DHI", "CMG", "ROST", "LEN", "GPC", "BBY", "POOL", "GRMN",
    # Consumer Staples
    "PG", "KO", "PEP", "COST", "WMT", "PM", "MO", "CL", "KMB", "GIS",
    "SJM", "HSY", "KHC", "MNST", "STZ", "KR", "TSN", "SYY", "CHD", "MKC",
    # Energy
    "XOM", "CVX", "COP", "SLB", "EOG", "MPC", "PSX", "VLO", "OXY", "HES",
    "DVN", "FANG", "HAL", "BKR", "CTRA", "APA", "TRGP", "KMI", "WMB", "OKE",
    # Financials
    "JPM", "BAC", "WFC", "GS", "MS", "BLK", "SCHW", "AXP", "USB", "PNC",
    "TFC", "AIG", "MET", "PRU", "ALL", "CB", "MMC", "AON", "ICE", "CME",
    # Healthcare
    "LLY", "UNH", "JNJ", "PFE", "ABT", "MRK", "TMO", "AMGN", "MDT", "ISRG",
    "BMY", "GILD", "VRTX", "REGN", "ZTS", "SYK", "BSX", "BDX", "EW", "HCA",
    # Industrials
    "CAT", "GE", "HON", "UNP", "RTX", "BA", "DE", "LMT", "MMM", "GD",
    "WM", "ITW", "EMR", "ETN", "FDX", "CSX", "NSC", "PCAR", "ROK", "FAST",
    # Materials
    "LIN", "APD", "SHW", "ECL", "NEM", "FCX", "NUE", "VMC", "MLM", "DOW",
    # Real Estate
    "PLD", "AMT", "CCI", "EQIX", "PSA", "SPG", "DLR", "O", "WELL", "ARE",
    # Utilities
    "NEE", "DUK", "SO", "D", "AEP", "SRE", "EXC", "XEL", "WEC", "ES",
]

UNIVERSE_SIZE = len(TICKERS)

# Subset for Sections 1/2/4 — 20 diverse stocks across caps and sectors
DEMO_TICKERS = [
    "AAPL", "NVDA", "JPM", "XOM", "JNJ", "KO", "TSLA", "PG", "GS", "AMD",
    "NEE", "PFE", "CAT", "COST", "NFLX", "WMT", "BA", "MO", "LLY", "T",
]


# ── Load functions ───────────────────────────────────────────────────────

def load_equity_prices() -> pd.DataFrame:
    """Load daily adjusted close prices for the full 200-ticker universe.

    Returns a DataFrame indexed by date with tickers as columns.
    """
    cache_file = CACHE_DIR / "equity_prices.parquet"
    if cache_file.exists():
        return pd.read_parquet(cache_file)

    print("Downloading equity prices (200 tickers, 2014-2024)...")
    data = yf.download(TICKERS, start=START, end=END, auto_adjust=True, progress=True)
    prices = data["Close"]

    # Drop tickers with >50% missing
    completeness = prices.notna().mean()
    valid = completeness[completeness > 0.50].index.tolist()
    prices = prices[valid]
    prices.index = pd.to_datetime(prices.index).tz_localize(None)

    print(f"  Tickers downloaded: {len(valid)}/{UNIVERSE_SIZE}")
    print(f"  Date range: {prices.index.min().date()} to {prices.index.max().date()}")
    print(f"  Rows: {len(prices)}")

    failed = set(TICKERS) - set(valid)
    if failed:
        print(f"  Failed/incomplete tickers ({len(failed)}): {sorted(failed)}")

    prices.to_parquet(cache_file)
    return prices


def load_monthly_returns() -> pd.DataFrame:
    """Compute monthly returns from daily prices (full universe).

    Returns a DataFrame indexed by month-end date with tickers as columns.
    """
    cache_file = CACHE_DIR / "monthly_returns.parquet"
    if cache_file.exists():
        return pd.read_parquet(cache_file)

    prices = load_equity_prices()
    monthly_prices = prices.resample("ME").last()
    monthly_returns = monthly_prices.pct_change().dropna(how="all")
    monthly_returns.to_parquet(cache_file)
    return monthly_returns


def load_factor_data(model: str = "5") -> pd.DataFrame:
    """Load official Fama-French factor data from getfactormodels.

    Args:
        model: '3', '5', or '6' (6 includes momentum UMD).

    Returns a DataFrame indexed by date with factor columns.
    """
    cache_file = CACHE_DIR / f"ff{model}_factors.parquet"
    if cache_file.exists():
        return pd.read_parquet(cache_file)

    from getfactormodels import FamaFrenchFactors
    print(f"Downloading FF{model} factor data...")
    ff = FamaFrenchFactors(model=model, frequency="M")
    df = ff.to_pandas()

    # Standardize index
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date")
    df.index = pd.to_datetime(df.index)
    df.index.name = "date"

    print(f"  FF{model} factors: {list(df.columns)}")
    print(f"  Date range: {df.index.min().date()} to {df.index.max().date()}")
    print(f"  Rows: {len(df)}")

    df.to_parquet(cache_file)
    return df


def load_carhart_factors() -> pd.DataFrame:
    """Load Carhart 4-factor model data (includes MOM)."""
    cache_file = CACHE_DIR / "carhart_factors.parquet"
    if cache_file.exists():
        return pd.read_parquet(cache_file)

    from getfactormodels import CarhartFactors
    print("Downloading Carhart factor data...")
    cf = CarhartFactors(frequency="M")
    df = cf.to_pandas()

    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date")
    df.index = pd.to_datetime(df.index)
    df.index.name = "date"

    print(f"  Carhart factors: {list(df.columns)}")
    print(f"  Date range: {df.index.min().date()} to {df.index.max().date()}")
    print(f"  Rows: {len(df)}")

    df.to_parquet(cache_file)
    return df


def load_fundamentals() -> dict:
    """Download annual balance sheet, income statement, and sector info for all tickers.

    Returns a dict with keys:
        'balance_sheet': DataFrame (multi-index: ticker, date)
        'income_stmt': DataFrame (multi-index: ticker, date)
        'sector': Series (index: ticker, values: sector string)
        'market_cap': Series (index: ticker, values: current market cap)
        'shares': DataFrame (multi-index: ticker, date) — shares outstanding
    """
    cache_file = CACHE_DIR / "fundamentals.parquet"
    sector_file = CACHE_DIR / "sector_info.parquet"
    income_file = CACHE_DIR / "income_stmt.parquet"
    shares_file = CACHE_DIR / "shares_outstanding.parquet"
    mcap_file = CACHE_DIR / "market_cap.parquet"

    if all(f.exists() for f in [cache_file, sector_file, income_file, shares_file, mcap_file]):
        return {
            "balance_sheet": pd.read_parquet(cache_file),
            "income_stmt": pd.read_parquet(income_file),
            "sector": pd.read_parquet(sector_file).squeeze(),
            "market_cap": pd.read_parquet(mcap_file).squeeze(),
            "shares": pd.read_parquet(shares_file),
        }

    prices = load_equity_prices()
    valid_tickers = prices.columns.tolist()

    bs_records = []
    inc_records = []
    shares_records = []
    sector_map = {}
    mcap_map = {}
    failures = []

    print(f"Downloading fundamentals for {len(valid_tickers)} tickers...")
    for ticker_str in tqdm(valid_tickers, desc="Fundamentals"):
        for attempt in range(3):
            try:
                tk = yf.Ticker(ticker_str)

                # Balance sheet
                bs = tk.balance_sheet
                if bs is not None and not bs.empty:
                    for col_date in bs.columns:
                        row = {}
                        for field in ["Stockholders Equity", "Total Assets",
                                      "Ordinary Shares Number"]:
                            if field in bs.index:
                                row[field] = bs.loc[field, col_date]
                        if row:
                            row["ticker"] = ticker_str
                            row["date"] = pd.Timestamp(col_date).tz_localize(None)
                            bs_records.append(row)

                        # Shares outstanding
                        if "Ordinary Shares Number" in bs.index:
                            shares_records.append({
                                "ticker": ticker_str,
                                "date": pd.Timestamp(col_date).tz_localize(None),
                                "shares": bs.loc["Ordinary Shares Number", col_date],
                            })

                # Income statement
                inc = tk.income_stmt
                if inc is not None and not inc.empty:
                    for col_date in inc.columns:
                        row = {"ticker": ticker_str,
                               "date": pd.Timestamp(col_date).tz_localize(None)}
                        for field in ["Operating Income", "Net Income",
                                      "Total Revenue", "Pretax Income"]:
                            if field in inc.index:
                                row[field] = inc.loc[field, col_date]
                        inc_records.append(row)

                # Sector & market cap
                info = tk.info
                sector_map[ticker_str] = info.get("sector", "Other")
                mcap_map[ticker_str] = info.get("marketCap", np.nan)

                break  # success
            except Exception as e:
                if attempt < 2:
                    time.sleep(2 ** (attempt + 1))
                else:
                    failures.append(ticker_str)
                    print(f"  FAILED {ticker_str}: {e}")

    # Build DataFrames
    bs_df = pd.DataFrame(bs_records)
    if not bs_df.empty:
        bs_df = bs_df.set_index(["ticker", "date"]).sort_index()

    inc_df = pd.DataFrame(inc_records)
    if not inc_df.empty:
        inc_df = inc_df.set_index(["ticker", "date"]).sort_index()

    shares_df = pd.DataFrame(shares_records)
    if not shares_df.empty:
        shares_df = shares_df.set_index(["ticker", "date"]).sort_index()

    sector_s = pd.Series(sector_map, name="sector")
    mcap_s = pd.Series(mcap_map, name="market_cap")

    print(f"  Balance sheet records: {len(bs_df)}")
    print(f"  Income statement records: {len(inc_df)}")
    print(f"  Sectors: {sector_s.nunique()} unique")
    print(f"  Failures: {len(failures)}")
    if failures:
        print(f"  Failed tickers: {sorted(failures)}")

    # Save
    bs_df.to_parquet(cache_file)
    inc_df.to_parquet(income_file)
    shares_df.to_parquet(shares_file)
    sector_s.to_frame().to_parquet(sector_file)
    mcap_s.to_frame().to_parquet(mcap_file)

    return {
        "balance_sheet": bs_df,
        "income_stmt": inc_df,
        "sector": sector_s,
        "market_cap": mcap_s,
        "shares": shares_df,
    }


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=FutureWarning)

    print("=" * 60)
    print("Week 3 — Data Setup")
    print("=" * 60)

    prices = load_equity_prices()
    monthly = load_monthly_returns()
    ff3 = load_factor_data("3")
    ff5 = load_factor_data("5")
    ff6 = load_factor_data("6")
    carhart = load_carhart_factors()
    fundamentals = load_fundamentals()

    print("\n" + "=" * 60)
    print("Data setup complete.")
    print(f"  Equity prices: {prices.shape}")
    print(f"  Monthly returns: {monthly.shape}")
    print(f"  FF3 factors: {ff3.shape}")
    print(f"  FF5 factors: {ff5.shape}")
    print(f"  FF6 factors: {ff6.shape}")
    print(f"  Carhart factors: {carhart.shape}")
    print(f"  Balance sheet: {fundamentals['balance_sheet'].shape}")
    print(f"  Income stmt: {fundamentals['income_stmt'].shape}")
    print(f"  Sectors: {fundamentals['sector'].nunique()} unique")
    print("=" * 60)
