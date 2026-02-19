"""
Week 3 Data Setup: Factor Models & Cross-Sectional Analysis

Downloads and caches all datasets needed by lecture, seminar, and homework code.
Run this file first: python data_setup.py

Data sources:
- yfinance: Daily adjusted close prices, quarterly balance sheet & income statement
- getfactormodels: Official Fama-French 5-factor + Momentum (monthly, decimal form)
"""
import warnings
warnings.filterwarnings("ignore")

import yfinance as yf
import pandas as pd
import numpy as np
from pathlib import Path

CACHE_DIR = Path(__file__).parent / ".cache"
CACHE_DIR.mkdir(exist_ok=True)

# ── Week-wide parameters ──────────────────────────────────

MARKET_PROXY = "SPY"

# 20 stocks spanning defensive to aggressive for CAPM demonstrations
DEMO_TICKERS = [
    "DUK", "SO",           # Utilities (expected beta < 0.6)
    "KO", "PG", "WMT",     # Consumer staples (expected beta 0.5-0.8)
    "JNJ", "PFE",          # Healthcare
    "JPM", "GS",           # Financials
    "XOM", "CVX",          # Energy
    "AAPL", "MSFT",        # Large-cap tech
    "GOOGL", "META",       # Large-cap tech (growth)
    "BA", "NFLX",          # Cyclical / growth
    "AMD", "NVDA",         # High-beta semiconductors
    "TSLA",                # Very high beta
]

# Full S&P 500 universe for factor construction (~500 stocks)
# Source: Wikipedia List of S&P 500 companies. Some recent additions may lack
# long price histories — the download step drops tickers with >50% missing data.
SP500_TICKERS = [
    "A", "AAPL", "ABBV", "ABNB", "ABT", "ACGL", "ACN", "ADBE", "ADI", "ADM",
    "ADP", "ADSK", "AEE", "AEP", "AES", "AFL", "AIG", "AIZ", "AJG", "AKAM",
    "ALB", "ALGN", "ALL", "ALLE", "AMAT", "AMCR", "AMD", "AME", "AMGN", "AMP",
    "AMT", "AMZN", "ANET", "AON", "AOS", "APA", "APD", "APH", "APO", "APTV",
    "ARE", "ARES", "ATO", "AVB", "AVGO", "AVY", "AWK", "AXON", "AXP", "AZO",
    "BA", "BAC", "BALL", "BAX", "BBY", "BDX", "BEN", "BF-B", "BG", "BIIB",
    "BK", "BKNG", "BKR", "BLDR", "BLK", "BMY", "BR", "BRK-B", "BRO", "BSX",
    "BX", "BXP", "C", "CAG", "CAH", "CARR", "CAT", "CB", "CBOE", "CBRE",
    "CCI", "CCL", "CDNS", "CDW", "CEG", "CF", "CFG", "CHD", "CHRW", "CHTR",
    "CI", "CINF", "CL", "CLX", "CMCSA", "CME", "CMG", "CMI", "CMS", "CNC",
    "CNP", "COF", "COO", "COP", "COR", "COST", "CPAY", "CPB", "CPRT", "CPT",
    "CRH", "CRL", "CRM", "CRWD", "CSCO", "CSGP", "CSX", "CTAS", "CTRA", "CTSH",
    "CTVA", "CVS", "CVX", "D", "DAL", "DD", "DE", "DECK", "DELL", "DG",
    "DGX", "DHI", "DHR", "DIS", "DLR", "DLTR", "DOV", "DOW", "DPZ", "DRI",
    "DTE", "DUK", "DVA", "DVN", "DXCM", "EA", "EBAY", "ECL", "ED", "EFX",
    "EIX", "EL", "ELV", "EME", "EMR", "EOG", "EQIX", "EQR", "EQT", "ERIE",
    "ES", "ESS", "ETN", "ETR", "EVRG", "EW", "EXC", "EXPD", "EXPE", "EXR",
    "F", "FANG", "FAST", "FCX", "FDS", "FDX", "FE", "FFIV", "FICO", "FIS",
    "FISV", "FITB", "FOX", "FOXA", "FRT", "FSLR", "FTNT", "FTV", "GD", "GDDY",
    "GE", "GEHC", "GEN", "GEV", "GILD", "GIS", "GL", "GLW", "GM", "GNRC",
    "GOOG", "GOOGL", "GPC", "GPN", "GRMN", "GS", "GWW", "HAL", "HAS", "HBAN",
    "HCA", "HD", "HIG", "HII", "HLT", "HOLX", "HON", "HPE", "HPQ", "HRL",
    "HSIC", "HST", "HSY", "HUBB", "HUM", "HWM", "IBM", "ICE", "IDXX", "IEX",
    "IFF", "INCY", "INTC", "INTU", "INVH", "IP", "IQV", "IR", "IRM", "ISRG",
    "IT", "ITW", "IVZ", "J", "JBHT", "JBL", "JCI", "JKHY", "JNJ", "JPM",
    "KDP", "KEY", "KEYS", "KHC", "KIM", "KKR", "KLAC", "KMB", "KMI", "KO",
    "KR", "KVUE", "L", "LDOS", "LEN", "LH", "LHX", "LIN", "LLY", "LMT",
    "LNT", "LOW", "LRCX", "LULU", "LUV", "LVS", "LW", "LYB", "LYV", "MA",
    "MAA", "MAR", "MAS", "MCD", "MCHP", "MCK", "MCO", "MDLZ", "MDT", "MET",
    "META", "MGM", "MKC", "MLM", "MMM", "MNST", "MO", "MOH", "MOS", "MPC",
    "MPWR", "MRK", "MRNA", "MS", "MSCI", "MSFT", "MSI", "MTB", "MTCH", "MTD",
    "MU", "NCLH", "NDAQ", "NDSN", "NEE", "NEM", "NFLX", "NI", "NKE", "NOC",
    "NOW", "NRG", "NSC", "NTAP", "NTRS", "NUE", "NVDA", "NVR", "NWS", "NWSA",
    "NXPI", "O", "ODFL", "OKE", "OMC", "ON", "ORCL", "ORLY", "OTIS", "OXY",
    "PANW", "PAYC", "PAYX", "PCAR", "PCG", "PEG", "PEP", "PFE", "PFG", "PG",
    "PGR", "PH", "PHM", "PKG", "PLD", "PLTR", "PM", "PNC", "PNR", "PNW",
    "PODD", "POOL", "PPG", "PPL", "PRU", "PSA", "PSX", "PTC", "PWR", "PYPL",
    "QCOM", "RCL", "REG", "REGN", "RJF", "RL", "RMD", "ROK", "ROL", "ROP",
    "ROST", "RSG", "RTX", "RVTY", "SBAC", "SBUX", "SCHW", "SHW", "SJM", "SLB",
    "SMCI", "SNA", "SNPS", "SO", "SPG", "SPGI", "SRE", "STE", "STLD", "STT",
    "STX", "STZ", "SWK", "SWKS", "SYF", "SYK", "SYY", "T", "TAP", "TDG",
    "TDY", "TECH", "TEL", "TER", "TFC", "TGT", "TJX", "TMO", "TMUS", "TPR",
    "TRGP", "TRMB", "TROW", "TRV", "TSCO", "TSLA", "TSN", "TT", "TTD", "TTWO",
    "TXN", "TXT", "TYL", "UAL", "UBER", "UDR", "UHS", "ULTA", "UNH", "UNP",
    "UPS", "URI", "USB", "V", "VICI", "VLO", "VLTO", "VMC", "VRSK", "VRSN",
    "VRTX", "VST", "VTR", "VTRS", "VZ", "WAB", "WAT", "WBD", "WDAY", "WDC",
    "WEC", "WELL", "WFC", "WM", "WMB", "WMT", "WRB", "WST", "WTW", "WY",
    "WYNN", "XEL", "XOM", "XYL", "YUM", "ZBH", "ZBRA", "ZTS",
]

START = "2004-01-01"
END = "2024-01-01"


# ── Internal helpers ───────────────────────────────────────

def _download_prices(tickers, cache_name):
    """Download and cache daily adjusted close prices for a list of tickers."""
    cache_file = CACHE_DIR / f"{cache_name}.parquet"
    if cache_file.exists():
        return pd.read_parquet(cache_file)

    raw = yf.download(tickers, start=START, end=END, auto_adjust=True, progress=False)

    if isinstance(raw.columns, pd.MultiIndex):
        prices = raw["Close"]
    else:
        prices = raw["Close"]
        if isinstance(prices, pd.Series):
            prices = prices.to_frame(name=tickers[0] if isinstance(tickers, list) else tickers)

    prices = prices.dropna(axis=1, thresh=len(prices) * 0.5)
    prices.to_parquet(cache_file)
    return prices


# ── Public data loaders ────────────────────────────────────

def load_demo_prices():
    """Daily adjusted close for 20 demo tickers + SPY.

    Returns a DataFrame with DatetimeIndex and one column per ticker (21 total).
    Used by: lecture S1 (CAPM).
    """
    return _download_prices(DEMO_TICKERS + [MARKET_PROXY], "demo_prices")


def load_sp500_prices():
    """Daily adjusted close for ~500 S&P 500 tickers.

    Returns a DataFrame with DatetimeIndex and one column per ticker.
    Tickers with >50% missing price days are dropped (recent IPOs, etc.).
    Used by: lecture S2-S7, seminar, homework.
    """
    return _download_prices(SP500_TICKERS, "sp500_prices")


def load_fundamentals():
    """Quarterly fundamental data for SP500_TICKERS.

    Returns a long-format DataFrame with columns:
        ticker, date, total_equity, total_assets, total_debt,
        shares_outstanding, operating_income, net_income

    Coverage: last ~4 years of quarterly reports (yfinance limitation).
    Used by: lecture S2-S3, S7, seminar, homework.
    """
    cache_file = CACHE_DIR / "fundamentals.parquet"
    if cache_file.exists():
        return pd.read_parquet(cache_file)

    BS_FIELDS = {
        "Total Assets": "total_assets",
        "Total Equity Gross Minority Interest": "total_equity",
        "Total Debt": "total_debt",
        "Ordinary Shares Number": "shares_outstanding",
    }
    IS_FIELDS = {
        "Operating Income": "operating_income",
        "Net Income": "net_income",
    }

    records = []
    import time
    for i, ticker in enumerate(SP500_TICKERS):
        if (i + 1) % 50 == 0:
            print(f"  Fundamentals: {i + 1}/{len(SP500_TICKERS)}")
            time.sleep(1)  # brief pause every 50 tickers to respect rate limits
        try:
            stock = yf.Ticker(ticker)
            bs = stock.quarterly_balance_sheet
            inc = stock.quarterly_income_stmt

            if bs is None or bs.empty:
                continue

            for date in bs.columns:
                row = {"ticker": ticker, "date": date}
                for src_name, dst_name in BS_FIELDS.items():
                    row[dst_name] = bs.loc[src_name, date] if src_name in bs.index else np.nan
                if inc is not None and not inc.empty and date in inc.columns:
                    for src_name, dst_name in IS_FIELDS.items():
                        row[dst_name] = inc.loc[src_name, date] if src_name in inc.index else np.nan
                else:
                    for dst_name in IS_FIELDS.values():
                        row[dst_name] = np.nan
                records.append(row)
        except Exception:
            continue

    fundamentals = pd.DataFrame(records)
    fundamentals["date"] = pd.to_datetime(fundamentals["date"])
    fundamentals = fundamentals.sort_values(["ticker", "date"]).reset_index(drop=True)
    fundamentals.to_parquet(cache_file)
    return fundamentals


def load_ken_french_factors():
    """Monthly Fama-French 5 factors + Momentum + RF.

    Returns a DataFrame with DatetimeIndex (month-end) and columns:
        Mkt-RF, SMB, HML, RMW, CMA, MOM, RF

    Values are in DECIMAL form (0.01 = 1%).
    Full history from 1963 onward.
    Used by: lecture S5-S7, seminar, homework.
    """
    cache_file = CACHE_DIR / "ken_french_factors.parquet"
    if cache_file.exists():
        return pd.read_parquet(cache_file)

    import getfactormodels as gfm

    ff5 = gfm.FamaFrenchFactors(frequency="m", model="5").to_pandas()
    carhart = gfm.CarhartFactors(frequency="m").to_pandas()

    factors = ff5.join(carhart[["MOM"]], how="inner")
    factors.index = pd.to_datetime(factors.index)
    factors.to_parquet(cache_file)
    return factors


def compute_monthly_returns(daily_prices):
    """Resample daily adjusted close prices to monthly simple returns.

    Uses month-end resampling: takes the last trading day's close each month,
    then computes pct_change. Drops only the first month (NaN from differencing).
    Individual tickers may still have NaN in early months if they IPO'd later —
    each code file handles per-ticker NaN as needed.
    """
    monthly_prices = daily_prices.resample("ME").last()
    return monthly_prices.pct_change().iloc[1:]


if __name__ == "__main__":
    print("=" * 60)
    print("Week 3 Data Setup: Factor Models & Cross-Sectional Analysis")
    print("=" * 60)

    print("\n1. Demo prices (20 tickers + SPY)...")
    demo = load_demo_prices()
    print(f"   {demo.shape[0]} days, {demo.shape[1]} tickers, "
          f"{demo.index[0].date()} to {demo.index[-1].date()}")

    print("\n2. S&P 500 prices (~500 tickers)...")
    sp500 = load_sp500_prices()
    print(f"   {sp500.shape[0]} days, {sp500.shape[1]} tickers, "
          f"{sp500.index[0].date()} to {sp500.index[-1].date()}")

    print("\n3. Fundamentals...")
    fund = load_fundamentals()
    print(f"   {len(fund)} records, {fund['ticker'].nunique()} tickers")

    print("\n4. Ken French factors (FF5 + MOM)...")
    kf = load_ken_french_factors()
    print(f"   {len(kf)} months, columns: {kf.columns.tolist()}")
    print(f"   {kf.index[0]} to {kf.index[-1]}")

    print("\n5. Monthly returns...")
    demo_ret = compute_monthly_returns(demo)
    sp500_ret = compute_monthly_returns(sp500)
    print(f"   Demo:  {demo_ret.shape}")
    print(f"   SP500: {sp500_ret.shape}")

    print("\n" + "=" * 60)
    print("Data setup complete. Cached files:")
    for f in sorted(CACHE_DIR.glob("*.parquet")):
        mb = f.stat().st_size / (1024 * 1024)
        print(f"  {f.name:30s} {mb:6.2f} MB")
