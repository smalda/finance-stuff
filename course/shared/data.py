"""Shared data downloads — cross-week cache layer.

Provides cached access to datasets used across multiple weeks.
Each week's data_setup.py imports from here instead of re-downloading.

Design:
    - Downloads the WIDEST possible data once (all tickers, full history)
    - Callers slice by date range / ticker subset at read time
    - Raw, untransformed data only (prices, factors, fundamentals)
    - Week-specific derived data (feature matrices, monthly returns for a
      particular subset) stays in the week's own .cache/
    - Cache lives at course/shared/.data_cache/
    - Functions are idempotent: download once, read from cache forever
    - Deleting .data_cache/ forces a clean re-download

Usage in a week's data_setup.py:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from shared.data import SP500_TICKERS, load_sp500_prices, load_ff_factors
"""

import json
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf
from tqdm import tqdm

# ── Cache location ────────────────────────────────────────────────────
SHARED_CACHE = Path(__file__).parent / ".data_cache"
SHARED_CACHE.mkdir(exist_ok=True)

# ── Superset download range ──────────────────────────────────────────
# All price data is downloaded once for this range. Callers get slices.
_SUPERSET_START = "2000-01-01"
_SUPERSET_END = "2026-12-31"

# ── S&P 500 Universe ─────────────────────────────────────────────────
# Full S&P 500 constituents (as of Feb 2026, sourced from Wikipedia).
# Some tickers may be delisted or have limited history — the download
# applies a completeness filter and drops tickers with >50% missing days.
# Weeks may use subsets (e.g., 20-ticker demo) but should draw from
# this list rather than defining their own.

SP500_TICKERS = [
    "A", "AAPL", "ABBV", "ABNB", "ABT", "ACGL", "ACN", "ADBE", "ADI", "ADM",
    "ADP", "ADSK", "AEE", "AEP", "AES", "AFL", "AIG", "AIZ", "AJG", "AKAM",
    "ALB", "ALGN", "ALL", "ALLE", "AMAT", "AMCR", "AMD", "AME", "AMGN", "AMP",
    "AMT", "AMZN", "ANET", "AON", "AOS", "APA", "APD", "APH", "APO", "APP",
    "APTV", "ARE", "ARES", "ATO", "AVB", "AVGO", "AVY", "AWK", "AXON", "AXP",
    "AZO", "BA", "BAC", "BALL", "BAX", "BBY", "BDX", "BEN", "BF-B", "BG",
    "BIIB", "BK", "BKNG", "BKR", "BLDR", "BLK", "BMY", "BR", "BRK-B", "BRO",
    "BSX", "BX", "BXP", "C", "CAG", "CAH", "CARR", "CAT", "CB", "CBOE",
    "CBRE", "CCI", "CCL", "CDNS", "CDW", "CEG", "CF", "CFG", "CHD", "CHRW",
    "CHTR", "CI", "CIEN", "CINF", "CL", "CLX", "CMCSA", "CME", "CMG", "CMI",
    "CMS", "CNC", "CNP", "COF", "COIN", "COO", "COP", "COR", "COST", "CPAY",
    "CPB", "CPRT", "CPT", "CRH", "CRL", "CRM", "CRWD", "CSCO", "CSGP", "CSX",
    "CTAS", "CTRA", "CTSH", "CTVA", "CVNA", "CVS", "CVX", "D", "DAL", "DASH",
    "DD", "DDOG", "DE", "DECK", "DELL", "DG", "DGX", "DHI", "DHR", "DIS",
    "DLR", "DLTR", "DOC", "DOV", "DOW", "DPZ", "DRI", "DTE", "DUK", "DVA",
    "DVN", "DXCM", "EA", "EBAY", "ECL", "ED", "EFX", "EG", "EIX", "EL",
    "ELV", "EME", "EMR", "EOG", "EPAM", "EQIX", "EQR", "EQT", "ERIE", "ES",
    "ESS", "ETN", "ETR", "EVRG", "EW", "EXC", "EXE", "EXPD", "EXPE", "EXR",
    "F", "FANG", "FAST", "FCX", "FDS", "FDX", "FE", "FFIV", "FICO", "FIS",
    "FISV", "FITB", "FIX", "FOX", "FOXA", "FRT", "FSLR", "FTNT", "FTV", "GD",
    "GDDY", "GE", "GEHC", "GEN", "GEV", "GILD", "GIS", "GL", "GLW", "GM",
    "GNRC", "GOOG", "GOOGL", "GPC", "GPN", "GRMN", "GS", "GWW", "HAL", "HAS",
    "HBAN", "HCA", "HD", "HIG", "HII", "HLT", "HOLX", "HON", "HOOD", "HPE",
    "HPQ", "HRL", "HSIC", "HST", "HSY", "HUBB", "HUM", "HWM", "IBKR", "IBM",
    "ICE", "IDXX", "IEX", "IFF", "INCY", "INTC", "INTU", "INVH", "IP", "IQV",
    "IR", "IRM", "ISRG", "IT", "ITW", "IVZ", "J", "JBHT", "JBL", "JCI",
    "JKHY", "JNJ", "JPM", "KDP", "KEY", "KEYS", "KHC", "KIM", "KKR", "KLAC",
    "KMB", "KMI", "KO", "KR", "KVUE", "L", "LDOS", "LEN", "LH", "LHX",
    "LII", "LIN", "LLY", "LMT", "LNT", "LOW", "LRCX", "LULU", "LUV", "LVS",
    "LW", "LYB", "LYV", "MA", "MAA", "MAR", "MAS", "MCD", "MCHP", "MCK",
    "MCO", "MDLZ", "MDT", "MET", "META", "MGM", "MKC", "MLM", "MMM", "MNST",
    "MO", "MOH", "MOS", "MPC", "MPWR", "MRK", "MRNA", "MRSH", "MS", "MSCI",
    "MSFT", "MSI", "MTB", "MTCH", "MTD", "MU", "NCLH", "NDAQ", "NDSN", "NEE",
    "NEM", "NFLX", "NI", "NKE", "NOC", "NOW", "NRG", "NSC", "NTAP", "NTRS",
    "NUE", "NVDA", "NVR", "NWS", "NWSA", "NXPI", "O", "ODFL", "OKE", "OMC",
    "ON", "ORCL", "ORLY", "OTIS", "OXY", "PANW", "PAYC", "PAYX", "PCAR", "PCG",
    "PEG", "PEP", "PFE", "PFG", "PG", "PGR", "PH", "PHM", "PKG", "PLD",
    "PLTR", "PM", "PNC", "PNR", "PNW", "PODD", "POOL", "PPG", "PPL", "PRU",
    "PSA", "PSKY", "PSX", "PTC", "PWR", "PYPL", "Q", "QCOM", "RCL", "REG",
    "REGN", "RF", "RJF", "RL", "RMD", "ROK", "ROL", "ROP", "ROST", "RSG",
    "RTX", "RVTY", "SBAC", "SBUX", "SCHW", "SHW", "SJM", "SLB", "SMCI", "SNA",
    "SNDK", "SNPS", "SO", "SOLV", "SPG", "SPGI", "SRE", "STE", "STLD", "STT",
    "STX", "STZ", "SW", "SWK", "SWKS", "SYF", "SYK", "SYY", "T", "TAP",
    "TDG", "TDY", "TECH", "TEL", "TER", "TFC", "TGT", "TJX", "TKO", "TMO",
    "TMUS", "TPL", "TPR", "TRGP", "TRMB", "TROW", "TRV", "TSCO", "TSLA", "TSN",
    "TT", "TTD", "TTWO", "TXN", "TXT", "TYL", "UAL", "UBER", "UDR", "UHS",
    "ULTA", "UNH", "UNP", "UPS", "URI", "USB", "V", "VICI", "VLO", "VLTO",
    "VMC", "VRSK", "VRSN", "VRTX", "VST", "VTR", "VTRS", "VZ", "WAB", "WAT",
    "WBD", "WDAY", "WDC", "WEC", "WELL", "WFC", "WM", "WMB", "WMT", "WRB",
    "WSM", "WST", "WTW", "WY", "WYNN", "XEL", "XOM", "XYL", "XYZ", "YUM",
    "ZBH", "ZBRA", "ZTS",
]

# ── Common subsets ────────────────────────────────────────────────────

DEMO_TICKERS = [
    "AAPL", "NVDA", "JPM", "XOM", "JNJ", "KO", "TSLA", "PG", "GS", "AMD",
    "NEE", "PFE", "CAT", "COST", "NFLX", "WMT", "BA", "MO", "LLY", "T",
]

# ── S&P 400 MidCap Universe ──────────────────────────────────────────
# Curated mid-cap stocks for broader cross-sectional coverage.
# Includes companies below S&P 500 market-cap threshold — essential for
# size-factor research, small-cap alpha, and survivorship-bias studies.
# Some tickers may overlap with SP500 due to index rebalancing;
# duplicates are removed via set operations in ALL_EQUITY_TICKERS.

SP400_TICKERS = [
    "ACLS", "AGCO", "AIT", "ALK", "ALSN", "AMH", "AN", "ARMK", "ATKR",
    "AVNT", "AX", "AZEK",
    "BC", "BECN", "BJ", "BMI", "BRBR", "BRX", "BWXT", "BYD",
    "CALM", "CARG", "CASY", "CBT", "CC", "CHE", "CHH", "CHDN", "CLH",
    "COLM", "CPRI", "CRC", "CRI", "CRS", "CUBE", "CW",
    "DAR", "DCI", "DDS", "DINO", "DKS", "DOCS", "DOOR", "DT",
    "EAT", "ENSG", "ESAB", "EWBC", "EXEL", "EXLS", "EXP",
    "FAF", "FBIN", "FHI", "FHN", "FL", "FLO", "FND", "FNF", "FOXF",
    "GATX", "GKOS", "GLOB", "GMS", "GNTX", "GTES",
    "HAE", "HELE", "HGV", "HLI", "HQY", "HRB", "HXL",
    "IBOC", "IDCC", "IPAR", "ITT",
    "JBLU", "JHG", "JLL",
    "KBR", "KEX", "KNSL", "KNX",
    "LANC", "LAUR", "LEA", "LFUS", "LITE", "LNTH", "LPLA", "LPX", "LSTR",
    "MANH", "MAN", "MATX", "MEDP", "MKTX", "MKSI", "MOD", "MTDR", "MTSI",
    "MUSA",
    "NBIX", "NEU", "NFG", "NOVT",
    "OGE", "OLED", "OLN", "OMF", "ORI", "OSK", "OZK",
    "PNFP", "PPC", "PRI", "PSN", "PVH",
    "QLYS",
    "RBC", "RGA", "RGLD", "RHP", "RLI", "RPM", "RRX",
    "SAM", "SAIA", "SEIC", "SF", "SITE", "SKX", "SLGN", "SM", "SNX",
    "SSD", "STN", "SWN",
    "TCBI", "TENB", "TNET", "TOL", "TPX", "TRN", "TTC", "TXRH",
    "UFPI", "UMBF", "USFD",
    "VCYT", "VIRT", "VRTS", "VSH",
    "WAL", "WBS", "WEN", "WEX", "WH", "WSC", "WTS",
    "X", "YETI",
]

# ── ETF Universe (comprehensive) ────────────────────────────────────
# Organized by category for easy subsetting by weeks.

CORE_ETFS = ["SPY", "QQQ", "IWM", "DIA", "RSP", "MDY", "IWB", "VTI"]

SECTOR_ETFS = [
    "XLB", "XLC", "XLE", "XLF", "XLI", "XLK", "XLP", "XLRE",
    "XLU", "XLV", "XLY",
]

FACTOR_ETFS = ["MTUM", "QUAL", "VLUE", "SIZE", "USMV", "MOAT", "COWZ"]

FIXED_INCOME_ETFS = [
    "AGG", "BND", "LQD", "HYG", "TLT", "IEF", "SHY", "TIP",
    "VCSH", "VCLT", "EMB", "MUB", "BNDX", "GOVT",
]

INTERNATIONAL_ETFS = [
    "EFA", "EEM", "VWO", "IEFA", "VGK", "AAXJ", "FXI", "EWJ",
    "EWZ", "EWG", "EWU", "INDA", "VEA", "IEMG",
]

COMMODITY_ETFS = ["GLD", "SLV", "GDX", "DBC", "USO", "UNG", "PDBC", "IAU"]

VOLATILITY_ETFS = ["VXX", "SVXY", "UVXY", "VIXM"]

REIT_ETFS = ["VNQ", "VNQI", "IYR", "REM"]

THEMATIC_ETFS = ["ARKK", "ICLN", "TAN", "XBI", "HACK", "ROBO", "KWEB"]

ETF_TICKERS = sorted(set(
    CORE_ETFS + SECTOR_ETFS + FACTOR_ETFS + FIXED_INCOME_ETFS
    + INTERNATIONAL_ETFS + COMMODITY_ETFS + VOLATILITY_ETFS
    + REIT_ETFS + THEMATIC_ETFS
))

# ── Universe composites ─────────────────────────────────────────────
ALL_EQUITY_TICKERS = sorted(set(SP500_TICKERS + SP400_TICKERS))
ALL_TICKERS = sorted(set(ALL_EQUITY_TICKERS + ETF_TICKERS))

# ── Crypto / DeFi tokens ─────────────────────────────────────────────
# Major crypto assets + DeFi governance tokens + stablecoins for Week 18.
CRYPTO_TICKERS = [
    # Layer 1 / major
    "BTC-USD", "ETH-USD", "SOL-USD", "BNB-USD", "ADA-USD", "AVAX-USD",
    "DOT-USD", "MATIC-USD", "ATOM-USD", "NEAR-USD",
    # DeFi governance
    "UNI-USD", "AAVE-USD", "LINK-USD", "CRV-USD", "MKR-USD", "COMP-USD",
    "SNX-USD", "SUSHI-USD", "LDO-USD",
    # Layer 2 / scaling
    "ARB-USD", "OP-USD",
]

# ── Common FRED series ───────────────────────────────────────────────
# Pre-downloaded for cross-week use. Weeks can request additional series
# via load_fred_series() — they'll be cached individually on first call.

FRED_TREASURY_YIELDS = [
    "DGS1MO", "DGS3MO", "DGS6MO", "DGS1", "DGS2", "DGS3",
    "DGS5", "DGS7", "DGS10", "DGS20", "DGS30",
]
FRED_TIPS_YIELDS = [
    "DFII5",        # 5-year TIPS real yield (daily)
    "DFII10",       # 10-year TIPS real yield (daily)
    "DFII20",       # 20-year TIPS real yield (daily)
    "DFII30",       # 30-year TIPS real yield (daily)
]
FRED_RATES = ["FEDFUNDS", "DFF"]  # fed funds monthly + daily
FRED_VOLATILITY = ["VIXCLS"]  # CBOE VIX (daily)
FRED_MACRO = [
    "GDP",        # quarterly GDP
    "GDPC1",      # real GDP (chained 2017 dollars, quarterly)
    "CPIAUCSL",   # CPI (monthly, seasonally adjusted)
    "CPILFESL",   # core CPI ex food & energy (monthly)
    "PCEPI",      # PCE price index (monthly) — Fed's preferred inflation
    "UNRATE",     # unemployment rate (monthly)
    "UMCSENT",    # consumer sentiment (monthly)
    "INDPRO",     # industrial production (monthly)
    "T10Y2Y",     # 10Y-2Y spread (daily, recession indicator)
    "T10Y3M",     # 10Y-3M spread (daily, alternative recession signal)
    "USREC",      # NBER recession indicator (monthly, 0/1)
    "M2SL",       # M2 money supply (monthly)
    "WALCL",      # Fed balance sheet total assets (weekly)
]
FRED_CREDIT = [
    "BAMLH0A0HYM2",  # ICE BofA high yield OAS
    "BAMLC0A4CBBB",   # ICE BofA BBB corporate OAS
    "BAMLC0A0CM",     # ICE BofA US corporate master OAS
    "TEDRATE",         # TED spread
    "DPRIME",          # bank prime loan rate
]
FRED_HOUSING = [
    "HOUST",        # housing starts (monthly, thousands)
    "CSUSHPISA",    # Case-Shiller home price index (monthly)
    "MORTGAGE30US", # 30-year fixed mortgage rate (weekly)
]
FRED_LABOR = [
    "PAYEMS",       # total nonfarm payrolls (monthly)
    "ICSA",         # initial jobless claims (weekly)
    "JTSJOL",       # job openings — JOLTS (monthly)
    "AWHAETP",      # avg weekly hours — private employees (monthly)
]
FRED_FINANCIAL_CONDITIONS = [
    "NFCI",         # Chicago Fed national financial conditions index
    "STLFSI2",      # St. Louis Fed financial stress index
    "DRTSCILM",     # bank lending standards — C&I loans (quarterly)
]
FRED_DOLLAR_FX = [
    "DTWEXBGS",     # broad trade-weighted USD index (daily)
    "DEXJPUS",      # USD/JPY (daily)
    "DEXUSEU",      # USD/EUR (daily)
    "DEXCHUS",      # USD/CNY (daily)
    "DEXUSUK",      # USD/GBP (daily)
]
FRED_COMMODITIES_PRICES = [
    "DCOILWTICO",        # WTI crude oil spot (daily)
    "DCOILBRENTEU",      # Brent crude oil spot (daily)
    "GOLDAMGBD228NLBM",  # gold London PM fix (daily)
    "DHHNGSP",           # Henry Hub natural gas spot (daily)
    "PCOPPUSDM",         # global copper price (monthly)
]
FRED_INFLATION_EXPECTATIONS = [
    "T5YIE",        # 5-year breakeven inflation (daily)
    "T10YIE",       # 10-year breakeven inflation (daily)
    "T5YIFR",       # 5-year forward inflation expectation (daily)
    "MICH",         # Michigan survey inflation expectations (monthly)
]

FRED_ALL = (
    FRED_TREASURY_YIELDS + FRED_TIPS_YIELDS + FRED_RATES
    + FRED_VOLATILITY + FRED_MACRO + FRED_CREDIT
    + FRED_HOUSING + FRED_LABOR + FRED_FINANCIAL_CONDITIONS
    + FRED_DOLLAR_FX + FRED_COMMODITIES_PRICES
    + FRED_INFLATION_EXPECTATIONS
)

# ── Ken French portfolio datasets ────────────────────────────────────
# Sorted portfolio returns from Ken French's data library.
# Used for factor model validation, industry analysis, momentum
# benchmarking, and cross-sectional factor research.
KF_PORTFOLIOS = {
    # Classic 5x5 double sorts
    "25_size_bm": "25_Portfolios_5x5",
    "25_size_mom": "25_Portfolios_ME_Prior_12_2",
    "25_size_inv": "25_Portfolios_ME_INV",
    "25_size_op": "25_Portfolios_ME_OP",
    # 2x3 sorts (Fama-French factor construction)
    "6_size_bm": "6_Portfolios_2x3",
    # Industry portfolios
    "17_industry": "17_Industry_Portfolios",
    "49_industry": "49_Industry_Portfolios",
    # Univariate decile sorts
    "10_momentum": "10_Portfolios_Prior_12_2",
    "10_size": "Portfolios_Formed_on_ME",
    "10_bm": "Portfolios_Formed_on_BE-ME",
    "10_ep": "Portfolios_Formed_on_E-P",
    "10_cfp": "Portfolios_Formed_on_CF-P",
    "10_dp": "Portfolios_Formed_on_D-P",
    "10_inv": "Portfolios_Formed_on_INV",
    "10_op": "Portfolios_Formed_on_OP",
    "10_st_reversal": "Portfolios_Formed_on_ST_Rev",
    "10_lt_reversal": "Portfolios_Formed_on_LT_Rev",
}


# ── Price data (OHLCV) ──────────────────────────────────────────────

def load_sp500_prices(
    start: str | None = None,
    end: str | None = None,
    tickers: list[str] | None = None,
    min_completeness: float = 0.50,
) -> pd.DataFrame:
    """Load daily adjusted close prices from the shared cache.

    Downloads the full ALL_TICKERS universe for the entire superset
    range (2000-2026) on first call. Subsequent calls read from cache
    and slice to the requested date range / ticker subset.

    Args:
        start: Start date (inclusive). None = beginning of cached data.
        end: End date (inclusive). None = end of cached data.
        tickers: Subset of tickers to return. None = SP500_TICKERS only
            (pass ALL_TICKERS or ETF_TICKERS explicitly if needed).
        min_completeness: Drop tickers with less than this fraction of
            non-null days. Applied at download time, not to the caller's
            date slice.

    Returns:
        DataFrame indexed by date, columns = tickers (adjusted close).
    """
    # Fast path: read Close-only cache written by _load_full_ohlcv()
    close_cache = SHARED_CACHE / "all_prices.parquet"
    if close_cache.exists():
        full = pd.read_parquet(close_cache)
    else:
        ohlcv = _load_full_ohlcv(min_completeness)
        if isinstance(ohlcv.columns, pd.MultiIndex):
            full = ohlcv["Close"]
        else:
            full = ohlcv

    if start is not None:
        full = full.loc[start:]
    if end is not None:
        full = full.loc[:end]
    if tickers is not None:
        available = [t for t in tickers if t in full.columns]
        full = full[available]
    return full


def load_sp500_ohlcv(
    start: str | None = None,
    end: str | None = None,
    tickers: list[str] | None = None,
    fields: list[str] | None = None,
    min_completeness: float = 0.50,
) -> pd.DataFrame:
    """Load full OHLCV data from the shared cache.

    Returns a DataFrame with MultiIndex columns (field, ticker).
    Fields: Close, High, Low, Open, Volume.

    Args:
        start: Start date (inclusive). None = beginning of cached data.
        end: End date (inclusive). None = end of cached data.
        tickers: Subset of tickers. None = all cached tickers.
        fields: Subset of fields, e.g. ["Close", "Volume"]. None = all.
        min_completeness: Applied at download time (see load_sp500_prices).

    Returns:
        DataFrame with MultiIndex columns (field, ticker).
    """
    full = _load_full_ohlcv(min_completeness)

    if start is not None:
        full = full.loc[start:]
    if end is not None:
        full = full.loc[:end]

    if isinstance(full.columns, pd.MultiIndex):
        if fields is not None:
            avail_fields = [f for f in fields
                           if f in full.columns.get_level_values(0)]
            full = full[avail_fields]
        if tickers is not None:
            avail_tickers = set(full.columns.get_level_values(1))
            keep = [t for t in tickers if t in avail_tickers]
            full = full.loc[:, full.columns.get_level_values(1).isin(keep)]

    return full


def _load_full_ohlcv(min_completeness: float) -> pd.DataFrame:
    """Load or download the full superset OHLCV cache."""
    cache_file = SHARED_CACHE / "all_ohlcv.parquet"
    if cache_file.exists():
        return pd.read_parquet(cache_file)

    print(f"[shared/data] Downloading OHLCV for {len(ALL_TICKERS)} tickers "
          f"({_SUPERSET_START} to {_SUPERSET_END})...")
    data = yf.download(
        ALL_TICKERS, start=_SUPERSET_START, end=_SUPERSET_END,
        auto_adjust=True, progress=True,
    )

    # Quality filter based on Close completeness
    if isinstance(data.columns, pd.MultiIndex):
        close = data["Close"]
    else:
        close = data[["Close"]].rename(columns={"Close": ALL_TICKERS[0]})

    completeness = close.notna().mean()
    valid = completeness[completeness >= min_completeness].index.tolist()

    if isinstance(data.columns, pd.MultiIndex):
        mask = data.columns.get_level_values(1).isin(valid)
        data = data.loc[:, mask]
    data.index = pd.to_datetime(data.index).tz_localize(None)

    failed = set(ALL_TICKERS) - set(valid)
    print(f"  Tickers OK: {len(valid)}/{len(ALL_TICKERS)}")
    if failed:
        print(f"  Dropped ({len(failed)}): {sorted(failed)}")
    print(f"  Date range: {data.index.min().date()} to {data.index.max().date()}")

    # Save full OHLCV
    data.to_parquet(cache_file)
    # Also save Close-only for fast reads by load_sp500_prices()
    close_only = data["Close"] if isinstance(data.columns, pd.MultiIndex) else data
    close_only.to_parquet(SHARED_CACHE / "all_prices.parquet")

    return data


# ── Factor data ───────────────────────────────────────────────────────

def load_ff_factors(model: str = "5", frequency: str = "M") -> pd.DataFrame:
    """Load Fama-French factor data from the shared cache.

    Downloads ALL available history on first call (the library returns
    its full dataset). Subsequent calls read from cache.

    Args:
        model: '3', '5', or '6' (6 includes momentum UMD).
        frequency: 'M' for monthly, 'D' for daily.

    Returns:
        DataFrame indexed by date with factor columns (Mkt-RF, SMB, etc.).
    """
    cache_file = SHARED_CACHE / f"ff{model}_{frequency}_factors.parquet"
    if cache_file.exists():
        return pd.read_parquet(cache_file)

    from getfactormodels import FamaFrenchFactors
    print(f"[shared/data] Downloading FF{model} ({frequency}) factor data...")
    ff = FamaFrenchFactors(model=model, frequency=frequency)
    df = ff.to_pandas()

    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date")
    df.index = pd.to_datetime(df.index)
    df.index.name = "date"

    print(f"  Factors: {list(df.columns)}")
    print(f"  Date range: {df.index.min().date()} to {df.index.max().date()}")

    df.to_parquet(cache_file)
    return df


def load_carhart_factors(frequency: str = "M") -> pd.DataFrame:
    """Load Carhart 4-factor model data from the shared cache."""
    cache_file = SHARED_CACHE / f"carhart_{frequency}_factors.parquet"
    if cache_file.exists():
        return pd.read_parquet(cache_file)

    from getfactormodels import CarhartFactors
    print(f"[shared/data] Downloading Carhart ({frequency}) factor data...")
    cf = CarhartFactors(frequency=frequency)
    df = cf.to_pandas()

    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date")
    df.index = pd.to_datetime(df.index)
    df.index.name = "date"

    print(f"  Factors: {list(df.columns)}")
    print(f"  Date range: {df.index.min().date()} to {df.index.max().date()}")

    df.to_parquet(cache_file)
    return df


# ── Ken French portfolios ────────────────────────────────────────────

_KF_BASE_URL = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp"


def load_ff_portfolios(
    name: str = "25_size_bm",
    frequency: str = "M",
) -> pd.DataFrame:
    """Load Ken French sorted portfolio returns from the shared cache.

    Downloads from Ken French's data library on first call.
    Returns value-weighted returns for the first table in the CSV.

    Args:
        name: Dataset key — one of KF_PORTFOLIOS keys:
            '25_size_bm'  — 25 portfolios formed on size and book-to-market
            '17_industry' — 17 industry portfolios
            '49_industry' — 49 industry portfolios
            '10_momentum' — 10 portfolios formed on prior 12-2 month returns
            Or a raw Ken French dataset name (e.g. 'Portfolios_Formed_on_ME').
        frequency: 'M' for monthly, 'D' for daily.

    Returns:
        DataFrame indexed by date, columns = portfolio names,
        values = returns in percent (e.g. 1.5 = 1.5%).
    """
    kf_name = KF_PORTFOLIOS.get(name, name)
    cache_key = f"kf_{name}_{frequency}"
    cache_file = SHARED_CACHE / f"{cache_key}.parquet"
    if cache_file.exists():
        return pd.read_parquet(cache_file)

    import requests
    import zipfile
    from io import BytesIO

    suffix = "_daily" if frequency == "D" else ""
    url = f"{_KF_BASE_URL}/{kf_name}{suffix}_CSV.zip"
    print(f"[shared/data] Downloading Ken French {name} ({frequency})...")
    print(f"  URL: {url}")

    r = requests.get(url, timeout=30,
                     headers={"User-Agent": "Mozilla/5.0 (research)"})
    r.raise_for_status()

    with zipfile.ZipFile(BytesIO(r.content)) as zf:
        csv_name = zf.namelist()[0]
        csv_text = zf.read(csv_name).decode("utf-8", errors="replace")

    df = _parse_kf_csv(csv_text, frequency)
    cols_preview = list(df.columns[:5])
    suffix_str = "..." if len(df.columns) > 5 else ""
    print(f"  Shape: {df.shape}, columns: {cols_preview}{suffix_str}")
    print(f"  Date range: {df.index.min().date()} to {df.index.max().date()}")

    df.to_parquet(cache_file)
    return df


def _parse_kf_csv(csv_text: str, frequency: str) -> pd.DataFrame:
    """Parse a Ken French CSV file, extracting the first data table.

    Ken French CSVs have description text, then tables separated by
    blank lines. We extract the first table (value-weighted returns).
    Monthly dates are YYYYMM, daily dates are YYYYMMDD.
    """
    lines = csv_text.split("\n")

    # Find first block of consecutive data rows (lines starting with a date)
    header_idx = None
    data_start = None
    data_end = None

    for i, line in enumerate(lines):
        stripped = line.strip()
        if not stripped:
            if data_start is not None:
                data_end = i
                break
            continue

        # Detect data rows: first comma/space-separated field is all digits, 6-8 chars
        if "," in stripped:
            parts = [p.strip() for p in stripped.split(",")]
        else:
            parts = stripped.split()
        first = parts[0].strip()

        if first.isdigit() and len(first) >= 6:
            if data_start is None:
                data_start = i
                # The previous non-empty line is the column header
                for j in range(i - 1, max(i - 5, -1), -1):
                    candidate = lines[j].strip()
                    if candidate:
                        # Make sure it's not another data line
                        cparts = candidate.split(",") if "," in candidate else candidate.split()
                        if not cparts[0].strip().isdigit():
                            header_idx = j
                            break
        elif data_start is not None:
            # Non-data line after data started — end of first table
            data_end = i
            break

    if data_start is None:
        raise ValueError("No data rows found in Ken French CSV")
    if data_end is None:
        data_end = len(lines)

    # Parse column header
    columns = None
    if header_idx is not None:
        h = lines[header_idx].strip()
        if "," in h:
            columns = [c.strip() for c in h.split(",") if c.strip()]
        else:
            columns = h.split()

    # Parse data rows
    dates, rows = [], []
    for line in lines[data_start:data_end]:
        stripped = line.strip()
        if not stripped:
            break
        if "," in stripped:
            parts = [p.strip() for p in stripped.split(",")]
        else:
            parts = stripped.split()
        date_str = parts[0].strip()
        if not date_str.isdigit():
            break

        try:
            if frequency == "M" and len(date_str) == 6:
                dt = pd.Timestamp(year=int(date_str[:4]),
                                  month=int(date_str[4:6]), day=1)
            elif len(date_str) == 8:
                dt = pd.Timestamp(date_str)
            else:
                continue
        except Exception:
            continue

        vals = []
        for v in parts[1:]:
            v = v.strip()
            try:
                fv = float(v)
                vals.append(np.nan if fv <= -99.0 else fv)
            except ValueError:
                vals.append(np.nan)

        dates.append(dt)
        rows.append(vals)

    if not rows:
        raise ValueError("No data rows parsed from Ken French CSV")

    n_cols = max(len(r) for r in rows)
    # Pad short rows
    for i, r in enumerate(rows):
        if len(r) < n_cols:
            rows[i] = r + [np.nan] * (n_cols - len(r))

    if columns:
        columns = columns[:n_cols]
        while len(columns) < n_cols:
            columns.append(f"col_{len(columns)}")
    else:
        columns = [f"col_{i}" for i in range(n_cols)]

    df = pd.DataFrame(rows, index=dates, columns=columns)
    df.index.name = "date"
    return df


# ── Fundamental data ──────────────────────────────────────────────────

# ── Fundamental field lists ──────────────────────────────────────────
# Expanded field sets for comprehensive fundamental analysis.

_BS_FIELDS = [
    "Stockholders Equity", "Total Assets", "Ordinary Shares Number",
    "Total Debt", "Net Debt", "Cash And Cash Equivalents",
    "Current Assets", "Current Liabilities",
    "Total Non Current Assets", "Total Non Current Liabilities Net Minority Interest",
    "Invested Capital", "Net Tangible Assets", "Working Capital",
    "Goodwill And Other Intangible Assets",
]

_INC_FIELDS = [
    "Operating Income", "Net Income", "Total Revenue", "Pretax Income",
    "EBITDA", "Gross Profit", "Interest Expense",
    "Research And Development", "Cost Of Revenue",
    "Selling General And Administration",
    "Diluted EPS", "Basic EPS",
]

_CF_FIELDS = [
    "Operating Cash Flow", "Capital Expenditure", "Free Cash Flow",
    "Dividends Paid", "Repurchase Of Capital Stock",
    "Issuance Of Debt", "Repayment Of Debt",
    "Changes In Cash",
]

_INFO_FIELDS = {
    "sector": "Other",
    "industry": "Other",
    "marketCap": np.nan,
    "trailingPE": np.nan,
    "forwardPE": np.nan,
    "priceToBook": np.nan,
    "dividendYield": np.nan,
    "trailingAnnualDividendYield": np.nan,
    "enterpriseValue": np.nan,
    "enterpriseToEbitda": np.nan,
    "enterpriseToRevenue": np.nan,
    "returnOnEquity": np.nan,
    "returnOnAssets": np.nan,
    "debtToEquity": np.nan,
    "currentRatio": np.nan,
    "quickRatio": np.nan,
    "beta": np.nan,
    "fiftyTwoWeekHigh": np.nan,
    "fiftyTwoWeekLow": np.nan,
    "shortRatio": np.nan,
    "shortPercentOfFloat": np.nan,
    "heldPercentInstitutions": np.nan,
    "heldPercentInsiders": np.nan,
    "bookValue": np.nan,
    "earningsGrowth": np.nan,
    "revenueGrowth": np.nan,
    "profitMargins": np.nan,
    "operatingMargins": np.nan,
    "grossMargins": np.nan,
}


def load_sp500_fundamentals(
    tickers: list[str] | None = None,
    pit_lag_days: int = 0,
) -> dict:
    """Load fundamental data from the shared cache.

    Downloads balance sheet, income statement, cash flow, sector, industry,
    market cap, shares outstanding, and valuation ratios for ALL_TICKERS.
    Results are cached; subsequent calls are instant.

    Args:
        tickers: Subset of tickers. None = all cached tickers.
        pit_lag_days: Number of days to shift fundamental dates forward
            to simulate reporting lag.  yfinance indexes by fiscal period
            end, but filings arrive ~55-90 days later.  Pass 90 for a
            conservative point-in-time adjustment (rigor.md §1.3 rule 4).
            Default 0 preserves raw dates.  The shift is applied at read
            time — cached data is never modified.

    Returns:
        dict with keys:
            'balance_sheet': DataFrame (multi-index: ticker, date)
            'income_stmt': DataFrame (multi-index: ticker, date)
            'cashflow': DataFrame (multi-index: ticker, date)
            'sector': Series (index: ticker, values: sector string)
            'industry': Series (index: ticker, values: industry string)
            'market_cap': Series (index: ticker, values: market cap)
            'shares': DataFrame (multi-index: ticker, date)
            'ratios': DataFrame (index: ticker, columns: ratio names)
    """
    cache_files = {
        "balance_sheet": SHARED_CACHE / "fundamentals_bs.parquet",
        "income_stmt": SHARED_CACHE / "fundamentals_inc.parquet",
        "cashflow": SHARED_CACHE / "fundamentals_cf.parquet",
        "sector": SHARED_CACHE / "fundamentals_sector.parquet",
        "industry": SHARED_CACHE / "fundamentals_industry.parquet",
        "market_cap": SHARED_CACHE / "fundamentals_mcap.parquet",
        "shares": SHARED_CACHE / "fundamentals_shares.parquet",
        "ratios": SHARED_CACHE / "fundamentals_ratios.parquet",
    }

    if all(f.exists() for f in cache_files.values()):
        result = {
            "balance_sheet": pd.read_parquet(cache_files["balance_sheet"]),
            "income_stmt": pd.read_parquet(cache_files["income_stmt"]),
            "cashflow": pd.read_parquet(cache_files["cashflow"]),
            "sector": pd.read_parquet(cache_files["sector"]).squeeze(),
            "industry": pd.read_parquet(cache_files["industry"]).squeeze(),
            "market_cap": pd.read_parquet(cache_files["market_cap"]).squeeze(),
            "shares": pd.read_parquet(cache_files["shares"]),
            "ratios": pd.read_parquet(cache_files["ratios"]),
        }
        if tickers:
            result = _subset_fundamentals(result, tickers)
        if pit_lag_days:
            result = _apply_pit_lag(result, pit_lag_days)
        return result

    # Download for full universe — use price cache to know valid tickers
    prices = load_sp500_prices()
    valid_tickers = sorted(set(prices.columns.tolist()))

    bs_records, inc_records, cf_records, shares_records = [], [], [], []
    info_records = []
    failures = []

    print(f"[shared/data] Downloading fundamentals for {len(valid_tickers)} tickers...")
    for ticker_str in tqdm(valid_tickers, desc="Fundamentals"):
        for attempt in range(3):
            try:
                tk = yf.Ticker(ticker_str)

                # Balance sheet (expanded fields)
                bs = tk.balance_sheet
                if bs is not None and not bs.empty:
                    for col_date in bs.columns:
                        row = {"ticker": ticker_str,
                               "date": pd.Timestamp(col_date).tz_localize(None)}
                        for field in _BS_FIELDS:
                            if field in bs.index:
                                row[field] = bs.loc[field, col_date]
                        if len(row) > 2:
                            bs_records.append(row)

                        if "Ordinary Shares Number" in bs.index:
                            shares_records.append({
                                "ticker": ticker_str,
                                "date": pd.Timestamp(col_date).tz_localize(None),
                                "shares": bs.loc["Ordinary Shares Number", col_date],
                            })

                # Income statement (expanded fields)
                inc = tk.income_stmt
                if inc is not None and not inc.empty:
                    for col_date in inc.columns:
                        row = {"ticker": ticker_str,
                               "date": pd.Timestamp(col_date).tz_localize(None)}
                        for field in _INC_FIELDS:
                            if field in inc.index:
                                row[field] = inc.loc[field, col_date]
                        inc_records.append(row)

                # Cash flow statement (NEW)
                cf = tk.cashflow
                if cf is not None and not cf.empty:
                    for col_date in cf.columns:
                        row = {"ticker": ticker_str,
                               "date": pd.Timestamp(col_date).tz_localize(None)}
                        for field in _CF_FIELDS:
                            if field in cf.index:
                                row[field] = cf.loc[field, col_date]
                        if len(row) > 2:
                            cf_records.append(row)

                # Info: sector, industry, market cap, ratios (expanded)
                info = tk.info
                info_row = {"ticker": ticker_str}
                for key, default in _INFO_FIELDS.items():
                    info_row[key] = info.get(key, default)
                info_records.append(info_row)

                break
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

    cf_df = pd.DataFrame(cf_records)
    if not cf_df.empty:
        cf_df = cf_df.set_index(["ticker", "date"]).sort_index()

    shares_df = pd.DataFrame(shares_records)
    if not shares_df.empty:
        shares_df = shares_df.set_index(["ticker", "date"]).sort_index()

    # Extract scalar info into separate structures
    info_df = pd.DataFrame(info_records).set_index("ticker")
    sector_s = info_df["sector"].rename("sector")
    industry_s = info_df["industry"].rename("industry")
    mcap_s = info_df["marketCap"].rename("market_cap")
    # Ratios: everything except sector/industry/marketCap
    ratio_cols = [c for c in info_df.columns
                  if c not in ("sector", "industry", "marketCap")]
    ratios_df = info_df[ratio_cols].apply(pd.to_numeric, errors="coerce")

    print(f"  Balance sheet records: {len(bs_df)}")
    print(f"  Income statement records: {len(inc_df)}")
    print(f"  Cash flow records: {len(cf_df)}")
    print(f"  Sectors: {sector_s.nunique()} unique")
    print(f"  Industries: {industry_s.nunique()} unique")
    print(f"  Ratios: {len(ratio_cols)} metrics for {len(ratios_df)} tickers")
    if failures:
        print(f"  Failed ({len(failures)}): {sorted(failures)}")

    # Save
    bs_df.to_parquet(cache_files["balance_sheet"])
    inc_df.to_parquet(cache_files["income_stmt"])
    cf_df.to_parquet(cache_files["cashflow"])
    shares_df.to_parquet(cache_files["shares"])
    sector_s.to_frame().to_parquet(cache_files["sector"])
    industry_s.to_frame().to_parquet(cache_files["industry"])
    mcap_s.to_frame().to_parquet(cache_files["market_cap"])
    ratios_df.to_parquet(cache_files["ratios"])

    result = {
        "balance_sheet": bs_df,
        "income_stmt": inc_df,
        "cashflow": cf_df,
        "sector": sector_s,
        "industry": industry_s,
        "market_cap": mcap_s,
        "shares": shares_df,
        "ratios": ratios_df,
    }
    if tickers:
        result = _subset_fundamentals(result, tickers)
    if pit_lag_days:
        result = _apply_pit_lag(result, pit_lag_days)
    return result


def _apply_pit_lag(result: dict, pit_lag_days: int) -> dict:
    """Shift fundamental dates forward to simulate reporting lag.

    Applied at read time — cached parquet files are never modified.
    Only affects date-indexed DataFrames (balance_sheet, income_stmt,
    cashflow, shares).  Scalar fields (sector, ratios, etc.) are
    returned unchanged.
    """
    lag = pd.Timedelta(days=pit_lag_days)
    for key in ("balance_sheet", "income_stmt", "cashflow", "shares"):
        df = result[key]
        if df.empty:
            continue
        idx = df.index
        new_dates = idx.get_level_values("date") + lag
        result[key] = df.set_index(
            [idx.get_level_values("ticker"), new_dates]
        )
        result[key].index.names = ["ticker", "date"]
    return result


def _subset_fundamentals(data: dict, tickers: list[str]) -> dict:
    """Filter a fundamentals dict to a subset of tickers."""
    result = {}
    # Multi-index DataFrames (ticker, date)
    for key in ("balance_sheet", "income_stmt", "cashflow", "shares"):
        df = data.get(key, pd.DataFrame())
        if not df.empty and "ticker" in df.index.names:
            avail = [t for t in tickers if t in df.index.get_level_values("ticker")]
            result[key] = df.loc[avail] if avail else df.iloc[:0]
        else:
            result[key] = df
    # Series indexed by ticker
    for key in ("sector", "industry", "market_cap"):
        s = data.get(key, pd.Series(dtype=object))
        avail = [t for t in tickers if t in s.index]
        result[key] = s[avail]
    # Ratios DataFrame indexed by ticker
    if "ratios" in data:
        r = data["ratios"]
        avail = [t for t in tickers if t in r.index]
        result["ratios"] = r.loc[avail] if avail else r.iloc[:0]
    return result


# ── FRED data ─────────────────────────────────────────────────────────

_FRED_CSV_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv"


def load_fred_series(
    series_ids: list[str],
    start: str = "2000-01-01",
    end: str = "2025-12-31",
) -> pd.DataFrame:
    """Load FRED economic data series from shared cache.

    Each series is cached individually (one file per FRED ID).
    Downloads via FRED's public CSV endpoint (no API key needed).

    Args:
        series_ids: FRED series IDs (e.g., ["DGS10", "DGS2", "FEDFUNDS"]).
        start: Start date.
        end: End date.

    Returns:
        DataFrame indexed by date, columns = series IDs.
    """
    import requests
    from io import StringIO

    frames = {}
    for sid in series_ids:
        cache_file = SHARED_CACHE / f"fred_{sid}.parquet"

        if cache_file.exists():
            cached = pd.read_parquet(cache_file)
            if (cached.index.min() <= pd.Timestamp(start) and
                    cached.index.max() >= pd.Timestamp(end) - pd.Timedelta(days=7)):
                frames[sid] = cached[sid]
                continue

        # Download via FRED public CSV endpoint
        dl_start = min(start, _SUPERSET_START)
        dl_end = max(end, _SUPERSET_END)
        try:
            print(f"[shared/data] Downloading FRED {sid}...")
            r = requests.get(
                _FRED_CSV_URL,
                params={"id": sid, "cosd": dl_start, "coed": dl_end},
                timeout=30,
            )
            r.raise_for_status()
            df_raw = pd.read_csv(
                StringIO(r.text),
                parse_dates=["observation_date"],
                index_col="observation_date",
                na_values=["."],
            )
            df_raw.index.name = "date"
            df_raw.to_parquet(cache_file)
            frames[sid] = df_raw[sid]
        except Exception as e:
            print(f"  FAILED {sid}: {e}")

    df = pd.DataFrame(frames)
    df.index.name = "date"
    df = df.loc[start:end]
    return df


# ── Crypto data ───────────────────────────────────────────────────────

def load_crypto_prices(
    start: str | None = None,
    end: str | None = None,
    tickers: list[str] | None = None,
) -> pd.DataFrame:
    """Load crypto/DeFi token daily close prices from the shared cache.

    Downloads all CRYPTO_TICKERS from 2014 onward on first call.

    Args:
        start: Start date (inclusive). None = beginning of cached data.
        end: End date (inclusive). None = end of cached data.
        tickers: Subset of crypto tickers. None = all CRYPTO_TICKERS.

    Returns:
        DataFrame indexed by date, columns = tickers (daily close).
    """
    cache_file = SHARED_CACHE / "crypto_prices.parquet"
    if cache_file.exists():
        df = pd.read_parquet(cache_file)
    else:
        crypto_start = "2014-01-01"
        fetch = CRYPTO_TICKERS
        print(f"[shared/data] Downloading {len(fetch)} crypto tickers "
              f"({crypto_start} to {_SUPERSET_END})...")
        raw = yf.download(
            fetch, start=crypto_start, end=_SUPERSET_END,
            auto_adjust=True, progress=True,
        )
        if isinstance(raw.columns, pd.MultiIndex):
            df = raw["Close"]
        else:
            df = raw[["Close"]].rename(columns={"Close": fetch[0]})

        df.index = pd.to_datetime(df.index).tz_localize(None)
        valid = df.columns[df.notna().any()]
        df = df[valid]

        print(f"  Tickers OK: {len(valid)}/{len(fetch)}")
        print(f"  Date range: {df.index.min().date()} to {df.index.max().date()}")
        df.to_parquet(cache_file)

    if start is not None:
        df = df.loc[start:]
    if end is not None:
        df = df.loc[:end]
    if tickers is not None:
        available = [t for t in tickers if t in df.columns]
        df = df[available]
    return df


# ── Options chain data ───────────────────────────────────────────────

def load_options_chain(
    tickers: list[str] | None = None,
    near_expiries: int = 4,
    max_age_days: int = 7,
) -> pd.DataFrame:
    """Load options chain data (current snapshot) from yfinance.

    Downloads option chains for the nearest `near_expiries` expiration
    dates. Cached per-ticker; cache refreshes when older than max_age_days.

    NOTE: This is a SNAPSHOT of current option chains, not historical data.
    Historical options data requires paid sources (OptionMetrics, IVolatility).
    For course purposes, current snapshots are sufficient for teaching BS,
    Greeks, vol surface fitting, and option strategy construction.

    Args:
        tickers: Tickers to download options for. None = DEMO_TICKERS.
        near_expiries: Number of nearest expiration dates to include.
        max_age_days: Re-download if cache is older than this many days.

    Returns:
        DataFrame with columns: ticker, expiry, strike, type (call/put),
        lastPrice, bid, ask, volume, openInterest, impliedVolatility,
        inTheMoney, contractSymbol, lastTradeDate.
    """
    if tickers is None:
        tickers = DEMO_TICKERS

    cache_file = SHARED_CACHE / "options_chains.parquet"

    # Check cache freshness
    if cache_file.exists():
        age = time.time() - cache_file.stat().st_mtime
        if age < max_age_days * 86400:
            df = pd.read_parquet(cache_file)
            cached_tickers = set(df["ticker"].unique())
            requested = set(tickers)
            if requested.issubset(cached_tickers):
                return df[df["ticker"].isin(tickers)]

    all_records = []
    print(f"[shared/data] Downloading option chains for {len(tickers)} tickers...")
    for ticker_str in tqdm(tickers, desc="Options"):
        try:
            tk = yf.Ticker(ticker_str)
            expiries = tk.options
            if not expiries:
                continue

            for exp_date in expiries[:near_expiries]:
                chain = tk.option_chain(exp_date)

                for opt_type, opt_df in [("call", chain.calls),
                                         ("put", chain.puts)]:
                    if opt_df.empty:
                        continue
                    chunk = opt_df.copy()
                    chunk["ticker"] = ticker_str
                    chunk["expiry"] = pd.Timestamp(exp_date)
                    chunk["type"] = opt_type
                    all_records.append(chunk)

        except Exception as e:
            print(f"  FAILED {ticker_str}: {e}")

    if not all_records:
        print("  No option data retrieved.")
        return pd.DataFrame()

    df = pd.concat(all_records, ignore_index=True)

    # Compute moneyness (S/K) using latest close prices
    prices = load_sp500_prices(tickers=tickers)
    if not prices.empty:
        latest = prices.iloc[-1]
        df["spot"] = df["ticker"].map(latest.to_dict())
        df["moneyness"] = df["spot"] / df["strike"]

    # Compute time to expiry in years
    df["tte_days"] = (df["expiry"] - pd.Timestamp.today()).dt.days
    df["tte_years"] = df["tte_days"] / 365.25

    cols_order = [
        "ticker", "expiry", "strike", "type", "lastPrice", "bid", "ask",
        "volume", "openInterest", "impliedVolatility", "inTheMoney",
        "spot", "moneyness", "tte_days", "tte_years",
        "contractSymbol", "lastTradeDate",
    ]
    cols_present = [c for c in cols_order if c in df.columns]
    df = df[cols_present]

    print(f"  Options: {len(df)} contracts across {df['ticker'].nunique()} tickers")
    print(f"  Expiries: {df['expiry'].nunique()} unique dates")

    df.to_parquet(cache_file)
    return df


# ── SEC EDGAR filings ────────────────────────────────────────────────

_EDGAR_FULL_TEXT_URL = "https://efts.sec.gov/LATEST/search-index"
_EDGAR_SEARCH_URL = "https://efts.sec.gov/LATEST/search-index"
_EDGAR_FILING_URL = "https://www.sec.gov/cgi-bin/browse-edgar"
_EDGAR_SUBMISSIONS_URL = "https://data.sec.gov/submissions"
_SEC_HEADERS = {"User-Agent": "FinanceCourse research@example.com"}


def _get_cik_map() -> dict[str, str]:
    """Load SEC CIK-to-ticker mapping from cache or download."""
    cache_file = SHARED_CACHE / "sec_cik_map.json"
    if cache_file.exists():
        return json.loads(cache_file.read_text())

    import requests
    print("[shared/data] Downloading SEC CIK-ticker mapping...")
    r = requests.get(
        "https://www.sec.gov/files/company_tickers.json",
        headers=_SEC_HEADERS, timeout=30,
    )
    r.raise_for_status()
    raw = r.json()

    # Build ticker -> CIK mapping (zero-padded to 10 digits)
    cik_map = {}
    for entry in raw.values():
        ticker = entry.get("ticker", "").upper()
        cik = str(entry.get("cik_str", "")).zfill(10)
        if ticker:
            cik_map[ticker] = cik

    cache_file.write_text(json.dumps(cik_map))
    print(f"  Mapped {len(cik_map)} tickers to CIKs")
    return cik_map


def load_sec_filings(
    tickers: list[str] | None = None,
    filing_type: str = "10-K",
    max_filings: int = 5,
) -> dict[str, list[dict]]:
    """Download SEC filing metadata and full text from EDGAR.

    Uses the SEC EDGAR full-text search and submissions APIs.
    Cached per ticker + filing type. Each filing includes the full
    text of the primary document (10-K or 10-Q).

    NOTE: SEC rate-limits to 10 requests/second. This function
    includes appropriate delays.

    Args:
        tickers: Tickers to download filings for. None = DEMO_TICKERS[:5].
        filing_type: '10-K' (annual) or '10-Q' (quarterly).
        max_filings: Maximum number of recent filings per ticker.

    Returns:
        dict mapping ticker -> list of dicts, each with keys:
            'filing_date': str (YYYY-MM-DD)
            'accession_number': str
            'primary_document': str (URL)
            'text': str (full filing text, HTML stripped)
    """
    import requests

    if tickers is None:
        tickers = DEMO_TICKERS[:5]

    cik_map = _get_cik_map()
    results = {}

    print(f"[shared/data] Downloading {filing_type} filings for "
          f"{len(tickers)} tickers...")
    for ticker_str in tqdm(tickers, desc=f"SEC {filing_type}"):
        cache_file = (SHARED_CACHE
                      / f"sec_{ticker_str}_{filing_type}.json")

        if cache_file.exists():
            results[ticker_str] = json.loads(cache_file.read_text())
            continue

        cik = cik_map.get(ticker_str)
        if not cik:
            print(f"  No CIK for {ticker_str}")
            continue

        try:
            # Fetch recent filings from submissions API
            sub_url = f"{_EDGAR_SUBMISSIONS_URL}/CIK{cik}.json"
            r = requests.get(sub_url, headers=_SEC_HEADERS, timeout=30)
            r.raise_for_status()
            sub_data = r.json()
            time.sleep(0.15)  # respect rate limit

            recent = sub_data.get("filings", {}).get("recent", {})
            forms = recent.get("form", [])
            dates = recent.get("filingDate", [])
            accessions = recent.get("accessionNumber", [])
            primary_docs = recent.get("primaryDocument", [])

            filings = []
            for i, form in enumerate(forms):
                if form != filing_type:
                    continue
                if len(filings) >= max_filings:
                    break

                acc = accessions[i].replace("-", "")
                doc_url = (f"https://www.sec.gov/Archives/edgar/data/"
                           f"{cik.lstrip('0')}/{acc}/{primary_docs[i]}")

                filing_entry = {
                    "filing_date": dates[i],
                    "accession_number": accessions[i],
                    "primary_document": doc_url,
                    "text": "",
                }

                # Download full text
                try:
                    r2 = requests.get(
                        doc_url, headers=_SEC_HEADERS, timeout=60)
                    r2.raise_for_status()
                    # Strip HTML tags for plain text
                    text = re.sub(r"<[^>]+>", " ", r2.text)
                    text = re.sub(r"\s+", " ", text).strip()
                    # Truncate to ~500K chars to keep cache reasonable
                    filing_entry["text"] = text[:500_000]
                    time.sleep(0.15)
                except Exception:
                    pass

                filings.append(filing_entry)

            results[ticker_str] = filings
            cache_file.write_text(json.dumps(filings))

        except Exception as e:
            print(f"  FAILED {ticker_str}: {e}")
            results[ticker_str] = []

    total_filings = sum(len(v) for v in results.values())
    total_text = sum(
        len(f["text"]) for v in results.values() for f in v
    )
    print(f"  Downloaded {total_filings} filings")
    print(f"  Total text: {total_text / 1e6:.1f} MB")
    return results


# ── Synthetic limit order book data ──────────────────────────────────

def generate_synthetic_lob(
    n_levels: int = 10,
    n_snapshots: int = 10_000,
    mid_price: float = 100.0,
    tick_size: float = 0.01,
    avg_spread_ticks: float = 3.0,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate synthetic limit order book snapshots.

    Produces realistic LOB data with bid/ask prices, sizes, and derived
    microstructure features. Useful for Weeks 1 (market structure intro)
    and 13 (microstructure & execution) when real tick data is unavailable.

    The generator models:
    - Bid-ask spread with mean-reverting dynamics
    - Price levels with exponentially decaying depth
    - Autocorrelated order flow
    - Realistic trade-through events

    Args:
        n_levels: Number of price levels on each side of the book.
        n_snapshots: Number of time snapshots to generate.
        mid_price: Starting mid price.
        tick_size: Minimum price increment.
        avg_spread_ticks: Average bid-ask spread in ticks.
        seed: Random seed for reproducibility.

    Returns:
        DataFrame with columns:
            timestamp: datetime index
            mid_price: float
            bid_price_1..N: best bid through N-th level
            ask_price_1..N: best ask through N-th level
            bid_size_1..N: size at each bid level
            ask_size_1..N: size at each ask level
            spread: best ask - best bid
            ofi: order flow imbalance (signed)
            trade_price: executed trade price (NaN if no trade)
            trade_size: executed trade size (NaN if no trade)
            trade_side: 'buy' or 'sell' (NaN if no trade)
    """
    rng = np.random.default_rng(seed)

    # Generate mid-price path (random walk with mean reversion)
    returns = rng.normal(0, 0.0002, n_snapshots)
    # Add mild mean reversion to starting price
    prices = np.zeros(n_snapshots)
    prices[0] = mid_price
    for i in range(1, n_snapshots):
        mr = -0.001 * (prices[i - 1] - mid_price)
        prices[i] = prices[i - 1] * (1 + returns[i] + mr)

    # Round to tick size
    prices = np.round(prices / tick_size) * tick_size

    # Generate spread (mean-reverting around avg_spread_ticks)
    spread_ticks = np.ones(n_snapshots) * avg_spread_ticks
    for i in range(1, n_snapshots):
        innov = rng.normal(0, 0.5)
        mr = 0.1 * (avg_spread_ticks - spread_ticks[i - 1])
        spread_ticks[i] = max(1, spread_ticks[i - 1] + innov + mr)
    spreads = np.round(spread_ticks) * tick_size

    # Best bid/ask
    best_bid = prices - spreads / 2
    best_ask = prices + spreads / 2
    best_bid = np.round(best_bid / tick_size) * tick_size
    best_ask = np.round(best_ask / tick_size) * tick_size
    # Ensure spread >= 1 tick
    mask = best_ask <= best_bid
    best_ask[mask] = best_bid[mask] + tick_size

    records = {
        "mid_price": prices,
        "spread": best_ask - best_bid,
    }

    # Generate depth at each level (exponential decay + noise)
    base_size = 500
    decay = 0.7
    for level in range(1, n_levels + 1):
        level_size = base_size * (decay ** (level - 1))
        bid_sizes = rng.poisson(level_size, n_snapshots).astype(float)
        ask_sizes = rng.poisson(level_size, n_snapshots).astype(float)

        records[f"bid_price_{level}"] = best_bid - (level - 1) * tick_size
        records[f"ask_price_{level}"] = best_ask + (level - 1) * tick_size
        records[f"bid_size_{level}"] = bid_sizes
        records[f"ask_size_{level}"] = ask_sizes

    # Order flow imbalance (autocorrelated)
    ofi = np.zeros(n_snapshots)
    ofi[0] = rng.normal(0, 50)
    for i in range(1, n_snapshots):
        ofi[i] = 0.7 * ofi[i - 1] + rng.normal(0, 30)
    records["ofi"] = ofi

    # Generate trades (roughly 60% of snapshots have a trade)
    trade_mask = rng.random(n_snapshots) < 0.6
    trade_prices = np.full(n_snapshots, np.nan)
    trade_sizes = np.full(n_snapshots, np.nan)
    trade_sides = np.full(n_snapshots, None, dtype=object)

    for i in np.where(trade_mask)[0]:
        side = "buy" if rng.random() < 0.5 + 0.001 * ofi[i] else "sell"
        if side == "buy":
            trade_prices[i] = best_ask[i]
        else:
            trade_prices[i] = best_bid[i]
        trade_sizes[i] = rng.poisson(200)
        trade_sides[i] = side

    records["trade_price"] = trade_prices
    records["trade_size"] = trade_sizes
    records["trade_side"] = trade_sides

    # Create timestamps (100ms intervals over a trading day)
    base_time = pd.Timestamp("2024-01-15 09:30:00")
    timestamps = pd.date_range(
        base_time, periods=n_snapshots, freq="100ms"
    )

    df = pd.DataFrame(records, index=timestamps[:n_snapshots])
    df.index.name = "timestamp"

    print(f"[shared/data] Generated synthetic LOB:")
    print(f"  Snapshots: {n_snapshots}, Levels: {n_levels}")
    print(f"  Price range: {prices.min():.2f} – {prices.max():.2f}")
    print(f"  Avg spread: {(best_ask - best_bid).mean():.4f}")
    print(f"  Trades: {trade_mask.sum()} ({trade_mask.mean():.0%})")

    return df


# ── CLI ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    import os

    parser = argparse.ArgumentParser(
        description="Pre-download and cache all shared datasets.")
    parser.add_argument(
        "--skip-fundamentals", action="store_true",
        help="Skip fundamentals download (slow, ~500+ API calls)")
    parser.add_argument(
        "--skip-options", action="store_true",
        help="Skip options chain download")
    parser.add_argument(
        "--skip-sec", action="store_true",
        help="Skip SEC EDGAR filing download")
    parser.add_argument(
        "--include-sec", action="store_true",
        help="Include SEC EDGAR filing download (off by default)")
    parser.add_argument(
        "--lob-snapshots", type=int, default=10_000,
        help="Number of synthetic LOB snapshots to generate")
    args = parser.parse_args()

    print("=" * 60)
    print("Shared Data — Full Pre-download")
    print(f"  Universe: {len(ALL_EQUITY_TICKERS)} equities + "
          f"{len(ETF_TICKERS)} ETFs = {len(ALL_TICKERS)} total")
    print(f"  Date range: {_SUPERSET_START} to {_SUPERSET_END}")
    print("=" * 60)

    # ── OHLCV ──
    print("\n── Equity Prices (OHLCV) ──")
    ohlcv = _load_full_ohlcv(0.50)
    if isinstance(ohlcv.columns, pd.MultiIndex):
        n_tickers = len(ohlcv["Close"].columns)
        fields = list(ohlcv.columns.get_level_values(0).unique())
    else:
        n_tickers = len(ohlcv.columns)
        fields = ["Close"]
    print(f"  Shape: {ohlcv.shape}")
    print(f"  Fields: {fields}")
    print(f"  Tickers: {n_tickers}")
    print(f"  Date range: {ohlcv.index.min().date()} to "
          f"{ohlcv.index.max().date()}")

    # ── Factor models (all variants, monthly + daily) ──
    print("\n── Factor Models ──")
    for model in ("3", "5", "6"):
        for freq in ("M", "D"):
            df = load_ff_factors(model, freq)
            print(f"  FF{model} ({freq}): {df.shape}")

    for freq in ("M", "D"):
        df = load_carhart_factors(freq)
        print(f"  Carhart ({freq}): {df.shape}")

    # ── Ken French portfolios ──
    print("\n── Ken French Portfolios ──")
    for name in KF_PORTFOLIOS:
        for freq in ("M",):  # daily is optional, monthly is primary
            try:
                df = load_ff_portfolios(name, freq)
                print(f"  KF {name} ({freq}): {df.shape}")
            except Exception as e:
                print(f"  KF {name} ({freq}): FAILED — {e}")

    # ── FRED series ──
    print(f"\n── FRED Series ({len(FRED_ALL)} series) ──")
    fred = load_fred_series(FRED_ALL)
    print(f"  Shape: {fred.shape}")
    print(f"  Categories: yields({len(FRED_TREASURY_YIELDS)}), "
          f"TIPS({len(FRED_TIPS_YIELDS)}), "
          f"rates({len(FRED_RATES)}), "
          f"vol({len(FRED_VOLATILITY)}), "
          f"macro({len(FRED_MACRO)}), "
          f"credit({len(FRED_CREDIT)}), "
          f"housing({len(FRED_HOUSING)}), "
          f"labor({len(FRED_LABOR)}), "
          f"fincon({len(FRED_FINANCIAL_CONDITIONS)}), "
          f"FX({len(FRED_DOLLAR_FX)}), "
          f"commodities({len(FRED_COMMODITIES_PRICES)}), "
          f"inflation_exp({len(FRED_INFLATION_EXPECTATIONS)})")

    # ── Fundamentals ──
    if not args.skip_fundamentals:
        print("\n── Fundamentals (enhanced) ──")
        fund = load_sp500_fundamentals()
        print(f"  Balance sheet: {fund['balance_sheet'].shape}")
        print(f"  Income stmt: {fund['income_stmt'].shape}")
        print(f"  Cash flow: {fund['cashflow'].shape}")
        print(f"  Sectors: {fund['sector'].nunique()} unique")
        print(f"  Industries: {fund['industry'].nunique()} unique")
        print(f"  Market caps: {fund['market_cap'].notna().sum()} tickers")
        print(f"  Ratios: {fund['ratios'].shape}")
    else:
        print("\n── Fundamentals: SKIPPED (--skip-fundamentals) ──")

    # ── Crypto ──
    print("\n── Crypto ──")
    crypto = load_crypto_prices()
    print(f"  Shape: {crypto.shape}")
    print(f"  Tickers: {list(crypto.columns)}")

    # ── Options ──
    if not args.skip_options:
        print("\n── Options Chains ──")
        opts = load_options_chain()
        print(f"  Contracts: {len(opts)}")
        if not opts.empty:
            print(f"  Tickers: {opts['ticker'].nunique()}")
            print(f"  Expiries: {opts['expiry'].nunique()}")
    else:
        print("\n── Options: SKIPPED (--skip-options) ──")

    # ── SEC EDGAR ──
    if args.include_sec:
        print("\n── SEC EDGAR Filings ──")
        filings = load_sec_filings()
        for ticker, docs in filings.items():
            total_chars = sum(len(d["text"]) for d in docs)
            print(f"  {ticker}: {len(docs)} filings, "
                  f"{total_chars / 1e3:.0f}K chars")
    else:
        print("\n── SEC EDGAR: SKIPPED (use --include-sec to enable) ──")

    # ── Synthetic LOB ──
    print("\n── Synthetic LOB ──")
    lob = generate_synthetic_lob(n_snapshots=args.lob_snapshots)
    print(f"  Shape: {lob.shape}")

    # ── Summary ──
    total_bytes = sum(
        os.path.getsize(SHARED_CACHE / f)
        for f in os.listdir(SHARED_CACHE)
        if (f.endswith(".parquet") or f.endswith(".json"))
    )
    n_parquet = len(list(SHARED_CACHE.glob("*.parquet")))
    n_json = len(list(SHARED_CACHE.glob("*.json")))
    print("\n" + "=" * 60)
    print("Shared cache fully populated.")
    print(f"  Cache location: {SHARED_CACHE}")
    print(f"  Total size: {total_bytes / 1e6:.1f} MB")
    print(f"  Files: {n_parquet} parquet + {n_json} json")
    print(f"  Universe: {len(ALL_EQUITY_TICKERS)} equities + "
          f"{len(ETF_TICKERS)} ETFs + {len(CRYPTO_TICKERS)} crypto")
    print(f"  FRED series: {len(FRED_ALL)}")
    print(f"  Ken French datasets: {len(KF_PORTFOLIOS)}")
    print("=" * 60)
