"""Week 4 — ML for Alpha: Shared data downloads and caching.

Loads the Week 3 feature matrix (179 tickers, 7 z-scored features, 130 months),
computes forward monthly returns as the prediction target, and caches all derived
data for section files.  Also loads FRED VIX for regime analysis (Ex1) and FF3
factors for excess return computation.

All section/exercise/homework files import from this module.  No other file in
this week calls yf.download() or web APIs directly.
"""
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

# ── Paths ────────────────────────────────────────────────────────────────
CACHE_DIR = Path(__file__).parent / ".cache"
PLOT_DIR = Path(__file__).parent / "logs" / "plots"
CACHE_DIR.mkdir(exist_ok=True)
PLOT_DIR.mkdir(parents=True, exist_ok=True)

# ── Shared data layer ────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from shared.data import (  # noqa: E402
    SP500_TICKERS,
    load_sp500_prices,
    load_sp500_ohlcv,
    load_ff_factors,
    load_fred_series,
    load_sp500_fundamentals,
    load_ff_portfolios,
)

# ── Constants ────────────────────────────────────────────────────────────
START = "2014-01-01"
END = "2024-12-31"
FEATURE_START = "2014-03-31"   # first month in Week 3 feature matrix
FEATURE_END = "2024-12-31"     # last month in Week 3 feature matrix
TRAIN_WINDOW = 60              # months in rolling training window
PURGE_GAP = 1                  # months purge gap between train and predict
N_OOS_MONTHS = 68              # OOS months (2019-04 to 2024-11 — last month
                               # has no forward return since Dec 2024 → Jan 2025
                               # exceeds our data boundary)
COST_BPS = 10                  # default one-way transaction cost (S&P 500)

# Week 3 feature matrix location (read-only — no runtime dependency)
W3_FEATURE_MATRIX = (
    Path(__file__).resolve().parents[2]
    / "week03_factor_models" / "code" / ".cache" / "feature_matrix_ml.parquet"
)


# ── Load functions ───────────────────────────────────────────────────────

def load_feature_matrix() -> pd.DataFrame:
    """Load the Week 3 feature matrix (7 z-scored features, 179 tickers, 130 months).

    Copies to the local .cache/ on first call so section files never touch
    the Week 3 directory at runtime.

    Returns:
        DataFrame with MultiIndex (date, ticker), columns = 7 z-scored features.
    """
    cache_file = CACHE_DIR / "feature_matrix.parquet"
    if cache_file.exists():
        return pd.read_parquet(cache_file)

    print("[data_setup] Loading Week 3 feature matrix...")
    fm = pd.read_parquet(W3_FEATURE_MATRIX)
    fm.to_parquet(cache_file)
    print(f"  Shape: {fm.shape}")
    print(f"  Columns: {list(fm.columns)}")
    print(f"  Unique dates: {fm.index.get_level_values('date').nunique()}")
    print(f"  Unique tickers: {fm.index.get_level_values('ticker').nunique()}")
    return fm


def load_forward_returns() -> pd.Series:
    """Compute forward 1-month returns for the feature matrix universe.

    For each month t in the feature matrix, the forward return is the
    total return from the close of month t to the close of month t+1.

    Returns:
        Series with MultiIndex (date, ticker), values = forward 1-month returns.
        Indexed by the FEATURE date (not the return date).  So the return at
        (2014-03-31, AAPL) is the return from 2014-03-31 to 2014-04-30.
    """
    cache_file = CACHE_DIR / "forward_returns.parquet"
    if cache_file.exists():
        s = pd.read_parquet(cache_file).squeeze()
        if isinstance(s, pd.DataFrame):
            s = s.iloc[:, 0]
        s.name = "fwd_return"
        return s

    print("[data_setup] Computing forward 1-month returns...")

    # Get daily prices from shared cache, compute monthly close
    prices = load_sp500_prices(start="2014-01-01", end="2025-01-31")
    monthly_close = prices.resample("ME").last()

    # Compute 1-month forward returns
    monthly_returns = monthly_close.pct_change().shift(-1)
    # Drop the last row (NaN from shift) and any dates beyond 2024-12
    monthly_returns = monthly_returns.loc[:"2024-12-31"]

    # Load feature matrix to get the exact (date, ticker) pairs
    fm = load_feature_matrix()
    fm_dates = fm.index.get_level_values("date").unique()

    # Stack to (date, ticker) Series
    returns_stacked = monthly_returns.stack()
    returns_stacked.index.names = ["date", "ticker"]
    returns_stacked.name = "fwd_return"

    # Align with feature matrix dates and tickers
    common = fm.index.intersection(returns_stacked.index)
    fwd_ret = returns_stacked.loc[common]

    # Remove the last feature month (no forward return available)
    last_date = fm_dates.max()
    fwd_ret = fwd_ret.drop(index=last_date, level="date", errors="ignore")

    print(f"  Forward returns shape: {len(fwd_ret)}")
    print(f"  Date range: {fwd_ret.index.get_level_values('date').min().date()} "
          f"to {fwd_ret.index.get_level_values('date').max().date()}")
    print(f"  Months covered: {fwd_ret.index.get_level_values('date').nunique()}")

    fwd_ret.to_frame().to_parquet(cache_file)
    return fwd_ret


def load_monthly_panel() -> pd.DataFrame:
    """Build the full monthly panel: features + forward return target.

    Returns:
        DataFrame with MultiIndex (date, ticker).
        Columns: 7 feature columns + 'fwd_return'.
    """
    cache_file = CACHE_DIR / "monthly_panel.parquet"
    if cache_file.exists():
        return pd.read_parquet(cache_file)

    print("[data_setup] Building monthly panel (features + target)...")
    fm = load_feature_matrix()
    fwd = load_forward_returns()

    panel = fm.join(fwd, how="inner")
    print(f"  Panel shape: {panel.shape}")
    print(f"  Months: {panel.index.get_level_values('date').nunique()}")
    print(f"  Tickers per month: "
          f"{panel.groupby(level='date').size().min()}–"
          f"{panel.groupby(level='date').size().max()}")

    panel.to_parquet(cache_file)
    return panel


def load_ff3_monthly() -> pd.DataFrame:
    """Load FF3 monthly factor returns (for excess return computation).

    Returns are in PERCENT (divide by 100 to match price returns).

    Returns:
        DataFrame indexed by date with columns [Mkt-RF, SMB, HML, RF].
    """
    cache_file = CACHE_DIR / "ff3_monthly.parquet"
    if cache_file.exists():
        return pd.read_parquet(cache_file)

    print("[data_setup] Loading FF3 monthly factors...")
    ff3 = load_ff_factors(model="3", frequency="M")
    ff3 = ff3.loc[START:END]
    print(f"  FF3 shape: {ff3.shape}")
    print(f"  Columns: {list(ff3.columns)}")
    ff3.to_parquet(cache_file)
    return ff3


def load_vix_monthly() -> pd.Series:
    """Load monthly VIX (CBOE implied volatility index) for regime classification.

    Resampled from daily to month-end.

    Returns:
        Series indexed by month-end date, values = month-end VIX level.
    """
    cache_file = CACHE_DIR / "vix_monthly.parquet"
    if cache_file.exists():
        s = pd.read_parquet(cache_file).squeeze()
        if isinstance(s, pd.DataFrame):
            s = s.iloc[:, 0]
        s.name = "VIX"
        return s

    print("[data_setup] Loading FRED VIX (VIXCLS)...")
    vix_daily = load_fred_series(["VIXCLS"], start="2019-01-01", end="2025-01-01")
    vix_monthly = vix_daily["VIXCLS"].resample("ME").last().dropna()
    vix_monthly.name = "VIX"
    print(f"  VIX monthly shape: {len(vix_monthly)}")
    print(f"  Date range: {vix_monthly.index.min().date()} to {vix_monthly.index.max().date()}")
    vix_monthly.to_frame().to_parquet(cache_file)
    return vix_monthly


def load_tickers() -> list:
    """Return the list of tickers in the feature matrix."""
    fm = load_feature_matrix()
    return sorted(fm.index.get_level_values("ticker").unique().tolist())


# ── Convenience: feature column names ────────────────────────────────────
FEATURE_COLS = [
    "pb_ratio_z", "roe_z", "asset_growth_z", "earnings_yield_z",
    "momentum_z", "reversal_z", "volatility_z",
]
TECHNICAL_FEATURES = ["momentum_z", "reversal_z", "volatility_z"]
FUNDAMENTAL_FEATURES = ["pb_ratio_z", "roe_z", "asset_growth_z", "earnings_yield_z"]

# Tickers will be populated after data load — section files should call
# load_tickers() or inspect the feature matrix directly.
TICKERS = None  # set dynamically


# ── Main: verify data pipeline ───────────────────────────────────────────
if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=FutureWarning)

    print("=" * 60)
    print("Week 4 — ML for Alpha: Data Setup")
    print("=" * 60)

    # 1. Feature matrix
    fm = load_feature_matrix()
    n_dates = fm.index.get_level_values("date").nunique()
    n_tickers = fm.index.get_level_values("ticker").nunique()
    tickers_per_month = fm.groupby(level="date").size()
    missing = fm.isna().mean()

    print(f"\n── Feature Matrix ──")
    print(f"  Shape: {fm.shape}")
    print(f"  Unique months: {n_dates}")
    print(f"  Unique tickers: {n_tickers}")
    print(f"  Tickers per month: {tickers_per_month.min()}–{tickers_per_month.max()}")
    print(f"  Date range: {fm.index.get_level_values('date').min().date()} "
          f"to {fm.index.get_level_values('date').max().date()}")
    print(f"  Missing values:")
    for col in fm.columns:
        print(f"    {col}: {missing[col]:.2%}")

    # 2. Forward returns
    fwd = load_forward_returns()
    fwd_dates = fwd.index.get_level_values("date").unique()
    print(f"\n── Forward Returns ──")
    print(f"  Observations: {len(fwd)}")
    print(f"  Months: {len(fwd_dates)}")
    print(f"  Date range: {fwd_dates.min().date()} to {fwd_dates.max().date()}")
    print(f"  Mean: {fwd.mean():.6f}")
    print(f"  Std: {fwd.std():.6f}")

    # 3. Monthly panel
    panel = load_monthly_panel()
    panel_dates = panel.index.get_level_values("date").unique()
    print(f"\n── Monthly Panel (features + target) ──")
    print(f"  Shape: {panel.shape}")
    print(f"  Months: {len(panel_dates)}")
    print(f"  Columns: {list(panel.columns)}")

    # 4. FF3 factors
    ff3 = load_ff3_monthly()
    print(f"\n── FF3 Monthly Factors ──")
    print(f"  Shape: {ff3.shape}")
    print(f"  Date range: {ff3.index.min().date()} to {ff3.index.max().date()}")

    # 5. VIX monthly
    vix = load_vix_monthly()
    print(f"\n── VIX Monthly ──")
    print(f"  Observations: {len(vix)}")
    print(f"  Date range: {vix.index.min().date()} to {vix.index.max().date()}")
    print(f"  Mean VIX: {vix.mean():.2f}")
    print(f"  Median VIX: {vix.median():.2f}")

    # 6. Verify OOS window
    train_end_date = panel_dates[TRAIN_WINDOW + PURGE_GAP - 1]
    oos_start = panel_dates[TRAIN_WINDOW + PURGE_GAP]
    oos_months = panel_dates[TRAIN_WINDOW + PURGE_GAP:]
    print(f"\n── Walk-Forward Configuration ──")
    print(f"  Training window: {TRAIN_WINDOW} months")
    print(f"  Purge gap: {PURGE_GAP} months")
    print(f"  OOS start: {oos_start.date()}")
    print(f"  OOS months: {len(oos_months)}")
    print(f"  Expected OOS months: {N_OOS_MONTHS}")

    # 7. Verify data integrity
    print(f"\n── Data Integrity Checks ──")

    # No dates beyond 2024-12-31 in forward returns
    max_fwd_date = fwd.index.get_level_values("date").max()
    assert max_fwd_date <= pd.Timestamp("2024-12-31"), \
        f"Forward return dates extend beyond 2024-12: {max_fwd_date}"
    print(f"  No future data beyond 2024-12: OK")

    # Feature matrix matches expectations
    assert fm.shape[1] == 7, f"Expected 7 features, got {fm.shape[1]}"
    assert n_dates == 130, f"Expected 130 months, got {n_dates}"
    assert 177 <= n_tickers <= 179, f"Expected 177-179 tickers, got {n_tickers}"
    print(f"  Feature matrix shape: OK (130 months, {n_tickers} tickers, 7 features)")

    # Forward returns cover 129 months (last month has no forward return)
    assert len(fwd_dates) == 129, \
        f"Expected 129 forward-return months, got {len(fwd_dates)}"
    print(f"  Forward returns: OK (129 months)")

    # OOS window
    assert len(oos_months) >= 65, \
        f"Expected >= 65 OOS months, got {len(oos_months)}"
    print(f"  OOS months: OK ({len(oos_months)} months)")

    # Panel has target column
    assert "fwd_return" in panel.columns, "Panel missing fwd_return column"
    print(f"  Panel target column: OK")

    # Cache files summary
    cache_files = list(CACHE_DIR.glob("*.parquet")) + list(CACHE_DIR.glob("*.json"))
    total_bytes = sum(f.stat().st_size for f in cache_files)
    print(f"\n── Cache Summary ──")
    print(f"  Cache location: {CACHE_DIR}")
    print(f"  Files: {len(cache_files)}")
    print(f"  Total size: {total_bytes / 1e6:.1f} MB")
    for f in sorted(cache_files):
        print(f"    {f.name}: {f.stat().st_size / 1e3:.1f} KB")

    print("\n" + "=" * 60)
    print("Data setup complete. All checks passed.")
    print("=" * 60)
