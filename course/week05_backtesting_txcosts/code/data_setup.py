"""Week 5 — Backtesting, Research Discipline & Transaction Costs.

Data setup: loads shared cache data, checks Week 4 dependency, prepares
per-week caches. Section files import from here; they never call yfinance directly.

Week 4 dependency:
  course/week04_ml_alpha/code/.cache/{gbm_predictions, gbm_ic_series,
  nn_predictions, nn_ic_series, expanded_features}.parquet

Synthetic fallback (activated if Week 4 cache is missing):
  12-1 month momentum signal on 449-ticker universe, rank-normalized,
  long-short decile portfolio. Methodology identical to Week 4 pipeline.
  Marked with WEEK4_AVAILABLE = False in exports.

Section file import pattern:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from data_setup import (
        TICKERS, START, END, CACHE_DIR, PLOT_DIR, WEEK4_AVAILABLE,
        load_equity_data, load_ohlcv_data, load_ff3, load_alpha_output,
        load_ls_portfolio,
    )
"""

import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# ── Path setup ────────────────────────────────────────────────────────
# shared layer: two levels up from code/
SHARED_DIR = Path(__file__).resolve().parents[2] / "shared"
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from shared.data import (
    SP500_TICKERS,
    DEMO_TICKERS,
    load_sp500_prices,
    load_sp500_ohlcv,
    load_ff_factors,
    load_sp500_fundamentals,
)
from shared.backtesting import (
    long_short_returns,
    portfolio_turnover,
    sharpe_ratio,
    net_returns as shared_net_returns,
    max_drawdown,
    cumulative_returns,
)
from shared.metrics import ic_summary, rank_ic

# ── Constants ─────────────────────────────────────────────────────────
START = "2012-01-01"
END = "2025-12-31"
COST_BPS = 10          # default one-way half-spread (basis points)
N_GROUPS = 10          # decile portfolios (consistent with Week 4)
TRAIN_WINDOW = 36      # 3-year initial train window for OOS from 2015
SEED = 42

TICKERS = SP500_TICKERS  # full universe; filtered to available on load

# ── Cache and plot directories ────────────────────────────────────────
CACHE_DIR = Path(__file__).parent / ".cache"
PLOT_DIR = Path(__file__).parent / "logs" / "plots"
CACHE_DIR.mkdir(exist_ok=True)
PLOT_DIR.mkdir(parents=True, exist_ok=True)

# ── Week 4 cache path ─────────────────────────────────────────────────
_W4_CACHE = Path(__file__).resolve().parents[2] / "week04_ml_alpha" / "code" / ".cache"
_W4_FILES = {
    "gbm_predictions": _W4_CACHE / "gbm_predictions.parquet",
    "gbm_ic_series":   _W4_CACHE / "gbm_ic_series.parquet",
    "nn_predictions":  _W4_CACHE / "nn_predictions.parquet",
    "nn_ic_series":    _W4_CACHE / "nn_ic_series.parquet",
    "expanded_features": _W4_CACHE / "expanded_features.parquet",
    "forward_returns": _W4_CACHE / "forward_returns.parquet",
}


# ══════════════════════════════════════════════════════════════════════
# Shared data loaders (wrappers around shared layer with local caching)
# ══════════════════════════════════════════════════════════════════════

def load_equity_data(start: str = START, end: str = END) -> pd.DataFrame:
    """Monthly close prices (adjusted), 2012–2025.

    Returns:
        DataFrame with DatetimeIndex (month-end), columns = tickers.
        Values are monthly close prices (not returns).
    """
    cache_file = CACHE_DIR / "monthly_prices.parquet"
    if cache_file.exists():
        return pd.read_parquet(cache_file)

    print("  Loading daily prices from shared cache...")
    prices_daily = load_sp500_prices(start=start, end=end, tickers=SP500_TICKERS)
    # Monthly close: last trading day of each month
    monthly = prices_daily.resample("ME").last()
    # Drop columns with > 30% missing
    monthly = monthly.dropna(axis=1, thresh=int(len(monthly) * 0.70))
    monthly.to_parquet(cache_file)
    print(f"  Monthly prices: {monthly.shape[1]} tickers × {len(monthly)} months")
    return monthly


def load_ohlcv_data(start: str = START, end: str = END) -> pd.DataFrame:
    """Daily OHLCV data, full history.

    Returns:
        MultiIndex DataFrame with columns (field, ticker):
        fields = Open, High, Low, Close, Volume.
    """
    cache_file = CACHE_DIR / "daily_ohlcv.parquet"
    if cache_file.exists():
        return pd.read_parquet(cache_file)

    print("  Loading OHLCV from shared cache...")
    ohlcv = load_sp500_ohlcv(start=start, end=end, tickers=SP500_TICKERS)
    ohlcv.to_parquet(cache_file)
    print(f"  OHLCV: {ohlcv.shape}")
    return ohlcv


def load_ff3(start: str = START, end: str = END) -> pd.DataFrame:
    """Fama-French 3-factor returns (monthly, 2012–2025).

    Returns:
        DataFrame with columns: Mkt-RF, SMB, HML, RF.
        Values in decimal form (divides by 100 from Ken French source).
    """
    cache_file = CACHE_DIR / "ff3_monthly.parquet"
    if cache_file.exists():
        return pd.read_parquet(cache_file)

    print("  Loading FF3 factors from shared cache...")
    ff3 = load_ff_factors(model="3", frequency="M")
    # Ken French data is in percent; convert to decimal
    ff3 = ff3 / 100.0
    ff3 = ff3.loc[start:end]
    ff3.to_parquet(cache_file)
    print(f"  FF3 factors: {ff3.shape[0]} months, columns: {list(ff3.columns)}")
    return ff3


def load_mcap_tiers() -> pd.Series:
    """Market-cap tier assignment per ticker (large / mid / small).

    Returns:
        Series indexed by ticker with values 'large', 'mid', 'small'.
        Used for tiered transaction cost assumptions in TC model.
    """
    cache_file = CACHE_DIR / "mcap_tiers.parquet"
    if cache_file.exists():
        return pd.read_parquet(cache_file)["tier"]

    print("  Loading fundamentals for market-cap tiers...")
    try:
        fundamentals = load_sp500_fundamentals(SP500_TICKERS)
        mcap = fundamentals.get("market_cap", pd.Series(dtype=float))
        if isinstance(mcap, dict):
            mcap = pd.Series(mcap)
        mcap = mcap.dropna()
    except Exception as e:
        print(f"  Warning: could not load fundamentals ({e}); using equal tiers.")
        mcap = pd.Series(dtype=float)

    if len(mcap) < 50:
        # Fallback: assign tiers based on ticker position (arbitrary but consistent)
        all_tickers = SP500_TICKERS
        n = len(all_tickers)
        tiers = (
            ["large"] * (n // 3) +
            ["mid"] * (n // 3) +
            ["small"] * (n - 2 * (n // 3))
        )
        mcap_tiers = pd.Series(tiers[:n], index=all_tickers[:n], name="tier")
    else:
        terciles = mcap.quantile([1/3, 2/3])
        mcap_tiers = pd.Series(index=mcap.index, dtype=str, name="tier")
        mcap_tiers[mcap >= terciles.iloc[1]] = "large"
        mcap_tiers[(mcap >= terciles.iloc[0]) & (mcap < terciles.iloc[1])] = "mid"
        mcap_tiers[mcap < terciles.iloc[0]] = "small"

    pd.DataFrame({"tier": mcap_tiers}).to_parquet(cache_file)
    print(f"  Market-cap tiers: {mcap_tiers.value_counts().to_dict()}")
    return mcap_tiers


# ══════════════════════════════════════════════════════════════════════
# Week 4 dependency: alpha predictions + IC series
# ══════════════════════════════════════════════════════════════════════

def _check_week4_cache() -> bool:
    """Return True if all required Week 4 cache files exist."""
    required = ["gbm_predictions", "gbm_ic_series", "expanded_features", "forward_returns"]
    missing = [k for k in required if not _W4_FILES[k].exists()]
    if missing:
        print(f"  Week 4 cache: MISSING files: {missing}")
        return False
    return True


def _build_synthetic_fallback(
    monthly_prices: pd.DataFrame,
) -> dict:
    """Build synthetic 12-1 momentum signal as fallback for Week 4 output.

    Uses 12-1 month momentum (11-month return, skipping most recent month)
    as the cross-sectional signal. Rank-normalized within each month.

    Args:
        monthly_prices: DataFrame (months × tickers) of monthly close prices.

    Returns:
        dict with keys:
            predictions: DataFrame MultiIndex (date, ticker) with 'prediction'
                         and 'actual' columns
            ic_series:   DataFrame with DatetimeIndex and 'ic' column
            forward_returns: Series with MultiIndex (date, ticker)
            model_name:  str, 'synthetic_momentum_12_1'
    """
    print("  Building synthetic 12-1 momentum fallback signal...")

    # Monthly returns
    monthly_ret = monthly_prices.pct_change()

    # Forward returns (1-month ahead)
    fwd_ret = monthly_ret.shift(-1)

    # 12-1 momentum: return from T-12 to T-1 (skip T)
    mom = monthly_prices.shift(1).pct_change(11)  # 11-month return ending T-1

    # Align: need at least 13 months of history for momentum + forward
    # OOS start: 2015-01-01 (3 years after START)
    oos_start = pd.Timestamp("2015-01-01")

    all_dates = monthly_ret.index[monthly_ret.index >= oos_start]
    # Exclude last month (no forward return available)
    all_dates = all_dates[:-1]

    predictions_list = []
    ic_list = []

    for date in all_dates:
        signal = mom.loc[date].dropna()
        actual = fwd_ret.loc[date].dropna()
        common = signal.index.intersection(actual.index)
        if len(common) < 20:
            continue

        sig = signal[common]
        act = actual[common]

        # Cross-sectional rank normalization → [0, 1]
        ranks = sig.rank(pct=True)

        # Compute IC
        ic_val = float(ranks.corr(act, method="spearman"))
        ic_list.append({"date": date, "ic": ic_val})

        for ticker in common:
            predictions_list.append({
                "date": date,
                "ticker": ticker,
                "prediction": float(ranks[ticker]),
                "actual": float(act[ticker]),
            })

    predictions_df = pd.DataFrame(predictions_list)
    predictions_df = predictions_df.set_index(["date", "ticker"])

    ic_df = pd.DataFrame(ic_list).set_index("date")
    ic_df.index = pd.DatetimeIndex(ic_df.index)

    # Forward returns as Series
    fwd_series = predictions_df["actual"].rename("fwd_return")

    print(f"  Synthetic signal: {len(all_dates)} OOS months, "
          f"{len(predictions_df.index.get_level_values('ticker').unique())} tickers")
    print(f"  Synthetic IC: mean={ic_df['ic'].mean():.4f}, "
          f"std={ic_df['ic'].std():.4f}")

    return {
        "predictions": predictions_df,
        "ic_series": ic_df,
        "forward_returns": fwd_series,
        "model_name": "synthetic_momentum_12_1",
    }


def load_alpha_output() -> dict:
    """Load alpha model predictions and IC series.

    Returns Week 4 GBM output if available; otherwise the synthetic
    12-1 momentum fallback.

    Returns:
        dict with keys:
            predictions: DataFrame MultiIndex (date, ticker) with
                         'prediction' and 'actual' columns
            ic_series:   DataFrame with DatetimeIndex and 'ic' column
            nn_predictions: (optional) NN model predictions if available
            nn_ic_series:   (optional) NN IC series if available
            expanded_features: DataFrame MultiIndex (date, ticker) features
            forward_returns: Series MultiIndex (date, ticker)
            model_name:  str
        Also sets module-level WEEK4_AVAILABLE flag.
    """
    cache_file = CACHE_DIR / "alpha_output.parquet"
    meta_file = CACHE_DIR / "alpha_output_meta.json"

    # Check if already processed
    if cache_file.exists() and meta_file.exists():
        import json
        with open(meta_file) as f:
            meta = json.load(f)
        predictions = pd.read_parquet(cache_file)
        ic_series = pd.read_parquet(CACHE_DIR / "ic_series.parquet")
        fwd = pd.read_parquet(CACHE_DIR / "forward_returns_w5.parquet")["fwd_return"]

        result = {
            "predictions": predictions,
            "ic_series": ic_series,
            "forward_returns": fwd,
            "model_name": meta.get("model_name", "unknown"),
        }

        # Load optional NN outputs
        nn_pred_file = CACHE_DIR / "nn_predictions_w5.parquet"
        nn_ic_file = CACHE_DIR / "nn_ic_series_w5.parquet"
        if nn_pred_file.exists():
            result["nn_predictions"] = pd.read_parquet(nn_pred_file)
        if nn_ic_file.exists():
            result["nn_ic_series"] = pd.read_parquet(nn_ic_file)

        # Load expanded features
        feat_file = CACHE_DIR / "expanded_features_w5.parquet"
        if feat_file.exists():
            result["expanded_features"] = pd.read_parquet(feat_file)

        return result

    # Build from source
    w4_ok = _check_week4_cache()

    if w4_ok:
        print("  Week 4 cache: AVAILABLE — loading GBM predictions...")
        predictions = pd.read_parquet(_W4_FILES["gbm_predictions"])
        ic_series = pd.read_parquet(_W4_FILES["gbm_ic_series"])
        fwd = pd.read_parquet(_W4_FILES["forward_returns"])["fwd_return"]
        expanded = pd.read_parquet(_W4_FILES["expanded_features"])
        model_name = "gbm_week4"

        result = {
            "predictions": predictions,
            "ic_series": ic_series,
            "forward_returns": fwd,
            "expanded_features": expanded,
            "model_name": model_name,
        }

        # NN predictions (optional — present in Week 4 cache)
        if _W4_FILES["nn_predictions"].exists():
            result["nn_predictions"] = pd.read_parquet(_W4_FILES["nn_predictions"])
        if _W4_FILES["nn_ic_series"].exists():
            result["nn_ic_series"] = pd.read_parquet(_W4_FILES["nn_ic_series"])

    else:
        print("  Week 4 cache: UNAVAILABLE — activating synthetic fallback.")
        print("  # SYNTHETIC FALLBACK — replace with AlphaModelPipeline "
              "output when available")
        monthly = load_equity_data()
        synthetic = _build_synthetic_fallback(monthly)
        result = {
            "predictions": synthetic["predictions"],
            "ic_series": synthetic["ic_series"],
            "forward_returns": synthetic["forward_returns"],
            "model_name": synthetic["model_name"],
        }

    # Write standardized caches
    result["predictions"].to_parquet(cache_file)
    result["ic_series"].to_parquet(CACHE_DIR / "ic_series.parquet")
    pd.DataFrame({"fwd_return": result["forward_returns"]}).to_parquet(
        CACHE_DIR / "forward_returns_w5.parquet"
    )
    if "nn_predictions" in result:
        result["nn_predictions"].to_parquet(CACHE_DIR / "nn_predictions_w5.parquet")
    if "nn_ic_series" in result:
        result["nn_ic_series"].to_parquet(CACHE_DIR / "nn_ic_series_w5.parquet")
    if "expanded_features" in result:
        result["expanded_features"].to_parquet(
            CACHE_DIR / "expanded_features_w5.parquet"
        )

    import json
    with open(meta_file, "w") as f:
        json.dump({"model_name": result["model_name"], "week4_available": w4_ok}, f)

    return result


def load_ls_portfolio() -> dict:
    """Build long-short portfolio from alpha predictions.

    Constructs equal-weight top-decile long / bottom-decile short portfolio
    from the cross-sectional predictions. Consistent with Week 4 approach.

    Returns:
        dict with keys:
            weights:       DataFrame (dates × tickers), portfolio weights
                           (+1/N_long for long leg, -1/N_short for short leg)
            gross_returns: Series (monthly), gross portfolio returns
            turnover:      Series (monthly), one-way turnover
    """
    weights_file = CACHE_DIR / "ls_weights.parquet"
    returns_file = CACHE_DIR / "gross_returns.parquet"
    turnover_file = CACHE_DIR / "turnover.parquet"

    if weights_file.exists() and returns_file.exists():
        return {
            "weights": pd.read_parquet(weights_file),
            "gross_returns": pd.read_parquet(returns_file)["gross_return"],
            "turnover": pd.read_parquet(turnover_file)["turnover"],
        }

    print("  Building long-short portfolio from alpha predictions...")
    alpha = load_alpha_output()
    predictions = alpha["predictions"]

    # Get dates from predictions
    dates = predictions.index.get_level_values("date").unique().sort_values()

    weights_list = []
    gross_list = []

    for date in dates:
        pred_date = predictions.loc[date]["prediction"].dropna()
        actual_date = predictions.loc[date]["actual"].dropna()
        common = pred_date.index.intersection(actual_date.index)
        if len(common) < 20:
            continue

        pred_s = pred_date[common]
        actual_s = actual_date[common]
        n = len(common)
        n_leg = max(1, n // N_GROUPS)  # top/bottom decile size

        # Sort by predicted return
        ranked = pred_s.rank(ascending=True)
        long_mask = ranked > (n - n_leg)    # top decile
        short_mask = ranked <= n_leg         # bottom decile

        w = pd.Series(0.0, index=common)
        if long_mask.sum() > 0:
            w[long_mask] = 1.0 / long_mask.sum()
        if short_mask.sum() > 0:
            w[short_mask] = -1.0 / short_mask.sum()

        # Gross return: dot product of weights with actual returns
        gross_ret = (w * actual_s).sum()
        weights_list.append(pd.Series(w, name=date))
        gross_list.append({"date": date, "gross_return": gross_ret})

    weights_df = pd.DataFrame(weights_list).fillna(0.0)
    weights_df.index = pd.DatetimeIndex(weights_df.index)

    gross_df = pd.DataFrame(gross_list).set_index("date")
    gross_df.index = pd.DatetimeIndex(gross_df.index)

    # Compute turnover
    turnover_vals = []
    for i in range(1, len(weights_df)):
        delta = weights_df.iloc[i] - weights_df.iloc[i - 1]
        one_way = delta.abs().sum() / 2.0
        turnover_vals.append({"date": weights_df.index[i], "turnover": one_way})
    if turnover_vals:
        turnover_df = pd.DataFrame(turnover_vals).set_index("date")
        turnover_df.index = pd.DatetimeIndex(turnover_df.index)
    else:
        turnover_df = pd.DataFrame({"turnover": pd.Series(dtype=float)})

    # Cache
    weights_df.to_parquet(weights_file)
    gross_df.to_parquet(returns_file)
    turnover_df.to_parquet(turnover_file)

    ann_sharpe = sharpe_ratio(gross_df["gross_return"], periods_per_year=12)
    mean_turnover = turnover_df["turnover"].mean() if len(turnover_df) > 0 else 0.0
    print(f"  Long-short portfolio: {len(gross_df)} months OOS")
    print(f"  Gross annualized Sharpe: {ann_sharpe:.3f}")
    print(f"  Mean one-way monthly turnover: {mean_turnover:.1%}")

    return {
        "weights": weights_df,
        "gross_returns": gross_df["gross_return"],
        "turnover": turnover_df["turnover"],
    }


def load_monthly_returns(start: str = START, end: str = END) -> pd.DataFrame:
    """Monthly log returns for all tickers.

    Returns:
        DataFrame (DatetimeIndex months × tickers), monthly log returns.
    """
    cache_file = CACHE_DIR / "monthly_returns.parquet"
    if cache_file.exists():
        return pd.read_parquet(cache_file)

    monthly = load_equity_data(start=start, end=end)
    # Use simple returns for consistency with portfolio math
    returns = monthly.pct_change().dropna(how="all")
    returns = returns.loc[start:end]
    returns.to_parquet(cache_file)
    return returns


# ══════════════════════════════════════════════════════════════════════
# Module-level flag (set from cache meta at import time; updated in __main__)
# ══════════════════════════════════════════════════════════════════════
def _read_week4_flag() -> bool:
    """Read WEEK4_AVAILABLE from cached meta file, or check directly."""
    meta_file = CACHE_DIR / "alpha_output_meta.json"
    if meta_file.exists():
        import json
        try:
            with open(meta_file) as f:
                return json.load(f).get("week4_available", False)
        except Exception:
            pass
    return _check_week4_cache()


WEEK4_AVAILABLE: bool = _read_week4_flag()


# ══════════════════════════════════════════════════════════════════════
# Main execution: run all downloads and print summary
# ══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import json

    print("=" * 60)
    print("Week 5 — Data Setup")
    print("=" * 60)

    # ── Step 1: Equity prices (monthly) ──────────────────────────────
    print("\n[1/6] Monthly equity prices")
    monthly = load_equity_data()
    print(f"  Shape: {monthly.shape[0]} months × {monthly.shape[1]} tickers")
    print(f"  Date range: {monthly.index[0].date()} to {monthly.index[-1].date()}")
    missing_pct = monthly.isnull().mean().mean()
    print(f"  Missing values: {missing_pct:.1%} overall")

    # ── Step 2: Daily OHLCV (for TC decomposition) ───────────────────
    print("\n[2/6] Daily OHLCV (for transaction cost estimation)")
    ohlcv = load_ohlcv_data()
    n_tickers_ohlcv = ohlcv.columns.get_level_values(1).nunique()
    print(f"  Shape: {ohlcv.shape[0]} days × {n_tickers_ohlcv} tickers × 5 fields")
    print(f"  Date range: {ohlcv.index[0].date()} to {ohlcv.index[-1].date()}")
    # High/Low coverage for Corwin-Schultz estimator
    if "High" in ohlcv.columns.get_level_values(0):
        high = ohlcv.xs("High", level=0, axis=1)
        cs_coverage = high.notnull().mean().mean()
        print(f"  High/Low coverage (Corwin-Schultz feasibility): {cs_coverage:.1%}")

    # ── Step 3: Fama-French 3 factors ────────────────────────────────
    print("\n[3/6] Fama-French 3 factors (monthly)")
    ff3 = load_ff3()
    print(f"  Shape: {ff3.shape}")
    print(f"  Date range: {ff3.index[0].date()} to {ff3.index[-1].date()}")
    print(f"  Columns: {list(ff3.columns)}")

    # ── Step 4: Market-cap tiers ──────────────────────────────────────
    print("\n[4/6] Market-cap tier assignments")
    mcap_tiers = load_mcap_tiers()
    print(f"  Tier distribution: {mcap_tiers.value_counts().to_dict()}")

    # ── Step 5: Alpha model output ────────────────────────────────────
    print("\n[5/6] Alpha model predictions")
    alpha = load_alpha_output()
    WEEK4_AVAILABLE = (alpha["model_name"] == "gbm_week4")

    # Update meta file flag
    meta_file = CACHE_DIR / "alpha_output_meta.json"
    if meta_file.exists():
        with open(meta_file) as f:
            meta = json.load(f)
        meta["week4_available"] = WEEK4_AVAILABLE
        with open(meta_file, "w") as f:
            json.dump(meta, f)

    predictions = alpha["predictions"]
    ic_series = alpha["ic_series"]
    oos_dates = ic_series.index
    n_oos_months = len(oos_dates)
    n_tickers_alpha = predictions.index.get_level_values("ticker").nunique()
    mean_ic = ic_series["ic"].mean()
    ic_std = ic_series["ic"].std()

    print(f"  Model: {alpha['model_name']}")
    print(f"  Week 4 cache available: {WEEK4_AVAILABLE}")
    print(f"  OOS months: {n_oos_months} "
          f"({oos_dates[0].date()} to {oos_dates[-1].date()})")
    print(f"  OOS tickers: {n_tickers_alpha}")
    print(f"  IC: mean={mean_ic:.4f}, std={ic_std:.4f}")

    if not WEEK4_AVAILABLE:
        print("  # SYNTHETIC FALLBACK ACTIVE — replace with AlphaModelPipeline "
              "output when Week 4 is available")

    # Additional model variants available?
    if "nn_predictions" in alpha:
        nn_ic = alpha["nn_ic_series"]["ic"]
        print(f"  NN model also available: IC mean={nn_ic.mean():.4f}")
    else:
        print("  NN model: not available (Ridge will be synthesized in S3/D3)")

    # ── Step 6: Long-short portfolio ──────────────────────────────────
    print("\n[6/6] Long-short portfolio construction")
    ls = load_ls_portfolio()
    gross_returns = ls["gross_returns"]
    turnover = ls["turnover"]

    ann_sharpe = sharpe_ratio(gross_returns, periods_per_year=12)
    ann_ret = gross_returns.mean() * 12
    mdd = max_drawdown(gross_returns)
    mean_to = turnover.mean() if len(turnover) > 0 else float("nan")

    print(f"  Gross annualized return: {ann_ret:.2%}")
    print(f"  Gross annualized Sharpe: {ann_sharpe:.3f}")
    print(f"  Max drawdown: {mdd:.2%}")
    print(f"  Mean one-way monthly turnover: {mean_to:.1%}")
    if mean_to > 0.50:
        monthly_tc_drag = mean_to * 2 * (COST_BPS / 10_000)
        annual_tc_drag = monthly_tc_drag * 12
        print(f"  ⚠ HIGH TURNOVER: {mean_to:.0%} one-way — "
              f"TC drag ≈ {annual_tc_drag:.2%}/year at {COST_BPS} bps")

    # ── Data quality summary ──────────────────────────────────────────
    print("\n" + "=" * 60)
    print("DATA QUALITY SUMMARY")
    print("=" * 60)
    monthly_ret = load_monthly_returns()
    coverage_by_period = monthly_ret.notna().sum(axis=1)
    max_stocks = coverage_by_period.max()
    min_stocks = coverage_by_period.min()
    n_low_cov = (coverage_by_period < 0.8 * max_stocks).sum()

    print(f"  Panel: {monthly.shape[1]} stocks × {len(monthly)} monthly periods")
    print(f"  Missing values: {missing_pct:.1%} overall")
    print(f"  Universe coverage: min {min_stocks} stocks, max {max_stocks} stocks")
    if n_low_cov > 0:
        print(f"  ⚠ {n_low_cov} periods with <80% of max coverage")
    print(f"  Universe: current S&P 500 — subject to survivorship bias. "
          f"Returns inflated by estimated 1–4% annually.")
    print(f"  OOS period: {n_oos_months} months "
          f"({oos_dates[0].date()} to {oos_dates[-1].date()})")
    print(f"  Week 4 dependency: {'SATISFIED' if WEEK4_AVAILABLE else 'FALLBACK ACTIVE'}")
    print()
    print("Data setup complete. All caches written to:")
    print(f"  {CACHE_DIR}")
    print()
    print("Exports available to section files:")
    print("  TICKERS, START, END, CACHE_DIR, PLOT_DIR, WEEK4_AVAILABLE, COST_BPS")
    print("  load_equity_data()    → monthly prices (months × tickers)")
    print("  load_ohlcv_data()     → daily OHLCV (MultiIndex)")
    print("  load_ff3()            → FF3 factors (months × factors)")
    print("  load_mcap_tiers()     → market-cap tier per ticker")
    print("  load_alpha_output()   → predictions, IC series, model variants")
    print("  load_ls_portfolio()   → weights, gross_returns, turnover")
    print("  load_monthly_returns()→ monthly simple returns (months × tickers)")
