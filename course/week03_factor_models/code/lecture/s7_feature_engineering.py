"""Section 7: From Factors to Features — Cross-Sectional Feature Engineering"""
import matplotlib
matplotlib.use("Agg")
import sys
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

sys.path.insert(0, str(Path(__file__).parent.parent))
from data_setup import (
    load_equity_prices, load_monthly_returns, load_fundamentals,
    CACHE_DIR,
)

warnings.filterwarnings("ignore", category=FutureWarning)

prices = load_equity_prices()
monthly_returns = load_monthly_returns()
fundamentals = load_fundamentals()
bs = fundamentals["balance_sheet"]
inc = fundamentals["income_stmt"]
mcap = fundamentals["market_cap"]

daily_returns = prices.pct_change().dropna(how="all")


# ── CELL: compute_fundamental_features ──────────────────────

# For each ticker, compute fundamental ratios from most recent data
fund_records = []

for ticker in monthly_returns.columns:
    row = {"ticker": ticker}

    # Book equity, total assets from balance sheet
    equity = np.nan
    total_assets = np.nan
    prev_assets = np.nan
    if ticker in bs.index.get_level_values("ticker"):
        tk_bs = bs.loc[ticker].sort_index()
        eq_vals = tk_bs["Stockholders Equity"].dropna()
        if len(eq_vals) > 0:
            equity = eq_vals.iloc[-1]
        asset_vals = tk_bs["Total Assets"].dropna()
        if len(asset_vals) > 0:
            total_assets = asset_vals.iloc[-1]
        if len(asset_vals) > 1:
            prev_assets = asset_vals.iloc[-2]

    # Net income, operating income, revenue
    net_income = np.nan
    op_income = np.nan
    revenue = np.nan
    if ticker in inc.index.get_level_values("ticker"):
        tk_inc = inc.loc[ticker].sort_index()
        ni_vals = tk_inc.get("Net Income", pd.Series(dtype=float)).dropna()
        if len(ni_vals) > 0:
            net_income = ni_vals.iloc[-1]
        oi_vals = tk_inc.get("Operating Income", pd.Series(dtype=float)).dropna()
        if len(oi_vals) > 0:
            op_income = oi_vals.iloc[-1]
        rev_vals = tk_inc.get("Total Revenue", pd.Series(dtype=float)).dropna()
        if len(rev_vals) > 0:
            revenue = rev_vals.iloc[-1]

    # Market cap
    mk = mcap.get(ticker, np.nan)

    # Ratios
    if mk > 0 and not np.isnan(net_income):
        row["earnings_yield"] = net_income / mk
    if mk > 0 and not np.isnan(equity) and equity > 0:
        row["pb_ratio"] = mk / equity
        row["book_to_market"] = equity / mk
    if not np.isnan(equity) and equity > 0 and not np.isnan(net_income):
        row["roe"] = net_income / equity
    if not np.isnan(total_assets) and not np.isnan(prev_assets) and prev_assets > 0:
        row["asset_growth"] = (total_assets / prev_assets) - 1

    fund_records.append(row)

fund_features = pd.DataFrame(fund_records).set_index("ticker")
print(f"Fundamental features computed for {len(fund_features)} tickers")
print(f"Non-null counts:")
for col in ["pe_ratio", "earnings_yield", "pb_ratio",
            "book_to_market", "roe", "asset_growth"]:
    if col in fund_features.columns:
        pct = fund_features[col].notna().mean()
        print(f"  {col}: {pct:.1%}")


# ── CELL: build_panel_features ──────────────────────────────

# Build monthly panel with both fundamental and price-based features
panel_records = []

for date in monthly_returns.index:
    # Momentum: 12-1 month
    mom_end = date - pd.DateOffset(months=1)
    mom_start = date - pd.DateOffset(months=12)
    mask = (prices.index >= mom_start) & (prices.index <= mom_end)
    if mask.sum() < 20:
        continue
    mom_prices = prices.loc[mask]
    momentum = (mom_prices.iloc[-1] / mom_prices.iloc[0]) - 1

    # Short-term reversal: 1-month return
    reversal = monthly_returns.loc[date]

    # Trailing volatility: 60-day
    vol_idx = prices.index.get_indexer([date], method="pad")[0]
    vol_start = max(0, vol_idx - 60)
    recent = daily_returns.iloc[vol_start:vol_idx]
    volatility = recent.std() * np.sqrt(252)

    for ticker in monthly_returns.columns:
        if pd.isna(monthly_returns.loc[date, ticker]):
            continue
        row = {
            "date": date,
            "ticker": ticker,
            "momentum": momentum.get(ticker, np.nan),
            "reversal": reversal.get(ticker, np.nan),
            "volatility": volatility.get(ticker, np.nan),
        }
        # Add fundamental features
        if ticker in fund_features.index:
            for col in fund_features.columns:
                row[col] = fund_features.loc[ticker, col]
        panel_records.append(row)

feature_panel = pd.DataFrame(panel_records)
feature_panel = feature_panel.set_index(["date", "ticker"])
print(f"\nRaw feature panel shape: {feature_panel.shape}")


# ── CELL: winsorize_and_standardize ─────────────────────────

feature_cols = [c for c in feature_panel.columns
                if c not in ["date", "ticker"]]

# Winsorize at [1st, 99th] percentiles within each month
def winsorize_cross_section(group):
    """Winsorize features at 1st and 99th percentiles."""
    for col in feature_cols:
        if col in group.columns:
            vals = group[col].dropna()
            if len(vals) > 10:
                lo = vals.quantile(0.01)
                hi = vals.quantile(0.99)
                group[col] = group[col].clip(lo, hi)
    return group

feature_win = feature_panel.groupby(level="date", group_keys=False).apply(
    winsorize_cross_section
)

# Cross-sectional z-score standardization
def zscore_cross_section(group):
    """Z-score standardize within each month."""
    for col in feature_cols:
        if col in group.columns:
            vals = group[col]
            mean, std = vals.mean(), vals.std()
            if std > 0:
                group[col + "_z"] = ((vals - mean) / std).clip(-3.0, 3.0)
            else:
                group[col + "_z"] = 0.0
    return group

feature_std = feature_win.groupby(level="date", group_keys=False).apply(
    zscore_cross_section
)

z_cols = [c for c in feature_std.columns if c.endswith("_z")]
print(f"\nStandardized feature matrix shape: {feature_std.shape}")
print(f"Z-score columns: {z_cols}")


# ── CELL: verify_standardization ────────────────────────────

# Check z-scores have mean ~0, std ~1 within each month
z_stats = []
for date in feature_std.index.get_level_values("date").unique()[:12]:
    month_data = feature_std.loc[date]
    for col in z_cols:
        vals = month_data[col].dropna()
        if len(vals) > 10:
            z_stats.append({
                "date": date, "feature": col,
                "mean": vals.mean(), "std": vals.std(),
            })

z_check = pd.DataFrame(z_stats)
print("\nZ-score verification (first 12 months):")
print(f"  Mean range: [{z_check['mean'].min():.4f}, "
      f"{z_check['mean'].max():.4f}]")
print(f"  Std range: [{z_check['std'].min():.4f}, "
      f"{z_check['std'].max():.4f}]")


# ── CELL: feature_correlation ───────────────────────────────

# Compute average cross-sectional correlation matrix
corr_matrices = []
for date in feature_std.index.get_level_values("date").unique()[-24:]:
    month_data = feature_std.loc[date][z_cols].dropna()
    if len(month_data) > 30:
        corr_matrices.append(month_data.corr())

avg_corr = pd.concat(corr_matrices).groupby(level=0).mean()
# Simplify column names for display
short_names = [c.replace("_z", "") for c in avg_corr.columns]

fig, ax = plt.subplots(figsize=(10, 8))
im = ax.imshow(avg_corr.values, cmap="RdBu_r", vmin=-1, vmax=1)
ax.set_xticks(range(len(short_names)))
ax.set_xticklabels(short_names, rotation=45, ha="right", fontsize=8)
ax.set_yticks(range(len(short_names)))
ax.set_yticklabels(short_names, fontsize=8)
for i in range(len(short_names)):
    for j in range(len(short_names)):
        ax.text(j, i, f"{avg_corr.values[i, j]:.2f}",
                ha="center", va="center", fontsize=7)
ax.set_title("Average Feature Correlation Matrix (Cross-Sectional)")
fig.colorbar(im, ax=ax, shrink=0.7)
plt.tight_layout()
plt.show()


if __name__ == "__main__":
    # ── ASSERTIONS ─────────────────────────────────
    n_tickers = fund_features.notna().any(axis=1).sum()
    assert n_tickers >= 150, \
        f"Features for only {n_tickers} tickers (expected ≥150)"

    # Feature panel shape
    n_rows = len(feature_panel)
    assert n_rows >= 5000, \
        f"Panel has only {n_rows} rows (expected ≥5000)"

    # Z-score verification
    assert z_check["mean"].abs().max() < 0.10, \
        f"Max |z-mean| = {z_check['mean'].abs().max():.4f} (expected < 0.10)"
    assert z_check["std"].min() > 0.80, \
        f"Min z-std = {z_check['std'].min():.4f} (expected > 0.80)"
    assert z_check["std"].max() < 1.15, \
        f"Max z-std = {z_check['std'].max():.4f} (expected < 1.15)"

    # Max |z| should be bounded by the hard cap
    max_z = feature_std[z_cols].abs().max().max()
    assert max_z <= 3.01, \
        f"Max |z-score| = {max_z:.2f} (expected ≤ 3.0 after clipping)"

    # Momentum-reversal correlation should be negative
    if "momentum_z" in avg_corr.index and "reversal_z" in avg_corr.index:
        mom_rev_corr = avg_corr.loc["momentum_z", "reversal_z"]
        assert -0.60 <= mom_rev_corr <= 0.10, \
            f"Momentum-reversal corr = {mom_rev_corr:.4f}, " \
            f"outside [-0.60, 0.10]"

    # ── RESULTS ────────────────────────────────────
    print(f"══ lecture/s7_feature_engineering ═══════════════════")
    print(f"  n_tickers_with_features: {n_tickers}")
    print(f"  panel_rows: {n_rows}")
    print(f"  n_feature_columns: {len(feature_cols)}")
    print(f"  n_z_columns: {len(z_cols)}")
    print(f"  z_mean_range: [{z_check['mean'].min():.4f}, "
          f"{z_check['mean'].max():.4f}]")
    print(f"  z_std_range: [{z_check['std'].min():.4f}, "
          f"{z_check['std'].max():.4f}]")
    print(f"  max_abs_z: {max_z:.4f}")

    # Key correlations
    if "momentum_z" in avg_corr.index and "reversal_z" in avg_corr.index:
        print(f"  momentum_reversal_corr: {mom_rev_corr:.4f}")

    # ── PLOT ───────────────────────────────────────
    fig.savefig(CACHE_DIR / "s7_feature_correlation.png",
                dpi=150, bbox_inches="tight")
    print(f"  ── plot: s7_feature_correlation.png ──")
    print(f"     type: correlation heatmap")
    print(f"     shape: {avg_corr.shape}")
    print(f"     title: {ax.get_title()}")
    print(f"✓ s7_feature_engineering: ALL PASSED")
