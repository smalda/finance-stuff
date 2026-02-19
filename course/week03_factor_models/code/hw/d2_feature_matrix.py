"""Deliverable 2: The Cross-Sectional Feature Matrix

A reusable FeatureEngineer class that constructs a standardized
cross-sectional feature matrix from raw stock data, ready for ML.
"""
import matplotlib
matplotlib.use("Agg")
import sys
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from data_setup import (
    load_equity_prices, load_monthly_returns, load_fundamentals,
    CACHE_DIR,
)

warnings.filterwarnings("ignore", category=FutureWarning)


# ── CELL: feature_engineer_class ────────────────────────────

class FeatureEngineer:
    """Construct a standardized cross-sectional feature matrix.

    Computes fundamental features (P/E, P/B, ROE, asset growth,
    earnings yield) and technical features (momentum, reversal,
    volatility), then applies winsorization and cross-sectional
    standardization.

    Attributes:
        feature_matrix: Panel DataFrame (MultiIndex: date, ticker)
            with standardized feature columns.
        quality_report: Dict with missing data statistics.
    """

    FUNDAMENTAL_FEATURES = [
        "pb_ratio", "roe", "asset_growth", "earnings_yield",
    ]
    TECHNICAL_FEATURES = [
        "momentum", "reversal", "volatility",
    ]
    ALL_FEATURES = FUNDAMENTAL_FEATURES + TECHNICAL_FEATURES

    def __init__(self, prices, monthly_returns, fundamentals):
        self.prices = prices
        self.monthly_returns = monthly_returns
        self.fundamentals = fundamentals
        self.feature_matrix = None
        self.quality_report = {}

    def _compute_fundamental_chars(self):
        """Compute fundamental ratios from balance sheet and income."""
        bs = self.fundamentals["balance_sheet"]
        inc = self.fundamentals["income_stmt"]
        mcap = self.fundamentals["market_cap"]

        records = {}
        for ticker in self.monthly_returns.columns:
            row = {}

            # Book equity and total assets
            equity = np.nan
            total_assets = np.nan
            prev_assets = np.nan

            if ticker in bs.index.get_level_values("ticker"):
                tk_bs = bs.loc[ticker].sort_index()
                eq = tk_bs["Stockholders Equity"].dropna()
                if len(eq) > 0 and eq.iloc[-1] > 0:
                    equity = eq.iloc[-1]
                assets = tk_bs["Total Assets"].dropna()
                if len(assets) > 0:
                    total_assets = assets.iloc[-1]
                if len(assets) > 1:
                    prev_assets = assets.iloc[-2]

            # Income items
            net_income = np.nan
            revenue = np.nan
            if ticker in inc.index.get_level_values("ticker"):
                tk_inc = inc.loc[ticker].sort_index()
                ni = tk_inc.get("Net Income", pd.Series(dtype=float)).dropna()
                if len(ni) > 0:
                    net_income = ni.iloc[-1]
                rev = tk_inc.get("Total Revenue", pd.Series(dtype=float)).dropna()
                if len(rev) > 0:
                    revenue = rev.iloc[-1]

            mk = mcap.get(ticker, np.nan)

            # P/B ratio
            if not np.isnan(mk) and mk > 0 and not np.isnan(equity) \
               and equity > 0:
                row["pb_ratio"] = mk / equity

            # ROE
            if not np.isnan(equity) and equity > 0 \
               and not np.isnan(net_income):
                row["roe"] = net_income / equity

            # Asset growth
            if not np.isnan(total_assets) and not np.isnan(prev_assets) \
               and prev_assets > 0:
                row["asset_growth"] = (total_assets / prev_assets) - 1

            # Earnings yield
            if not np.isnan(mk) and mk > 0 and not np.isnan(net_income):
                row["earnings_yield"] = net_income / mk

            records[ticker] = row

        return pd.DataFrame(records).T

    def build(self):
        """Build the full feature matrix panel."""
        fund_chars = self._compute_fundamental_chars()
        daily_returns = self.prices.pct_change().dropna(how="all")

        panel_records = []
        for date in self.monthly_returns.index:
            # Momentum: 12-1 month return
            mom_end = date - pd.DateOffset(months=1)
            mom_start = date - pd.DateOffset(months=12)
            mask = (self.prices.index >= mom_start) & \
                   (self.prices.index <= mom_end)
            if mask.sum() < 20:
                continue
            mom_prices = self.prices.loc[mask]
            if len(mom_prices) < 2:
                continue
            momentum = (mom_prices.iloc[-1] / mom_prices.iloc[0]) - 1

            # Short-term reversal (1-month return)
            reversal = self.monthly_returns.loc[date]

            # Trailing volatility (60-day)
            vol_idx = self.prices.index.get_indexer([date], method="pad")[0]
            vol_start = max(0, vol_idx - 60)
            recent = daily_returns.iloc[vol_start:vol_idx]
            volatility = recent.std() * np.sqrt(252)

            for ticker in self.monthly_returns.columns:
                if pd.isna(self.monthly_returns.loc[date, ticker]):
                    continue

                row = {
                    "date": date,
                    "ticker": ticker,
                    "momentum": momentum.get(ticker, np.nan),
                    "reversal": reversal.get(ticker, np.nan),
                    "volatility": volatility.get(ticker, np.nan),
                }

                # Add fundamental features
                if ticker in fund_chars.index:
                    for col in self.FUNDAMENTAL_FEATURES:
                        if col in fund_chars.columns:
                            row[col] = fund_chars.loc[ticker, col]

                panel_records.append(row)

        panel = pd.DataFrame(panel_records)
        panel = panel.set_index(["date", "ticker"])

        # Winsorize and standardize
        self.feature_matrix = self._winsorize_and_zscore(panel)

        # Quality report
        self._build_quality_report()

        return self.feature_matrix

    def _winsorize_and_zscore(self, panel):
        """Winsorize at [1,99] percentiles, then z-score per month."""
        feature_cols = [c for c in self.ALL_FEATURES if c in panel.columns]

        def process_month(group):
            """Winsorize and z-score one month."""
            for col in feature_cols:
                vals = group[col].dropna()
                if len(vals) > 10:
                    lo = vals.quantile(0.01)
                    hi = vals.quantile(0.99)
                    group[col] = group[col].clip(lo, hi)

                    mean, std = group[col].mean(), group[col].std()
                    if std > 0:
                        group[col + "_z"] = ((group[col] - mean) / std).clip(-3.0, 3.0)
                    else:
                        group[col + "_z"] = 0.0

                    # Rank transform (0 to 1)
                    group[col + "_rank"] = group[col].rank(pct=True)

            return group

        result = panel.groupby(level="date", group_keys=False).apply(
            process_month
        )
        return result

    def _build_quality_report(self):
        """Report missing data statistics."""
        fm = self.feature_matrix
        dates = fm.index.get_level_values("date").unique()

        for col in self.ALL_FEATURES:
            if col in fm.columns:
                missing_pct = fm[col].isna().mean()
                self.quality_report[col] = {
                    "missing_pct": missing_pct,
                    "n_non_null": fm[col].notna().sum(),
                }

        # Monthly missing rates
        worst_month_missing = {}
        for col in self.ALL_FEATURES:
            if col in fm.columns:
                monthly_missing = fm.groupby(level="date")[col].apply(
                    lambda x: x.isna().mean()
                )
                worst_month_missing[col] = monthly_missing.max()
        self.quality_report["worst_month_missing"] = worst_month_missing

    def get_ml_ready_matrix(self):
        """Return a clean panel for ML consumption.

        Drops rows with too many missing features and returns
        only z-scored columns.
        """
        z_cols = [c for c in self.feature_matrix.columns if c.endswith("_z")]
        clean = self.feature_matrix[z_cols].dropna(thresh=len(z_cols) - 2)
        return clean


# ── CELL: run_feature_engineer ──────────────────────────────

prices = load_equity_prices()
monthly_returns = load_monthly_returns()
fundamentals = load_fundamentals()

fe = FeatureEngineer(prices, monthly_returns, fundamentals)
feature_matrix = fe.build()

print(f"Feature matrix shape: {feature_matrix.shape}")
print(f"Columns: {feature_matrix.columns.tolist()}")


# ── CELL: quality_summary ──────────────────────────────────

print(f"\nMissing Data Report:")
for feat in fe.ALL_FEATURES:
    if feat in fe.quality_report:
        info = fe.quality_report[feat]
        print(f"  {feat}: {info['missing_pct']:.1%} missing "
              f"({info['n_non_null']} non-null)")

print(f"\nWorst monthly missing rates:")
for feat, rate in fe.quality_report.get("worst_month_missing", {}).items():
    print(f"  {feat}: {rate:.1%}")


# ── CELL: verify_standardization ────────────────────────────

z_cols = [c for c in feature_matrix.columns if c.endswith("_z")]
sample_dates = feature_matrix.index.get_level_values("date").unique()[:12]

z_means = []
z_stds = []
for date in sample_dates:
    month = feature_matrix.loc[date]
    for col in z_cols:
        vals = month[col].dropna()
        if len(vals) > 10:
            z_means.append(vals.mean())
            z_stds.append(vals.std())

print(f"\nZ-score verification (first 12 months):")
print(f"  Mean range: [{min(z_means):.4f}, {max(z_means):.4f}]")
print(f"  Std range: [{min(z_stds):.4f}, {max(z_stds):.4f}]")


# ── CELL: ml_ready_output ──────────────────────────────────

ml_matrix = fe.get_ml_ready_matrix()
print(f"\nML-ready matrix shape: {ml_matrix.shape}")
print(f"Columns: {ml_matrix.columns.tolist()}")

# Save for downstream use
ml_matrix.to_parquet(CACHE_DIR / "feature_matrix_ml.parquet")
print(f"Saved to: {CACHE_DIR / 'feature_matrix_ml.parquet'}")


# ── CELL: feature_distribution_plot ─────────────────────────

fig, axes = plt.subplots(2, 4, figsize=(16, 8))
axes = axes.flatten()

for i, col in enumerate(z_cols[:8]):
    ax = axes[i]
    vals = feature_matrix[col].dropna()
    ax.hist(vals, bins=50, alpha=0.7, edgecolor="k", linewidth=0.3)
    ax.set_title(col.replace("_z", ""), fontsize=10)
    ax.axvline(0, color="red", lw=1, ls="--")

plt.suptitle("Feature Distributions (Z-scored)", fontsize=12)
plt.tight_layout()
plt.show()


if __name__ == "__main__":
    # ── ASSERTIONS ─────────────────────────────────
    assert feature_matrix.shape[0] >= 5000, \
        f"Panel too small: {feature_matrix.shape[0]} (expected ≥5000)"

    # At least 3 fundamental features computed
    fund_computed = sum(1 for f in fe.FUNDAMENTAL_FEATURES
                        if f in feature_matrix.columns)
    assert fund_computed >= 3, \
        f"Only {fund_computed} fundamental features (expected ≥3)"

    # Technical features should have <5% missing
    for f in fe.TECHNICAL_FEATURES:
        if f in fe.quality_report:
            missing = fe.quality_report[f]["missing_pct"]
            assert missing < 0.10, \
                f"{f}: {missing:.1%} missing (expected <10%)"

    # Z-score verification
    assert max(abs(m) for m in z_means) < 0.10, \
        f"Z-score means not centered: max |mean| = {max(abs(m) for m in z_means):.4f}"
    assert min(z_stds) > 0.80, \
        f"Z-score std too low: min std = {min(z_stds):.4f}"

    # Rank features exist and are in [0, 1]
    rank_cols = [c for c in feature_matrix.columns if c.endswith("_rank")]
    assert len(rank_cols) >= 4, \
        f"Only {len(rank_cols)} rank columns (expected ≥4)"
    for col in rank_cols[:3]:
        vals = feature_matrix[col].dropna()
        assert vals.min() >= -0.01, f"{col} min = {vals.min():.4f}"
        assert vals.max() <= 1.01, f"{col} max = {vals.max():.4f}"

    # ML matrix is consumable
    assert ml_matrix.shape[0] >= 5000, \
        f"ML matrix too small: {ml_matrix.shape[0]}"
    assert ml_matrix.shape[1] >= 6, \
        f"ML matrix too few columns: {ml_matrix.shape[1]}"

    # Handles at least 150 tickers
    n_tickers = feature_matrix.index.get_level_values("ticker").nunique()
    assert n_tickers >= 150, \
        f"Only {n_tickers} tickers (expected ≥150)"

    # ── RESULTS ────────────────────────────────────
    print(f"══ hw/d2_feature_matrix ═════════════════════════════")
    print(f"  feature_matrix_shape: {feature_matrix.shape}")
    print(f"  ml_matrix_shape: {ml_matrix.shape}")
    print(f"  n_tickers: {n_tickers}")
    print(f"  n_months: {feature_matrix.index.get_level_values('date').nunique()}")
    print(f"  n_fund_features: {fund_computed}")
    print(f"  n_tech_features: {len(fe.TECHNICAL_FEATURES)}")
    print(f"  n_z_columns: {len(z_cols)}")
    print(f"  n_rank_columns: {len(rank_cols)}")
    print(f"  z_mean_range: [{min(z_means):.4f}, {max(z_means):.4f}]")
    print(f"  z_std_range: [{min(z_stds):.4f}, {max(z_stds):.4f}]")

    # ── PLOT ───────────────────────────────────────
    fig.savefig(CACHE_DIR / "d2_feature_distributions.png",
                dpi=150, bbox_inches="tight")
    print(f"  ── plot: d2_feature_distributions.png ──")
    print(f"     type: histogram grid")
    print(f"     n_panels: {min(8, len(z_cols))}")
    print(f"     title: Feature Distributions (Z-scored)")
    print(f"✓ d2_feature_matrix: ALL PASSED")
