"""Section 1: Cross-Sectional Setup — The Prediction Problem Structure"""
import matplotlib
matplotlib.use("Agg")
import sys
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore", category=FutureWarning)

sys.path.insert(0, str(Path(__file__).parent.parent))
from data_setup import (
    load_feature_matrix,
    load_forward_returns,
    load_monthly_panel,
    CACHE_DIR,
    PLOT_DIR,
    FEATURE_COLS,
)


# ── Load data ───────────────────────────────────────────────────────────
fm = load_feature_matrix()
fwd = load_forward_returns()
panel = load_monthly_panel()

dates = panel.index.get_level_values("date").unique().sort_values()
fm_tickers_per_month = fm.groupby(level="date").size()
fm_dates = fm.index.get_level_values("date").unique().sort_values()


# ── CELL: data_quality_block ────────────────────────────────────────────

print("── DATA QUALITY ──────────────────────────────────")
print(f"  Feature matrix shape: {fm.shape}")
print(f"  Features: {list(fm.columns)}")
print(f"  Unique months: {fm.index.get_level_values('date').nunique()}")
print(f"  Tickers per month: {fm_tickers_per_month.min()}–{fm_tickers_per_month.max()}")
print(f"  Date range: {fm_dates.min().strftime('%Y-%m')} to {fm_dates.max().strftime('%Y-%m')}")
missing_pct = fm.isna().mean()
print(f"  Missing values:")
for col in fm.columns:
    print(f"    {col}: {missing_pct[col]:.1%}")
print(f"  Forward return months: {fwd.index.get_level_values('date').nunique()}")
print(f"  Forward return mean: {fwd.mean():.6f}")
print(f"  Forward return std:  {fwd.std():.6f}")
print()
print("  ⚠ SURVIVORSHIP BIAS: Universe is current S&P 500 constituents")
print("    projected backward. Stocks that were removed from the index")
print("    (due to decline, acquisition, bankruptcy) are excluded. This")
print("    upward-biases mean returns and likely inflates any cross-")
print("    sectional signal that correlates with survival probability.")
print("    Production systems require point-in-time membership data.")
print("──────────────────────────────────────────────────")


# ── CELL: cross_sectional_excess_returns ────────────────────────────────

excess_returns = fwd.groupby(level="date").transform(lambda x: x - x.mean())
excess_returns.name = "excess_return"
cs_means = excess_returns.groupby(level="date").mean()


# ── CELL: representative_month_scatter ──────────────────────────────────

mid_date = dates[len(dates) // 2]
cs = panel.loc[mid_date].dropna(subset=["momentum_z", "fwd_return"]).copy()
cs["excess_return"] = cs["fwd_return"] - cs["fwd_return"].mean()

fig_scatter, ax_scatter = plt.subplots(figsize=(8, 6))
ax_scatter.scatter(cs["momentum_z"], cs["excess_return"], alpha=0.5, s=20, c="steelblue")
slope, intercept = np.polyfit(cs["momentum_z"], cs["excess_return"], 1)
x_line = np.linspace(cs["momentum_z"].min(), cs["momentum_z"].max(), 100)
ax_scatter.plot(x_line, slope * x_line + intercept, "r-", linewidth=2, label=f"OLS slope={slope:.4f}")
ax_scatter.axhline(0, color="grey", linewidth=0.5, linestyle="--")
ax_scatter.axvline(0, color="grey", linewidth=0.5, linestyle="--")
ax_scatter.set(
    title=f"Cross-Sectional: Momentum vs Excess Return ({mid_date.strftime('%Y-%m')})",
    xlabel="Momentum (z-score)",
    ylabel="Excess return (demeaned)",
)
ax_scatter.legend(fontsize=9)
plt.tight_layout()
plt.show()


# ── CELL: return_distribution_histogram ─────────────────────────────────

fig_hist, ax_hist = plt.subplots(figsize=(8, 5))
ax_hist.hist(cs["fwd_return"], bins=30, alpha=0.7, color="steelblue", edgecolor="white")
ax_hist.axvline(cs["fwd_return"].mean(), color="red", linewidth=1.5, linestyle="--", label="Mean")
ax_hist.axvline(cs["fwd_return"].median(), color="orange", linewidth=1.5, linestyle="--", label="Median")
ax_hist.set(
    title=f"Cross-Sectional Return Distribution ({mid_date.strftime('%Y-%m')})",
    xlabel="Forward 1-month return",
    ylabel="Count",
)
ax_hist.legend()
plt.tight_layout()
plt.show()


# ── CELL: monthly_spearman_series ───────────────────────────────────────

monthly_spearman = []
for d in dates:
    cross = panel.loc[d].dropna(subset=["momentum_z", "fwd_return"])
    if len(cross) < 20:
        continue
    r, _ = stats.spearmanr(cross["momentum_z"], cross["fwd_return"])
    monthly_spearman.append({"date": d, "spearman": r})

spearman_df = pd.DataFrame(monthly_spearman).set_index("date")
median_spearman = spearman_df["spearman"].median()
mean_spearman = spearman_df["spearman"].mean()


if __name__ == "__main__":
    # ── ASSERTIONS ─────────────────────────────────────────────────────
    # S1-1: Feature matrix shape and structure
    assert 22000 <= fm.shape[0] <= 24000, (
        f"S1-1: Feature matrix rows {fm.shape[0]} outside [22000, 24000]"
    )
    assert fm.shape[1] == 7, (
        f"S1-1: Expected 7 features, got {fm.shape[1]}"
    )
    n_months_fm = fm.index.get_level_values("date").nunique()
    assert n_months_fm == 130, (
        f"S1-1: Expected 130 unique months, got {n_months_fm}"
    )
    tpm = fm.groupby(level="date").size()
    assert 177 <= tpm.min() and tpm.max() <= 179, (
        f"S1-1: Tickers per month {tpm.min()}–{tpm.max()} outside [177, 179]"
    )

    # S1-2: Forward returns cover 129 months, no future data
    fwd_dates = fwd.index.get_level_values("date").unique()
    assert len(fwd_dates) == 129, (
        f"S1-2: Expected 129 forward-return months, got {len(fwd_dates)}"
    )
    max_fwd_date = fwd_dates.max()
    assert max_fwd_date <= pd.Timestamp("2024-12-31"), (
        f"S1-2: Forward returns extend beyond 2024-12: {max_fwd_date}"
    )

    # S1-3: Cross-sectional excess returns mean ~0 per month
    max_abs_mean = cs_means.abs().max()
    assert max_abs_mean < 1e-10, (
        f"S1-3: Max |cross-sectional mean| = {max_abs_mean:.2e}, expected < 1e-10"
    )

    # S1-4: Median monthly Spearman of momentum_z vs forward return
    assert -0.02 <= median_spearman <= 0.10, (
        f"S1-4: Median monthly Spearman {median_spearman:.4f} outside [-0.02, 0.10]"
    )

    # S1-5: DATA QUALITY block (presence verified by stdout capture)
    # S1-6: Survivorship bias (presence verified by stdout capture)

    # ── RESULTS ────────────────────────────────────────────────────────
    print(f"══ lecture/s1_cross_sectional_setup ══════════════════════")
    print(f"  feature_matrix_shape: {fm.shape}")
    print(f"  n_features: {fm.shape[1]}")
    print(f"  n_months: {n_months_fm}")
    print(f"  tickers_per_month: {tpm.min()}–{tpm.max()}")
    print(f"  forward_return_months: {len(fwd_dates)}")
    print(f"  max_fwd_date: {max_fwd_date.date()}")
    print(f"  excess_return_max_abs_mean: {max_abs_mean:.2e}")
    print(f"  median_monthly_spearman: {median_spearman:.4f}")
    print(f"  mean_monthly_spearman: {mean_spearman:.4f}")
    print(f"  spearman_months_computed: {len(spearman_df)}")
    print(f"  representative_month: {mid_date.strftime('%Y-%m')}")
    print(f"  representative_month_n_stocks: {len(cs)}")
    print(f"  ols_slope_representative: {slope:.6f}")
    print(f"  fwd_return_mean: {fwd.mean():.6f}")
    print(f"  fwd_return_std: {fwd.std():.6f}")
    print(f"  survivorship_bias_acknowledged: yes")

    # ── PLOT: scatter ─────────────────────────────────────────────────
    fig_scatter.savefig(PLOT_DIR / "s1_momentum_scatter.png", dpi=150, bbox_inches="tight")
    print(f"  ── plot: s1_momentum_scatter.png ──")
    print(f"     type: scatter (momentum_z vs excess return)")
    print(f"     n_points: {len(cs)}")
    print(f"     x_range: [{cs['momentum_z'].min():.2f}, {cs['momentum_z'].max():.2f}]")
    print(f"     y_range: [{ax_scatter.get_ylim()[0]:.4f}, {ax_scatter.get_ylim()[1]:.4f}]")
    print(f"     title: {ax_scatter.get_title()}")

    # ── PLOT: histogram ───────────────────────────────────────────────
    fig_hist.savefig(PLOT_DIR / "s1_return_distribution.png", dpi=150, bbox_inches="tight")
    print(f"  ── plot: s1_return_distribution.png ──")
    print(f"     type: histogram (cross-sectional forward returns)")
    print(f"     n_observations: {len(cs)}")
    print(f"     n_bins: {len(ax_hist.patches)}")
    print(f"     x_range: [{ax_hist.get_xlim()[0]:.4f}, {ax_hist.get_xlim()[1]:.4f}]")
    print(f"     title: {ax_hist.get_title()}")

    print(f"✓ s1_cross_sectional_setup: ALL PASSED")
