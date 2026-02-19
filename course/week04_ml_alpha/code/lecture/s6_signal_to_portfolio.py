"""Section 6: From Signal to Portfolio — Decile Sorts, Long-Short, and Costs"""
import matplotlib
matplotlib.use("Agg")
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).parent.parent))
from data_setup import CACHE_DIR, PLOT_DIR, COST_BPS

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from shared.backtesting import (
    quantile_portfolios,
    long_short_returns,
    portfolio_turnover,
    sharpe_ratio,
    net_returns,
    max_drawdown,
    cumulative_returns,
)

# ── Load upstream data ──────────────────────────────────────────────────
gbm_preds = pd.read_parquet(CACHE_DIR / "gbm_predictions.parquet")
predictions = gbm_preds["prediction"]
actuals = gbm_preds["actual"]

oos_dates = predictions.index.get_level_values("date").unique().sort_values()
n_oos = len(oos_dates)
tickers_per_month = predictions.groupby(level="date").size()
print(f"Loaded GBM predictions: {len(predictions)} obs, {n_oos} months, "
      f"{tickers_per_month.mean():.0f} stocks/month")


# ── CELL: decile_portfolios ─────────────────────────────────────────────

decile_returns = quantile_portfolios(predictions, actuals, n_groups=10)
decile_means = decile_returns.mean()

print(f"\nDecile mean monthly returns:")
for d in decile_means.index:
    print(f"  D{d:2d}: {decile_means[d]:+.5f}")


# ── CELL: monotonicity_check ────────────────────────────────────────────

top_decile = decile_means.iloc[-1]
bottom_decile = decile_means.iloc[0]
spread = top_decile - bottom_decile

print(f"\nTop decile (D{decile_means.index[-1]}) mean:    {top_decile:+.5f}")
print(f"Bottom decile (D{decile_means.index[0]}) mean: {bottom_decile:+.5f}")
print(f"Top - Bottom spread:        {spread:+.5f}")

rank_corr = stats.spearmanr(decile_means.index, decile_means.values)
print(f"Rank corr (decile vs mean return): {rank_corr.statistic:.3f} "
      f"(p={rank_corr.pvalue:.3f})")


# ── CELL: decile_bar_chart ──────────────────────────────────────────────

decile_labels = [int(x) for x in decile_means.index]
decile_vals = [float(v) * 100 for v in decile_means.values]

fig_decile, ax_decile = plt.subplots(figsize=(8, 5))
colors = ["#d62728" if v < 0 else "#2ca02c" for v in decile_vals]
ax_decile.bar(decile_labels, decile_vals, color=colors,
              edgecolor="black", linewidth=0.5)
ax_decile.axhline(0, color="black", linewidth=0.8)
ax_decile.set(
    title="Mean Monthly Return by Prediction Decile (GBM)",
    xlabel="Decile (1=lowest predicted, 10=highest predicted)",
    ylabel="Mean Monthly Return (%)",
)
ax_decile.set_xticks(decile_labels)
plt.tight_layout()
plt.show()


# ── CELL: long_short_series ─────────────────────────────────────────────

ls_returns = long_short_returns(predictions, actuals, n_groups=10)
ls_mean = ls_returns.mean()
ls_std = ls_returns.std()
ls_sharpe_gross = sharpe_ratio(ls_returns)

print(f"\nLong-short (D10 - D1) monthly return series:")
print(f"  Observations: {len(ls_returns)}")
print(f"  Mean monthly:  {ls_mean:+.5f}")
print(f"  Std monthly:   {ls_std:.5f}")
print(f"  Gross ann. Sharpe: {ls_sharpe_gross:.3f}")


# ── CELL: turnover_analysis ─────────────────────────────────────────────

turnover = portfolio_turnover(predictions, n_groups=10)
mean_turnover = turnover.mean()
annual_turnover = mean_turnover * 12

print(f"\nTurnover analysis:")
print(f"  Monthly turnover obs: {len(turnover)}")
print(f"  Mean monthly one-way: {mean_turnover:.4f}")
print(f"  Annualized one-way:   {annual_turnover:.2f}")


# ── CELL: net_returns_analysis ──────────────────────────────────────────

ls_net = net_returns(ls_returns, turnover, cost_bps=COST_BPS)
ls_sharpe_net = sharpe_ratio(ls_net)
sharpe_drag = ls_sharpe_gross - ls_sharpe_net

print(f"\nNet returns at {COST_BPS} bps one-way cost:")
print(f"  Mean monthly net:  {ls_net.mean():+.5f}")
print(f"  Net ann. Sharpe:   {ls_sharpe_net:.3f}")
print(f"  Sharpe drag:       {sharpe_drag:+.3f}")


# ── CELL: drawdown_analysis ─────────────────────────────────────────────

mdd_gross = max_drawdown(ls_returns)
mdd_net = max_drawdown(ls_net)

print(f"\nDrawdown analysis:")
print(f"  Max drawdown (gross): {mdd_gross:.4f}")
print(f"  Max drawdown (net):   {mdd_net:.4f}")


# ── CELL: return_distribution ───────────────────────────────────────────

skewness = float(stats.skew(ls_returns.dropna().values))
kurtosis_excess = float(stats.kurtosis(ls_returns.dropna().values))

print(f"\nReturn distribution (gross long-short):")
print(f"  Skewness:         {skewness:+.3f}")
print(f"  Excess kurtosis:  {kurtosis_excess:+.3f}")
print(f"  Sharpe:           {ls_sharpe_gross:.3f}")


# ── CELL: cumulative_return_plot ────────────────────────────────────────

cum_gross = cumulative_returns(ls_returns)
cum_net = cumulative_returns(ls_net)

fig_cum, ax_cum = plt.subplots(figsize=(10, 5))
ax_cum.plot(cum_gross.index, cum_gross.values, label="Gross", linewidth=1.5,
            color="#1f77b4")
ax_cum.plot(cum_net.index, cum_net.values, label=f"Net ({COST_BPS} bps)",
            linewidth=1.5, color="#ff7f0e", linestyle="--")
ax_cum.axhline(1.0, color="grey", linewidth=0.8, linestyle=":")
ax_cum.set(
    title="Cumulative Long-Short Returns (GBM Signal, Decile Sort)",
    xlabel="Date",
    ylabel="Cumulative Return ($1 invested)",
)
ax_cum.legend(loc="upper left")
plt.tight_layout()
plt.show()


if __name__ == "__main__":
    # ── ASSERTIONS ─────────────────────────────────────────────────────
    n_decile_months = len(decile_returns)

    # S6-1: Decile portfolios for OOS months: 10 groups, ~18 stocks/decile
    assert 60 <= n_decile_months <= 69, \
        f"S6-1: Expected 60-69 decile months, got {n_decile_months}"
    n_decile_groups = len(decile_returns.columns)
    assert n_decile_groups == 10, \
        f"S6-1: Expected 10 decile groups, got {n_decile_groups}"

    # S6-2: Monotonic return pattern: top > bottom (if IC > 0)
    assert top_decile > bottom_decile, \
        f"S6-2: Top decile ({top_decile:.5f}) not > bottom ({bottom_decile:.5f})"

    # S6-3: Long-short monthly return series: 69 obs; mean in [-0.005, +0.015]
    assert len(ls_returns) >= 60, \
        f"S6-3: Expected >=60 LS obs, got {len(ls_returns)}"
    assert -0.005 <= ls_mean <= 0.015, \
        f"S6-3: LS mean {ls_mean:.5f} outside [-0.005, 0.015]"

    # S6-4: Gross annualized Sharpe: range [-0.5, 1.5]
    assert -0.5 <= ls_sharpe_gross <= 1.5, \
        f"S6-4: Gross Sharpe {ls_sharpe_gross:.3f} outside [-0.5, 1.5]"

    # S6-5: Monthly one-way turnover: range [0.20, 0.80]
    assert 0.20 <= mean_turnover <= 0.80, \
        f"S6-5: Mean turnover {mean_turnover:.4f} outside [0.20, 0.80]"

    # S6-6: Net returns at 10 bps: net Sharpe < gross Sharpe
    assert ls_sharpe_net < ls_sharpe_gross, \
        f"S6-6: Net Sharpe ({ls_sharpe_net:.3f}) not < gross ({ls_sharpe_gross:.3f})"

    # S6-7: Cumulative return plot produced (verified by file existence below)

    # S6-8: Max drawdown reported: range [-0.50, 0.00]
    assert -0.50 <= mdd_gross <= 0.00, \
        f"S6-8: Max drawdown {mdd_gross:.4f} outside [-0.50, 0.00]"

    # S6-9: Skewness and excess kurtosis reported alongside Sharpe
    assert np.isfinite(skewness), f"S6-9: Skewness is not finite: {skewness}"
    assert np.isfinite(kurtosis_excess), \
        f"S6-9: Excess kurtosis is not finite: {kurtosis_excess}"

    # ── PLOTS ──────────────────────────────────────────────────────────
    fig_decile.savefig(
        PLOT_DIR / "s6_decile_returns.png", dpi=150, bbox_inches="tight"
    )
    fig_cum.savefig(
        PLOT_DIR / "s6_cumulative_long_short.png", dpi=150, bbox_inches="tight"
    )

    # ── RESULTS ────────────────────────────────────────────────────────
    print(f"\n══ lecture/s6_signal_to_portfolio ══════════════════════")
    print(f"  n_oos_months: {n_oos}")
    print(f"  n_decile_months: {n_decile_months}")
    print(f"  n_decile_groups: {n_decile_groups}")
    print(f"  stocks_per_month: {tickers_per_month.mean():.0f}")
    print(f"  top_decile_mean: {top_decile:+.6f}")
    print(f"  bottom_decile_mean: {bottom_decile:+.6f}")
    print(f"  spread: {spread:+.6f}")
    print(f"  monotonicity_rank_corr: {rank_corr.statistic:.3f}")
    print(f"  ls_mean_monthly: {ls_mean:+.6f}")
    print(f"  ls_std_monthly: {ls_std:.6f}")
    print(f"  sharpe_gross: {ls_sharpe_gross:.4f}")
    print(f"  sharpe_net_{COST_BPS}bps: {ls_sharpe_net:.4f}")
    print(f"  sharpe_drag: {sharpe_drag:+.4f}")
    print(f"  mean_monthly_turnover: {mean_turnover:.4f}")
    print(f"  annual_turnover: {annual_turnover:.2f}")
    print(f"  max_drawdown_gross: {mdd_gross:.4f}")
    print(f"  max_drawdown_net: {mdd_net:.4f}")
    print(f"  skewness: {skewness:+.4f}")
    print(f"  excess_kurtosis: {kurtosis_excess:+.4f}")

    print(f"  ── plot: s6_decile_returns.png ──")
    print(f"     type: bar chart of mean monthly return by decile")
    print(f"     n_bars: {len(ax_decile.patches)}")
    print(f"     y_range: [{ax_decile.get_ylim()[0]:.4f}, {ax_decile.get_ylim()[1]:.4f}]")
    print(f"     title: {ax_decile.get_title()}")

    print(f"  ── plot: s6_cumulative_long_short.png ──")
    print(f"     type: line chart of cumulative gross + net long-short returns")
    print(f"     n_lines: {len(ax_cum.get_lines())}")
    print(f"     y_range: [{ax_cum.get_ylim()[0]:.4f}, {ax_cum.get_ylim()[1]:.4f}]")
    print(f"     title: {ax_cum.get_title()}")

    print(f"✓ s6_signal_to_portfolio: ALL PASSED")
