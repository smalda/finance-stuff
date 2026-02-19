"""Section 6: The Responsible Backtest — Putting It All Together.

Integrates purged CV (from S2), transaction costs (from S4), and DSR (from S5)
into a unified "naive vs. responsible" evaluation framework. Teaches the
practitioner discipline of reporting the full degradation stack.
"""

import matplotlib
matplotlib.use("Agg")
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.model_selection import TimeSeriesSplit

_CODE_DIR = Path(__file__).resolve().parent.parent
_COURSE_DIR = _CODE_DIR.parent.parent
sys.path.insert(0, str(_CODE_DIR))
sys.path.insert(0, str(_COURSE_DIR))
from data_setup import CACHE_DIR, PLOT_DIR, load_alpha_output, load_ls_portfolio
from shared.backtesting import (
    cumulative_returns,
    max_drawdown,
    net_returns,
    sharpe_ratio,
)
from shared.metrics import deflated_sharpe_ratio, ic_summary
from shared.temporal import PurgedWalkForwardCV


# ── Load upstream caches ──────────────────────────────────────────────

_alpha = load_alpha_output()
_predictions = _alpha["predictions"]
_ic_series = _alpha["ic_series"]

_ls = load_ls_portfolio()
_gross_returns = _ls["gross_returns"]       # full OOS series
_turnover = _ls["turnover"]

# Net-of-cost returns from S4 cache.
# Use tiered (10/20/30 bps by market-cap tier) as the responsible baseline:
# it reflects realistic differentiated costs across large/mid/small-cap stocks.
_tc = pd.read_parquet(CACHE_DIR / "tc_results.parquet")
_net_returns_tiered = _tc["net_return_tiered"]   # 10/20/30 bps tiered

# Walk-forward and purged IC fold series from S2 cache
_wf_ic_folds = pd.read_parquet(CACHE_DIR / "wf_ic.parquet")["wf_ic"]
_purged_ic_folds = pd.read_parquet(CACHE_DIR / "purged_ic.parquet")["purged_ic"]

# OOS date range (April 2019 – November 2024)
_oos_start = _gross_returns.index[0]
_oos_end = _gross_returns.index[-1]

# Sub-period boundary: 2019-04 to 2021-11 (first 32 months) vs 2021-12 to 2024-11
_sub1_end = pd.Timestamp("2021-11-30")
_sub2_start = pd.Timestamp("2021-12-31")


# ── CELL: naive_metrics ──────────────────────────────────────────────

# Naive evaluation: gross returns, TimeSeriesSplit (no purging), no DSR filter.
# This is what an uncritical researcher reports: just the equity curve from
# rolling cross-sectional IC with no methodology adjustments.

gross_returns = _gross_returns.copy()

naive_sharpe = sharpe_ratio(gross_returns, periods_per_year=12)
naive_ret_ann = gross_returns.mean() * 12
naive_mdd = max_drawdown(gross_returns)
naive_skew = float(stats.skew(gross_returns.dropna()))
naive_kurt = float(stats.kurtosis(gross_returns.dropna(), fisher=True))

# TimeSeriesSplit IC (no purging) — use fold series cached from S2
tss_mean_ic = float(_wf_ic_folds.mean())
tss_std_ic = float(_wf_ic_folds.std())


# ── CELL: naive_print ────────────────────────────────────────────────

print("── IS / OOS BOUNDARY ─────────────────────────────────────────")
print(f"  IS period:  pre-April 2019 (model training in Week 4)")
print(f"  OOS period: {_oos_start.date()} to {_oos_end.date()} ({len(gross_returns)} months)")
print()
print("── NAIVE EVALUATION (IS label; gross returns; no purging) ────")
print(f"  ann_return:      {naive_ret_ann:.4f}")
print(f"  ann_sharpe:      {naive_sharpe:.4f}")
print(f"  max_drawdown:    {naive_mdd:.4f}")
print(f"  skewness:        {naive_skew:.4f}")
print(f"  excess_kurtosis: {naive_kurt:.4f}")
print(f"  cv_mean_ic:      {tss_mean_ic:.4f}  (TimeSeriesSplit, no purging)")


# ── CELL: responsible_net_metrics ────────────────────────────────────

# Responsible evaluation adds three disciplines:
#   1. Purged CV IC (from S2) — accounts for look-ahead in fold construction
#   2. Net-of-cost returns (from S4, tiered 10/20/30 bps) — realistic TC
#   3. DSR at M=10 — adjusts Sharpe for the number of model variants tested

net_ret_series = _net_returns_tiered.reindex(_gross_returns.index).dropna()
resp_sharpe = sharpe_ratio(net_ret_series, periods_per_year=12)
resp_ret_ann = net_ret_series.mean() * 12
resp_mdd = max_drawdown(net_ret_series)
resp_skew = float(stats.skew(net_ret_series.dropna()))
resp_kurt = float(stats.kurtosis(net_ret_series.dropna(), fisher=True))


# ── CELL: responsible_ic_and_dsr ─────────────────────────────────────

# Purged CV IC
purged_mean_ic = float(_purged_ic_folds.mean())
purged_std_ic = float(_purged_ic_folds.std())

# DSR at M=10 on the responsible return series
# deflated_sharpe_ratio() expects per-period (monthly) SR, not annualized
n_obs = len(net_ret_series)
resp_monthly_sr = net_ret_series.mean() / net_ret_series.std() if net_ret_series.std() > 0 else 0.0
dsr_m10 = deflated_sharpe_ratio(
    observed_sr=resp_monthly_sr,
    n_trials=10,
    n_obs=n_obs,
    skew=resp_skew,
    excess_kurt=resp_kurt,
)

sharpe_gap = naive_sharpe - resp_sharpe


# ── CELL: responsible_print ──────────────────────────────────────────

print()
print("── RESPONSIBLE EVALUATION (OOS label; net-tiered; purged CV) ──")
print(f"  ann_return:      {resp_ret_ann:.4f}")
print(f"  ann_sharpe:      {resp_sharpe:.4f}")
print(f"  monthly_sharpe:  {resp_monthly_sr:.4f}")
print(f"  max_drawdown:    {resp_mdd:.4f}")
print(f"  skewness:        {resp_skew:.4f}")
print(f"  excess_kurtosis: {resp_kurt:.4f}")
print(f"  cv_mean_ic:      {purged_mean_ic:.4f}  (PurgedKFold)")
print(f"  dsr_m10:         {dsr_m10:.4f}")
print(f"  sharpe_gap (naive - resp): {sharpe_gap:.4f}")


# ── CELL: sub_period_ic_split ─────────────────────────────────────────

# Sub-period IC degradation: a practitioner always checks whether the
# signal is stable across time or has decayed. Here we split the OOS
# period at the midpoint and compare IC in the two halves.

ic_full = _ic_series["ic"]
ic_sub1 = ic_full.loc[_oos_start:_sub1_end]
ic_sub2 = ic_full.loc[_sub2_start:_oos_end]

ic_sub1_mean = float(ic_sub1.mean())
ic_sub2_mean = float(ic_sub2.mean())
ic_sub1_std = float(ic_sub1.std())
ic_sub2_std = float(ic_sub2.std())
ic_degradation = ic_sub1_mean - ic_sub2_mean

print()
print("── SUB-PERIOD IC SPLIT (OOS stability check) ─────────────────")
print(f"  Sub-period 1 (OOS: {ic_sub1.index[0].date()} to {ic_sub1.index[-1].date()}):")
print(f"    n_months: {len(ic_sub1)}")
print(f"    mean_ic:  {ic_sub1_mean:.4f}  std_ic: {ic_sub1_std:.4f}")
print(f"  Sub-period 2 (OOS: {ic_sub2.index[0].date()} to {ic_sub2.index[-1].date()}):")
print(f"    n_months: {len(ic_sub2)}")
print(f"    mean_ic:  {ic_sub2_mean:.4f}  std_ic: {ic_sub2_std:.4f}")
print(f"  IC degradation (sub1 - sub2): {ic_degradation:.4f}")


# ── CELL: equity_curve_comparison ────────────────────────────────────

# Side-by-side naive vs. responsible equity curves.
# IS/OOS boundary marked with a vertical line.
# The gap between the curves is the aggregate cost of responsible reporting.

cum_naive = cumulative_returns(gross_returns)
# align net to gross index for plotting
net_aligned = net_ret_series.reindex(gross_returns.index).fillna(0.0)
cum_resp = cumulative_returns(net_aligned)

fig_curve, ax_curve = plt.subplots(figsize=(10, 5))
ax_curve.plot(cum_naive.index, cum_naive.values, label="Naive (gross, WF CV)", lw=2)
ax_curve.plot(cum_resp.index, cum_resp.values,
              label="Responsible (net-tiered, purged CV)", lw=2, linestyle="--")
ax_curve.axvline(x=_oos_start, color="gray", lw=1.2, linestyle=":", label="OOS start")
ax_curve.axvline(x=_sub1_end, color="lightcoral", lw=1.0, linestyle=":",
                 label="Sub-period split")
ax_curve.set(
    title="Naive vs. Responsible Equity Curve (IS/OOS labeled)",
    xlabel="Date",
    ylabel="Cumulative Return",
)
ax_curve.legend(fontsize=9)
plt.tight_layout()
plt.show()


# ── CELL: ic_bar_comparison ───────────────────────────────────────────

# Monthly IC bar chart: walk-forward IC vs. purged IC side by side.
# Both are fold-level summaries from S2. Purged IC is systematically lower
# (reduced look-ahead) — the visual gap motivates careful CV construction.

fold_indices = np.arange(len(_wf_ic_folds))
bar_width = 0.35

fig_bar, ax_bar = plt.subplots(figsize=(10, 5))
ax_bar.bar(fold_indices - bar_width / 2, _wf_ic_folds.values,
           width=bar_width, label="Walk-Forward IC (naive)", alpha=0.8)
ax_bar.bar(fold_indices + bar_width / 2, _purged_ic_folds.values,
           width=bar_width, label="Purged CV IC (responsible)", alpha=0.8)
ax_bar.axhline(0, color="black", lw=0.8)
ax_bar.set(
    title="Fold-Level IC: Walk-Forward vs. Purged CV (OOS folds)",
    xlabel="Fold Index",
    ylabel="IC (Spearman)",
    xticks=fold_indices,
)
ax_bar.legend()
plt.tight_layout()
plt.show()


# ── VERIFICATION BLOCK ────────────────────────────────────────────────

if __name__ == "__main__":

    # ── ASSERTIONS ─────────────────────────────────────────────────
    assert 0.5 <= naive_sharpe <= 1.8, (
        f"S6-1: Naive Sharpe {naive_sharpe:.4f} outside [0.5, 1.8]"
    )
    assert 0.1 <= resp_sharpe <= 1.2, (
        f"S6-2: Responsible Sharpe {resp_sharpe:.4f} outside [0.1, 1.2]"
    )
    assert resp_sharpe < naive_sharpe, (
        f"S6-2: Responsible Sharpe {resp_sharpe:.4f} must be < naive {naive_sharpe:.4f}"
    )
    assert 0.1 <= sharpe_gap <= 0.8, (
        f"S6-3: Sharpe gap {sharpe_gap:.4f} outside [0.1, 0.8]"
    )
    # S6-4: sub-period IC split was printed above (structural)
    assert len(ic_sub1) >= 10, (
        f"S6-4: sub-period 1 has only {len(ic_sub1)} months (need ≥10)"
    )
    assert len(ic_sub2) >= 10, (
        f"S6-4: sub-period 2 has only {len(ic_sub2)} months (need ≥10)"
    )
    # S6-5: IS/OOS labeling is in all prints and plots above (structural)

    # ── RESULTS ────────────────────────────────────────────────────
    print()
    print(f"══ lecture/s6_responsible_backtest ══════════════════════════")
    print(f"  naive_sharpe:      {naive_sharpe:.4f}")
    print(f"  resp_sharpe:       {resp_sharpe:.4f}")
    print(f"  resp_monthly_sr:   {resp_monthly_sr:.4f}")
    print(f"  sharpe_gap:        {sharpe_gap:.4f}")
    print(f"  naive_ann_ret:     {naive_ret_ann:.4f}")
    print(f"  resp_ann_ret:      {resp_ret_ann:.4f}")
    print(f"  naive_mdd:         {naive_mdd:.4f}")
    print(f"  resp_mdd:          {resp_mdd:.4f}")
    print(f"  naive_skew:        {naive_skew:.4f}")
    print(f"  naive_excess_kurt: {naive_kurt:.4f}")
    print(f"  resp_skew:         {resp_skew:.4f}")
    print(f"  resp_excess_kurt:  {resp_kurt:.4f}")
    print(f"  wf_cv_mean_ic:     {tss_mean_ic:.4f}")
    print(f"  purged_cv_mean_ic: {purged_mean_ic:.4f}")
    print(f"  ic_degradation:    {ic_degradation:.4f}")
    print(f"  dsr_m10:           {dsr_m10:.4f}")
    print(f"  n_oos_months:      {len(gross_returns)}")
    print(f"  sub1_n:            {len(ic_sub1)}")
    print(f"  sub2_n:            {len(ic_sub2)}")
    print(f"  sub1_mean_ic:      {ic_sub1_mean:.4f}")
    print(f"  sub2_mean_ic:      {ic_sub2_mean:.4f}")

    # ── PLOTS ───────────────────────────────────────────────────────
    fig_curve.savefig(
        PLOT_DIR / "s6_equity_curve_comparison.png", dpi=150, bbox_inches="tight"
    )
    print(f"  ── plot: s6_equity_curve_comparison.png ──")
    print(f"     type: dual cumulative return lines (naive vs. responsible)")
    print(f"     n_lines: {len(ax_curve.get_lines())}")
    print(f"     y_range: [{ax_curve.get_ylim()[0]:.2f}, {ax_curve.get_ylim()[1]:.2f}]")
    print(f"     title: {ax_curve.get_title()}")

    fig_bar.savefig(
        PLOT_DIR / "s6_ic_bar_comparison.png", dpi=150, bbox_inches="tight"
    )
    print(f"  ── plot: s6_ic_bar_comparison.png ──")
    print(f"     type: grouped bar chart (fold-level IC, WF vs. purged)")
    print(f"     n_groups: {len(fold_indices)}")
    print(f"     y_range: [{ax_bar.get_ylim()[0]:.2f}, {ax_bar.get_ylim()[1]:.2f}]")
    print(f"     title: {ax_bar.get_title()}")

    print(f"✓ s6_responsible_backtest: ALL PASSED")
