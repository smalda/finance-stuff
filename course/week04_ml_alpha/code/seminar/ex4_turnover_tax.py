"""Exercise 4: The Turnover Tax — Transaction Costs and Signal Decay

Quantifies how transaction costs erode alpha strategy profitability.
Loads GBM predictions, constructs long-short quintile portfolios, and sweeps
one-way cost from 0 to 100 bps to map out the net Sharpe frontier.
"""
import matplotlib
matplotlib.use("Agg")
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from data_setup import CACHE_DIR, PLOT_DIR

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from shared.backtesting import (
    long_short_returns,
    portfolio_turnover,
    sharpe_ratio,
    net_returns,
)

N_GROUPS = 5  # quintile portfolios — standard academic long-short sort

# ── Load upstream data ──────────────────────────────────────
gbm = pd.read_parquet(CACHE_DIR / "gbm_predictions.parquet")
predictions = gbm["prediction"]
actuals = gbm["actual"]


# ── CELL: compute_turnover ──────────────────────────────────

turnover = portfolio_turnover(predictions, n_groups=N_GROUPS)
gross_ls = long_short_returns(predictions, actuals, n_groups=N_GROUPS)

# Align gross returns to turnover dates for fair cost comparison
# (turnover has n-1 observations since it measures change between months)
common_dates = gross_ls.index.intersection(turnover.index)
gross_ls_aligned = gross_ls.loc[common_dates]

print(f"Turnover series length: {len(turnover)}")
print(f"Long-short series length (aligned): {len(gross_ls_aligned)}")
print(f"Mean monthly one-way turnover: {turnover.mean():.4f}")
print(f"Annualized one-way turnover: {turnover.mean() * 12:.2f}")


# ── CELL: turnover_warning ──────────────────────────────────

mean_turnover = turnover.mean()
annualized_turnover = mean_turnover * 12

if mean_turnover > 0.50:
    print("⚠ HIGH TURNOVER: monthly one-way turnover exceeds 0.50 — "
          "strategy replaces >50% of holdings each month")


# ── CELL: gross_sharpe ──────────────────────────────────────

gross_sharpe = sharpe_ratio(gross_ls_aligned)
print(f"Gross annualized Sharpe: {gross_sharpe:.4f}")


# ── CELL: net_sharpe_sweep ──────────────────────────────────

cost_levels_bps = np.arange(0, 101, 1)
net_sharpes = []

for cost in cost_levels_bps:
    net_ret = net_returns(gross_ls, turnover, cost_bps=cost)
    sr = sharpe_ratio(net_ret)
    net_sharpes.append(sr)

net_sharpes = np.array(net_sharpes)


# ── CELL: key_cost_levels ───────────────────────────────────

cost_labels = {5: "5 bps", 20: "20 bps", 50: "50 bps"}
net_sharpe_5 = net_sharpes[5]
net_sharpe_20 = net_sharpes[20]
net_sharpe_50 = net_sharpes[50]

diff_5 = gross_sharpe - net_sharpe_5
diff_20 = gross_sharpe - net_sharpe_20
diff_50 = gross_sharpe - net_sharpe_50

print(f"Net Sharpe at  5 bps: {net_sharpe_5:.4f}  (reduction: {diff_5:.4f})")
print(f"Net Sharpe at 20 bps: {net_sharpe_20:.4f}  (reduction: {diff_20:.4f})")
print(f"Net Sharpe at 50 bps: {net_sharpe_50:.4f}  (reduction: {diff_50:.4f})")


# ── CELL: breakeven_cost ────────────────────────────────────

breakeven_bps = np.nan
for i, sr in enumerate(net_sharpes):
    if sr <= 0:
        if i > 0 and net_sharpes[i - 1] > 0:
            # Linear interpolation for precision
            breakeven_bps = (i - 1) + net_sharpes[i - 1] / (
                net_sharpes[i - 1] - sr
            )
        else:
            breakeven_bps = float(i)
        break

if np.isnan(breakeven_bps) and net_sharpes[-1] > 0:
    breakeven_bps = float("inf")
    print("Breakeven cost: >100 bps (strategy remains profitable across sweep)")
else:
    print(f"Breakeven one-way cost: {breakeven_bps:.1f} bps")


# ── CELL: plot_sharpe_curve ─────────────────────────────────

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Panel A: Net Sharpe vs cost level
ax1 = axes[0]
ax1.plot(cost_levels_bps, net_sharpes, "b-", linewidth=1.5, label="Net Sharpe")
ax1.axhline(0, color="gray", linestyle="--", linewidth=0.8)
if not np.isinf(breakeven_bps):
    ax1.axvline(breakeven_bps, color="red", linestyle=":", linewidth=1.2,
                label=f"Breakeven = {breakeven_bps:.0f} bps")
for bps, label in cost_labels.items():
    ax1.plot(bps, net_sharpes[bps], "ro", markersize=6)
    ax1.annotate(f"{label}\n({net_sharpes[bps]:.2f})",
                 (bps, net_sharpes[bps]),
                 textcoords="offset points", xytext=(8, 8), fontsize=8)
ax1.set(title="Net Sharpe Ratio vs Transaction Cost",
        xlabel="One-way cost (bps)", ylabel="Annualized Sharpe")
ax1.legend(fontsize=9)

# Panel B: Bar chart at key cost levels
ax2 = axes[1]
bar_costs = [0, 5, 20, 50]
bar_sharpes = [net_sharpes[c] for c in bar_costs]
bar_labels = ["0 (gross)", "5", "20", "50"]
colors = ["#2ecc71" if s > 0 else "#e74c3c" for s in bar_sharpes]
ax2.bar(bar_labels, bar_sharpes, color=colors, edgecolor="black", linewidth=0.5)
ax2.axhline(0, color="gray", linestyle="--", linewidth=0.8)
ax2.set(title="Sharpe at Key Cost Levels (bps)",
        xlabel="One-way cost (bps)", ylabel="Annualized Sharpe")
for i, (lbl, val) in enumerate(zip(bar_labels, bar_sharpes)):
    ax2.text(i, val + 0.02 * np.sign(val), f"{val:.2f}",
             ha="center", va="bottom" if val >= 0 else "top", fontsize=9)

plt.tight_layout()
plt.show()


if __name__ == "__main__":
    from scipy.stats import skew, kurtosis

    # ── ASSERTIONS ─────────────────────────────────────

    # EX4-1: Turnover series length and mean
    assert 60 <= len(turnover) <= 68, \
        f"Expected 60–68 turnover values, got {len(turnover)}"
    assert 0.20 <= mean_turnover <= 0.80, \
        f"Mean monthly turnover {mean_turnover:.4f} outside [0.20, 0.80]"

    # EX4-2: Annualized one-way turnover
    assert 2.4 <= annualized_turnover <= 9.6, \
        f"Annualized turnover {annualized_turnover:.2f} outside [2.4, 9.6]"

    # EX4-3: Net Sharpe at 5 bps — reduction < 0.3
    assert diff_5 < 0.3, \
        f"Sharpe reduction at 5 bps = {diff_5:.4f}, expected < 0.3"

    # EX4-4: Net Sharpe at 20 bps — reduction between 0.2 and 1.0
    assert 0.2 <= diff_20 <= 1.0, \
        f"Sharpe reduction at 20 bps = {diff_20:.4f}, outside [0.2, 1.0]"

    # EX4-5: Net Sharpe at 50 bps — expected negative or near zero
    assert net_sharpe_50 <= 0.2, \
        f"Net Sharpe at 50 bps = {net_sharpe_50:.4f}, expected <= 0.2"

    # EX4-6: Breakeven cost level computed
    assert not np.isnan(breakeven_bps), "Breakeven cost not computed"

    # EX4-7: HIGH TURNOVER warning (conditional — already printed above)
    # The warning is printed conditionally in CELL: turnover_warning

    # EX4-8: Plot exists (saved below)

    # ── RESULTS ────────────────────────────────────────
    print(f"══ seminar/ex4_turnover_tax ══════════════════════════")
    print(f"  n_groups: {N_GROUPS}")
    print(f"  n_turnover_months: {len(turnover)}")
    print(f"  mean_monthly_turnover: {mean_turnover:.4f}")
    print(f"  annualized_turnover: {annualized_turnover:.2f}")
    print(f"  gross_sharpe: {gross_sharpe:.4f}")
    print(f"  net_sharpe_5bps: {net_sharpe_5:.4f}")
    print(f"  net_sharpe_20bps: {net_sharpe_20:.4f}")
    print(f"  net_sharpe_50bps: {net_sharpe_50:.4f}")
    print(f"  sharpe_reduction_5bps: {diff_5:.4f}")
    print(f"  sharpe_reduction_20bps: {diff_20:.4f}")
    print(f"  sharpe_reduction_50bps: {diff_50:.4f}")
    print(f"  breakeven_bps: {breakeven_bps:.1f}")
    if mean_turnover > 0.50:
        print(f"  HIGH_TURNOVER_WARNING: yes (monthly={mean_turnover:.4f})")
    else:
        print(f"  HIGH_TURNOVER_WARNING: no (monthly={mean_turnover:.4f})")

    # Distribution statistics of gross long-short returns
    ls_skew = skew(gross_ls_aligned.dropna())
    ls_kurt = kurtosis(gross_ls_aligned.dropna(), fisher=True)
    print(f"  ls_skewness: {ls_skew:.4f}")
    print(f"  ls_excess_kurtosis: {ls_kurt:.4f}")

    # ── PLOT ───────────────────────────────────────────
    fig.savefig(PLOT_DIR / "ex4_sharpe_vs_cost.png", dpi=150, bbox_inches="tight")
    print(f"  ── plot: ex4_sharpe_vs_cost.png ──")
    print(f"     type: dual-panel (line chart + bar chart)")
    print(f"     panel_a_n_points: {len(cost_levels_bps)}")
    print(f"     panel_a_y_range: [{axes[0].get_ylim()[0]:.2f}, {axes[0].get_ylim()[1]:.2f}]")
    print(f"     panel_a_title: {axes[0].get_title()}")
    print(f"     panel_b_n_bars: {len(bar_costs)}")
    print(f"     panel_b_title: {axes[1].get_title()}")
    print(f"✓ ex4_turnover_tax: ALL PASSED")
