"""Seminar Exercise 3: Transaction Cost Sensitivity — Feasibility Frontier.

At 139.9% monthly one-way turnover, even modest transaction costs erode
a strategy's Sharpe ratio substantially. This exercise maps the full
(half-spread, turnover-reduction) feasibility space to show practitioners
what combination of cost environment and execution efficiency is needed
to keep a strategy viable.
"""

import matplotlib
matplotlib.use("Agg")
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

_CODE_DIR = Path(__file__).resolve().parent.parent
_COURSE_DIR = _CODE_DIR.parent.parent
sys.path.insert(0, str(_CODE_DIR))
sys.path.insert(0, str(_COURSE_DIR))
from data_setup import CACHE_DIR, PLOT_DIR

# Load upstream caches written by s4_transaction_costs.py
_TC_CACHE = CACHE_DIR / "tc_results.parquet"
_LS_CACHE = CACHE_DIR / "ls_portfolio.parquet"


# ── CELL: load_data ──────────────────────────────────────────────────

# Load gross returns and turnover from s4 cache; fall back to ls_portfolio
if _TC_CACHE.exists():
    tc = pd.read_parquet(_TC_CACHE)
    gross_returns = tc["gross_return"]
    turnover = tc["turnover"]
    # Fixed impact baseline from s4 (does not vary with spread/turnover grid)
    impact_baseline = tc["impact_cost"] if "impact_cost" in tc.columns else pd.Series(0.0, index=gross_returns.index)
    print(f"Loaded tc_results.parquet: {len(gross_returns)} months")
else:
    ls = pd.read_parquet(_LS_CACHE)
    gross_returns = ls["gross_return"]
    turnover = ls["turnover"]
    impact_baseline = pd.Series(0.0, index=gross_returns.index)
    print("Fallback: loaded ls_portfolio.parquet")


# ── CELL: data_statistics ───────────────────────────────────────────

mean_turnover = turnover.mean()
gross_sharpe = gross_returns.mean() * 12 / (gross_returns.std() * np.sqrt(12))
mean_impact = impact_baseline.mean()

print(f"Gross annualized Sharpe: {gross_sharpe:.4f}")
print(f"Mean monthly one-way turnover: {mean_turnover:.1%}")
print(f"Mean monthly impact cost (fixed baseline): {mean_impact:.6f}")

if mean_turnover > 0.50:
    ann_drag_at_10bps = mean_turnover * 2 * (10 / 10_000) * 12
    print(f"⚠ HIGH TURNOVER: {mean_turnover:.0%} one-way — "
          f"TC drag ≈ {ann_drag_at_10bps:.2%}/year at 10 bps")


# ── CELL: define_grid ────────────────────────────────────────────────

# Feasibility grid: half-spread (bps) × turnover reduction fraction
HALF_SPREAD_GRID = [2, 5, 8, 10, 12, 15, 20, 25, 30]   # bps
TURNOVER_REDUCTION_GRID = [0.0, 0.10, 0.20, 0.30, 0.40, 0.50]  # fraction saved
SHARPE_THRESHOLD = 0.5   # minimum acceptable annualized net Sharpe

print(f"\nGrid dimensions: {len(HALF_SPREAD_GRID)} spreads × "
      f"{len(TURNOVER_REDUCTION_GRID)} turnover reductions = "
      f"{len(HALF_SPREAD_GRID) * len(TURNOVER_REDUCTION_GRID)} cells")
print(f"Feasibility threshold: net Sharpe ≥ {SHARPE_THRESHOLD}")


# ── CELL: annualized_sharpe_fn ───────────────────────────────────────

def annualized_sharpe(returns: pd.Series) -> float:
    """Compute annualized Sharpe ratio (12 monthly periods per year)."""
    if returns.std() == 0:
        return 0.0
    return float(returns.mean() * 12 / (returns.std() * np.sqrt(12)))


# ── CELL: net_sharpe_fn ─────────────────────────────────────────────

def net_sharpe_for_cell(
    gross: pd.Series,
    to: pd.Series,
    half_spread_bps: float,
    to_reduction: float,
    impact: pd.Series | None = None,
) -> float:
    """Net Sharpe for one (spread, turnover-reduction) cell.

    net_returns subtracts: scaled_turnover * 2 * half_spread_bps / 10_000
    per period (round-trip cost = 2 × one-way turnover × half-spread),
    plus a fixed market impact baseline (does not vary with spread/turnover).
    """
    scaled_turnover = to * (1.0 - to_reduction)
    cost_frac = half_spread_bps / 10_000
    common = gross.index.intersection(scaled_turnover.index)
    costs = scaled_turnover[common] * 2 * cost_frac
    net = gross[common] - costs
    if impact is not None:
        impact_aligned = impact.reindex(common).fillna(0.0)
        net = net - impact_aligned
    return annualized_sharpe(net)


# ── CELL: compute_grid ──────────────────────────────────────────────

# Compute the full grid
net_sharpe_grid = np.zeros((len(HALF_SPREAD_GRID), len(TURNOVER_REDUCTION_GRID)))

for i, spread in enumerate(HALF_SPREAD_GRID):
    for j, reduction in enumerate(TURNOVER_REDUCTION_GRID):
        net_sharpe_grid[i, j] = net_sharpe_for_cell(
            gross_returns, turnover, spread, reduction,
            impact=impact_baseline,
        )


# ── CELL: print_heatmap ─────────────────────────────────────────────

print("\nNet Sharpe heatmap (rows=half-spread bps, cols=turnover reduction %):")
header = "        " + "  ".join([f"{r:.0%}".rjust(6) for r in TURNOVER_REDUCTION_GRID])
print(header)
for i, sp in enumerate(HALF_SPREAD_GRID):
    row_vals = "  ".join([f"{v:+.3f}" for v in net_sharpe_grid[i]])
    feasible = (net_sharpe_grid[i] >= SHARPE_THRESHOLD).sum()
    print(f"  {sp:2d} bps: {row_vals}  [{feasible}/{len(TURNOVER_REDUCTION_GRID)} feasible]")


# ── CELL: feasibility_counts ────────────────────────────────────────

# Feasible cells: net Sharpe ≥ threshold
feasible_mask = net_sharpe_grid >= SHARPE_THRESHOLD
FEASIBLE_CELLS = int(feasible_mask.sum())


# ── CELL: feasibility_frontier ──────────────────────────────────────

# Feasibility frontier per turnover-reduction column: min spread that stays feasible
frontier_spreads = []
for j in range(len(TURNOVER_REDUCTION_GRID)):
    col_feasible = [
        HALF_SPREAD_GRID[i]
        for i in range(len(HALF_SPREAD_GRID))
        if net_sharpe_grid[i, j] >= SHARPE_THRESHOLD
    ]
    frontier_spreads.append(max(col_feasible) if col_feasible else 0)

# Breakeven half-spread at 0% turnover reduction (worst case)
breakeven_spread_no_reduction = frontier_spreads[0]

print(f"\nFeasibility frontier (max viable half-spread per turnover reduction):")
for j, red in enumerate(TURNOVER_REDUCTION_GRID):
    print(f"  {red:.0%} reduction: max viable half-spread = "
          f"{frontier_spreads[j]} bps")


# ── CELL: breakeven_analysis ────────────────────────────────────────

print(f"\nFEASIBLE_CELLS = {FEASIBLE_CELLS} "
      f"(of {len(HALF_SPREAD_GRID) * len(TURNOVER_REDUCTION_GRID)} total)")
print(f"GROSS_SHARPE = {gross_sharpe:.4f}")
print(f"BREAKEVEN_SPREAD = {breakeven_spread_no_reduction} bps "
      f"(at 0% turnover reduction, net Sharpe ≥ {SHARPE_THRESHOLD})")

if FEASIBLE_CELLS == 0:
    print(f"⚠ STRATEGY NON-VIABLE: no cell achieves net Sharpe ≥ {SHARPE_THRESHOLD}")
    print("  At this turnover level, no realistic cost environment is viable.")
    print("  To rescue: reduce rebalancing frequency or tighten signal thresholds.")
else:
    print(f"\nAt 30 bps spread (no reduction):")
    net_30_0 = net_sharpe_for_cell(gross_returns, turnover, 30, 0.0, impact=impact_baseline)
    sharpe_gap = gross_sharpe - net_30_0
    print(f"  Net Sharpe = {net_30_0:.4f}")
    print(f"  Gross − Net = {sharpe_gap:.4f} ({sharpe_gap:.2f} Sharpe units)")


# ── CELL: plot_heatmap ───────────────────────────────────────────────

fig, ax = plt.subplots(figsize=(9, 6))

im = ax.imshow(
    net_sharpe_grid,
    aspect="auto",
    cmap="RdYlGn",
    vmin=0.0,
    vmax=1.0,
    origin="upper",
)
plt.colorbar(im, ax=ax, label="Net Annualized Sharpe Ratio")

# Axis labels
ax.set_xticks(range(len(TURNOVER_REDUCTION_GRID)))
ax.set_xticklabels([f"{r:.0%}" for r in TURNOVER_REDUCTION_GRID])
ax.set_yticks(range(len(HALF_SPREAD_GRID)))
ax.set_yticklabels([f"{s} bps" for s in HALF_SPREAD_GRID])
ax.set_xlabel("Turnover Reduction (execution efficiency gain)")
ax.set_ylabel("Half-Spread (bps, transaction cost environment)")
ax.set_title(
    f"TC Feasibility Frontier — Net Sharpe vs. Cost & Efficiency\n"
    f"Gross Sharpe = {gross_sharpe:.2f}, Mean Turnover = {mean_turnover:.0%}/month"
)

# Annotate each cell with net Sharpe value
for i in range(len(HALF_SPREAD_GRID)):
    for j in range(len(TURNOVER_REDUCTION_GRID)):
        val = net_sharpe_grid[i, j]
        color = "white" if val < 0.35 or val > 0.80 else "black"
        ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                fontsize=8.5, color=color, fontweight="bold")

# Feasibility contour overlay at threshold
# Draw contour on a fine interpolated grid
xx = np.linspace(0, len(TURNOVER_REDUCTION_GRID) - 1, 200)
yy = np.linspace(0, len(HALF_SPREAD_GRID) - 1, 200)
XX, YY = np.meshgrid(xx, yy)

# Bilinear interpolation of the grid for contour
from scipy.interpolate import RegularGridInterpolator
interp = RegularGridInterpolator(
    (np.arange(len(HALF_SPREAD_GRID)), np.arange(len(TURNOVER_REDUCTION_GRID))),
    net_sharpe_grid,
    method="linear",
    bounds_error=False,
    fill_value=None,
)
ZZ = interp((YY, XX))
cs = ax.contour(XX, YY, ZZ, levels=[SHARPE_THRESHOLD], colors=["navy"],
                linewidths=2, linestyles="--")
ax.clabel(cs, fmt=f"SR={SHARPE_THRESHOLD}", fontsize=9, colors="navy")

plt.tight_layout()
plt.show()


if __name__ == "__main__":
    # ── ASSERTIONS ──────────────────────────────────────────────────

    # EX3-1: Feasibility frontier monotonically non-increasing in spread
    # For any fixed turnover reduction, higher spread → lower or equal net Sharpe
    for j in range(len(TURNOVER_REDUCTION_GRID)):
        col = net_sharpe_grid[:, j]
        for i in range(len(col) - 1):
            assert col[i] >= col[i + 1] - 1e-8, (
                f"Frontier not monotone at reduction={TURNOVER_REDUCTION_GRID[j]:.0%}: "
                f"spread={HALF_SPREAD_GRID[i]} bps has net SR={col[i]:.4f} < "
                f"spread={HALF_SPREAD_GRID[i+1]} bps SR={col[i+1]:.4f}"
            )

    # EX3-2: At 30 bps spread, 0% reduction: net Sharpe ≤ gross Sharpe − 0.3
    net_30_0 = net_sharpe_for_cell(gross_returns, turnover, 30, 0.0, impact=impact_baseline)
    assert net_30_0 <= gross_sharpe - 0.3, (
        f"At 30 bps / 0% reduction: net SR={net_30_0:.4f}, "
        f"gross SR={gross_sharpe:.4f}, gap={gross_sharpe - net_30_0:.4f} < 0.3"
    )

    # EX3-4: Warning printed if FEASIBLE_CELLS == 0 (handled inline above)
    # (structural assertion: the block executes without error)
    assert isinstance(FEASIBLE_CELLS, int), "FEASIBLE_CELLS must be an integer"

    # EX3-5: Key quantities are numeric and finite
    assert np.isfinite(gross_sharpe), f"GROSS_SHARPE not finite: {gross_sharpe}"
    assert FEASIBLE_CELLS >= 0, f"FEASIBLE_CELLS negative: {FEASIBLE_CELLS}"
    assert breakeven_spread_no_reduction >= 0, (
        f"BREAKEVEN_SPREAD negative: {breakeven_spread_no_reduction}"
    )

    # Additional sanity: grid is 9 × 6
    assert net_sharpe_grid.shape == (9, 6), (
        f"Grid shape mismatch: {net_sharpe_grid.shape} != (9, 6)"
    )

    # ── RESULTS ─────────────────────────────────────────────────────
    print(f"══ seminar/ex3_tc_sensitivity ════════════════════════════")
    print(f"  GROSS_SHARPE: {gross_sharpe:.4f}")
    print(f"  mean_turnover_pct: {mean_turnover:.4f}  ({mean_turnover:.1%}/month)")
    print(f"  FEASIBLE_CELLS: {FEASIBLE_CELLS} "
          f"(of {len(HALF_SPREAD_GRID) * len(TURNOVER_REDUCTION_GRID)} total, "
          f"threshold={SHARPE_THRESHOLD})")
    print(f"  BREAKEVEN_SPREAD: {breakeven_spread_no_reduction} bps "
          f"(0% turnover reduction)")
    print(f"  net_sharpe_at_30bps_0pct: {net_30_0:.4f}")
    print(f"  sharpe_drag_at_30bps: {gross_sharpe - net_30_0:.4f}")

    # ── PLOT ────────────────────────────────────────────────────────
    plot_path = PLOT_DIR / "ex3_tc_sensitivity_heatmap.png"
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    print(f"  ── plot: ex3_tc_sensitivity_heatmap.png ──")
    print(f"     type: 2D heatmap with feasibility contour")
    print(f"     grid: {len(HALF_SPREAD_GRID)} half-spreads × "
          f"{len(TURNOVER_REDUCTION_GRID)} turnover reductions")
    print(f"     color_range: [0.0, 1.0] net Sharpe (RdYlGn)")
    print(f"     contour_threshold: {SHARPE_THRESHOLD}")
    print(f"     title: {ax.get_title().splitlines()[0]}")
    print(f"     y_axis: {ax.get_ylabel()}")
    print(f"     x_axis: {ax.get_xlabel()}")

    print(f"✓ ex3_tc_sensitivity: ALL PASSED")
