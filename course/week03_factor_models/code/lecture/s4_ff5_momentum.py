"""Section 4: The Five-Factor Model and the Momentum Orphan"""
import matplotlib
matplotlib.use("Agg")
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm

sys.path.insert(0, str(Path(__file__).parent.parent))
from data_setup import (
    load_monthly_returns, load_factor_data, DEMO_TICKERS, CACHE_DIR,
)

# Load data
demo_subset = DEMO_TICKERS[:15]
monthly_returns = load_monthly_returns()[demo_subset]
ff5 = load_factor_data("5")
ff6 = load_factor_data("6")
ff3 = load_factor_data("3")

# Align dates (2014-2024)
common_idx = monthly_returns.index.intersection(ff5.index)
returns = monthly_returns.loc[common_idx]
rf = ff5.loc[common_idx, "RF"]
excess_returns = returns.sub(rf, axis=0)
factors5 = ff5.loc[common_idx, ["Mkt-RF", "SMB", "HML", "RMW", "CMA"]]
factors3 = ff3.loc[common_idx, ["Mkt-RF", "SMB", "HML"]]


# ── CELL: ff3_vs_ff5_comparison ─────────────────────────────

results = []
for ticker in excess_returns.columns:
    y = excess_returns[ticker].dropna()
    idx = y.index

    # FF3
    X3 = sm.add_constant(factors3.loc[idx])
    m3 = sm.OLS(y, X3).fit()

    # FF5
    X5 = sm.add_constant(factors5.loc[idx])
    m5 = sm.OLS(y, X5).fit()

    results.append({
        "ticker": ticker,
        "ff3_r2": m3.rsquared,
        "ff5_r2": m5.rsquared,
        "r2_gain": m5.rsquared - m3.rsquared,
        "ff3_alpha": m3.params["const"] * 12 * 100,
        "ff5_alpha": m5.params["const"] * 12 * 100,
        "rmw_loading": m5.params["RMW"],
        "cma_loading": m5.params["CMA"],
    })

comp = pd.DataFrame(results).set_index("ticker")
print(comp[["ff3_r2", "ff5_r2", "r2_gain"]].round(4).to_string())


# ── CELL: diminishing_returns ───────────────────────────────

median_ff3 = comp["ff3_r2"].median()
median_ff5 = comp["ff5_r2"].median()
median_gain = comp["r2_gain"].median()
pct_improved = (comp["r2_gain"] > 0).mean()

alpha_shrank = (comp["ff5_alpha"].abs() < comp["ff3_alpha"].abs()).mean()

print(f"\nMedian FF3 R²: {median_ff3:.4f}")
print(f"Median FF5 R²: {median_ff5:.4f}")
print(f"Median R² gain (FF3→FF5): {median_gain:.4f}")
print(f"Stocks improved: {pct_improved:.0%}")
print(f"Alpha shrank (FF3→FF5): {alpha_shrank:.0%}")


# ── CELL: factor_correlation_matrix ─────────────────────────

all_factors = ff6.loc[common_idx, ["Mkt-RF", "SMB", "HML", "RMW", "CMA", "UMD"]]
corr_matrix = all_factors.corr()

fig_corr, ax_corr = plt.subplots(figsize=(8, 6))
im = ax_corr.imshow(corr_matrix.values, cmap="RdBu_r", vmin=-1, vmax=1)
labels = corr_matrix.columns.tolist()
ax_corr.set_xticks(range(len(labels)))
ax_corr.set_xticklabels(labels, rotation=45, ha="right")
ax_corr.set_yticks(range(len(labels)))
ax_corr.set_yticklabels(labels)
for i in range(len(labels)):
    for j in range(len(labels)):
        ax_corr.text(j, i, f"{corr_matrix.values[i, j]:.2f}",
                     ha="center", va="center", fontsize=9)
ax_corr.set_title("Factor Correlation Matrix (2014-2024)")
fig_corr.colorbar(im, ax=ax_corr, shrink=0.8)
plt.tight_layout()
plt.show()


# ── CELL: momentum_cumulative_returns ───────────────────────

ff6_full = load_factor_data("6")
umd_full = ff6_full["UMD"].dropna()
umd_cum = (1 + umd_full).cumprod()

fig_mom, ax_mom = plt.subplots(figsize=(10, 5))
ax_mom.plot(umd_cum.index, umd_cum, lw=1.5, color="purple")
ax_mom.set_yscale("log")
ax_mom.set(title="Momentum (UMD) Cumulative Returns — Strong Growth, Crash Risk",
           xlabel="Date", ylabel="Cumulative Return ($1 invested)")
ax_mom.axvspan(pd.Timestamp("2009-01-01"), pd.Timestamp("2009-06-30"),
               alpha=0.2, color="red", label="2009 momentum crash")
ax_mom.legend()
ax_mom.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()


if __name__ == "__main__":
    # ── ASSERTIONS ─────────────────────────────────
    n_stocks = len(comp)
    assert n_stocks >= 10, f"Expected ≥10 stocks, got {n_stocks}"

    assert 0.25 <= median_ff5 <= 0.70, \
        f"Median FF5 R² = {median_ff5:.4f}, outside [0.25, 0.70]"

    assert 0.005 <= median_gain <= 0.12, \
        f"Median R² gain = {median_gain:.4f}, outside [0.005, 0.12]"

    assert pct_improved >= 0.50, \
        f"Only {pct_improved:.0%} improved (expected ≥50%)"

    assert alpha_shrank >= 0.40, \
        f"Alpha shrank for only {alpha_shrank:.0%} (expected ≥40%)"

    # Factor correlations: RMW/CMA low correlation with MKT
    for factor in ["RMW", "CMA"]:
        r = abs(corr_matrix.loc[factor, "Mkt-RF"])
        assert r < 0.50, f"|corr({factor}, Mkt-RF)| = {r:.2f}, expected < 0.50"

    # Momentum crash visible: 2009 max drawdown
    umd_2009 = umd_full.loc["2009"]
    assert umd_2009.min() < -0.10, \
        f"2009 UMD min = {umd_2009.min():.4f}, expected large crash month"

    # ── RESULTS ────────────────────────────────────
    print(f"══ lecture/s4_ff5_momentum ══════════════════════════")
    print(f"  n_stocks: {n_stocks}")
    print(f"  median_ff3_r2: {median_ff3:.4f}")
    print(f"  median_ff5_r2: {median_ff5:.4f}")
    print(f"  median_r2_gain: {median_gain:.4f}")
    print(f"  pct_ff5_improved: {pct_improved:.2f}")
    print(f"  pct_alpha_shrank: {alpha_shrank:.2f}")
    print(f"  umd_hml_corr: {corr_matrix.loc['UMD', 'HML']:.4f}")

    # ── PLOT ───────────────────────────────────────
    fig_corr.savefig(CACHE_DIR / "s4_factor_correlation.png",
                     dpi=150, bbox_inches="tight")
    print(f"  ── plot: s4_factor_correlation.png ──")
    print(f"     type: correlation heatmap")
    print(f"     shape: {corr_matrix.shape}")
    print(f"     title: {ax_corr.get_title()}")

    fig_mom.savefig(CACHE_DIR / "s4_momentum_cumulative.png",
                    dpi=150, bbox_inches="tight")
    print(f"  ── plot: s4_momentum_cumulative.png ──")
    print(f"     type: cumulative return (log scale)")
    print(f"     n_lines: {len(ax_mom.get_lines())}")
    print(f"     y_range: [{ax_mom.get_ylim()[0]:.2f}, {ax_mom.get_ylim()[1]:.2f}]")
    print(f"     title: {ax_mom.get_title()}")
    print(f"✓ s4_ff5_momentum: ALL PASSED")
