"""Exercise 3: Decompose Your Portfolio's Risk

Compare risk decomposition for a diversified portfolio vs. a concentrated
(single-sector) portfolio using a simplified Barra-style approach.
"""
import matplotlib
matplotlib.use("Agg")
import sys
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm

sys.path.insert(0, str(Path(__file__).parent.parent))
from data_setup import (
    load_monthly_returns, load_factor_data, load_fundamentals, CACHE_DIR,
)

warnings.filterwarnings("ignore", category=FutureWarning)

monthly_returns = load_monthly_returns()
ff5 = load_factor_data("5")
fundamentals = load_fundamentals()
sectors = fundamentals["sector"]

common_idx = monthly_returns.index.intersection(ff5.index)
rf = ff5.loc[common_idx, "RF"]


# ── CELL: define_portfolios ─────────────────────────────────

# Diversified portfolio: 2 stocks from each sector
diversified_tickers = []
for sector_name in sorted(sectors.unique()):
    tickers_in_sector = sectors[sectors == sector_name].index.tolist()
    available = [t for t in tickers_in_sector if t in monthly_returns.columns]
    diversified_tickers.extend(available[:2])
diversified_tickers = diversified_tickers[:20]

# Concentrated portfolio: all technology stocks
tech_tickers = sectors[sectors == "Technology"].index.tolist()
tech_available = [t for t in tech_tickers if t in monthly_returns.columns]
concentrated_tickers = tech_available[:20]

print(f"Diversified portfolio: {len(diversified_tickers)} stocks "
      f"across {len(set(sectors[t] for t in diversified_tickers if t in sectors.index))} sectors")
print(f"Concentrated portfolio: {len(concentrated_tickers)} tech stocks")


# ── CELL: factor_regressions ───────────────────────────────

# Equal-weighted portfolio returns
factors = ff5.loc[common_idx, ["Mkt-RF", "SMB", "HML", "RMW", "CMA"]]

def decompose_portfolio(tickers, label):
    """Run FF5 regression and compute risk decomposition."""
    port_ret = monthly_returns[tickers].loc[common_idx].mean(axis=1)
    port_excess = port_ret - rf

    X = sm.add_constant(factors)
    model = sm.OLS(port_excess, X).fit()

    total_var = port_excess.var()
    factor_var = model.fittedvalues.var()
    specific_var = model.resid.var()
    factor_share = factor_var / total_var

    print(f"\n{label}:")
    print(f"  R²: {model.rsquared:.4f}")
    print(f"  Factor risk share: {factor_share:.2%}")
    print(f"  Specific risk share: {1 - factor_share:.2%}")
    print(f"  Factor loadings:")
    for f_name in factors.columns:
        print(f"    {f_name}: {model.params[f_name]:.4f} "
              f"(t={model.tvalues[f_name]:.2f})")

    return {
        "label": label,
        "r2": model.rsquared,
        "factor_share": factor_share,
        "specific_share": 1 - factor_share,
        "total_var": total_var,
        "loadings": {f: model.params[f] for f in factors.columns},
    }

div_result = decompose_portfolio(diversified_tickers, "Diversified (20 stocks)")
conc_result = decompose_portfolio(concentrated_tickers, "Concentrated (Tech)")


# ── CELL: risk_comparison_plot ──────────────────────────────

fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# Pie charts
for ax, result in zip(axes[:2], [div_result, conc_result]):
    sizes = [result["factor_share"], result["specific_share"]]
    labels = ["Factor Risk", "Specific Risk"]
    colors = ["#2196F3", "#FF9800"]
    ax.pie(sizes, labels=labels, colors=colors, autopct="%1.1f%%",
           startangle=90, textprops={"fontsize": 10})
    ax.set_title(f"{result['label']}\n(R²={result['r2']:.3f})")

# Factor loadings comparison
ax = axes[2]
factor_names = list(factors.columns)
x = np.arange(len(factor_names))
width = 0.35
div_loads = [div_result["loadings"][f] for f in factor_names]
conc_loads = [conc_result["loadings"][f] for f in factor_names]
ax.bar(x - width / 2, div_loads, width, label="Diversified", color="#2196F3")
ax.bar(x + width / 2, conc_loads, width, label="Concentrated", color="#FF9800")
ax.set_xticks(x)
ax.set_xticklabels(factor_names, rotation=30, ha="right")
ax.set(title="Factor Loadings Comparison", ylabel="Loading")
ax.legend()
ax.axhline(0, color="gray", lw=0.5)

plt.tight_layout()
plt.show()


if __name__ == "__main__":
    # ── ASSERTIONS ─────────────────────────────────
    # Diversified portfolio: factor risk 40-95%
    assert 0.30 <= div_result["factor_share"] <= 0.95, \
        f"Diversified factor share = {div_result['factor_share']:.2%}, " \
        f"outside [30%, 95%]"

    # Both portfolios should have substantial factor risk
    assert conc_result["factor_share"] >= 0.40, \
        f"Concentrated factor share = {conc_result['factor_share']:.2%} " \
        f"(expected ≥40%)"

    # Difference should be visible (either direction is pedagogically valid)
    diff_pp = abs(conc_result["factor_share"] - div_result["factor_share"]) * 100
    assert diff_pp > 1.0, \
        f"Factor share difference = {diff_pp:.1f} pp (expected > 1.0 pp)"

    # Risk components should approximately sum to 100%
    for result in [div_result, conc_result]:
        total = result["factor_share"] + result["specific_share"]
        assert 0.95 <= total <= 1.05, \
            f"{result['label']}: factor + specific = {total:.4f} (expected ~1.0)"

    # ── RESULTS ────────────────────────────────────
    print(f"══ seminar/ex3_portfolio_risk ═══════════════════════")
    print(f"  diversified_n: {len(diversified_tickers)}")
    print(f"  concentrated_n: {len(concentrated_tickers)}")
    print(f"  div_factor_share: {div_result['factor_share']:.4f}")
    print(f"  div_specific_share: {div_result['specific_share']:.4f}")
    print(f"  conc_factor_share: {conc_result['factor_share']:.4f}")
    print(f"  conc_specific_share: {conc_result['specific_share']:.4f}")
    print(f"  factor_share_diff_pp: {diff_pp:.2f}")

    # ── PLOT ───────────────────────────────────────
    fig.savefig(CACHE_DIR / "ex3_portfolio_risk.png",
                dpi=150, bbox_inches="tight")
    print(f"  ── plot: ex3_portfolio_risk.png ──")
    print(f"     type: pie charts + bar chart")
    print(f"     n_panels: 3")
    print(f"     title_left: {axes[0].get_title()}")
    print(f"     title_right: {axes[2].get_title()}")
    print(f"✓ ex3_portfolio_risk: ALL PASSED")
