"""Section 2: When One Factor Fails — The Fama-French Three-Factor Model"""
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

# Load data — use 10-15 of the demo tickers for regressions
demo_subset = DEMO_TICKERS[:15]
monthly_returns = load_monthly_returns()[demo_subset]
ff3 = load_factor_data("3")

# Align on 2014-2024
common_idx = monthly_returns.index.intersection(ff3.index)
returns = monthly_returns.loc[common_idx]
mkt_rf = ff3.loc[common_idx, "Mkt-RF"]
smb = ff3.loc[common_idx, "SMB"]
hml = ff3.loc[common_idx, "HML"]
rf = ff3.loc[common_idx, "RF"]
excess_returns = returns.sub(rf, axis=0)


# ── CELL: capm_vs_ff3_regressions ───────────────────────────

results_list = []

for ticker in excess_returns.columns:
    y = excess_returns[ticker].dropna()
    idx = y.index

    # CAPM
    X_capm = sm.add_constant(mkt_rf.loc[idx])
    capm = sm.OLS(y, X_capm).fit()

    # FF3
    X_ff3 = sm.add_constant(
        pd.DataFrame({"Mkt-RF": mkt_rf, "SMB": smb, "HML": hml}).loc[idx]
    )
    ff3_model = sm.OLS(y, X_ff3).fit()

    results_list.append({
        "ticker": ticker,
        "capm_r2": capm.rsquared,
        "ff3_r2": ff3_model.rsquared,
        "r2_improvement": ff3_model.rsquared - capm.rsquared,
        "capm_alpha": capm.params["const"] * 12 * 100,
        "ff3_alpha": ff3_model.params["const"] * 12 * 100,
        "smb_loading": ff3_model.params["SMB"],
        "hml_loading": ff3_model.params["HML"],
    })

comparison = pd.DataFrame(results_list).set_index("ticker")
print(comparison[["capm_r2", "ff3_r2", "r2_improvement"]].round(3).to_string())


# ── CELL: alpha_shrinkage ───────────────────────────────────

alpha_df = comparison[["capm_alpha", "ff3_alpha"]].copy()
alpha_df["shrinkage"] = alpha_df["capm_alpha"].abs() - alpha_df["ff3_alpha"].abs()
alpha_shrank = (alpha_df["shrinkage"] > 0).mean()
print(f"\nAlpha shrank (|FF3 alpha| < |CAPM alpha|) for "
      f"{alpha_shrank:.0%} of stocks")


# ── CELL: cumulative_factor_returns ─────────────────────────

ff3_full = load_factor_data("3")
# Use full history for cumulative plot
factors_to_plot = ff3_full[["Mkt-RF", "SMB", "HML"]]
cumulative = (1 + factors_to_plot).cumprod()

fig, ax = plt.subplots(figsize=(10, 6))
for col in cumulative.columns:
    ax.plot(cumulative.index, cumulative[col], label=col, lw=1.5)
ax.set(title="Cumulative Factor Returns (1926-2025)",
       xlabel="Date", ylabel="Cumulative Return ($1 invested)")
ax.set_yscale("log")
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()


# ── CELL: factor_loadings_heatmap ───────────────────────────

loadings = comparison[["smb_loading", "hml_loading"]].copy()
loadings.columns = ["SMB", "HML"]

fig2, ax2 = plt.subplots(figsize=(6, 8))
im = ax2.imshow(loadings.values, cmap="RdYlBu_r", aspect="auto",
                vmin=-1.5, vmax=1.5)
ax2.set_xticks(range(len(loadings.columns)))
ax2.set_xticklabels(loadings.columns)
ax2.set_yticks(range(len(loadings.index)))
ax2.set_yticklabels(loadings.index, fontsize=8)
for i in range(len(loadings.index)):
    for j in range(len(loadings.columns)):
        ax2.text(j, i, f"{loadings.values[i, j]:.2f}",
                 ha="center", va="center", fontsize=8)
ax2.set_title("FF3 Factor Loadings")
fig2.colorbar(im, ax=ax2, shrink=0.6)
plt.tight_layout()
plt.show()


if __name__ == "__main__":
    # ── ASSERTIONS ─────────────────────────────────
    n_stocks = len(comparison)
    assert n_stocks >= 10, f"Expected ≥10 stocks, got {n_stocks}"

    median_ff3_r2 = comparison["ff3_r2"].median()
    assert 0.15 <= median_ff3_r2 <= 0.75, \
        f"FF3 median R² = {median_ff3_r2:.4f}, outside [0.15, 0.75]"

    median_improvement = comparison["r2_improvement"].median()
    assert 0.01 <= median_improvement <= 0.20, \
        f"Median R² improvement = {median_improvement:.4f}, outside [0.01, 0.20]"

    pct_improved = (comparison["r2_improvement"] > 0).mean()
    assert pct_improved >= 0.60, \
        f"Only {pct_improved:.0%} stocks improved R² (expected ≥60%)"

    assert alpha_shrank >= 0.50, \
        f"Alpha shrank for only {alpha_shrank:.0%} (expected ≥50%)"

    # Cumulative HML — recent period may be negative
    hml_cum_recent = (1 + ff3_full.loc["2014":, "HML"]).cumprod().iloc[-1]
    # Just check it's finite — sign is uncertain
    assert np.isfinite(hml_cum_recent), "HML cumulative not finite"

    # ── RESULTS ────────────────────────────────────
    print(f"══ lecture/s2_ff3_model ═════════════════════════════")
    print(f"  n_stocks: {n_stocks}")
    print(f"  median_capm_r2: {comparison['capm_r2'].median():.4f}")
    print(f"  median_ff3_r2: {median_ff3_r2:.4f}")
    print(f"  median_r2_improvement: {median_improvement:.4f}")
    print(f"  pct_r2_improved: {pct_improved:.2f}")
    print(f"  pct_alpha_shrank: {alpha_shrank:.2f}")
    print(f"  hml_cumulative_2014_2024: {hml_cum_recent:.4f}")

    # ── PLOT ───────────────────────────────────────
    fig.savefig(CACHE_DIR / "s2_cumulative_factors.png", dpi=150, bbox_inches="tight")
    print(f"  ── plot: s2_cumulative_factors.png ──")
    print(f"     type: multi-line cumulative returns (log scale)")
    print(f"     n_lines: {len(ax.get_lines())}")
    print(f"     y_range: [{ax.get_ylim()[0]:.2f}, {ax.get_ylim()[1]:.2f}]")
    print(f"     title: {ax.get_title()}")

    fig2.savefig(CACHE_DIR / "s2_factor_loadings.png", dpi=150, bbox_inches="tight")
    print(f"  ── plot: s2_factor_loadings.png ──")
    print(f"     type: heatmap of factor loadings")
    print(f"     shape: {loadings.shape}")
    print(f"     title: {ax2.get_title()}")
    print(f"✓ s2_ff3_model: ALL PASSED")
