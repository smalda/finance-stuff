"""Section 1: The One-Factor World — CAPM Betas and the Security Market Line"""
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
monthly_returns = load_monthly_returns()[DEMO_TICKERS]
ff3 = load_factor_data("3")

# Align dates
common_idx = monthly_returns.index.intersection(ff3.index)
returns = monthly_returns.loc[common_idx]
mkt_rf = ff3.loc[common_idx, "Mkt-RF"]
rf = ff3.loc[common_idx, "RF"]
excess_returns = returns.sub(rf, axis=0)


# ── CELL: estimate_betas ────────────────────────────────────

betas = {}
alphas = {}
r_squared = {}

for ticker in excess_returns.columns:
    y = excess_returns[ticker].dropna()
    x = mkt_rf.loc[y.index]
    X = sm.add_constant(x)
    model = sm.OLS(y, X).fit()
    betas[ticker] = model.params["Mkt-RF"]
    alphas[ticker] = model.params["const"]
    r_squared[ticker] = model.rsquared

betas = pd.Series(betas, name="beta")
alphas = pd.Series(alphas, name="alpha")
r_squared = pd.Series(r_squared, name="r_squared")


# ── CELL: beta_summary ─────────────────────────────────────

beta_df = pd.DataFrame({
    "Beta": betas,
    "Ann. Alpha (%)": alphas * 12 * 100,
    "R²": r_squared,
}).sort_values("Beta")

print(beta_df.round(3).to_string())


# ── CELL: security_market_line ──────────────────────────────

avg_excess = excess_returns.mean() * 12 * 100  # annualized %
sml_X = sm.add_constant(betas)
sml_model = sm.OLS(avg_excess, sml_X).fit()
sml_slope = sml_model.params["beta"]
sml_intercept = sml_model.params["const"]
sml_r2 = sml_model.rsquared

fig, ax = plt.subplots(figsize=(9, 6))
ax.scatter(betas, avg_excess, s=60, alpha=0.7, edgecolors="k", linewidths=0.5)
for ticker in betas.index:
    ax.annotate(ticker, (betas[ticker], avg_excess[ticker]),
                fontsize=7, ha="left", va="bottom")

beta_range = np.linspace(betas.min() - 0.1, betas.max() + 0.1, 50)
ax.plot(beta_range, sml_intercept + sml_slope * beta_range, "r-", lw=2,
        label=f"Empirical SML (slope={sml_slope:.1f}%, R²={sml_r2:.3f})")

# Theoretical SML
avg_mkt_premium = mkt_rf.mean() * 12 * 100
ax.plot(beta_range, avg_mkt_premium * beta_range, "b--", lw=1.5, alpha=0.6,
        label=f"Theoretical SML (slope={avg_mkt_premium:.1f}%)")

ax.set(title="CAPM Security Market Line — Empirical vs. Theoretical",
       xlabel="Estimated Beta", ylabel="Annualized Excess Return (%)")
ax.legend(fontsize=9)
ax.axhline(0, color="gray", lw=0.5, ls="--")
plt.tight_layout()
plt.show()


if __name__ == "__main__":
    # ── ASSERTIONS ─────────────────────────────────
    assert len(betas) >= 15, f"Expected ≥15 stocks, got {len(betas)}"
    assert betas.min() < 0.8, f"Min beta {betas.min():.2f} not < 0.8 (need defensive stocks)"
    assert betas.max() > 1.3, f"Max beta {betas.max():.2f} not > 1.3 (need high-beta stocks)"

    median_r2 = r_squared.median()
    assert 0.10 <= median_r2 <= 0.55, f"Median R² = {median_r2:.4f}, outside [0.10, 0.55]"

    ann_alphas = alphas * 12 * 100
    assert ann_alphas.min() < -5, f"Min ann. alpha {ann_alphas.min():.1f}% not < -5%"
    assert ann_alphas.max() > 5, f"Max ann. alpha {ann_alphas.max():.1f}% not > 5%"

    assert 0.00 <= sml_r2 <= 0.65, f"SML R² = {sml_r2:.4f}, outside [0.00, 0.65]"

    # ── RESULTS ────────────────────────────────────
    print(f"══ lecture/s1_capm ══════════════════════════════════")
    print(f"  n_stocks: {len(betas)}")
    print(f"  beta_range: [{betas.min():.3f}, {betas.max():.3f}]")
    print(f"  median_r_squared: {median_r2:.4f}")
    print(f"  r_squared_range: [{r_squared.min():.4f}, {r_squared.max():.4f}]")
    print(f"  ann_alpha_range: [{ann_alphas.min():.1f}%, {ann_alphas.max():.1f}%]")
    print(f"  sml_slope: {sml_slope:.4f}")
    print(f"  sml_r_squared: {sml_r2:.4f}")
    print(f"  theoretical_slope: {avg_mkt_premium:.4f}")
    print(f"  n_months: {len(common_idx)}")

    # ── PLOT ───────────────────────────────────────
    fig.savefig(CACHE_DIR / "s1_capm_sml.png", dpi=150, bbox_inches="tight")
    print(f"  ── plot: s1_capm_sml.png ──")
    print(f"     type: scatter with regression lines")
    print(f"     n_points: {len(betas)}")
    print(f"     x_range: [{ax.get_xlim()[0]:.2f}, {ax.get_xlim()[1]:.2f}]")
    print(f"     y_range: [{ax.get_ylim()[0]:.2f}, {ax.get_ylim()[1]:.2f}]")
    print(f"     title: {ax.get_title()}")
    print(f"✓ s1_capm: ALL PASSED")
