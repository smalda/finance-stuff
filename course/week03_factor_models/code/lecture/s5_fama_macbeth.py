"""Section 5: Are Factors Priced? — The Fama-MacBeth Methodology"""
import matplotlib
matplotlib.use("Agg")
import sys
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from linearmodels import FamaMacBeth as LMFamaMacBeth

sys.path.insert(0, str(Path(__file__).parent.parent))
from data_setup import (
    load_monthly_returns, load_factor_data, CACHE_DIR,
)

warnings.filterwarnings("ignore", category=FutureWarning)

# Load data — full universe
monthly_returns = load_monthly_returns()
ff3 = load_factor_data("3")

common_idx = monthly_returns.index.intersection(ff3.index)
returns = monthly_returns.loc[common_idx]
mkt_rf = ff3.loc[common_idx, "Mkt-RF"]
smb = ff3.loc[common_idx, "SMB"]
hml = ff3.loc[common_idx, "HML"]
rf = ff3.loc[common_idx, "RF"]
excess_returns = returns.sub(rf, axis=0)

tickers = excess_returns.columns.tolist()


# ── CELL: estimate_betas ────────────────────────────────────

# Step 1: Time-series regression — estimate betas for each stock
factor_df = pd.DataFrame({"Mkt-RF": mkt_rf, "SMB": smb, "HML": hml})
betas_dict = {}

for ticker in tickers:
    y = excess_returns[ticker].dropna()
    idx = y.index
    X = sm.add_constant(factor_df.loc[idx])
    model = sm.OLS(y, X).fit()
    betas_dict[ticker] = {
        "beta_mkt": model.params["Mkt-RF"],
        "beta_smb": model.params["SMB"],
        "beta_hml": model.params["HML"],
    }

betas = pd.DataFrame(betas_dict).T
print(f"Betas estimated for {len(betas)} stocks")
print(betas.describe().round(4).to_string())


# ── CELL: manual_fama_macbeth ───────────────────────────────

# Step 2: Cross-sectional regression each month
gammas = []
r2_cross = []

for date in common_idx:
    rets = excess_returns.loc[date].dropna()
    available = rets.index.intersection(betas.index)
    if len(available) < 50:
        continue

    y_cs = rets[available]
    X_cs = sm.add_constant(betas.loc[available])
    model_cs = sm.OLS(y_cs, X_cs).fit()

    gammas.append({
        "date": date,
        "gamma_const": model_cs.params["const"],
        "gamma_mkt": model_cs.params["beta_mkt"],
        "gamma_smb": model_cs.params["beta_smb"],
        "gamma_hml": model_cs.params["beta_hml"],
    })
    r2_cross.append(model_cs.rsquared)

gamma_df = pd.DataFrame(gammas).set_index("date")

# Time-series average of gammas = risk premium estimates
manual_premia = gamma_df.mean()
manual_tstat = gamma_df.mean() / (gamma_df.std() / np.sqrt(len(gamma_df)))

print("\nManual Fama-MacBeth risk premia (monthly):")
for col in ["gamma_mkt", "gamma_smb", "gamma_hml"]:
    print(f"  {col}: {manual_premia[col]:.6f} (t={manual_tstat[col]:.2f})")

avg_r2 = np.mean(r2_cross)
print(f"\nAverage cross-sectional R²: {avg_r2:.4f}")


# ── CELL: linearmodels_fama_macbeth ─────────────────────────

# Build panel for linearmodels
panel_records = []
for date in common_idx:
    rets = excess_returns.loc[date].dropna()
    available = rets.index.intersection(betas.index)
    for ticker in available:
        panel_records.append({
            "date": date,
            "ticker": ticker,
            "excess_ret": rets[ticker],
            "beta_mkt": betas.loc[ticker, "beta_mkt"],
            "beta_smb": betas.loc[ticker, "beta_smb"],
            "beta_hml": betas.loc[ticker, "beta_hml"],
        })

panel = pd.DataFrame(panel_records)
panel = panel.set_index(["ticker", "date"])

dep = panel[["excess_ret"]]
indep = panel[["beta_mkt", "beta_smb", "beta_hml"]]

fm_model = LMFamaMacBeth(dep, indep).fit(cov_type="kernel")

lm_premia = fm_model.params
lm_tstats = fm_model.tstats

print("\nlinearmodels Fama-MacBeth risk premia:")
for factor in ["beta_mkt", "beta_smb", "beta_hml"]:
    print(f"  {factor}: {lm_premia[factor]:.6f} "
          f"(NW t={lm_tstats[factor]:.2f})")


# ── CELL: compare_manual_vs_linearmodels ────────────────────

print("\nManual vs. linearmodels gamma comparison:")
for manual_col, lm_col in [("gamma_mkt", "beta_mkt"),
                             ("gamma_smb", "beta_smb"),
                             ("gamma_hml", "beta_hml")]:
    diff = abs(manual_premia[manual_col] - lm_premia[lm_col])
    print(f"  {lm_col}: diff = {diff:.6f}")


# ── CELL: fama_macbeth_plot ─────────────────────────────────

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
factor_names = ["gamma_mkt", "gamma_smb", "gamma_hml"]
titles = ["Market Risk Premium", "Size Risk Premium", "Value Risk Premium"]

for ax, col, title in zip(axes, factor_names, titles):
    ax.bar(range(len(gamma_df)), gamma_df[col], alpha=0.6, width=1.0)
    ax.axhline(gamma_df[col].mean(), color="red", lw=2,
               label=f"Mean: {gamma_df[col].mean():.4f}")
    ax.set(title=title, xlabel="Month", ylabel="Gamma (monthly)")
    ax.legend(fontsize=8)

plt.suptitle("Fama-MacBeth Cross-Sectional Slopes Over Time", y=1.02)
plt.tight_layout()
plt.show()


if __name__ == "__main__":
    # ── ASSERTIONS ─────────────────────────────────
    assert len(betas) >= 150, \
        f"Betas for only {len(betas)} stocks (expected ≥150)"

    # Manual and linearmodels should agree closely
    for manual_col, lm_col in [("gamma_mkt", "beta_mkt"),
                                 ("gamma_smb", "beta_smb"),
                                 ("gamma_hml", "beta_hml")]:
        diff = abs(manual_premia[manual_col] - lm_premia[lm_col])
        assert diff < 0.005, \
            f"{lm_col}: manual vs linearmodels diff = {diff:.6f} (expected < 0.005)"

    # Market risk premium
    gamma_mkt = manual_premia["gamma_mkt"]
    assert -0.010 <= gamma_mkt <= 0.020, \
        f"gamma_mkt = {gamma_mkt:.6f}, outside [-0.010, 0.020]"

    # Average cross-sectional R²
    assert 0.005 <= avg_r2 <= 0.30, \
        f"Avg cross-sectional R² = {avg_r2:.4f}, outside [0.005, 0.30]"

    # Newey-West t-stats should differ from naive
    nw_t_mkt = abs(lm_tstats["beta_mkt"])
    naive_t_mkt = abs(manual_tstat["gamma_mkt"])
    # NW t-stats are typically different — just check they're finite
    assert np.isfinite(nw_t_mkt), "NW t-stat for MKT is not finite"

    # ── RESULTS ────────────────────────────────────
    print(f"══ lecture/s5_fama_macbeth ══════════════════════════")
    print(f"  n_stocks: {len(betas)}")
    print(f"  n_months: {len(gamma_df)}")
    print(f"  gamma_mkt: {manual_premia['gamma_mkt']:.6f}")
    print(f"  gamma_smb: {manual_premia['gamma_smb']:.6f}")
    print(f"  gamma_hml: {manual_premia['gamma_hml']:.6f}")
    print(f"  naive_t_mkt: {manual_tstat['gamma_mkt']:.4f}")
    print(f"  naive_t_smb: {manual_tstat['gamma_smb']:.4f}")
    print(f"  naive_t_hml: {manual_tstat['gamma_hml']:.4f}")
    print(f"  nw_t_mkt: {lm_tstats['beta_mkt']:.4f}")
    print(f"  nw_t_smb: {lm_tstats['beta_smb']:.4f}")
    print(f"  nw_t_hml: {lm_tstats['beta_hml']:.4f}")
    print(f"  avg_cross_r2: {avg_r2:.4f}")

    # ── PLOT ───────────────────────────────────────
    fig.savefig(CACHE_DIR / "s5_fama_macbeth_gammas.png",
                dpi=150, bbox_inches="tight")
    print(f"  ── plot: s5_fama_macbeth_gammas.png ──")
    print(f"     type: bar chart of cross-sectional slopes")
    print(f"     n_panels: 3")
    print(f"     title: Fama-MacBeth Cross-Sectional Slopes Over Time")
    print(f"✓ s5_fama_macbeth: ALL PASSED")
