"""
Section 5: Fama-MacBeth Regression — Testing If Factors Are Priced

Acceptance criteria (from README):
- Step 1: Factor loadings (betas) estimated for >= 50 stocks via time-series
- Step 2: Cross-sectional regression run for each month (>= 60 months)
- Lambda time series has the same length as the number of months
- Mean lambda computed and t-statistic calculated for each factor
- At least one factor (MKT or SMB or HML) has |t-stat| > 2.0
- Summary table: Factor name, Mean lambda, Std error, t-stat, p-value
- linearmodels.FamaMacBeth validation shown (results similar to manual)
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats as sp_stats

sys.path.insert(0, str(Path(__file__).parent.parent))
from data_setup import (
    load_sp500_prices, load_ken_french_factors, compute_monthly_returns,
)

prices = load_sp500_prices()
kf = load_ken_french_factors()
monthly_returns = compute_monthly_returns(prices)

# Align returns with Ken French factors (both month-end)
common_months = monthly_returns.index.intersection(kf.index)
monthly_returns = monthly_returns.loc[common_months]
kf_aligned = kf.loc[common_months]

# Use 2010-2023 for the analysis (enough data for stable betas)
start_date = "2010-01-01"
monthly_returns = monthly_returns.loc[start_date:]
kf_aligned = kf_aligned.loc[start_date:]


# ── CELL: step1_time_series_betas ───────────────────────
# Purpose: Step 1 of Fama-MacBeth: for each stock, regress its excess returns
#   on the 3 Fama-French factors (MKT, SMB, HML) using the FULL time series.
#   This gives each stock's factor loadings (betas).
# Takeaway: Each stock gets three betas — its sensitivity to market, size, and
#   value factors. Market betas range ~0.5-1.8. SMB betas are mostly near zero
#   (our S&P 500 universe is large-cap-biased). HML betas range from -0.5
#   (growth stocks) to +0.5 (value stocks). These betas are the "X" variables
#   in Step 2's cross-sectional regressions.

factor_cols = ["Mkt-RF", "SMB", "HML"]
rf = kf_aligned["RF"]

betas = {}
for ticker in monthly_returns.columns:
    excess_ret = monthly_returns[ticker] - rf
    y = excess_ret.dropna()
    X = kf_aligned[factor_cols].loc[y.index]
    valid = y.index.intersection(X.dropna().index)
    if len(valid) < 36:
        continue
    y_v = y.loc[valid].values
    X_v = np.column_stack([np.ones(len(valid)), X.loc[valid].values])
    coeffs, _, _, _ = np.linalg.lstsq(X_v, y_v, rcond=None)
    betas[ticker] = dict(zip(["alpha"] + factor_cols, coeffs))

beta_df = pd.DataFrame(betas).T
print(f"Step 1: Estimated betas for {len(beta_df)} stocks")
print(f"  Months used: {len(kf_aligned)}")
print(f"\nBeta summary:")
print(beta_df[factor_cols].describe().round(3))


# ── CELL: step2_cross_sectional_lambdas ─────────────────
# Purpose: Step 2: For each month, regress that month's cross-section of
#   stock returns on the betas from Step 1. The coefficients (lambdas) are
#   the risk premia — how much return each unit of factor exposure earned.
# Takeaway: We get a time series of lambdas for each factor. If the average
#   lambda is significantly positive, that factor is "priced" — investors
#   earn a premium for bearing that risk. The MKT lambda is typically
#   positive and significant (equity risk premium exists). SMB and HML
#   lambdas are noisier but historically positive.

lambda_results = {col: [] for col in ["const"] + factor_cols}
lambda_dates = []

for month in monthly_returns.index:
    ret_month = monthly_returns.loc[month]
    valid_tickers = ret_month.dropna().index.intersection(beta_df.index)
    if len(valid_tickers) < 30:
        continue

    y = (ret_month.loc[valid_tickers] - rf.loc[month]).values
    X = np.column_stack([
        np.ones(len(valid_tickers)),
        beta_df.loc[valid_tickers, factor_cols].values,
    ])

    mask = np.isfinite(y) & np.all(np.isfinite(X), axis=1)
    if mask.sum() < 30:
        continue

    coeffs, _, _, _ = np.linalg.lstsq(X[mask], y[mask], rcond=None)
    for i, col in enumerate(["const"] + factor_cols):
        lambda_results[col].append(coeffs[i])
    lambda_dates.append(month)

lambda_df = pd.DataFrame(lambda_results, index=lambda_dates)

print(f"\nStep 2: Cross-sectional regressions for {len(lambda_df)} months")


# ── CELL: aggregate_and_test ────────────────────────────
# Purpose: Average the lambdas over time and compute t-statistics. A factor
#   is "priced" if mean(lambda) is significantly different from zero.
# Takeaway: MKT typically has a t-stat > 2.0 (the equity risk premium is
#   real). SMB and HML may or may not be significant depending on the sample
#   period. In recent data (2010-2023), the value premium (HML) has weakened
#   substantially. The summary table is the output you'd show in an academic
#   paper or a research report to a portfolio manager.

summary = []
for col in factor_cols:
    lam = lambda_df[col]
    mean_lam = lam.mean()
    se = lam.std() / np.sqrt(len(lam))
    t_stat = mean_lam / se
    p_val = 2 * (1 - sp_stats.t.cdf(abs(t_stat), df=len(lam) - 1))
    summary.append({
        "Factor": col,
        "Mean Lambda": mean_lam,
        "Std Error": se,
        "t-stat": t_stat,
        "p-value": p_val,
        "Significant": "***" if abs(t_stat) > 2.58 else "**" if abs(t_stat) > 1.96 else "",
    })

summary_df = pd.DataFrame(summary).set_index("Factor")
print("\nFama-MacBeth Risk Premia (manual implementation):")
print(summary_df.round(4))


# ── CELL: linearmodels_validation ───────────────────────
# Purpose: Validate the manual implementation using the linearmodels library's
#   FamaMacBeth class, which automates the two-step procedure and computes
#   standard errors correctly.
# Takeaway: The library results should closely match our manual implementation.
#   Small differences arise from how linearmodels handles standard errors
#   (it uses Driscoll-Kraay by default, which corrects for autocorrelation
#   and cross-sectional dependence). The key check: are the mean lambdas
#   similar? Are the same factors significant?

from linearmodels.panel import FamaMacBeth as FM

# Build panel data for linearmodels
panel_data = []
for month in monthly_returns.index:
    ret = monthly_returns.loc[month]
    valid = ret.dropna().index.intersection(beta_df.index)
    for ticker in valid:
        panel_data.append({
            "date": month,
            "ticker": ticker,
            "excess_ret": ret[ticker] - rf.loc[month],
            **beta_df.loc[ticker, factor_cols].to_dict(),
        })

panel = pd.DataFrame(panel_data)
panel = panel.set_index(["ticker", "date"])
panel = panel.dropna()

fm_model = FM(panel["excess_ret"], panel[factor_cols])
fm_result = fm_model.fit()

print("\nlinearmodels FamaMacBeth validation:")
lib_summary = pd.DataFrame({
    "Mean Lambda (lib)": fm_result.params,
    "t-stat (lib)": fm_result.tstats,
    "p-value (lib)": fm_result.pvalues,
})
print(lib_summary.round(4))


# ── CELL: plot_lambda_timeseries ────────────────────────
# Purpose: Plot the time series of monthly lambdas for each factor.
# Visual: Three panels of monthly bar charts. Mkt-RF (top): mean=0.0055,
#   bars range −0.15 to +0.18 with the deepest negative spike in March 2020
#   and the tallest positive bars in the 2020–2021 recovery. SMB (middle):
#   mean=0.0036, extremely noisy with sharp spikes to ±0.10 and no visual
#   trend. HML (bottom): mean=−0.0076, bars are predominantly negative with
#   the red dashed mean line clearly below zero — value was consistently
#   punished over 2010–2024. The noisiness of all three panels explains why
#   you need many months to detect significant risk premia.

fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
for i, col in enumerate(factor_cols):
    axes[i].bar(lambda_df.index, lambda_df[col], width=20, alpha=0.6)
    axes[i].axhline(lambda_df[col].mean(), color="red", linestyle="--",
                     linewidth=1.5, label=f"Mean = {lambda_df[col].mean():.4f}")
    axes[i].set_ylabel(col)
    axes[i].legend(fontsize=9)
    axes[i].grid(True, alpha=0.3)

axes[0].set_title("Fama-MacBeth Monthly Risk Premia (Lambda)",
                   fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(Path(__file__).parent.parent / ".cache" / "s5_fm_lambdas.png",
            dpi=150, bbox_inches="tight")
plt.close()


if __name__ == "__main__":
    # ── ASSERTIONS ─────────────────────────────────────
    assert len(beta_df) >= 50, (
        f"Expected >= 50 stocks with betas, got {len(beta_df)}")

    assert len(lambda_df) >= 60, (
        f"Expected >= 60 months of lambdas, got {len(lambda_df)}")

    # At least one factor should be significant (|t| > 2)
    max_t = summary_df["t-stat"].abs().max()
    assert max_t > 2.0, (
        f"No factor has |t-stat| > 2.0 (max = {max_t:.2f})")

    # Manual and library results should be in the same ballpark
    for col in factor_cols:
        manual = summary_df.loc[col, "Mean Lambda"]
        lib = fm_result.params[col]
        diff = abs(manual - lib)
        assert diff < 0.01, (
            f"{col}: manual={manual:.4f}, lib={lib:.4f}, diff={diff:.4f}")

    print("\n✓ Section 5 (FM): All acceptance criteria passed")
    print(f"  Stocks: {len(beta_df)}, Months: {len(lambda_df)}")
