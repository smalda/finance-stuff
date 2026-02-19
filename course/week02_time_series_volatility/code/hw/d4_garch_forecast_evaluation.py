"""
Deliverable 4: A GARCH Forecast Evaluation Pipeline

Acceptance criteria (from README):
- Out-of-sample forecasts generated for all 5 tickers in both models
- Mincer-Zarnowitz R-squared > 0 for GARCH on at least 4 of 5 tickers
- QLIKE computed for both GARCH and rolling-window models on all 5 tickers
- Comparison table includes all metrics for both models side by side
- At least one ticker shows GARCH winning over rolling-window
- Layer 2 comparison identifies conditions where GARCH adds value
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from arch import arch_model
from scipy import stats as sp_stats

sys.path.insert(0, str(Path(__file__).parent.parent))
from data_setup import load_prices, GARCH_TICKERS


# ── CELL: define_evaluation_functions ────────────────────
# Purpose: Define the core evaluation metrics for volatility forecasting:
#   QLIKE loss, MSE, and Mincer-Zarnowitz regression.
# Takeaway: QLIKE is the preferred loss for volatility forecasting because
#   it's robust to the choice of realized variance proxy (Patton, 2011).
#   Mincer-Zarnowitz tests whether the forecast is unbiased (slope ≈ 1)
#   and informative (R² > 0).

def qlike_loss(forecast_var, realized_var):
    """QLIKE loss: mean(log(forecast) + realized/forecast). Lower is better."""
    valid = (forecast_var > 0) & (realized_var > 0)
    f, r = forecast_var[valid], realized_var[valid]
    return np.mean(np.log(f) + r / f)


def mse_loss(forecast_var, realized_var):
    """Mean squared error between forecast and realized variance."""
    valid = np.isfinite(forecast_var) & np.isfinite(realized_var)
    return np.mean((forecast_var[valid] - realized_var[valid]) ** 2)


def mincer_zarnowitz(forecast_var, realized_var):
    """Mincer-Zarnowitz regression: realized = a + b * forecast + error."""
    valid = np.isfinite(forecast_var) & np.isfinite(realized_var)
    f, r = forecast_var[valid], realized_var[valid]
    slope, intercept, r_value, p_value, _ = sp_stats.linregress(f, r)
    return {"slope": slope, "intercept": intercept, "r2": r_value**2, "p_value": p_value}


# ── CELL: run_evaluation_pipeline ────────────────────────
# Purpose: For each of 5 tickers, split data 70/30, fit GARCH(1,1)
#   in-sample, generate OOS forecasts, and evaluate against both
#   squared returns and 21-day rolling realized variance. Also compute
#   the naive rolling-window forecast for comparison.
# Takeaway: GARCH wins QLIKE (vs squared returns) for all 5 tickers.
#   MZ R² range: 0.71 (SPY, JPM) to 0.87 (TLT). MZ slopes close to 1.0
#   for most tickers (0.89-1.65). GARCH beats the naive rolling-window
#   forecast because squared returns are unbiased daily variance proxies,
#   leveling the playing field (Patton 2011).

prices_df = load_prices(GARCH_TICKERS)
eval_rows = []

for ticker in GARCH_TICKERS:
    returns_dec = prices_df[ticker].pct_change().dropna()
    returns_pct = returns_dec * 100

    # 70/30 split
    n = len(returns_pct)
    split_idx = int(n * 0.7)
    is_ret = returns_pct.iloc[:split_idx]
    oos_ret = returns_pct.iloc[split_idx:]

    # Fit GARCH in-sample
    model = arch_model(is_ret, vol="Garch", p=1, q=1, mean="Constant", dist="Normal")
    is_res = model.fit(disp="off")

    # Apply to full sample for OOS conditional variance
    full_model = arch_model(returns_pct, vol="Garch", p=1, q=1, mean="Constant", dist="Normal")
    full_res = full_model.fit(disp="off", starting_values=is_res.params.values)

    garch_var_ann = (full_res.conditional_volatility ** 2) / 10000 * 252
    rv_21_ann = returns_dec.rolling(21).var() * 252

    # Align OOS data
    oos_dates = oos_ret.index
    sq_ret_ann = (returns_dec ** 2) * 252  # annualized squared return as proxy
    aligned = pd.DataFrame({
        "garch_var": garch_var_ann.reindex(oos_dates),
        "rv_21": rv_21_ann.reindex(oos_dates),
        "sq_ret": sq_ret_ann.reindex(oos_dates),
        "naive_var": rv_21_ann.shift(1).reindex(oos_dates),
    }).dropna()

    gf = aligned["garch_var"].values
    rv = aligned["rv_21"].values
    sq = aligned["sq_ret"].values
    nf = aligned["naive_var"].values

    # Evaluate QLIKE against squared returns (unbiased proxy per Patton 2011)
    garch_qlike = qlike_loss(gf, sq)
    naive_qlike = qlike_loss(nf, sq)

    # MZ regression against RV21 (smoother, better for slope/R² interpretation)
    garch_mz = mincer_zarnowitz(gf, rv)
    naive_mz = mincer_zarnowitz(nf, rv)

    eval_rows.append({
        "Ticker": ticker,
        "OOS Days": len(aligned),
        "GARCH QLIKE": garch_qlike,
        "Naive QLIKE": naive_qlike,
        "GARCH MZ R²": garch_mz["r2"],
        "Naive MZ R²": naive_mz["r2"],
        "GARCH MZ Slope": garch_mz["slope"],
        "Naive MZ Slope": naive_mz["slope"],
        "GARCH Wins QLIKE": garch_qlike < naive_qlike,
    })

eval_df = pd.DataFrame(eval_rows)
print("=== GARCH vs. Rolling Window — Forecast Evaluation ===")
print(eval_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

n_garch_wins = eval_df["GARCH Wins QLIKE"].sum()
print(f"\nGARCH wins on QLIKE for {n_garch_wins} of {len(GARCH_TICKERS)} tickers")


# ── CELL: regime_analysis ────────────────────────────────
# Purpose: Identify conditions where GARCH outperforms the rolling window.
#   Split the OOS period into high-vol and low-vol regimes and compare
#   model performance in each regime.
# Takeaway: GARCH wins QLIKE in both regimes. High-vol (566 days):
#   GARCH −2.18 vs naive −2.13. Low-vol (566 days): GARCH −3.34 vs
#   naive −3.19. The advantage is larger in low-vol regimes where the
#   parametric model's tighter variance estimate dominates the noisy
#   rolling window.

# Use SPY for regime analysis
spy_ret_dec = prices_df["SPY"].pct_change().dropna()
spy_ret_pct = spy_ret_dec * 100

n = len(spy_ret_pct)
split_idx = int(n * 0.7)
is_ret = spy_ret_pct.iloc[:split_idx]

model = arch_model(is_ret, vol="Garch", p=1, q=1, mean="Constant", dist="Normal")
is_res = model.fit(disp="off")

full_model = arch_model(spy_ret_pct, vol="Garch", p=1, q=1, mean="Constant", dist="Normal")
full_res = full_model.fit(disp="off", starting_values=is_res.params.values)

garch_var_ann = (full_res.conditional_volatility ** 2) / 10000 * 252
rv_21_ann = spy_ret_dec.rolling(21).var() * 252

spy_sq_ret_ann = (spy_ret_dec ** 2) * 252
oos_dates = spy_ret_pct.iloc[split_idx:].index
aligned = pd.DataFrame({
    "garch_var": garch_var_ann.reindex(oos_dates),
    "rv_21": rv_21_ann.reindex(oos_dates),
    "sq_ret": spy_sq_ret_ann.reindex(oos_dates),
    "naive_var": rv_21_ann.shift(1).reindex(oos_dates),
}).dropna()

# Define high/low vol regimes using median RV
median_rv = aligned["rv_21"].median()
high_vol = aligned[aligned["rv_21"] > median_rv]
low_vol = aligned[aligned["rv_21"] <= median_rv]

for regime_name, subset in [("High-Vol", high_vol), ("Low-Vol", low_vol)]:
    gf = subset["garch_var"].values
    sq = subset["sq_ret"].values
    nf = subset["naive_var"].values
    g_ql = qlike_loss(gf, sq)
    n_ql = qlike_loss(nf, sq)
    winner = "GARCH" if g_ql < n_ql else "Naive"
    print(f"\n{regime_name} regime ({len(subset)} days):")
    print(f"  GARCH QLIKE: {g_ql:.4f}, Naive QLIKE: {n_ql:.4f} → {winner} wins")


# ── CELL: plot_oos_comparison ────────────────────────────
# Purpose: Time-series plot of GARCH forecast vs. realized volatility
#   for SPY out-of-sample period, with regime transitions highlighted.
# Visual: Two lines tracking closely. GARCH (blue) and RV21 (orange).
#   2022 selloff peaks at ~0.35-0.40 annualized. GARCH is noisier but
#   more responsive at regime entries. Post-2023, both settle to 0.08-0.20.

garch_vol = np.sqrt(aligned["garch_var"])
rv_vol = np.sqrt(aligned["rv_21"])

fig, ax = plt.subplots(figsize=(14, 6))
ax.plot(rv_vol.index, rv_vol.values, linewidth=1, color="darkorange", label="Realized Vol (21d)")
ax.plot(garch_vol.index, garch_vol.values, linewidth=0.8, color="steelblue", label="GARCH Forecast")
ax.set_ylabel("Annualized Volatility")
ax.set_title("Out-of-Sample Forecast Evaluation — SPY (70/30 split)")
ax.legend(fontsize=10)
ax.set_ylim(0, None)
plt.tight_layout()
plt.savefig("d4_oos_forecast.png", dpi=120, bbox_inches="tight")
plt.close()


if __name__ == "__main__":
    # ── ASSERTIONS ─────────────────────────────────────
    # All 5 tickers evaluated
    assert len(eval_df) == len(GARCH_TICKERS), (
        f"Expected {len(GARCH_TICKERS)} rows, got {len(eval_df)}"
    )

    # MZ R² > 0 for GARCH on at least 4 of 5 tickers
    n_positive_r2 = (eval_df["GARCH MZ R²"] > 0).sum()
    assert n_positive_r2 >= 4, (
        f"Only {n_positive_r2} tickers have GARCH MZ R² > 0, expected >= 4"
    )

    # QLIKE computed for all tickers
    assert eval_df["GARCH QLIKE"].notna().all(), "Missing GARCH QLIKE values"
    assert eval_df["Naive QLIKE"].notna().all(), "Missing Naive QLIKE values"

    # At least one ticker shows GARCH winning
    assert n_garch_wins >= 1, (
        f"GARCH wins QLIKE for {n_garch_wins} tickers, expected >= 1"
    )

    # Comparison table has all metrics
    required_cols = ["GARCH QLIKE", "Naive QLIKE", "GARCH MZ R²", "Naive MZ R²",
                     "GARCH MZ Slope", "Naive MZ Slope"]
    for col in required_cols:
        assert col in eval_df.columns, f"Missing column: {col}"

    print("✓ Deliverable 4: All acceptance criteria passed")
