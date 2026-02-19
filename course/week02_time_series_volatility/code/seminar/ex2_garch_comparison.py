"""
Exercise 2: GARCH Model Comparison — Which Flavor of Volatility Wins?

Acceptance criteria (from README):
- All 15 model fits converge (5 tickers x 3 models)
- Best model by BIC identified for each ticker
- At least 3 of 5 tickers show an asymmetric model as best by BIC
- Persistence (alpha + beta) > 0.90 for all equity tickers
- TLT squared-return ACF at lag 1 is lower than SPY squared-return ACF at lag 1
- Comparison table complete with all parameters, log-likelihoods, and information criteria
- 5-panel conditional volatility figure produced
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from arch import arch_model
from statsmodels.tsa.stattools import acf

sys.path.insert(0, str(Path(__file__).parent.parent))
from data_setup import load_prices, GARCH_TICKERS

prices = load_prices(GARCH_TICKERS)


# ── CELL: fit_all_models ─────────────────────────────────
# Purpose: Fit GARCH(1,1), EGARCH(1,1), and GJR-GARCH(1,1) to 5 tickers
#   with diverse volatility profiles: SPY (index), AAPL (large-cap tech),
#   JPM (financials), TSLA (high-vol growth), TLT (bonds). Compare
#   parameters, log-likelihood, AIC, and BIC across all 15 fits.
# Takeaway: EGARCH wins for SPY, AAPL, JPM (3 of 5 tickers). TSLA's
#   asymmetry is negligible (γ ≈ 0) — its vol is so extreme that the
#   leverage effect is swamped. TLT shows no leverage (bonds are different).
#   Persistence: TSLA = 0.991, SPY = 0.950, TLT = 0.981, all > 0.90.

specs = {
    "GARCH": {"vol": "Garch", "p": 1, "q": 1},
    "EGARCH": {"vol": "EGARCH", "p": 1, "o": 1, "q": 1},
    "GJR-GARCH": {"vol": "Garch", "p": 1, "o": 1, "q": 1},
}

all_results = {}
comp_rows = []

for ticker in GARCH_TICKERS:
    returns_pct = prices[ticker].pct_change().dropna() * 100
    all_results[ticker] = {}

    for model_name, kwargs in specs.items():
        m = arch_model(returns_pct, mean="Constant", dist="Normal", **kwargs)
        r = m.fit(disp="off")
        all_results[ticker][model_name] = r

        params = r.params.to_dict()
        alpha = params.get("alpha[1]", np.nan)
        beta = params.get("beta[1]", np.nan)
        gamma = params.get("gamma[1]", np.nan)

        # Persistence calculation depends on model type
        if model_name == "EGARCH":
            persistence = beta if not np.isnan(beta) else np.nan
        elif model_name == "GJR-GARCH":
            persistence = alpha + 0.5 * gamma + beta if not np.isnan(gamma) else alpha + beta
        else:
            persistence = alpha + beta

        comp_rows.append({
            "Ticker": ticker,
            "Model": model_name,
            "omega": params.get("omega", np.nan),
            "alpha": alpha,
            "beta": beta,
            "gamma": gamma,
            "Persistence": persistence,
            "Log-Lik": r.loglikelihood,
            "AIC": r.aic,
            "BIC": r.bic,
        })

comp_df = pd.DataFrame(comp_rows)
print("=== Full Comparison Table ===")
print(comp_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))


# ── CELL: identify_best_models ───────────────────────────
# Purpose: For each ticker, identify the best model by BIC and report
#   whether asymmetric models dominate across different asset types.
# Takeaway: EGARCH wins for SPY (BIC 9505), AAPL (14326), JPM (13771).
#   GARCH wins for TSLA (BIC 19408 — EGARCH γ ≈ 0, no leverage) and
#   TLT (BIC 9839 — bonds lack leverage effect). The BIC gap is 120-160
#   points for SPY, a decisive difference.

best_rows = []
for ticker in GARCH_TICKERS:
    sub = comp_df[comp_df["Ticker"] == ticker]
    best_idx = sub["BIC"].idxmin()
    best = sub.loc[best_idx]
    best_rows.append({
        "Ticker": ticker,
        "Best Model": best["Model"],
        "BIC": best["BIC"],
        "Persistence": best["Persistence"],
    })

best_df = pd.DataFrame(best_rows)
print("\n=== Best Model by BIC ===")
print(best_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

n_asymmetric = sum(1 for r in best_rows if r["Best Model"] != "GARCH")
print(f"\nAsymmetric model wins for {n_asymmetric} of {len(GARCH_TICKERS)} tickers")


# ── CELL: squared_return_acf_comparison ──────────────────
# Purpose: Compare the strength of volatility clustering between SPY
#   (equity index) and TLT (bonds) using the ACF of squared returns.
# Takeaway: SPY squared-return ACF at lag 1 = 0.447, TLT = 0.397. The
#   difference (0.05) confirms equity clustering is stronger, though TLT
#   still shows significant clustering (both well above 95% bands).

spy_ret = prices["SPY"].pct_change().dropna()
tlt_ret = prices["TLT"].pct_change().dropna()
acf_spy_sq = acf(spy_ret**2, nlags=5, fft=True)
acf_tlt_sq = acf(tlt_ret**2, nlags=5, fft=True)
print(f"\nSquared-return ACF at lag 1:")
print(f"  SPY: {acf_spy_sq[1]:.4f}")
print(f"  TLT: {acf_tlt_sq[1]:.4f}")


# ── CELL: plot_conditional_volatility ────────────────────
# Purpose: Produce a 5-panel figure showing the conditional volatility
#   from the best-fit model for each ticker, overlaid on absolute returns.
# Visual: Five rows, one per ticker. TSLA daily vol peaks at ~25% (highest),
#   TLT at ~4% (lowest), SPY at ~10% during COVID. Each panel shows grey
#   |return| bars and a blue conditional vol line. COVID 2020 spike dominates
#   all equity panels. The scale differences across assets are stark.

fig, axes = plt.subplots(5, 1, figsize=(14, 16), sharex=True)

for ax, row in zip(axes, best_rows):
    ticker = row["Ticker"]
    model_name = row["Best Model"]
    r = all_results[ticker][model_name]
    returns_pct = prices[ticker].pct_change().dropna() * 100

    cond_vol = r.conditional_volatility / 100
    abs_ret = returns_pct.abs() / 100

    ax.bar(abs_ret.index, abs_ret.values, width=1, color="lightgray",
           alpha=0.6, label="|Return|")
    ax.plot(cond_vol.index, cond_vol.values, linewidth=0.6, color="steelblue",
            label=f"{model_name} σ")
    ax.set_ylabel("Vol (daily)")
    ax.set_title(f"{ticker} — {model_name}")
    ax.legend(fontsize=8, loc="upper right")

axes[-1].set_xlabel("Date")
plt.tight_layout()
plt.savefig("ex2_garch_comparison.png", dpi=120, bbox_inches="tight")
plt.close()


if __name__ == "__main__":
    # ── ASSERTIONS ─────────────────────────────────────
    # All 15 fits converge
    for ticker in GARCH_TICKERS:
        for model_name in specs:
            r = all_results[ticker][model_name]
            assert r.convergence_flag == 0, (
                f"{ticker} {model_name} did not converge"
            )

    # At least 3 of 5 tickers have asymmetric best model
    assert n_asymmetric >= 3, (
        f"Only {n_asymmetric} tickers have asymmetric best model, expected >= 3"
    )

    # Persistence > 0.90 for equity tickers
    equity_tickers = ["SPY", "AAPL", "JPM", "TSLA"]
    for row in best_rows:
        if row["Ticker"] in equity_tickers:
            assert row["Persistence"] > 0.90, (
                f"{row['Ticker']} persistence = {row['Persistence']:.4f}, expected > 0.90"
            )

    # TLT squared ACF at lag 1 < SPY
    assert acf_tlt_sq[1] < acf_spy_sq[1], (
        f"TLT sq ACF1 ({acf_tlt_sq[1]:.4f}) should be < SPY ({acf_spy_sq[1]:.4f})"
    )

    # Comparison table has correct dimensions
    assert len(comp_df) == len(GARCH_TICKERS) * len(specs), (
        f"Expected {len(GARCH_TICKERS) * len(specs)} rows, got {len(comp_df)}"
    )

    print("✓ Exercise 2: All acceptance criteria passed")
