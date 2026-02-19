"""
Section 6: Asymmetric GARCH — The Leverage Effect

Acceptance criteria (from README):
- All three models converge successfully
- EGARCH gamma parameter is negative (negative shocks increase volatility more)
- GJR-GARCH gamma parameter is positive (indicator for negative shocks adds volatility)
- BIC: at least one asymmetric model has lower (better) BIC than GARCH(1,1)
- News impact curve plot shows visible asymmetry for EGARCH and GJR, symmetry for GARCH
- Comparison table includes model name, omega, alpha, beta, gamma, log-likelihood, BIC
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from arch import arch_model

sys.path.insert(0, str(Path(__file__).parent.parent))
from data_setup import load_prices

prices = load_prices(["SPY"])
returns = prices["SPY"].pct_change().dropna() * 100


# ── CELL: fit_three_models ──────────────────────────────────
# Purpose: Fit GARCH(1,1), EGARCH(1,1), and GJR-GARCH(1,1) to the same
#   SPY return data and compare them head-to-head. BIC penalizes
#   complexity, so if an asymmetric model wins, the leverage effect
#   is real and material — not just a statistical artifact.
# Takeaway: EGARCH BIC = 9,505, GJR BIC = 9,544, GARCH BIC = 9,665 —
#   both asymmetric models win decisively. EGARCH γ = -0.17, confirming
#   that negative shocks increase log-volatility more than positive shocks.
#   GJR γ = 0.24, adding a substantial "bad news premium" (α_eff doubles
#   for negative shocks: 0.026 + 0.239 = 0.265). The leverage effect is
#   not a minor detail — it improves BIC by 120-160 points.

specs = {
    "GARCH(1,1)": {"vol": "Garch", "p": 1, "q": 1},
    "EGARCH(1,1)": {"vol": "EGARCH", "p": 1, "o": 1, "q": 1},
    "GJR-GARCH(1,1)": {"vol": "Garch", "p": 1, "o": 1, "q": 1},
}

results = {}
rows = []

for name, kwargs in specs.items():
    m = arch_model(returns, mean="Constant", dist="Normal", **kwargs)
    r = m.fit(disp="off")
    results[name] = r

    params = r.params.to_dict()
    row = {
        "Model": name,
        "omega": params.get("omega", np.nan),
        "alpha": params.get("alpha[1]", np.nan),
        "beta": params.get("beta[1]", np.nan),
        "gamma": params.get("gamma[1]", np.nan),
        "Log-Lik": r.loglikelihood,
        "BIC": r.bic,
    }
    rows.append(row)

comparison = pd.DataFrame(rows)
print(comparison.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

best = comparison.loc[comparison["BIC"].idxmin(), "Model"]
print(f"\nBest model by BIC: {best}")


# ── CELL: news_impact_curves ────────────────────────────────
# Purpose: Plot the "news impact curve" for each model — how a shock
#   of size epsilon affects next-period variance. GARCH produces a
#   symmetric V-shape; EGARCH and GJR tilt toward negative shocks.
# Visual: Three curves on one plot. GARCH is symmetric around zero.
#   EGARCH and GJR are asymmetric — the left side (negative shocks)
#   rises more steeply than the right side. This is the leverage effect
#   visualized: crashes spike volatility more than rallies.

shocks = np.linspace(-4, 4, 200)

# GARCH: sigma2 = omega + alpha * eps^2 + beta * sigma2_bar
garch_r = results["GARCH(1,1)"]
g_omega = garch_r.params["omega"]
g_alpha = garch_r.params["alpha[1]"]
g_beta = garch_r.params["beta[1]"]
g_long_var = g_omega / (1 - g_alpha - g_beta)
garch_nic = g_omega + g_alpha * shocks**2 + g_beta * g_long_var

# GJR: sigma2 = omega + (alpha + gamma * I[eps<0]) * eps^2 + beta * sigma2_bar
gjr_r = results["GJR-GARCH(1,1)"]
j_omega = gjr_r.params["omega"]
j_alpha = gjr_r.params["alpha[1]"]
j_beta = gjr_r.params["beta[1]"]
j_gamma = gjr_r.params["gamma[1]"]
j_long_var = j_omega / (1 - j_alpha - 0.5 * j_gamma - j_beta)
gjr_nic = j_omega + (j_alpha + j_gamma * (shocks < 0)) * shocks**2 + j_beta * j_long_var

# EGARCH: log(sigma2) = omega + alpha * |z| + gamma * z + beta * log(sigma2_bar)
egarch_r = results["EGARCH(1,1)"]
e_omega = egarch_r.params["omega"]
e_alpha = egarch_r.params["alpha[1]"]
e_beta = egarch_r.params["beta[1]"]
e_gamma = egarch_r.params["gamma[1]"]
e_long_logvar = e_omega / (1 - e_beta)
egarch_nic = np.exp(e_omega + e_alpha * np.abs(shocks) + e_gamma * shocks + e_beta * e_long_logvar)

fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(shocks, garch_nic, label="GARCH(1,1)", linewidth=2, color="steelblue")
ax.plot(shocks, gjr_nic, label="GJR-GARCH(1,1)", linewidth=2, color="darkorange")
ax.plot(shocks, egarch_nic, label="EGARCH(1,1)", linewidth=2, color="green")
ax.axvline(0, color="black", linewidth=0.5, linestyle="--")
ax.set_xlabel("Shock (standardized)")
ax.set_ylabel("Next-period conditional variance")
ax.set_title("News Impact Curves — Leverage Effect Visualization")
ax.legend()
plt.tight_layout()
plt.savefig("s6_news_impact_curves.png", dpi=120, bbox_inches="tight")
plt.close()


if __name__ == "__main__":
    # ── ASSERTIONS ─────────────────────────────────────
    for name, r in results.items():
        assert r.convergence_flag == 0, f"{name} did not converge"

    # EGARCH gamma is negative (leverage effect)
    egarch_gamma = results["EGARCH(1,1)"].params["gamma[1]"]
    assert egarch_gamma < 0, f"EGARCH gamma = {egarch_gamma:.4f}, expected < 0"

    # GJR-GARCH gamma is positive (bad news indicator)
    gjr_gamma = results["GJR-GARCH(1,1)"].params["gamma[1]"]
    assert gjr_gamma > 0, f"GJR gamma = {gjr_gamma:.4f}, expected > 0"

    # At least one asymmetric model beats GARCH by BIC
    garch_bic = results["GARCH(1,1)"].bic
    asym_bics = [results[n].bic for n in ["EGARCH(1,1)", "GJR-GARCH(1,1)"]]
    assert min(asym_bics) < garch_bic, (
        f"No asymmetric model beats GARCH BIC ({garch_bic:.1f}): "
        f"EGARCH={asym_bics[0]:.1f}, GJR={asym_bics[1]:.1f}"
    )

    print("✓ Section 6: All acceptance criteria passed")
