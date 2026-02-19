"""
Section 5: GARCH — The Industry's Canonical Volatility Model

Acceptance criteria (from README):
- GARCH(1,1) converges successfully
- alpha > 0 and beta > 0 (both positive)
- alpha + beta > 0.90 and < 1.0 (high persistence, stationary)
- Conditional volatility plot shows visible spikes during crisis periods
- Long-run annualized volatility between 10% and 30%
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
returns = prices["SPY"].pct_change().dropna() * 100  # arch expects pct returns


# ── CELL: fit_garch ─────────────────────────────────────────
# Purpose: Fit GARCH(1,1) to SPY daily returns — the single most widely
#   used parametric volatility model in finance. Display estimated
#   parameters with standard errors and interpret each one.
# Takeaway: SPY GARCH(1,1) yields α = 0.168, β = 0.800,
#   persistence = 0.968. Yesterday's shock explains ~17% of today's
#   variance, and ~80% comes from persistence of the previous variance
#   estimate. ω = 0.037. Long-run annualized volatility = 17.1%.

model = arch_model(returns, vol="Garch", p=1, q=1, mean="Constant", dist="Normal")
res = model.fit(disp="off")

alpha = res.params["alpha[1]"]
beta = res.params["beta[1]"]
omega = res.params["omega"]
persistence = alpha + beta
long_run_var = omega / (1 - persistence)
long_run_vol_annual = np.sqrt(long_run_var * 252) / 100  # convert from pct

print(res.summary().tables[1])
print(f"\nPersistence (α + β): {persistence:.4f}")
print(f"Long-run annualized volatility: {long_run_vol_annual:.1%}")


# ── CELL: plot_conditional_vol ──────────────────────────────
# Purpose: Overlay GARCH conditional volatility on absolute returns
#   to show that the model tracks volatility regimes in real time.
# Visual: Grey bars show daily absolute returns (volatile spikes in 2011,
#   2015-16, 2018, COVID March 2020 at ~9%, 2022). The blue GARCH
#   conditional volatility line tracks these regimes smoothly — rising
#   during crises and falling during calm periods. The model responds
#   within days, not weeks.

cond_vol = res.conditional_volatility / 100  # back to decimal
abs_ret = returns.abs() / 100

fig, ax = plt.subplots(figsize=(14, 5))
ax.bar(abs_ret.index, abs_ret.values, width=1, color="lightgray",
       alpha=0.6, label="|Daily Return|")
ax.plot(cond_vol.index, cond_vol.values, linewidth=0.8, color="steelblue",
        label="GARCH(1,1) σ")
ax.set_ylabel("Volatility (daily)")
ax.set_title("GARCH(1,1) Conditional Volatility — SPY")
ax.legend(fontsize=9)
plt.tight_layout()
plt.savefig("s5_garch_cond_vol.png", dpi=120, bbox_inches="tight")
plt.close()


if __name__ == "__main__":
    # ── ASSERTIONS ─────────────────────────────────────
    assert res.convergence_flag == 0, (
        f"GARCH(1,1) did not converge: flag = {res.convergence_flag}"
    )
    assert alpha > 0, f"alpha = {alpha:.4f}, expected > 0"
    assert beta > 0, f"beta = {beta:.4f}, expected > 0"
    assert 0.90 < persistence < 1.0, (
        f"persistence = {persistence:.4f}, expected in (0.90, 1.0)"
    )
    assert 0.10 < long_run_vol_annual < 0.30, (
        f"long-run vol = {long_run_vol_annual:.2%}, expected 10%-30%"
    )

    print("✓ Section 5: All acceptance criteria passed")
