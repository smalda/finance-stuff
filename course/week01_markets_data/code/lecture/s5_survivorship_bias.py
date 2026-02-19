"""
Section 5: The Ghost Stocks — Survivorship Bias

Acceptance criteria (from README):
- Simulation uses exactly 10 stocks: 7 survivors, 3 bankruptcies
- Survivor-only portfolio return > full-universe portfolio return (bias is always positive)
- Gap between survivor and full-universe returns is > 1% annualized
- Bankrupt stocks reach $0 or near-zero — they're not just "underperformers"
"""
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

np.random.seed(42)


# ── CELL: simulate_universe ──────────────────────────────
# Purpose: Create a 10-stock universe over 14 years where 7 stocks
#   survive and 3 go bankrupt. This is the thought experiment that
#   makes survivorship bias concrete.
# Takeaway: The 10 stocks mimic a realistic outcome — most companies
#   survive and grow, but a meaningful minority fail completely. The
#   question is: what happens to your backtest when you pretend the
#   failures never existed?

n_years = 14
n_days = n_years * 252
dates = pd.bdate_range("2010-01-04", periods=n_days)

survivor_names = [f"Stock_{i}" for i in range(1, 8)]
bankrupt_names = [f"Bankrupt_{i}" for i in range(1, 4)]
all_names = survivor_names + bankrupt_names

daily_returns = pd.DataFrame(index=dates, columns=all_names, dtype=float)

for name in survivor_names:
    mu = np.random.uniform(0.0002, 0.0005)
    sigma = np.random.uniform(0.015, 0.025)
    daily_returns[name] = np.random.normal(mu, sigma, n_days)

bankruptcy_days = [int(n_days * f) for f in [0.4, 0.6, 0.8]]
for name, death_day in zip(bankrupt_names, bankruptcy_days):
    mu = np.random.uniform(-0.0003, 0.0001)
    sigma = np.random.uniform(0.02, 0.035)
    rets = np.random.normal(mu, sigma, n_days)
    rets[death_day - 20 : death_day] = np.random.normal(-0.08, 0.04, 20)
    rets[death_day:] = 0.0
    daily_returns[name] = rets

prices = (1 + daily_returns).cumprod() * 100

for name, death_day in zip(bankrupt_names, bankruptcy_days):
    prices.loc[prices.index[death_day:], name] = 0.0


# ── CELL: plot_all_stocks ────────────────────────────────
# Purpose: Plot all 10 stock price paths — survivors in blue, bankruptcies in red.
# Visual: Seven lines trend upward (with noise). Three lines crash to zero
#   at different points. The bankruptcies are dramatic, not subtle — these
#   aren't just underperformers, they're total losses.

fig, ax = plt.subplots(figsize=(12, 6))

for name in survivor_names:
    ax.plot(prices[name], color="#1565C0", alpha=0.6, linewidth=0.8)
for name in bankrupt_names:
    ax.plot(prices[name], color="#C62828", alpha=0.8, linewidth=1.2, linestyle="--")

legend_elements = [
    Line2D([0], [0], color="#1565C0", alpha=0.6, linewidth=0.8, label="Survivors"),
    Line2D([0], [0], color="#C62828", alpha=0.8, linewidth=1.2, linestyle="--", label="Bankruptcies")
]
ax.set_xlabel("Date")
ax.set_ylabel("Price ($)")
ax.set_title("10-Stock Universe: 7 Survivors (blue) + 3 Bankruptcies (red)")
ax.legend(handles=legend_elements, loc="upper left")
plt.tight_layout()
plt.show()


# ── CELL: measure_bias ──────────────────────────────────
# Purpose: Compute the annualized return of the full universe vs.
#   the survivor-only portfolio. The difference is the survivorship bias.
# Takeaway: The gap is not trivial. Looking only at survivors inflates
#   annualized returns by several percentage points — compounding to
#   a massive cumulative overstatement over 14 years. If your backtest
#   shows 15% annualized returns, some of those percentage points are ghosts.

survivor_total = prices[survivor_names].iloc[-1].mean() / 100
full_total = prices[all_names].iloc[-1].mean() / 100

ann_return_survivors = survivor_total ** (1 / n_years) - 1
ann_return_full = full_total ** (1 / n_years) - 1
bias = ann_return_survivors - ann_return_full

cum_survivors = (survivor_total - 1) * 100
cum_full = (full_total - 1) * 100

print(f"Survivor-only annualized return: {ann_return_survivors:.2%}")
print(f"Full-universe annualized return: {ann_return_full:.2%}")
print(f"Survivorship bias: {bias:.2%} per year")
print(f"Cumulative: survivors {cum_survivors:.0f}% vs full {cum_full:.0f}%")


if __name__ == "__main__":
    # ── ASSERTIONS ─────────────────────────────────────
    assert len(survivor_names) == 7, f"Expected 7 survivors, got {len(survivor_names)}"
    assert len(bankrupt_names) == 3, f"Expected 3 bankruptcies, got {len(bankrupt_names)}"
    assert len(all_names) == 10, f"Expected 10 total stocks, got {len(all_names)}"

    for name in bankrupt_names:
        final = prices[name].iloc[-1]
        assert final < 1.0, f"{name} should be near zero, got {final:.2f}"

    assert ann_return_survivors > ann_return_full, (
        f"Survivors ({ann_return_survivors:.4f}) should beat full universe ({ann_return_full:.4f})"
    )
    assert bias > 0.01, f"Bias should be > 1% annualized, got {bias:.4f}"

    print("Section 5: All acceptance criteria passed")
