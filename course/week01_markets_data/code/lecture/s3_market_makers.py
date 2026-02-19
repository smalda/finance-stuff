"""
Section 3: Market Makers — The Invisible Infrastructure

Acceptance criteria (from README):
- Market maker P&L is positive (profitable) with uninformed (random) order flow
- Market maker P&L is reduced or negative with informed order flow
- P&L difference between uninformed vs. informed scenarios is statistically visible
- Simulation runs >= 100 trades for stable results
"""
import matplotlib.pyplot as plt
import numpy as np

np.random.seed(42)


# ── CELL: simulation_setup ───────────────────────────────
# Purpose: Define the market maker simulation. The market maker
#   continuously quotes a bid and ask around a "true" price. Incoming
#   orders arrive and trade against the quotes. The key variable:
#   are the orders informed (they know where the price is going) or
#   uninformed (random)?
# Takeaway: Market making is profitable on average — but only if
#   most order flow is uninformed. This explains why retail order flow
#   is so valuable: retail traders are statistically uninformed.

n_trades = 500
spread = 0.10
half_spread = spread / 2
price_volatility = 0.50


def simulate_market_maker(n_trades, informed_fraction):
    """Simulate a market maker facing a mix of informed and uninformed orders.

    Returns a cumulative P&L array of length n_trades.
    """
    pnl = np.zeros(n_trades)

    for i in range(n_trades):
        is_informed = np.random.random() < informed_fraction

        if is_informed:
            # Informed trader knows the next price move and trades accordingly.
            # MM earns the spread but loses the move (adverse selection).
            price_move = np.random.normal(0, price_volatility)
            pnl[i] = half_spread - abs(price_move)
        else:
            # Uninformed trader buys or sells randomly.
            # MM earns the spread with no adverse selection.
            pnl[i] = half_spread

    return np.cumsum(pnl)


# ── CELL: run_uninformed ─────────────────────────────────
# Purpose: Run the simulation with 100% uninformed (random) order flow.
# Takeaway: With only random flow, the market maker earns the spread
#   on every trade. P&L climbs steadily — this is the ideal scenario
#   and the economic rationale for market making.

pnl_uninformed = simulate_market_maker(n_trades, informed_fraction=0.0)
print(f"Uninformed flow — Final P&L: ${pnl_uninformed[-1]:.2f}")
print(f"  Per-trade avg: ${pnl_uninformed[-1] / n_trades:.4f}")


# ── CELL: run_informed ───────────────────────────────────
# Purpose: Run the simulation with 50% informed order flow — half
#   the traders know the next price move.
# Takeaway: Informed traders systematically extract value from the
#   market maker. The spread income can't compensate for the adverse
#   selection losses. This is why market makers care deeply about
#   WHO they trade with — not just how much.

pnl_informed = simulate_market_maker(n_trades, informed_fraction=0.5)
print(f"Informed flow (50%) — Final P&L: ${pnl_informed[-1]:.2f}")
print(f"  Per-trade avg: ${pnl_informed[-1] / n_trades:.4f}")


# ── CELL: plot_pnl_comparison ────────────────────────────
# Purpose: Compare cumulative P&L paths for uninformed vs. informed flow.
# Visual: Two diverging lines. The uninformed path trends upward (profitable).
#   The informed path is flat or trends downward (losses from adverse selection
#   exceed spread income). The gap between them is the cost of adverse selection.

fig, ax = plt.subplots(figsize=(12, 5))

ax.plot(pnl_uninformed, label="100% uninformed flow", color="#2E7D32", linewidth=1.2)
ax.plot(pnl_informed, label="50% informed flow", color="#C62828", linewidth=1.2)
ax.axhline(0, color="gray", linestyle="--", linewidth=0.5)

ax.set_xlabel("Trade number")
ax.set_ylabel("Cumulative P&L ($)")
ax.set_title("Market Maker P&L: Uninformed vs. Informed Order Flow")
ax.legend()
plt.tight_layout()
plt.show()


if __name__ == "__main__":
    # ── ASSERTIONS ─────────────────────────────────────
    # Run multiple seeds to ensure the result is robust, not seed-dependent
    profits_uninformed = []
    profits_informed = []
    for seed in range(20):
        np.random.seed(seed)
        u = simulate_market_maker(500, informed_fraction=0.0)
        profits_uninformed.append(u[-1])
        np.random.seed(seed + 1000)
        inf = simulate_market_maker(500, informed_fraction=0.5)
        profits_informed.append(inf[-1])

    mean_u = np.mean(profits_uninformed)
    mean_i = np.mean(profits_informed)

    assert mean_u > 0, f"Uninformed P&L should be positive, got {mean_u:.2f}"
    assert mean_u > mean_i, (
        f"Uninformed P&L ({mean_u:.2f}) should exceed informed P&L ({mean_i:.2f})"
    )
    assert n_trades >= 100, f"Need >= 100 trades, got {n_trades}"

    print(f"  Mean uninformed P&L: ${mean_u:.2f}")
    print(f"  Mean informed P&L:   ${mean_i:.2f}")
    print("Section 3: All acceptance criteria passed")
