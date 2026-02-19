"""
Section 2: The Order Book — Where Price Discovery Lives

Acceptance criteria (from README):
- Order book has >= 5 price levels on each side (bid and ask)
- Bid-ask spread > 0 and visible in visualization
- After large market buy: top-of-book ask price increases
  (demonstrates walking the book)
- Spread widens after the large order vs. before
"""
import matplotlib.pyplot as plt
import numpy as np

np.random.seed(42)


# ── CELL: build_order_book ───────────────────────────────
# Purpose: Build a synthetic limit order book for a stock trading
#   near $150. The book has 10 price levels on each side, with
#   volumes that taper off away from the mid-price.
# Takeaway: The order book is the real-time supply/demand curve for
#   a stock. Bids stack below (buyers willing to pay less), asks stack
#   above (sellers demanding more). The gap between the best bid and
#   best ask is the spread — the market's admission fee.

mid_price = 150.00
tick_size = 0.01
n_levels = 10

bid_prices = np.array([mid_price - tick_size * (i + 1) for i in range(n_levels)])
ask_prices = np.array([mid_price + tick_size * (i + 1) for i in range(n_levels)])

bid_volumes = np.random.randint(100, 800, size=n_levels).astype(float)
ask_volumes = np.random.randint(100, 800, size=n_levels).astype(float)

best_bid = bid_prices[0]
best_ask = ask_prices[0]
spread_before = best_ask - best_bid
print(f"Best bid: ${best_bid:.2f}  |  Best ask: ${best_ask:.2f}")
print(f"Spread: ${spread_before:.4f}  ({spread_before / mid_price * 10000:.1f} bps)")
print(f"Mid-price: ${mid_price:.2f}")


# ── CELL: plot_order_book ────────────────────────────────
# Purpose: Visualize the order book as a horizontal bar chart —
#   bids on the left (green), asks on the right (red).
# Visual: A symmetric structure centered on the mid-price. Volume
#   bars extend horizontally. The gap in the center is the spread.

fig, ax = plt.subplots(figsize=(10, 6))

ax.barh(bid_prices, -bid_volumes, height=tick_size * 0.8,
        color="#2E7D32", alpha=0.7, label="Bids (buyers)")
ax.barh(ask_prices, ask_volumes, height=tick_size * 0.8,
        color="#C62828", alpha=0.7, label="Asks (sellers)")

ax.axhline(mid_price, color="gray", linestyle="--", linewidth=0.8, label="Mid-price")
ax.set_xlabel("Volume (shares)")
ax.set_ylabel("Price ($)")
ax.set_title("Limit Order Book — Before Large Market Buy")
ax.legend(loc="lower right")
plt.tight_layout()
plt.show()


# ── CELL: market_order_impact ────────────────────────────
# Purpose: Simulate a large market buy order (2000 shares) that
#   "walks the book" — consuming multiple ask levels and moving
#   the price up.
# Takeaway: Large orders don't execute at a single price. They eat
#   through resting limit orders, pushing the price against the buyer.
#   This is market impact — and it's why institutional traders slice
#   large orders into smaller pieces.

market_buy_size = 2000
remaining = market_buy_size
filled_prices = []
filled_volumes = []
post_ask_volumes = ask_volumes.copy()

for i in range(n_levels):
    if remaining <= 0:
        break
    fill = min(remaining, post_ask_volumes[i])
    filled_prices.append(ask_prices[i])
    filled_volumes.append(fill)
    post_ask_volumes[i] -= fill
    remaining -= fill

avg_fill_price = np.average(filled_prices, weights=filled_volumes)
levels_consumed = sum(1 for v in filled_volumes if v > 0)

new_best_ask = ask_prices[np.argmax(post_ask_volumes > 0)]
spread_after = new_best_ask - best_bid

print(f"Market buy: {market_buy_size} shares")
print(f"Levels consumed: {levels_consumed}")
print(f"Average fill price: ${avg_fill_price:.4f}")
print(f"New best ask: ${new_best_ask:.2f}  (was ${best_ask:.2f})")
print(f"Spread after: ${spread_after:.4f}  (was ${spread_before:.4f})")


# ── CELL: plot_after_impact ──────────────────────────────
# Purpose: Visualize the order book after the large market buy.
# Visual: The ask side is depleted near the top — several levels
#   consumed or partially filled. The spread is visibly wider.
#   Contrasting with the "before" chart above shows how large orders
#   move markets.

fig, ax = plt.subplots(figsize=(10, 6))

ax.barh(bid_prices, -bid_volumes, height=tick_size * 0.8,
        color="#2E7D32", alpha=0.7, label="Bids (buyers)")
ax.barh(ask_prices, post_ask_volumes, height=tick_size * 0.8,
        color="#C62828", alpha=0.7, label="Asks (sellers)")

ax.axhline(mid_price, color="gray", linestyle="--", linewidth=0.8, label="Mid-price")
ax.set_xlabel("Volume (shares)")
ax.set_ylabel("Price ($)")
ax.set_title("Limit Order Book — After 2,000-Share Market Buy")
ax.legend(loc="lower right")
plt.tight_layout()
plt.show()


if __name__ == "__main__":
    # ── ASSERTIONS ─────────────────────────────────────
    assert n_levels >= 5, f"Need >= 5 levels per side, got {n_levels}"
    assert spread_before > 0, f"Spread must be > 0, got {spread_before}"
    assert new_best_ask > best_ask, (
        f"After large buy, best ask should increase: {new_best_ask} vs {best_ask}"
    )
    assert spread_after > spread_before, (
        f"Spread should widen: {spread_after:.4f} vs {spread_before:.4f}"
    )
    print("Section 2: All acceptance criteria passed")
