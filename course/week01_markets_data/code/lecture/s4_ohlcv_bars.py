"""
Section 4: From Tick Data to OHLCV Bars — The Aggregation Pipeline

Acceptance criteria (from README):
- DataFrame has >= 2500 rows per ticker (10+ years of daily data)
- All 5 expected columns present (Open, High, Low, Close, Volume)
- Mean daily return within +/-0.003 for each ticker (approximately mean-zero)
- Price series and return series plotted side-by-side
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from data_setup import load_or_download, TICKERS, START, END

raw = load_or_download()


# ── CELL: inspect_ohlcv ─────────────────────────────────
# Purpose: Examine the raw DataFrame structure — columns, dtypes, shape.
# Takeaway: Daily OHLCV is the atomic unit of most quant research.
#   Five columns, one row per trading day. Simple — but every number
#   hides an aggregation (thousands of trades compressed into 4 prices
#   and a volume count).

close = raw["Close"]
print(f"Shape: {raw.shape}")
print(f"Date range: {raw.index[0].date()} to {raw.index[-1].date()}")
print(f"Columns: {list(raw.columns.get_level_values(0).unique())}")
close.head()


# ── CELL: compute_returns ────────────────────────────────
# Purpose: Compute simple daily returns from Close prices.
# Takeaway: Returns are roughly symmetric and mean-zero. Prices are not.
#   This is the single most important transformation in quantitative
#   finance — we almost never model prices directly.

returns = close.pct_change().dropna()
returns.describe().round(4)


# ── CELL: plot_price_vs_returns ──────────────────────────
# Purpose: Side-by-side plot of price series and return series.
# Visual: Left panel shows trending, non-stationary prices (each ticker
#   at a different scale). Right panel shows mean-zero, noisy returns.
#   The visual contrast motivates why we work with returns, not prices.

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

close.plot(ax=axes[0], linewidth=0.8)
axes[0].set_title("Price Series (Close)")
axes[0].set_ylabel("Price ($)")

returns.plot(ax=axes[1], linewidth=0.3, alpha=0.7, legend=False)
axes[1].set_title("Daily Returns")
axes[1].set_ylabel("Return")

plt.tight_layout()
plt.show()


if __name__ == "__main__":
    # ── ASSERTIONS ─────────────────────────────────────
    for ticker in TICKERS:
        n = close[ticker].dropna().shape[0]
        assert n >= 2500, f"{ticker}: expected >= 2500 rows, got {n}"

    expected_cols = {"Open", "High", "Low", "Close", "Volume"}
    actual_cols = set(raw.columns.get_level_values(0).unique())
    assert expected_cols <= actual_cols, f"Missing columns: {expected_cols - actual_cols}"

    for ticker in TICKERS:
        mu = returns[ticker].mean()
        assert abs(mu) < 0.003, f"{ticker}: mean return {mu:.5f} outside +/-0.003"

    print("Section 4: All acceptance criteria passed")
