"""
Exercise 3: Fractional Differentiation in Practice — Finding the Minimum d

Acceptance criteria (from README):
- Fractional differentiation computed at all 11 d values for all 5 tickers
- Minimum d for stationarity is between 0.1 and 0.9 for all 5 tickers
- For each ticker: correlation at minimum d > correlation at d=1.0
- Memory gain (correlation difference) > 0.05 for at least 4 of 5 tickers
- ADF p-value plot shows clear transition from > 0.05 to < 0.05 as d increases
- Summary table complete with all 5 tickers
- 3-panel SPY visualization produced (d=0, d_min, d=1)
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller

sys.path.insert(0, str(Path(__file__).parent.parent))
from data_setup import load_prices, FRACDIFF_TICKERS


# ── CELL: fracdiff_implementation ────────────────────────
# Purpose: Implement fractional differentiation using the truncated
#   binomial series (Lopez de Prado, AFML Chapter 5). Same implementation
#   as the lecture — factored here for reuse across 5 tickers.
# Takeaway: The weights decay as a power law: w_k ~ k^(-d-1). For d<1,
#   many weights are non-zero, so the differenced series retains a long
#   "memory" of past prices. The window parameter controls truncation.

def fracdiff_weights(d, window, threshold=1e-5):
    """Compute fractional differencing weights using the binomial series."""
    weights = [1.0]
    for k in range(1, window):
        w = -weights[-1] * (d - k + 1) / k
        if abs(w) < threshold:
            break
        weights.append(w)
    return np.array(weights)


def fracdiff(series, d, window=500):
    """Apply fractional differencing of order d to a pandas Series."""
    weights = fracdiff_weights(d, window)
    width = len(weights)
    result = pd.Series(index=series.index, dtype=float)
    for t in range(width - 1, len(series)):
        result.iloc[t] = np.dot(weights, series.values[t - width + 1:t + 1][::-1])
    return result.dropna()


# ── CELL: sweep_all_tickers ──────────────────────────────
# Purpose: For each of 5 tickers, sweep d from 0.0 to 1.0 in steps of
#   0.1. At each d, run the ADF test and measure correlation with the
#   original log price series. Find the minimum d for stationarity.
# Takeaway: Minimum d: SPY = 0.4, AAPL = 0.5, TSLA = 0.3, TLT = 0.3,
#   GLD = 0.4. Correlation at min d: SPY = 0.88, TSLA = 0.91, AAPL = 0.73.
#   At d=1, all correlations collapse to <0.04. Memory gains are enormous
#   (0.75-0.90) — fractional differentiation preserves meaningful level info.

prices_df = load_prices(FRACDIFF_TICKERS)
d_values = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

all_sweep_rows = []
summary_rows = []

for ticker in FRACDIFF_TICKERS:
    log_p = np.log(prices_df[ticker].dropna())

    for d in d_values:
        if d == 0.0:
            fd = log_p
        elif d == 1.0:
            fd = log_p.diff().dropna()
        else:
            fd = fracdiff(log_p, d)

        adf_p = adfuller(fd.dropna(), maxlag=20, autolag="AIC")[1]
        corr = fd.corr(log_p.reindex(fd.index))
        all_sweep_rows.append({
            "Ticker": ticker, "d": d, "ADF p-value": adf_p, "Corr": corr,
        })

    # Find minimum d for stationarity
    ticker_rows = [r for r in all_sweep_rows if r["Ticker"] == ticker]
    stationary = [r for r in ticker_rows if r["ADF p-value"] < 0.05]
    if stationary:
        min_d = min(r["d"] for r in stationary)
        corr_min = next(r["Corr"] for r in ticker_rows if r["d"] == min_d)
    else:
        min_d = 1.0
        corr_min = next(r["Corr"] for r in ticker_rows if r["d"] == 1.0)
    corr_d1 = next(r["Corr"] for r in ticker_rows if r["d"] == 1.0)

    summary_rows.append({
        "Ticker": ticker,
        "Min d": min_d,
        "Corr @ min d": corr_min,
        "Corr @ d=1": corr_d1,
        "Memory Gain": corr_min - corr_d1,
    })

sweep_df = pd.DataFrame(all_sweep_rows)
summary_df = pd.DataFrame(summary_rows)

print("=== Summary: Minimum d for Stationarity ===")
print(summary_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))


# ── CELL: plot_adf_vs_d ─────────────────────────────────
# Purpose: Plot d (x-axis) vs. ADF p-value (y-axis) for all 5 tickers
#   on the same figure. Shows the transition from non-stationary to
#   stationary as d increases, with ticker-specific stationarity thresholds.
# Visual: Five curves declining from ~0.9 at d=0 toward 0. TLT crosses
#   the red p=0.05 line earliest at d=0.3 (weakest trend, starting ADF
#   p=0.18). AAPL crosses latest at d=0.5 (strongest trend, starting
#   p=0.88). SPY crosses between d=0.3 and d=0.4.

fig, ax = plt.subplots(figsize=(10, 6))
for ticker in FRACDIFF_TICKERS:
    sub = sweep_df[sweep_df["Ticker"] == ticker]
    ax.plot(sub["d"], sub["ADF p-value"], "o-", linewidth=1.5, markersize=4, label=ticker)

ax.axhline(0.05, color="red", linestyle="--", linewidth=1, label="p = 0.05")
ax.set_xlabel("Fractional differentiation order (d)")
ax.set_ylabel("ADF p-value")
ax.set_title("Stationarity Transition: ADF p-value vs. d (5 Tickers)")
ax.legend(fontsize=9)
ax.set_ylim(-0.05, 1.0)
plt.tight_layout()
plt.savefig("ex3_adf_vs_d.png", dpi=120, bbox_inches="tight")
plt.close()


# ── CELL: plot_spy_three_panel ───────────────────────────
# Purpose: Visualize SPY at d=0 (log prices), d=min_d (fractionally
#   differenced), and d=1.0 (log returns) side by side to show the
#   progression from trending non-stationary to "just stationary enough"
#   to fully stationary but memoryless.
# Visual: Three panels. Top: log prices trending 4.5→6.4. Middle: d=0.4
#   oscillates 0.15-0.40, still drifts upward but with stationarity.
#   Bottom: flat mean-zero log returns, COVID spike at -0.10.

spy_log = np.log(prices_df["SPY"].dropna())
spy_min_d = next(r["Min d"] for r in summary_rows if r["Ticker"] == "SPY")
fd_min = fracdiff(spy_log, spy_min_d)
fd_returns = spy_log.diff().dropna()

fig, axes = plt.subplots(3, 1, figsize=(14, 9), sharex=True)
panels = [
    (spy_log, f"d = 0.0 (Log Prices)", "steelblue"),
    (fd_min, f"d = {spy_min_d:.1f} (Minimum for Stationarity)", "darkorange"),
    (fd_returns, "d = 1.0 (Log Returns)", "green"),
]
for ax, (series, title, color) in zip(axes, panels):
    ax.plot(series.index, series.values, linewidth=0.5, color=color)
    ax.set_title(title)
    ax.set_ylabel("Value")

axes[-1].set_xlabel("Date")
plt.tight_layout()
plt.savefig("ex3_spy_fracdiff_comparison.png", dpi=120, bbox_inches="tight")
plt.close()


if __name__ == "__main__":
    # ── ASSERTIONS ─────────────────────────────────────
    # All 11 d values for all 5 tickers computed
    assert len(sweep_df) == len(FRACDIFF_TICKERS) * len(d_values), (
        f"Expected {len(FRACDIFF_TICKERS) * len(d_values)} rows, got {len(sweep_df)}"
    )

    # Minimum d in [0.1, 0.9] for all tickers
    for _, row in summary_df.iterrows():
        assert 0.1 <= row["Min d"] <= 0.9, (
            f"{row['Ticker']}: min d = {row['Min d']}, expected in [0.1, 0.9]"
        )

    # Correlation at min d > correlation at d=1 for each ticker
    for _, row in summary_df.iterrows():
        assert row["Corr @ min d"] > row["Corr @ d=1"], (
            f"{row['Ticker']}: corr at min d ({row['Corr @ min d']:.3f}) "
            f"should exceed corr at d=1 ({row['Corr @ d=1']:.3f})"
        )

    # Memory gain > 0.05 for at least 4 of 5 tickers
    n_significant_gain = (summary_df["Memory Gain"] > 0.05).sum()
    assert n_significant_gain >= 4, (
        f"Only {n_significant_gain} tickers have memory gain > 0.05, expected >= 4"
    )

    print("✓ Exercise 3: All acceptance criteria passed")
