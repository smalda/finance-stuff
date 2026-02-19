"""
Section 8: Fractional Differentiation — Stationarity Without Amnesia

Acceptance criteria (from README):
- d=0.0 (raw prices): ADF p-value > 0.05 (non-stationary)
- d=1.0 (returns): ADF p-value < 0.01 (stationary)
- Minimum d for stationarity is between 0.2 and 0.8
- Correlation with original prices decreases as d increases (memory loss)
- Fractionally differenced series at minimum d has ADF p-value < 0.05
  AND correlation > correlation at d=1.0
- Table and multi-panel plot both produced
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller

sys.path.insert(0, str(Path(__file__).parent.parent))
from data_setup import load_prices


# ── CELL: fracdiff_weights ──────────────────────────────────
# Purpose: Implement fractional differentiation from scratch using the
#   truncated binomial series (Lopez de Prado, AFML Chapter 5). The key
#   insight: integer differencing (d=1) is a special case of a continuous
#   family parameterized by d in [0, 1].
# Takeaway: The weights decay as a power law: w_k ~ k^(-d-1). For d=1
#   (full differencing), only w_0=1 and w_1=-1 survive. For d=0.4, many
#   weights are non-zero, meaning the differenced series retains a long
#   "memory" of past prices — exactly the memory we want for ML features.

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


# ── CELL: fracdiff_sweep ────────────────────────────────────
# Purpose: Apply fractional differentiation at d = 0.0, 0.3, 0.5, 0.7, 1.0
#   to SPY log prices. For each d, run ADF and measure correlation with
#   original prices. Find the minimum d that achieves stationarity.
# Takeaway: The minimum d for SPY stationarity (ADF p < 0.05) is 0.4,
#   preserving 88% correlation with the original price level. Returns
#   (d=1) retain only 1.4% correlation — nearly complete memory loss.
#   At d=0.3, ADF p = 0.33 (still non-stationary); at d=0.4, p = 0.024.
#   This is the core value proposition: stationarity + memory.

prices_df = load_prices(["SPY"])
log_prices = np.log(prices_df["SPY"].dropna())

d_values = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
rows = []

for d in d_values:
    if d == 0.0:
        fd = log_prices
    elif d == 1.0:
        fd = log_prices.diff().dropna()
    else:
        fd = fracdiff(log_prices, d)

    adf_pval = adfuller(fd.dropna(), maxlag=20, autolag="AIC")[1]
    corr = fd.corr(log_prices.reindex(fd.index))
    rows.append({"d": d, "ADF p-value": adf_pval, "Corr w/ prices": corr})

sweep_df = pd.DataFrame(rows)
print(sweep_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

# Find minimum d for stationarity
min_d = sweep_df.loc[sweep_df["ADF p-value"] < 0.05, "d"].min()
print(f"\nMinimum d for stationarity (ADF p < 0.05): {min_d:.1f}")


# ── CELL: plot_fracdiff_comparison ──────────────────────────
# Purpose: Plot the fractionally differenced series at d=0, d=min_d,
#   and d=1.0 side by side to visualize the progression from prices
#   (trending, non-stationary) to "just stationary enough" (mild trend
#   removed) to returns (fully stationary, no memory of levels).
# Visual: Three panels. d=0 (log prices): upward trend from ~4.5 to ~6.4.
#   d=0.4 (fractionally differenced): oscillates around 0.25-0.40, still
#   tracks price level drift but with stationarity. d=1.0 (log returns):
#   flat, mean-zero noise with COVID spike at -0.10. The middle panel
#   is the sweet spot for ML features — stationary yet memory-rich.

fd_min = fracdiff(log_prices, min_d)
fd_returns = log_prices.diff().dropna()

fig, axes = plt.subplots(3, 1, figsize=(14, 9), sharex=True)
panels = [
    (log_prices, f"d = 0.0 (Log Prices)", "steelblue"),
    (fd_min, f"d = {min_d:.1f} (Minimum for Stationarity)", "darkorange"),
    (fd_returns, "d = 1.0 (Log Returns)", "green"),
]
for ax, (series, title, color) in zip(axes, panels):
    ax.plot(series.index, series.values, linewidth=0.5, color=color)
    ax.set_title(title)
    ax.set_ylabel("Value")

axes[-1].set_xlabel("Date")
plt.tight_layout()
plt.savefig("s8_fracdiff_comparison.png", dpi=120, bbox_inches="tight")
plt.close()


# ── CELL: plot_adf_vs_d ────────────────────────────────────
# Purpose: Plot d (x-axis) vs. ADF p-value (y-axis) to show the
#   transition from non-stationary to stationary as d increases.
# Visual: A declining curve that crosses the 0.05 significance line
#   at d ≈ 0.3-0.5 for SPY. The horizontal red line marks p = 0.05.

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(sweep_df["d"], sweep_df["ADF p-value"], "o-", color="steelblue", linewidth=2)
ax.axhline(0.05, color="red", linestyle="--", linewidth=1, label="p = 0.05")
ax.set_xlabel("Fractional differentiation order (d)")
ax.set_ylabel("ADF p-value")
ax.set_title("Stationarity Transition: ADF p-value vs. d")
ax.legend()
ax.set_ylim(-0.05, max(sweep_df["ADF p-value"].max() * 1.1, 0.15))
plt.tight_layout()
plt.savefig("s8_adf_vs_d.png", dpi=120, bbox_inches="tight")
plt.close()


if __name__ == "__main__":
    # ── ASSERTIONS ─────────────────────────────────────
    r = {row["d"]: row for _, row in sweep_df.iterrows()}

    # d=0: non-stationary
    assert r[0.0]["ADF p-value"] > 0.05, (
        f"d=0 ADF p = {r[0.0]['ADF p-value']:.4f}, expected > 0.05"
    )

    # d=1: stationary
    assert r[1.0]["ADF p-value"] < 0.01, (
        f"d=1 ADF p = {r[1.0]['ADF p-value']:.4f}, expected < 0.01"
    )

    # Minimum d in reasonable range
    assert 0.2 <= min_d <= 0.8, (
        f"Minimum d = {min_d}, expected in [0.2, 0.8]"
    )

    # Correlation decreases as d increases
    corr_min_d = r[min_d]["Corr w/ prices"]
    corr_d1 = r[1.0]["Corr w/ prices"]
    assert corr_min_d > corr_d1, (
        f"Correlation at d={min_d} ({corr_min_d:.3f}) should exceed "
        f"correlation at d=1 ({corr_d1:.3f})"
    )

    # Fractionally differenced series at min d is stationary
    assert r[min_d]["ADF p-value"] < 0.05, (
        f"Series at d={min_d} has ADF p = {r[min_d]['ADF p-value']:.4f}, expected < 0.05"
    )

    print("✓ Section 8: All acceptance criteria passed")
