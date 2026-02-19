"""
Deliverable 4: Return Statistics Summary (Bridge to Week 2)

Acceptance criteria (from README):
- Statistics computed for all 10 tickers (not fewer)
- Kurtosis > 3 for every ticker (fat tails are universal — should be 4-30+)
- At least one volatile ticker (e.g., TSLA) has kurtosis > 10
- Number of > 3σ days exceeds the normal distribution expectation (~0.3%) for every ticker
- Summary table is clean and readable with consistent formatting
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats

sys.path.insert(0, str(Path(__file__).parent.parent))
from data_setup import CACHE_DIR

# Import the loader
from hw.d1_data_loader_class import FinancialDataLoader

# ── CELL: select_representative_tickers ─────────────────
# Purpose: Choose 10 tickers spanning sectors and volatility profiles.
# Takeaway: Returns are NOT homogeneous. A utility (DUK) behaves
#   differently from a tech growth stock (TSLA). Sector and market
#   cap matter. This diversity shows up in the statistics.

TICKERS = [
    "AAPL",   # Tech, large-cap
    "TSLA",   # Tech/auto, high volatility
    "JPM",    # Financials
    "JNJ",    # Healthcare, defensive
    "XOM",    # Energy
    "WMT",    # Consumer staples, defensive
    "NFLX",   # Media/entertainment (replacing DIS which wasn't in d1)
    "BA",     # Industrials
    "SPY",    # S&P 500 ETF (market benchmark)
    "GLD"     # Gold ETF (alternative asset)
]

print(f"Computing return statistics for {len(TICKERS)} representative tickers\n")


# ── CELL: load_data_and_compute_returns ─────────────────
# Purpose: Load adjusted close prices and compute daily log returns.
# Takeaway: Use log returns for statistical analysis (they're additive
#   across time, approximately normal for small moves). Use simple
#   returns for dollar P&L. The distinction matters.

loader = FinancialDataLoader(
    tickers=TICKERS,
    start_date="2010-01-01",
    end_date="2025-01-01"
)

# Load only the selected tickers
data = loader.load(tickers=TICKERS, format="wide")

close = data["Adj Close"]
returns = np.log(close / close.shift(1)).dropna()

print(f"Loaded {len(returns)} daily returns per ticker")
print(f"Date range: {returns.index[0].date()} to {returns.index[-1].date()}\n")


# ── CELL: compute_statistics ────────────────────────────
# Purpose: For each ticker, compute: mean, volatility (annualized),
#   skewness, kurtosis, min/max, and tail behavior (> 3σ days).
# Takeaway: These statistics preview Week 2's core questions:
#   Are returns normal? (No — kurtosis >> 3)
#   Are they symmetric? (No — skewness != 0)
#   Are tails Gaussian? (No — far more outliers than expected)

stats_rows = []

for ticker in TICKERS:
    ret = returns[ticker].dropna()

    # Basic moments
    mean_daily = ret.mean()
    std_daily = ret.std()
    vol_annual = std_daily * np.sqrt(252)  # Annualize volatility

    # Higher moments
    skew = stats.skew(ret)
    kurt = stats.kurtosis(ret)  # Excess kurtosis (normal = 0)

    # Extremes
    min_ret = ret.min()
    max_ret = ret.max()

    # Tail behavior: count days with |return| > 3σ
    threshold = 3 * std_daily
    extreme_days = (ret.abs() > threshold).sum()
    extreme_pct = extreme_days / len(ret)

    # Normal distribution expectation: P(|X| > 3σ) ≈ 0.27%
    normal_expectation = 0.0027
    extreme_ratio = extreme_pct / normal_expectation

    stats_rows.append({
        "Ticker": ticker,
        "Mean (daily)": mean_daily,
        "Vol (annual)": vol_annual,
        "Skewness": skew,
        "Kurtosis": kurt + 3,  # Convert to absolute kurtosis (normal = 3)
        "Min": min_ret,
        "Max": max_ret,
        "> 3σ days": extreme_days,
        "> 3σ %": extreme_pct,
        "vs Normal": extreme_ratio
    })

stats_df = pd.DataFrame(stats_rows)


# ── CELL: format_summary_table ──────────────────────────
# Purpose: Display statistics in a clean, readable table.
# Takeaway: The table makes the non-normality immediately obvious:
#   kurtosis >> 3, extreme days >> 0.3%, skewness != 0. Every ticker
#   violates the Gaussian assumption. This motivates Week 2.

print("=" * 100)
print("DAILY RETURN STATISTICS (2010-2025)")
print("=" * 100)
print()

# Format for display
display_df = stats_df.copy()
display_df["Mean (daily)"] = display_df["Mean (daily)"].apply(lambda x: f"{x:.4f}")
display_df["Vol (annual)"] = display_df["Vol (annual)"].apply(lambda x: f"{x:.1%}")
display_df["Skewness"] = display_df["Skewness"].apply(lambda x: f"{x:+.2f}")
display_df["Kurtosis"] = display_df["Kurtosis"].apply(lambda x: f"{x:.1f}")
display_df["Min"] = display_df["Min"].apply(lambda x: f"{x:.1%}")
display_df["Max"] = display_df["Max"].apply(lambda x: f"{x:.1%}")
display_df["> 3σ %"] = display_df["> 3σ %"].apply(lambda x: f"{x:.2%}")
display_df["vs Normal"] = display_df["vs Normal"].apply(lambda x: f"{x:.1f}x")

print(display_df.to_string(index=False))
print()


# ── CELL: key_observations ──────────────────────────────
# Purpose: Highlight the key patterns that set up Week 2.
# Takeaway: Financial returns are NOT normal — they have fat tails
#   (high kurtosis), asymmetry (nonzero skewness), and far more
#   extreme events than a Gaussian model predicts. This breaks
#   many standard ML assumptions.

print("=" * 100)
print("KEY OBSERVATIONS")
print("=" * 100)
print()

max_kurt_ticker = stats_df.loc[stats_df["Kurtosis"].idxmax(), "Ticker"]
max_kurt_value = stats_df["Kurtosis"].max()

print("1. Fat Tails (Kurtosis)")
print(f"   - All tickers have kurtosis > 3 (normal distribution has kurtosis = 3)")
print(f"   - Highest: {max_kurt_ticker} with kurtosis = {max_kurt_value:.1f}")
print(f"   - This means: extreme events (crashes, spikes) occur FAR more often than normal models predict")
print()

print("2. Asymmetry (Skewness)")
print(f"   - Most tickers have negative skewness (left tail is fatter)")
print(f"   - This means: big losses happen more often than big gains of the same magnitude")
print(f"   - Average skewness: {stats_df['Skewness'].mean():+.2f}")
print()

print("3. Extreme Events")
print(f"   - Normal distribution predicts ~0.27% of days with |return| > 3σ")
print(f"   - Actual: {stats_df['> 3σ %'].mean():.2%} on average ({stats_df['vs Normal'].mean():.1f}x more)")
print(f"   - TSLA: {stats_df[stats_df['Ticker'] == 'TSLA']['> 3σ %'].values[0]:.2%} extreme days")
print()

print("4. Implications for ML")
print("   - Assumption: 'Returns are normally distributed' — VIOLATED by every ticker")
print("   - Assumption: 'Extreme events are rare' — VIOLATED (they're 2-10x more common)")
print("   - Assumption: 'Up and down moves are symmetric' — VIOLATED (crashes > booms)")
print()
print("These violations are the subject of Week 2: Stylized Facts of Financial Time Series")
print()


# ── CELL: visualize_distributions ───────────────────────
# Purpose: Plot return distributions for 3 representative tickers
#   (defensive, benchmark, volatile) against a Gaussian fit.
# Visual: The histograms show pronounced peaks (leptokurtosis) and
#   fat tails. The Gaussian overlay badly underestimates tail
#   probability. The visual makes the non-normality undeniable.

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

tickers_to_plot = ["JNJ", "SPY", "TSLA"]  # Defensive, benchmark, volatile

for ax, ticker in zip(axes, tickers_to_plot):
    ret = returns[ticker].dropna()

    # Histogram
    ax.hist(ret, bins=50, density=True, alpha=0.7, label="Empirical", edgecolor="black")

    # Gaussian fit
    mu, sigma = ret.mean(), ret.std()
    x = np.linspace(ret.min(), ret.max(), 100)
    gaussian = stats.norm.pdf(x, mu, sigma)
    ax.plot(x, gaussian, 'r-', linewidth=2, label=f"Gaussian fit (μ={mu:.4f}, σ={sigma:.4f})")

    ax.set_title(f"{ticker} Daily Returns")
    ax.set_xlabel("Return")
    ax.set_ylabel("Density")
    ax.legend()
    ax.grid(alpha=0.3)

plt.tight_layout()
plt.show()


if __name__ == "__main__":
    # ── ASSERTIONS ─────────────────────────────────────

    # Verify statistics computed for all 10 tickers
    assert len(stats_df) == 10, (
        f"Expected statistics for 10 tickers, got {len(stats_df)}"
    )

    # Verify kurtosis > 3 for all tickers (fat tails)
    kurtosis_values = stats_df["Kurtosis"].values
    assert all(k > 3 for k in kurtosis_values), (
        f"All tickers should have kurtosis > 3, got min {kurtosis_values.min():.1f}"
    )

    # Verify at least one volatile ticker has kurtosis > 10
    max_kurtosis = kurtosis_values.max()
    max_kurt_ticker = stats_df.loc[stats_df["Kurtosis"] == max_kurtosis, "Ticker"].values[0]
    assert max_kurtosis > 10, (
        f"Expected at least one ticker with kurtosis > 10, got max {max_kurtosis:.1f} ({max_kurt_ticker})"
    )

    # Verify > 3σ days exceed normal expectation for all tickers
    normal_expectation = 0.0027
    extreme_pcts = stats_df["> 3σ %"].values
    assert all(pct > normal_expectation for pct in extreme_pcts), (
        f"All tickers should have > 3σ % > {normal_expectation:.2%}, got min {extreme_pcts.min():.2%}"
    )

    # Verify summary table is properly formatted
    assert len(display_df.columns) == len(stats_df.columns), (
        "Display table should have same columns as stats table"
    )

    print("\n✓ Deliverable 4: All acceptance criteria passed")
