"""
Exercise 1: The Survivorship Trap — Measuring Ghost Returns

Acceptance criteria (from README):
- >= 400 tickers successfully downloaded (some S&P 500 tickers may fail in yfinance)
- At least 20 tickers identified as having data starting after 2011 (late entrants)
- Survivorship-biased portfolio annualized return > survivors-only portfolio return
- Annual bias magnitude > 0.5% (should be 1-4%)
- Cumulative bias over the full period > 5%
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

sys.path.insert(0, str(Path(__file__).parent.parent))
from data_setup import CACHE_DIR

# ── CELL: scrape_sp500_tickers ──────────────────────────
# Purpose: Download the current S&P 500 constituent list from Wikipedia.
#   This is the survivorship-biased universe — it includes companies that
#   succeeded (are in the index NOW) and excludes companies that failed.
# Takeaway: Wikipedia's S&P 500 list is a snapshot of TODAY's index.
#   Projecting it backward creates survivorship bias because companies
#   that were in the index in 2010 but failed before 2025 are missing.

url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
# Need to set a user agent to avoid 403 errors from Wikipedia
headers = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"
}
tables = pd.read_html(url, storage_options={"User-Agent": headers["User-Agent"]})
sp500_table = tables[0]
sp500_tickers = sp500_table["Symbol"].str.replace(".", "-", regex=False).tolist()

print(f"Scraped {len(sp500_tickers)} tickers from Wikipedia")
print(f"First 10: {sp500_tickers[:10]}")


# ── CELL: download_historical_data ──────────────────────
# Purpose: Download adjusted close prices for all S&P 500 constituents
#   from 2010 to present. This reveals which tickers have incomplete
#   histories — they joined the index after 2010 (late entrants).
# Takeaway: yfinance fails gracefully for some tickers (ticker changes,
#   delistings, data quality issues). We handle failures by catching them
#   and continuing — real production code does the same.

start_date = "2010-01-01"
end_date = "2025-01-01"

cache_file = CACHE_DIR / "sp500_survivorship.parquet"
if cache_file.exists():
    print("Loading cached S&P 500 data...")
    prices = pd.read_parquet(cache_file)
else:
    print(f"Downloading {len(sp500_tickers)} tickers (this takes 2-5 minutes)...")
    prices = yf.download(
        sp500_tickers,
        start=start_date,
        end=end_date,
        auto_adjust=True,
        progress=False
    )["Close"]
    prices.to_parquet(cache_file)

n_tickers_downloaded = prices.shape[1]
print(f"Successfully downloaded {n_tickers_downloaded} tickers")
print(f"Date range: {prices.index[0].date()} to {prices.index[-1].date()}")


# ── CELL: identify_late_entrants ────────────────────────
# Purpose: Find tickers that have data starting significantly after 2010.
#   These are companies that IPO'd or were added to the S&P 500 after 2010
#   — they're late entrants. Including them in a backtest inflates returns
#   because they were selected for success AFTER they succeeded.
# Takeaway: A ticker with data starting in 2015 is a company that either
#   didn't exist in 2010 or wasn't in the S&P 500 then. A 2010 backtest
#   couldn't have invested in it — but the survivorship-biased approach does.

late_entry_threshold = pd.Timestamp("2011-01-01")
first_valid = prices.apply(lambda col: col.first_valid_index())

late_entrants = first_valid[first_valid > late_entry_threshold].sort_values()
full_history_tickers = first_valid[first_valid <= late_entry_threshold]

print(f"\nLate entrants ({len(late_entrants)} tickers):")
print(late_entrants.head(10))
print(f"\nFull-history tickers: {len(full_history_tickers)}")


# ── CELL: compute_portfolio_returns ─────────────────────
# Purpose: Compute equal-weight portfolio returns for two universes:
#   (1) ALL current constituents (survivorship-biased)
#   (2) ONLY tickers with complete data from 2010 (survivors-only, less biased)
# Takeaway: Universe (1) includes future winners that weren't available in 2010.
#   Universe (2) is closer to what a 2010 investor could actually buy, but it's
#   still biased because it excludes companies that WERE in the index in 2010
#   but failed before 2025. The difference measures survivorship bias.

# Biased portfolio: equal-weight all tickers, forward-fill missing data
biased_prices = prices.ffill().dropna(axis=1, how="all")
biased_returns = biased_prices.pct_change().mean(axis=1)
biased_cumret = (1 + biased_returns).cumprod()

# Survivors-only portfolio: only tickers with data starting <= 2011-01-01
survivor_tickers = full_history_tickers.index.tolist()
survivor_prices = prices[survivor_tickers].ffill()
survivor_returns = survivor_prices.pct_change().mean(axis=1)
survivor_cumret = (1 + survivor_returns).cumprod()

print(f"\nBiased portfolio: {biased_prices.shape[1]} tickers")
print(f"Survivor portfolio: {len(survivor_tickers)} tickers")


# ── CELL: compute_bias_metrics ──────────────────────────
# Purpose: Quantify the survivorship bias: annualized return difference
#   and cumulative return difference over the full period.
# Takeaway: The bias is NOT trivial. Typical results show 1-4% per year —
#   over 14 years, that compounds to 15-70% cumulative overstatement.
#   If your backtest shows 15% annualized return, 2-4 of those percentage
#   points might be ghosts from survivorship bias alone.

n_years = (prices.index[-1] - prices.index[0]).days / 365.25

biased_total = biased_cumret.iloc[-1] - 1
survivor_total = survivor_cumret.iloc[-1] - 1

biased_annualized = (1 + biased_total) ** (1 / n_years) - 1
survivor_annualized = (1 + survivor_total) ** (1 / n_years) - 1

annual_bias = biased_annualized - survivor_annualized
cumulative_bias = biased_total - survivor_total

print("\nSurvivorship Bias Results:")
print(f"Biased portfolio annualized return: {biased_annualized:.2%}")
print(f"Survivor portfolio annualized return: {survivor_annualized:.2%}")
print(f"Annual bias: {annual_bias:.2%}")
print(f"Cumulative bias over {n_years:.1f} years: {cumulative_bias:.2%}")
print(f"\nInterpretation: If your backtest shows {biased_annualized:.2%} annualized,")
print(f"{annual_bias:.2%} of that might be survivorship bias — returns from")
print(f"companies that succeeded AFTER your backtest period began.")


# ── CELL: visualize_cumulative_returns ──────────────────
# Purpose: Plot the cumulative return paths for both portfolios.
# Visual: The biased portfolio (all current constituents) shows higher
#   cumulative returns. The gap widens over time because late entrants
#   are companies that succeeded — including them is like picking winners
#   with hindsight. The visual makes the bias concrete and dramatic.

import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(12, 6))

ax.plot(biased_cumret.index, (biased_cumret - 1) * 100,
        label="Survivorship-biased (all current constituents)",
        color="#C62828", linewidth=1.2)
ax.plot(survivor_cumret.index, (survivor_cumret - 1) * 100,
        label="Survivors-only (complete data from 2010)",
        color="#1565C0", linewidth=1.2)

ax.axhline(0, color="gray", linestyle="--", alpha=0.3)
ax.set_xlabel("Date")
ax.set_ylabel("Cumulative Return (%)")
ax.set_title("Survivorship Bias in S&P 500 Equal-Weight Portfolios (2010-2025)")
ax.legend()
ax.grid(alpha=0.2)
plt.tight_layout()
plt.show()


if __name__ == "__main__":
    # ── ASSERTIONS ─────────────────────────────────────
    assert n_tickers_downloaded >= 400, (
        f"Expected >= 400 tickers downloaded, got {n_tickers_downloaded}"
    )

    assert len(late_entrants) >= 20, (
        f"Expected >= 20 late entrants (data starting after 2011), got {len(late_entrants)}"
    )

    assert biased_annualized > survivor_annualized, (
        f"Biased portfolio ({biased_annualized:.2%}) should outperform survivor portfolio "
        f"({survivor_annualized:.2%}) — this is the survivorship bias signal"
    )

    assert annual_bias > 0.005, (
        f"Annual bias should be > 0.5%, got {annual_bias:.2%}"
    )

    assert cumulative_bias > 0.05, (
        f"Cumulative bias over {n_years:.1f} years should be > 5%, got {cumulative_bias:.2%}"
    )

    print("✓ Exercise 1: All acceptance criteria passed")
