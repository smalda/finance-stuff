"""Section 3: Building Factors from Scratch — Portfolio Sorts"""
import matplotlib
matplotlib.use("Agg")
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from data_setup import (
    load_equity_prices, load_monthly_returns, load_fundamentals,
    load_factor_data, CACHE_DIR,
)

# Load data
prices = load_equity_prices()
monthly_returns = load_monthly_returns()
fundamentals = load_fundamentals()
ff3_official = load_factor_data("3")

bs = fundamentals["balance_sheet"]
shares_df = fundamentals["shares"]
mcap_series = fundamentals["market_cap"]


# ── CELL: compute_book_to_market ────────────────────────────

# Compute book equity and book-to-market for each ticker-year
book_equity = bs["Stockholders Equity"].dropna()
book_equity = book_equity[book_equity > 0]  # must be positive

# Get shares outstanding for market cap computation
shares = shares_df["shares"].dropna()

# Build annual B/M using most recent book equity and market cap
# For each ticker, use most recent annual balance sheet
bm_records = []
for ticker in book_equity.index.get_level_values("ticker").unique():
    be = book_equity.loc[ticker]
    if len(be) == 0:
        continue
    # Use most recent positive book equity
    latest_be = be.sort_index().iloc[-1]

    # Market cap from data_setup (current)
    if ticker in mcap_series.index and mcap_series[ticker] > 0:
        mkt_cap = mcap_series[ticker]
        bm = latest_be / mkt_cap
        bm_records.append({
            "ticker": ticker,
            "book_equity": latest_be,
            "market_cap": mkt_cap,
            "book_to_market": bm,
        })

bm_df = pd.DataFrame(bm_records).set_index("ticker")
print(f"B/M computed for {len(bm_df)} tickers")
print(f"B/M range: [{bm_df['book_to_market'].min():.4f}, "
      f"{bm_df['book_to_market'].max():.4f}]")
print(f"B/M median: {bm_df['book_to_market'].median():.4f}")


# ── CELL: double_sort_portfolios ────────────────────────────

# 2x3 sort: median market cap for size, 30/70 B/M for value
size_median = bm_df["market_cap"].median()
bm_30 = bm_df["book_to_market"].quantile(0.30)
bm_70 = bm_df["book_to_market"].quantile(0.70)

def assign_portfolio(row):
    """Assign a stock to one of six portfolios based on size and B/M."""
    size = "S" if row["market_cap"] < size_median else "B"
    if row["book_to_market"] <= bm_30:
        value = "L"
    elif row["book_to_market"] >= bm_70:
        value = "H"
    else:
        value = "M"
    return f"{size}/{value}"

bm_df["portfolio"] = bm_df.apply(assign_portfolio, axis=1)
portfolio_counts = bm_df["portfolio"].value_counts().sort_index()
print("\nPortfolio counts:")
print(portfolio_counts.to_string())


# ── CELL: compute_factor_returns ────────────────────────────

# Compute value-weighted monthly returns for each portfolio
valid_tickers = set(bm_df.index) & set(monthly_returns.columns)
bm_valid = bm_df.loc[bm_df.index.isin(valid_tickers)]

# Weight by market cap within each portfolio
smb_monthly = []
hml_monthly = []

for date in monthly_returns.index:
    rets_month = monthly_returns.loc[date, list(valid_tickers)].dropna()
    if len(rets_month) < 20:
        continue

    portfolio_returns = {}
    for pf_name in ["S/L", "S/M", "S/H", "B/L", "B/M", "B/H"]:
        tickers_in_pf = bm_valid[bm_valid["portfolio"] == pf_name].index
        tickers_avail = [t for t in tickers_in_pf if t in rets_month.index]
        if len(tickers_avail) == 0:
            portfolio_returns[pf_name] = np.nan
            continue
        weights = bm_valid.loc[tickers_avail, "market_cap"]
        weights = weights / weights.sum()
        portfolio_returns[pf_name] = (rets_month[tickers_avail] * weights).sum()

    if any(np.isnan(v) for v in portfolio_returns.values()):
        continue

    # SMB = (S/L + S/M + S/H)/3 - (B/L + B/M + B/H)/3
    smb_val = (portfolio_returns["S/L"] + portfolio_returns["S/M"]
               + portfolio_returns["S/H"]) / 3 \
              - (portfolio_returns["B/L"] + portfolio_returns["B/M"]
                 + portfolio_returns["B/H"]) / 3

    # HML = (S/H + B/H)/2 - (S/L + B/L)/2
    hml_val = (portfolio_returns["S/H"] + portfolio_returns["B/H"]) / 2 \
              - (portfolio_returns["S/L"] + portfolio_returns["B/L"]) / 2

    smb_monthly.append({"date": date, "SMB_self": smb_val})
    hml_monthly.append({"date": date, "HML_self": hml_val})

smb_series = pd.DataFrame(smb_monthly).set_index("date")["SMB_self"]
hml_series = pd.DataFrame(hml_monthly).set_index("date")["HML_self"]

print(f"\nSelf-built factor returns: {len(smb_series)} months")
print(f"SMB mean: {smb_series.mean():.6f}, std: {smb_series.std():.6f}")
print(f"HML mean: {hml_series.mean():.6f}, std: {hml_series.std():.6f}")


# ── CELL: validate_against_official ─────────────────────────

# Align self-built and official factors
common_dates = smb_series.index.intersection(ff3_official.index)
smb_corr = smb_series.loc[common_dates].corr(ff3_official.loc[common_dates, "SMB"])
hml_corr = hml_series.loc[common_dates].corr(ff3_official.loc[common_dates, "HML"])

smb_te = (smb_series.loc[common_dates]
          - ff3_official.loc[common_dates, "SMB"]).std() * np.sqrt(12)
hml_te = (hml_series.loc[common_dates]
          - ff3_official.loc[common_dates, "HML"]).std() * np.sqrt(12)

print(f"\nValidation against Ken French ({len(common_dates)} months):")
print(f"  SMB correlation: {smb_corr:.4f}")
print(f"  HML correlation: {hml_corr:.4f}")
print(f"  SMB tracking error (ann.): {smb_te:.4f}")
print(f"  HML tracking error (ann.): {hml_te:.4f}")


# ── CELL: factor_comparison_plot ────────────────────────────

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# SMB comparison
ax = axes[0]
cum_smb_self = (1 + smb_series.loc[common_dates]).cumprod()
cum_smb_off = (1 + ff3_official.loc[common_dates, "SMB"]).cumprod()
ax.plot(cum_smb_self.index, cum_smb_self, label="Self-built", lw=1.5)
ax.plot(cum_smb_off.index, cum_smb_off, label="Ken French", lw=1.5, ls="--")
ax.set(title=f"SMB: Self-Built vs. Official (r={smb_corr:.2f})",
       ylabel="Cumulative Return")
ax.legend()
ax.grid(True, alpha=0.3)

# HML comparison
ax = axes[1]
cum_hml_self = (1 + hml_series.loc[common_dates]).cumprod()
cum_hml_off = (1 + ff3_official.loc[common_dates, "HML"]).cumprod()
ax.plot(cum_hml_self.index, cum_hml_self, label="Self-built", lw=1.5)
ax.plot(cum_hml_off.index, cum_hml_off, label="Ken French", lw=1.5, ls="--")
ax.set(title=f"HML: Self-Built vs. Official (r={hml_corr:.2f})",
       ylabel="Cumulative Return")
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()


if __name__ == "__main__":
    # ── ASSERTIONS ─────────────────────────────────
    n_bm = len(bm_df)
    assert n_bm >= 150, f"B/M computed for only {n_bm} tickers (expected ≥150)"

    bm_median = bm_df["book_to_market"].median()
    assert 0.03 <= bm_median <= 0.50, \
        f"B/M median = {bm_median:.4f}, outside [0.03, 0.50]"

    for pf in ["S/L", "S/M", "S/H", "B/L", "B/M", "B/H"]:
        count = portfolio_counts.get(pf, 0)
        assert count >= 10, f"Portfolio {pf} has only {count} stocks (expected ≥10)"

    assert len(smb_series) >= 50, \
        f"SMB has only {len(smb_series)} months (expected ≥50)"

    assert -0.02 <= smb_series.mean() <= 0.01, \
        f"SMB mean = {smb_series.mean():.6f}, outside [-0.02, +0.01]"
    assert 0.005 <= smb_series.std() <= 0.05, \
        f"SMB std = {smb_series.std():.6f}, outside [0.005, 0.05]"

    assert -0.02 <= hml_series.mean() <= 0.02, \
        f"HML mean = {hml_series.mean():.6f}, outside [-0.02, +0.02]"
    assert 0.005 <= hml_series.std() <= 0.06, \
        f"HML std = {hml_series.std():.6f}, outside [0.005, 0.06]"

    assert 0.05 <= smb_corr <= 0.65, \
        f"SMB correlation = {smb_corr:.4f}, outside [0.05, 0.65]"
    assert 0.05 <= hml_corr <= 0.90, \
        f"HML correlation = {hml_corr:.4f}, outside [0.05, 0.90]"

    assert 0.03 <= smb_te <= 0.25, \
        f"SMB tracking error = {smb_te:.4f}, outside [0.03, 0.25]"
    assert 0.03 <= hml_te <= 0.25, \
        f"HML tracking error = {hml_te:.4f}, outside [0.03, 0.25]"

    # ── RESULTS ────────────────────────────────────
    print(f"══ lecture/s3_factor_construction ═══════════════════")
    print(f"  n_tickers_with_bm: {n_bm}")
    print(f"  bm_median: {bm_median:.4f}")
    print(f"  bm_range: [{bm_df['book_to_market'].min():.4f}, "
          f"{bm_df['book_to_market'].max():.4f}]")
    print(f"  portfolio_counts: {portfolio_counts.to_dict()}")
    print(f"  n_factor_months: {len(smb_series)}")
    print(f"  smb_mean: {smb_series.mean():.6f}")
    print(f"  smb_std: {smb_series.std():.6f}")
    print(f"  hml_mean: {hml_series.mean():.6f}")
    print(f"  hml_std: {hml_series.std():.6f}")
    print(f"  smb_kf_corr: {smb_corr:.4f}")
    print(f"  hml_kf_corr: {hml_corr:.4f}")
    print(f"  smb_tracking_error_ann: {smb_te:.4f}")
    print(f"  hml_tracking_error_ann: {hml_te:.4f}")

    # ── PLOT ───────────────────────────────────────
    fig.savefig(CACHE_DIR / "s3_factor_comparison.png", dpi=150, bbox_inches="tight")
    print(f"  ── plot: s3_factor_comparison.png ──")
    print(f"     type: dual cumulative return comparison")
    print(f"     n_lines_per_panel: 2")
    print(f"     title_left: {axes[0].get_title()}")
    print(f"     title_right: {axes[1].get_title()}")
    print(f"✓ s3_factor_construction: ALL PASSED")
