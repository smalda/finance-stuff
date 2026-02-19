"""Section 6: The Practitioner's Lens — Barra-Style Risk Models"""
import matplotlib
matplotlib.use("Agg")
import sys
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

sys.path.insert(0, str(Path(__file__).parent.parent))
from data_setup import (
    load_equity_prices, load_monthly_returns, load_fundamentals,
    load_factor_data, CACHE_DIR,
)

warnings.filterwarnings("ignore", category=FutureWarning)

# Load data
prices = load_equity_prices()
monthly_returns = load_monthly_returns()
fundamentals = load_fundamentals()
ff3 = load_factor_data("3")

bs = fundamentals["balance_sheet"]
inc = fundamentals["income_stmt"]
sectors = fundamentals["sector"]
mcap = fundamentals["market_cap"]


# ── CELL: build_characteristics ─────────────────────────────

# Compute characteristics for each ticker
char_records = []
for ticker in monthly_returns.columns:
    # Log market cap
    if ticker in mcap.index and mcap[ticker] > 0:
        log_mcap = np.log(mcap[ticker])
    else:
        log_mcap = np.nan

    # Book-to-market from most recent balance sheet
    bm = np.nan
    roe = np.nan
    if ticker in bs.index.get_level_values("ticker"):
        tk_bs = bs.loc[ticker].dropna(subset=["Stockholders Equity"])
        if len(tk_bs) > 0:
            equity = tk_bs["Stockholders Equity"].iloc[-1]
            if equity > 0 and ticker in mcap.index and mcap[ticker] > 0:
                bm = equity / mcap[ticker]

            # ROE from income statement
            if ticker in inc.index.get_level_values("ticker"):
                tk_inc = inc.loc[ticker]
                if "Net Income" in tk_inc.columns:
                    net_inc_vals = tk_inc["Net Income"].dropna()
                    if len(net_inc_vals) > 0 and equity > 0:
                        roe = net_inc_vals.iloc[-1] / equity

    # Sector
    sector = sectors.get(ticker, "Other")

    char_records.append({
        "ticker": ticker,
        "log_mcap": log_mcap,
        "book_to_market": bm,
        "roe": roe,
        "sector": sector,
    })

chars = pd.DataFrame(char_records).set_index("ticker")

# Add price-based characteristics per month
# Momentum (12-1 month) and volatility (trailing 60-day)
daily_returns = prices.pct_change().dropna(how="all")


# ── CELL: monthly_characteristics ───────────────────────────

# Build panel of monthly characteristics
panel_months = []

for date in monthly_returns.index:
    # Momentum: 12-1 month return
    mom_end = date - pd.DateOffset(months=1)
    mom_start = date - pd.DateOffset(months=12)
    mask = (prices.index >= mom_start) & (prices.index <= mom_end)
    if mask.sum() < 20:
        continue
    mom_prices = prices.loc[mask]
    if len(mom_prices) >= 2:
        momentum = (mom_prices.iloc[-1] / mom_prices.iloc[0]) - 1
    else:
        continue

    # Volatility: trailing 60-day std
    vol_end_idx = prices.index.get_indexer([date], method="pad")[0]
    vol_start_idx = max(0, vol_end_idx - 60)
    recent_rets = daily_returns.iloc[vol_start_idx:vol_end_idx]
    volatility = recent_rets.std() * np.sqrt(252)

    for ticker in monthly_returns.columns:
        if pd.isna(monthly_returns.loc[date, ticker]):
            continue
        row = {
            "date": date,
            "ticker": ticker,
            "ret": monthly_returns.loc[date, ticker],
            "momentum": momentum.get(ticker, np.nan),
            "volatility": volatility.get(ticker, np.nan),
        }
        # Static characteristics
        for col in ["log_mcap", "book_to_market", "roe", "sector"]:
            row[col] = chars.loc[ticker, col] if ticker in chars.index else np.nan
        panel_months.append(row)

panel = pd.DataFrame(panel_months)
panel = panel.dropna(subset=["ret"])
print(f"Panel shape: {panel.shape}")
print(f"Date range: {panel['date'].min()} to {panel['date'].max()}")


# ── CELL: cross_sectional_regression ────────────────────────

# Standardize characteristics cross-sectionally each month
style_factors = ["log_mcap", "book_to_market", "momentum", "volatility", "roe"]
dates = sorted(panel["date"].unique())

# Use only dates where we have enough data
regression_dates = dates[-48:]  # last ~4 years (fundamental window)

factor_returns = []
r2_per_month = []

for date in regression_dates:
    month_data = panel[panel["date"] == date].copy()
    if len(month_data) < 50:
        continue

    # Standardize style factors within the month
    for col in style_factors:
        vals = month_data[col].astype(float)
        month_data[col] = (vals - vals.mean()) / vals.std()

    # Sector dummies
    sector_dummies = pd.get_dummies(month_data["sector"], prefix="sec", dtype=float)
    # Drop one for full rank
    if len(sector_dummies.columns) > 1:
        sector_dummies = sector_dummies.iloc[:, 1:]

    # Build regressor matrix
    X = pd.concat([
        month_data[style_factors].reset_index(drop=True),
        sector_dummies.reset_index(drop=True),
    ], axis=1).astype(float)
    X = X.dropna(axis=1, how="all")
    y = month_data["ret"].reset_index(drop=True).astype(float)

    # Drop rows with NaN
    valid = X.notna().all(axis=1) & y.notna()
    X = X.loc[valid]
    y = y.loc[valid]

    if len(y) < 30:
        continue

    X_with_const = sm.add_constant(X)
    model = sm.OLS(y, X_with_const).fit()

    # Extract style factor returns (coefficients)
    fret = {"date": date}
    for col in style_factors:
        if col in model.params:
            fret[col] = model.params[col]
    factor_returns.append(fret)
    r2_per_month.append(model.rsquared)

factor_ret_df = pd.DataFrame(factor_returns).set_index("date")
r2_series = pd.Series(r2_per_month, index=factor_ret_df.index)

print(f"\nCross-sectional regressions: {len(factor_ret_df)} months")
print(f"R² range: [{r2_series.min():.4f}, {r2_series.max():.4f}]")
print(f"R² median: {r2_series.median():.4f}")
print(f"\nFactor return means (monthly):")
print(factor_ret_df.mean().round(6).to_string())


# ── CELL: risk_decomposition ────────────────────────────────

# Pick a sample diversified portfolio (2 stocks per sector)
portfolio_tickers = []
for sector_name in sectors.unique():
    tickers_in_sector = sectors[sectors == sector_name].index.tolist()
    available = [t for t in tickers_in_sector
                 if t in monthly_returns.columns]
    portfolio_tickers.extend(available[:2])

portfolio_tickers = portfolio_tickers[:20]

# Use official FF5 factors for a proper time-series risk decomposition
# This gives a cleaner factor vs. specific variance breakdown
ff5 = load_factor_data("5")
port_ret_monthly = monthly_returns[portfolio_tickers].mean(axis=1)
common_fm = port_ret_monthly.index.intersection(ff5.index)
port_excess = port_ret_monthly.loc[common_fm] - ff5.loc[common_fm, "RF"]
X_risk = sm.add_constant(
    ff5.loc[common_fm, ["Mkt-RF", "SMB", "HML", "RMW", "CMA"]]
)
risk_model = sm.OLS(port_excess, X_risk).fit()

# Factor-explained variance = var(fitted) / var(actual)
fitted_returns = risk_model.fittedvalues
total_var = port_excess.var()
factor_var = fitted_returns.var()
specific_var = risk_model.resid.var()
factor_risk_share = factor_var / total_var

print(f"\nPortfolio risk decomposition ({len(portfolio_tickers)} stocks):")
print(f"  R² from FF5 regression: {risk_model.rsquared:.4f}")
print(f"  Total variance: {total_var:.8f}")
print(f"  Factor variance: {factor_var:.8f}")
print(f"  Specific variance: {specific_var:.8f}")
print(f"  Factor risk share: {factor_risk_share:.2%}")
print(f"  Specific risk share: {1 - factor_risk_share:.2%}")


# ── CELL: barra_plot ────────────────────────────────────────

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# R² over time
ax = axes[0]
ax.bar(range(len(r2_series)), r2_series.values, alpha=0.6, width=1.0)
ax.axhline(r2_series.median(), color="red", lw=2,
           label=f"Median: {r2_series.median():.3f}")
ax.set(title="Cross-Sectional R² by Month",
       xlabel="Month", ylabel="R²")
ax.legend()

# Risk decomposition pie
ax = axes[1]
labels = ["Factor Risk", "Specific Risk"]
sizes = [factor_risk_share, 1 - factor_risk_share]
colors = ["#2196F3", "#FF9800"]
ax.pie(sizes, labels=labels, colors=colors, autopct="%1.1f%%",
       startangle=90, textprops={"fontsize": 11})
ax.set_title(f"Portfolio Risk Decomposition\n({len(portfolio_tickers)} stocks)")

plt.tight_layout()
plt.show()


if __name__ == "__main__":
    # ── ASSERTIONS ─────────────────────────────────
    n_months = len(factor_ret_df)
    assert n_months >= 20, \
        f"Only {n_months} regression months (expected ≥20)"

    median_r2 = r2_series.median()
    assert 0.05 <= median_r2 <= 0.50, \
        f"Median R² = {median_r2:.4f}, outside [0.05, 0.50]"

    # Size factor returns near zero for large-cap universe
    size_mean = factor_ret_df["log_mcap"].mean()
    assert -0.010 <= size_mean <= 0.010, \
        f"Size factor mean = {size_mean:.6f}, outside [-0.010, 0.010]"

    # Factor risk share for diversified portfolio
    assert 0.10 <= factor_risk_share <= 0.95, \
        f"Factor risk share = {factor_risk_share:.2%}, outside [10%, 95%]"

    # ── RESULTS ────────────────────────────────────
    print(f"══ lecture/s6_barra_risk_models ═════════════════════")
    print(f"  n_regression_months: {n_months}")
    print(f"  median_cross_r2: {median_r2:.4f}")
    print(f"  r2_range: [{r2_series.min():.4f}, {r2_series.max():.4f}]")
    print(f"  size_factor_mean: {size_mean:.6f}")
    print(f"  momentum_factor_mean: {factor_ret_df['momentum'].mean():.6f}")
    print(f"  portfolio_tickers: {len(portfolio_tickers)}")
    print(f"  factor_risk_share: {factor_risk_share:.4f}")
    print(f"  specific_risk_share: {1 - factor_risk_share:.4f}")

    # ── PLOT ───────────────────────────────────────
    fig.savefig(CACHE_DIR / "s6_barra_risk.png", dpi=150, bbox_inches="tight")
    print(f"  ── plot: s6_barra_risk.png ──")
    print(f"     type: bar chart + pie chart")
    print(f"     n_panels: 2")
    print(f"     title_left: {axes[0].get_title()}")
    print(f"     title_right: {axes[1].get_title()}")
    print(f"✓ s6_barra_risk_models: ALL PASSED")
