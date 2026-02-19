"""Exercise 2: Which Factors Carry a Risk Premium?

Run Fama-MacBeth regressions with 5 characteristics:
market beta, size, value, profitability, and momentum.
"""
import matplotlib
matplotlib.use("Agg")
import sys
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from linearmodels import FamaMacBeth as LMFamaMacBeth

sys.path.insert(0, str(Path(__file__).parent.parent))
from data_setup import (
    load_equity_prices, load_monthly_returns, load_fundamentals,
    load_factor_data, CACHE_DIR,
)

warnings.filterwarnings("ignore", category=FutureWarning)

prices = load_equity_prices()
monthly_returns = load_monthly_returns()
fundamentals = load_fundamentals()
ff3 = load_factor_data("3")
bs = fundamentals["balance_sheet"]
inc = fundamentals["income_stmt"]
mcap_current = fundamentals["market_cap"]


# ── CELL: estimate_rolling_betas ────────────────────────────

# Market betas from rolling 60-month windows
common_idx = monthly_returns.index.intersection(ff3.index)
excess_returns = monthly_returns.loc[common_idx].sub(ff3.loc[common_idx, "RF"], axis=0)
mkt_rf = ff3.loc[common_idx, "Mkt-RF"]

beta_panel = {}
window = 60

for i in range(window, len(common_idx)):
    date = common_idx[i]
    mkt_window = mkt_rf.iloc[i - window:i]
    betas = {}
    for ticker in excess_returns.columns:
        y = excess_returns[ticker].iloc[i - window:i].dropna()
        if len(y) < 36:
            continue
        x = mkt_window.loc[y.index]
        X = sm.add_constant(x)
        model = sm.OLS(y, X).fit()
        betas[ticker] = model.params["Mkt-RF"]
    beta_panel[date] = betas

beta_df = pd.DataFrame(beta_panel).T
print(f"Rolling beta panel: {beta_df.shape}")


# ── CELL: build_characteristics_panel ───────────────────────

# Static fundamental characteristics
fund_chars = {}
for ticker in monthly_returns.columns:
    chars = {}
    # Log market cap
    if ticker in mcap_current.index and mcap_current[ticker] > 0:
        chars["log_mcap"] = np.log(mcap_current[ticker])

    # Book-to-market
    if ticker in bs.index.get_level_values("ticker"):
        tk_bs = bs.loc[ticker].sort_index()
        eq = tk_bs["Stockholders Equity"].dropna()
        if len(eq) > 0 and eq.iloc[-1] > 0:
            if ticker in mcap_current.index and mcap_current[ticker] > 0:
                chars["bm"] = eq.iloc[-1] / mcap_current[ticker]

    # Profitability (operating income / book equity)
    if ticker in inc.index.get_level_values("ticker"):
        tk_inc = inc.loc[ticker].sort_index()
        oi = tk_inc.get("Operating Income", pd.Series(dtype=float)).dropna()
        if len(oi) == 0:
            oi = tk_inc.get("Net Income", pd.Series(dtype=float)).dropna()
        if len(oi) > 0 and "bm" in chars:
            # Use equity from above
            eq_val = bs.loc[ticker].sort_index()["Stockholders Equity"].dropna()
            if len(eq_val) > 0 and eq_val.iloc[-1] > 0:
                chars["profitability"] = oi.iloc[-1] / eq_val.iloc[-1]

    fund_chars[ticker] = chars

fund_df = pd.DataFrame(fund_chars).T
print(f"Fundamental chars: {fund_df.shape}")
print(f"  log_mcap non-null: {fund_df['log_mcap'].notna().sum()}")
print(f"  bm non-null: {fund_df['bm'].notna().sum()}")
print(f"  profitability non-null: {fund_df['profitability'].notna().sum()}")


# ── CELL: build_fama_macbeth_panel ──────────────────────────

# Build panel: for each month, characteristics + returns
panel_records = []

for date in beta_df.index:
    # Momentum: 12-1 month return
    mom_end = date - pd.DateOffset(months=1)
    mom_start = date - pd.DateOffset(months=12)
    mask = (prices.index >= mom_start) & (prices.index <= mom_end)
    if mask.sum() < 20:
        continue
    mom_prices = prices.loc[mask]
    if len(mom_prices) < 2:
        continue
    momentum = (mom_prices.iloc[-1] / mom_prices.iloc[0]) - 1

    for ticker in monthly_returns.columns:
        if date not in excess_returns.index:
            continue
        ret = excess_returns.loc[date, ticker]
        if pd.isna(ret):
            continue

        beta_val = beta_df.loc[date].get(ticker, np.nan)
        if pd.isna(beta_val):
            continue

        row = {
            "date": date,
            "ticker": ticker,
            "excess_ret": ret,
            "beta": beta_val,
            "momentum": momentum.get(ticker, np.nan),
        }

        if ticker in fund_df.index:
            for col in fund_df.columns:
                row[col] = fund_df.loc[ticker, col]

        panel_records.append(row)

panel = pd.DataFrame(panel_records)
panel = panel.set_index(["ticker", "date"]).sort_index()
print(f"\nPanel shape: {panel.shape}")


# ── CELL: cross_sectional_standardize ───────────────────────

# Standardize characteristics cross-sectionally each month
char_cols = ["beta", "log_mcap", "bm", "profitability", "momentum"]

def standardize_month(group):
    """Z-score standardize within each month."""
    for col in char_cols:
        if col in group.columns:
            vals = group[col]
            mean, std = vals.mean(), vals.std()
            if std > 0:
                group[col] = (vals - mean) / std
    return group

panel_std = panel.groupby(level="date", group_keys=False).apply(standardize_month)
panel_clean = panel_std.dropna(subset=char_cols + ["excess_ret"])
print(f"Clean panel: {panel_clean.shape}")


# ── CELL: run_fama_macbeth ──────────────────────────────────

dep = panel_clean[["excess_ret"]]
indep = panel_clean[char_cols]

fm = LMFamaMacBeth(dep, indep).fit(cov_type="kernel")

print("\nFama-MacBeth Risk Premia:")
print(f"{'Factor':<16} {'Gamma':>10} {'NW t-stat':>10} {'p-value':>10}")
print("-" * 50)
for factor in char_cols:
    g = fm.params[factor]
    t = fm.tstats[factor]
    p = fm.pvalues[factor]
    sig = "*" if p < 0.05 else ""
    print(f"{factor:<16} {g:>10.6f} {t:>10.2f} {p:>10.4f} {sig}")


# ── CELL: significance_plot ─────────────────────────────────

fig, ax = plt.subplots(figsize=(10, 5))
factors = char_cols
gammas = [fm.params[f] for f in factors]
tstats = [fm.tstats[f] for f in factors]
colors = ["#2196F3" if abs(t) >= 2.0 else "#BDBDBD" for t in tstats]

bars = ax.bar(range(len(factors)), tstats, color=colors, edgecolor="k",
              linewidth=0.5)
ax.axhline(2.0, color="red", ls="--", lw=1, label="t = 2.0")
ax.axhline(-2.0, color="red", ls="--", lw=1)
ax.set_xticks(range(len(factors)))
ax.set_xticklabels(factors, rotation=30, ha="right")
ax.set(title="Fama-MacBeth t-Statistics — Which Factors Are Priced?",
       ylabel="Newey-West t-statistic")
ax.legend()
plt.tight_layout()
plt.show()


if __name__ == "__main__":
    # ── ASSERTIONS ─────────────────────────────────
    assert len(panel_clean) >= 5000, \
        f"Panel too small: {len(panel_clean)} (expected ≥5000)"

    # Gamma ranges
    gamma_beta = fm.params["beta"]
    assert -0.015 <= gamma_beta <= 0.020, \
        f"gamma_beta = {gamma_beta:.6f}, outside [-0.015, 0.020]"

    gamma_mcap = fm.params["log_mcap"]
    assert -0.010 <= gamma_mcap <= 0.010, \
        f"gamma_mcap = {gamma_mcap:.6f}, outside [-0.010, 0.010]"

    # At least one factor significant
    sig_count = sum(1 for f in char_cols if abs(fm.tstats[f]) >= 2.0)
    # At least one factor insignificant
    insig_count = sum(1 for f in char_cols if abs(fm.tstats[f]) < 2.0)

    assert sig_count >= 1, "No factors are significant at t=2.0"
    assert insig_count >= 1, "All factors are significant — expected some to be weak"

    # ── RESULTS ────────────────────────────────────
    print(f"══ seminar/ex2_factor_premia ════════════════════════")
    print(f"  panel_size: {len(panel_clean)}")
    print(f"  n_months: {panel_clean.index.get_level_values('date').nunique()}")
    for f in char_cols:
        print(f"  gamma_{f}: {fm.params[f]:.6f} "
              f"(NW t={fm.tstats[f]:.2f}, p={fm.pvalues[f]:.4f})")
    print(f"  n_significant: {sig_count}")
    print(f"  n_insignificant: {insig_count}")

    # ── PLOT ───────────────────────────────────────
    fig.savefig(CACHE_DIR / "ex2_factor_premia.png",
                dpi=150, bbox_inches="tight")
    print(f"  ── plot: ex2_factor_premia.png ──")
    print(f"     type: bar chart of t-statistics")
    print(f"     n_bars: {len(factors)}")
    print(f"     title: {ax.get_title()}")
    print(f"✓ ex2_factor_premia: ALL PASSED")
