"""Section 1: The Seven Sins of Backtesting — Look-Ahead Bias & Survivorship Bias"""
import matplotlib
matplotlib.use("Agg")
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

_CODE_DIR = Path(__file__).resolve().parent.parent
_COURSE_DIR = _CODE_DIR.parent.parent
sys.path.insert(0, str(_CODE_DIR))
sys.path.insert(0, str(_COURSE_DIR))
from data_setup import (
    CACHE_DIR,
    PLOT_DIR,
    START,
    END,
    load_equity_data,
    load_monthly_returns,
)

# ── Constants ──────────────────────────────────────────────────────────
IS_START  = "2012-01-01"
IS_END    = "2017-12-31"
OOS_START = "2018-01-01"
OOS_END   = "2024-12-31"
# Annual churn rate used in survivorship simulation (5% of stocks exit per year)
ANNUAL_CHURN_RATE = 0.05
# Return assigned to a stock in the month it delists
DELIST_RETURN = -0.50
SEED = 42


# ── CELL: data_quality_block ──────────────────────────────────────────

monthly_prices  = load_equity_data()
# Compute returns from prices to preserve NaN structure for coverage tracking
monthly_returns_raw = monthly_prices.pct_change()

# Use the cached monthly_returns for IC computation (consistent with rest of course)
monthly_returns = load_monthly_returns()

n_stocks  = monthly_returns.shape[1]
n_periods = monthly_returns.shape[0]
missing_pct = monthly_prices.isnull().mean().mean()

# Survivor count: tickers present in the very first price observation
first_month_prices = monthly_prices.iloc[0]
n_survivors_strict = int(first_month_prices.notna().sum())
n_all              = n_stocks

print("── Data Quality ──────────────────────────────────────────────────")
print(f"  N_stocks:          {n_stocks}")
print(f"  N_periods:         {n_periods} monthly observations")
print(f"  Missing (prices):  {missing_pct:.4%}")
print(f"  Present at start:  {n_survivors_strict} / {n_all} tickers")
print(f"  Survivorship note: S&P 500 universe = current constituents only.")
print(f"                     Delisted stocks not in dataset — bias understated.")
print()


# ── CELL: look_ahead_signal ──────────────────────────────────────────

# Flawed signal: rank by THIS month's return → predicts THIS month's return.
# This is a pure look-ahead bug — the signal equals the outcome variable.
flawed_signal = monthly_returns.rank(axis=1, pct=True)

# Corrected signal: rank by PREVIOUS month's return → predicts THIS month's return.
# Only T-1 information is used: no future leakage.
corrected_signal = monthly_returns.shift(1).rank(axis=1, pct=True)


# ── CELL: monthly_ic_func ─────────────────────────────────────────────

def monthly_ic(signal: pd.DataFrame, returns: pd.DataFrame) -> pd.Series:
    """Cross-sectional Spearman IC for each date in signal's index."""
    ic_vals = {}
    for date in signal.index:
        sig = signal.loc[date].dropna()
        ret = returns.loc[date].dropna()
        common = sig.index.intersection(ret.index)
        if len(common) < 10:
            continue
        corr, _ = spearmanr(sig[common], ret[common])
        ic_vals[date] = corr
    return pd.Series(ic_vals, name="ic")


# ── CELL: compute_oos_ic ──────────────────────────────────────────────

# IS and OOS masks
oos_mask = (monthly_returns.index >= OOS_START) & (monthly_returns.index <= OOS_END)
oos_returns = monthly_returns.loc[oos_mask]

flawed_ic_oos    = monthly_ic(flawed_signal.loc[oos_mask],    oos_returns)
corrected_ic_oos = monthly_ic(corrected_signal.loc[oos_mask], oos_returns)

mean_flawed_ic_oos    = flawed_ic_oos.mean()
mean_corrected_ic_oos = corrected_ic_oos.mean()

print(f"  Flawed signal OOS IC:    {mean_flawed_ic_oos:.4f}  ← signal == outcome")
print(f"  Corrected signal OOS IC: {mean_corrected_ic_oos:.4f}")


# ── CELL: long_short_func ─────────────────────────────────────────────

def long_short_monthly(
    signal: pd.DataFrame,
    returns: pd.DataFrame,
    n_leg: int = 20,
) -> pd.Series:
    """Equal-weight top-n long / bottom-n short monthly return."""
    port_rets = []
    for date in signal.index:
        sig = signal.loc[date].dropna()
        ret = returns.loc[date].dropna()
        common = sig.index.intersection(ret.index)
        if len(common) < n_leg * 2:
            continue
        ranked = sig[common].rank(ascending=True)
        n = len(ranked)
        long_ret  = ret[common][ranked > (n - n_leg)].mean()
        short_ret = ret[common][ranked <= n_leg].mean()
        port_rets.append({"date": date, "ret": long_ret - short_ret})
    return pd.DataFrame(port_rets).set_index("date")["ret"]


# ── CELL: equity_curves ──────────────────────────────────────────────

flawed_ret_oos    = long_short_monthly(flawed_signal.loc[oos_mask],    oos_returns)
corrected_ret_oos = long_short_monthly(corrected_signal.loc[oos_mask], oos_returns)

ann_flawed    = flawed_ret_oos.mean()    * 12
ann_corrected = corrected_ret_oos.mean() * 12
annual_gap    = ann_flawed - ann_corrected

cum_flawed    = (1 + flawed_ret_oos).cumprod()
cum_corrected = (1 + corrected_ret_oos).cumprod()

print(f"  Flawed ann. return:    {ann_flawed:.2%}")
print(f"  Corrected ann. return: {ann_corrected:.2%}")
print(f"  Annual return gap:     {annual_gap:.2%}")


# ── CELL: equity_curve_plot ──────────────────────────────────────────

fig_eq, ax_eq = plt.subplots(figsize=(10, 5))
ax_eq.semilogy(cum_flawed.index,    cum_flawed.values,    label="Flawed (look-ahead)", linewidth=2)
ax_eq.semilogy(cum_corrected.index, cum_corrected.values, label="Corrected (lagged signal)", linewidth=2)
ax_eq.set(
    title="Equity Curve: Flawed vs. Corrected Signal (OOS 2018–2024, log scale)",
    xlabel="Date",
    ylabel="Cumulative Return (log scale)",
)
ax_eq.legend()
ax_eq.axhline(1.0, color="gray", linestyle="--", linewidth=0.8)
plt.tight_layout()
plt.show()


# ── CELL: survivorship_setup ──────────────────────────────────────────

# Demonstration of survivorship bias using simulated delisting.
#
# The S&P 500 dataset contains only current constituents (already survivorship-
# biased). To quantify the bias, we simulate a realistic universe that includes
# stocks that would have been delisted: each year, ANNUAL_CHURN_RATE of stocks
# receive a one-time delisting return (DELIST_RETURN) then exit. The difference
# between the clean S&P universe and this simulated unbiased universe measures
# the annual survivorship premium.

oos_raw = monthly_returns_raw.loc[OOS_START:OOS_END]

# Survivor portfolio: equal-weight all tickers (already survivorship-biased)
ew_survivors = oos_raw.mean(axis=1)
ann_survivors = ew_survivors.mean() * 12

# Simulated non-biased portfolio: inject delisting events
rng = np.random.default_rng(SEED)
oos_simulated = oos_raw.copy()
n_tickers = oos_raw.shape[1]


# ── CELL: survivorship_simulation ─────────────────────────────────────

for year in oos_raw.index.year.unique():
    year_dates = oos_raw.index[oos_raw.index.year == year]
    if len(year_dates) == 0:
        continue
    n_exit = max(1, int(n_tickers * ANNUAL_CHURN_RATE))
    exit_cols = rng.choice(n_tickers, size=n_exit, replace=False)
    exit_tickers = oos_raw.columns[exit_cols]
    # Month of delisting: large loss, then treated as absent (0 return)
    delist_month = year_dates[0]
    oos_simulated.loc[delist_month, exit_tickers] = DELIST_RETURN
    if len(year_dates) > 1:
        oos_simulated.loc[year_dates[1:], exit_tickers] = 0.0


# ── CELL: survivorship_results ────────────────────────────────────────

ew_all = oos_simulated.mean(axis=1)
ann_all = ew_all.mean() * 12
survivorship_premium = ann_survivors - ann_all

print(f"  Survivor EW annual return:     {ann_survivors:.4%}")
print(f"  Sim unbiased EW annual return: {ann_all:.4%}")
print(f"  Survivorship premium:          {survivorship_premium:.4%} annualized")
print(f"  (Simulates {ANNUAL_CHURN_RATE:.0%}/yr churn, "
      f"{DELIST_RETURN:.0%} delist return — consistent with historical estimates)")


# ── CELL: survivorship_bar_chart ─────────────────────────────────────

fig_sv, ax_sv = plt.subplots(figsize=(7, 5))
labels   = ["Unbiased\n(simulated delisting)", "Survivor-Only\n(S&P 500 universe)"]
returns_ = [ann_all, ann_survivors]
colors   = ["steelblue", "tomato"]

bars = ax_sv.bar(labels, [r * 100 for r in returns_], color=colors, alpha=0.85, width=0.5)
ax_sv.bar_label(bars, fmt="%.1f%%", padding=3)
ax_sv.set(
    title=f"Survivorship Bias: EW Annual Return OOS 2018–2024\n"
          f"(Simulated {ANNUAL_CHURN_RATE:.0%}/yr churn, {DELIST_RETURN:.0%} delist return)",
    ylabel="Annualized Return (%)",
)
ax_sv.axhline(0, color="gray", linewidth=0.8)
plt.tight_layout()
plt.show()


if __name__ == "__main__":
    # ── ASSERTIONS ────────────────────────────────────────────────
    # S1-1: Flawed signal OOS IC must be ≥ 0.30 (look-ahead → signal ≈ outcome)
    assert mean_flawed_ic_oos >= 0.30, (
        f"S1-1: Flawed OOS IC = {mean_flawed_ic_oos:.4f}; expected ≥ 0.30"
    )

    # S1-2: Corrected signal OOS IC ∈ [-0.05, 0.06] (weak momentum)
    assert -0.05 <= mean_corrected_ic_oos <= 0.06, (
        f"S1-2: Corrected OOS IC = {mean_corrected_ic_oos:.4f}; expected ∈ [-0.05, 0.06]"
    )

    # S1-3: Annual return gap ≥ 15 pp (look-ahead premium must be dramatic)
    assert annual_gap >= 0.15, (
        f"S1-3: Annual return gap = {annual_gap:.4f}; expected ≥ 0.15"
    )

    # S1-4: Survivorship premium ∈ [0.5%, 4.0%] and positive
    assert 0.005 <= survivorship_premium <= 0.040, (
        f"S1-4: Survivorship premium = {survivorship_premium:.4f}; expected ∈ [0.005, 0.040]"
    )

    # S1-5: Data quality block — key variables computed and non-trivial
    assert n_stocks > 0,  f"S1-5: n_stocks = {n_stocks}; must be > 0"
    assert n_periods > 0, f"S1-5: n_periods = {n_periods}; must be > 0"
    assert 0.0 <= missing_pct <= 1.0, f"S1-5: missing_pct={missing_pct:.4f}; invalid"
    assert n_survivors_strict > 0, f"S1-5: n_survivors_strict={n_survivors_strict}; must be > 0"

    # ── RESULTS ───────────────────────────────────────────────────
    print(f"══ lecture/s1_seven_sins ════════════════════════════════════")
    print(f"  n_stocks:                 {n_stocks}")
    print(f"  n_periods:                {n_periods}")
    print(f"  missing_pct_prices:       {missing_pct:.6f}")
    print(f"  n_survivors_strict:       {n_survivors_strict}")
    print(f"  mean_flawed_ic_oos:       {mean_flawed_ic_oos:.4f}")
    print(f"  mean_corrected_ic_oos:    {mean_corrected_ic_oos:.4f}")
    print(f"  ann_flawed_return:        {ann_flawed:.4f}")
    print(f"  ann_corrected_return:     {ann_corrected:.4f}")
    print(f"  annual_gap:               {annual_gap:.4f}")
    print(f"  ann_survivors_return:     {ann_survivors:.6f}")
    print(f"  ann_all_return:           {ann_all:.6f}")
    print(f"  survivorship_premium:     {survivorship_premium:.6f}")

    # ── PLOTS ──────────────────────────────────────────────────────
    fig_eq.savefig(PLOT_DIR / "s1_equity_curves.png", dpi=150, bbox_inches="tight")
    print(f"  ── plot: s1_equity_curves.png ──")
    print(f"     type: line chart (cumulative returns, flawed vs corrected)")
    print(f"     n_lines: {len(ax_eq.get_lines())}")
    print(f"     y_range: [{ax_eq.get_ylim()[0]:.2f}, {ax_eq.get_ylim()[1]:.2f}]")
    print(f"     title: {ax_eq.get_title()}")

    fig_sv.savefig(PLOT_DIR / "s1_survivorship_bar.png", dpi=150, bbox_inches="tight")
    print(f"  ── plot: s1_survivorship_bar.png ──")
    print(f"     type: bar chart (annualized return: all vs survivors)")
    print(f"     n_bars: {len(ax_sv.patches)}")
    print(f"     y_range: [{ax_sv.get_ylim()[0]:.2f}, {ax_sv.get_ylim()[1]:.2f}]")
    print(f"     title: {ax_sv.get_title()}")

    print(f"✓ s1_seven_sins: ALL PASSED")
