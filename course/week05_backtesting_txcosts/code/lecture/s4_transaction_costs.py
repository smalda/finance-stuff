"""Section 4: Transaction Costs — From Gross to Net Returns"""
import matplotlib
matplotlib.use("Agg")
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

_CODE_DIR = Path(__file__).resolve().parent.parent
_COURSE_DIR = _CODE_DIR.parent.parent
sys.path.insert(0, str(_CODE_DIR))
sys.path.insert(0, str(_COURSE_DIR))
from data_setup import (
    CACHE_DIR,
    PLOT_DIR,
    load_ls_portfolio,
    load_ohlcv_data,
    load_mcap_tiers,
)

# ── Load portfolio data ───────────────────────────────────────────────────────

ls = load_ls_portfolio()
gross_returns = ls["gross_returns"]
turnover = ls["turnover"]
weights_df = ls["weights"]

# Align gross_returns and turnover on common dates
common_dates = gross_returns.index.intersection(turnover.index)
gross_returns = gross_returns.loc[common_dates]
turnover = turnover.loc[common_dates]


# ── CELL: turnover_analysis ───────────────────────────────────────────────────

mean_turnover = turnover.mean()
max_turnover = turnover.max()

print(f"Mean one-way monthly turnover: {mean_turnover:.1%}")
print(f"Max one-way monthly turnover:  {max_turnover:.1%}")

if mean_turnover > 0.50:
    drag_monthly = mean_turnover * 2 * (5 / 10_000)
    drag_annual = drag_monthly * 12
    print(f"⚠ HIGH TURNOVER: {mean_turnover:.0%} one-way — "
          f"TC drag ≈ {drag_annual:.2%}/year at 5 bps (optimistic)")


# ── CELL: spread_cost_func ───────────────────────────────────────────────────

def net_returns_from_spread(gross: pd.Series, to: pd.Series,
                             cost_bps: float) -> pd.Series:
    """Subtract round-trip spread cost from gross returns."""
    cost_frac = cost_bps / 10_000
    costs = to * 2 * cost_frac
    return (gross - costs).rename("net_return")


# ── CELL: spread_regime_zero_fixed ───────────────────────────────────────────

# Regime (a): zero TC — gross returns unchanged
returns_zero = gross_returns.rename("zero_tc")

# Regime (b): optimistic fixed spread — 5 bps (best-in-class execution)
returns_fixed = net_returns_from_spread(gross_returns, turnover, cost_bps=5.0)


# ── CELL: spread_regime_tiered ───────────────────────────────────────────────

# Regime (c): market-cap tiered spread — realistic per-stock liquidity costs
#   Large-cap: 10 bps (liquid S&P components), Mid-cap: 20 bps, Small-cap: 30 bps.
#   These represent realistic all-in one-way costs including commissions.
mcap_tiers = load_mcap_tiers()
spread_map = {"large": 10.0, "mid": 20.0, "small": 30.0}

# Compute weighted-average spread for each rebalance period
tiered_spreads = []
for date in common_dates:
    if date not in weights_df.index:
        tiered_spreads.append(np.nan)
        continue
    w = weights_df.loc[date]
    active_tickers = w[w.abs() > 1e-8].index
    spreads = active_tickers.map(
        lambda t: spread_map.get(mcap_tiers.get(t, "mid"), 15.0)
    )
    avg_spread = np.mean(spreads) if len(spreads) > 0 else 15.0
    tiered_spreads.append(avg_spread)

tiered_spread_series = pd.Series(tiered_spreads, index=common_dates)


# ── CELL: spread_regime_tiered_apply ─────────────────────────────────────────

# Apply tiered spread cost period-by-period
tiered_costs = turnover * 2 * (tiered_spread_series / 10_000)
returns_tiered = (gross_returns - tiered_costs).rename("tiered_tc")

# Annualized spread drag for regime (b): fixed 5 bps (one-way)
spread_drag_fixed = (turnover * 2 * (5 / 10_000)).mean() * 12


# ── CELL: market_impact_one_period ───────────────────────────────────────────

ASSUMED_AUM = 100_000_000  # $100M — standard research portfolio assumption

def _compute_impact_one_period(
    weights_df, vol_30d, dollar_vol, date, prev_date, i
):
    """Compute Almgren-Chriss sqrt-law impact for one rebalance period.

    Returns dict with keys 'date' and 'impact_cost'.
    """
    delta_w = (weights_df.iloc[i] - weights_df.iloc[i - 1]).abs()
    active = delta_w[delta_w > 1e-8].index

    if len(active) == 0:
        return {"date": date, "impact_cost": 0.0}

    # Find nearest trading day at or before rebalance date
    available_days = vol_30d.index[vol_30d.index <= date]
    if len(available_days) == 0:
        return {"date": date, "impact_cost": 0.0}
    nearest_day = available_days[-1]

    sigma = vol_30d.loc[nearest_day, active].dropna()
    adv = dollar_vol.loc[nearest_day, active].dropna()
    common_t = sigma.index.intersection(adv.index)

    if len(common_t) == 0:
        return {"date": date, "impact_cost": 0.0}

    dw = delta_w[common_t]
    sig = sigma[common_t]
    adv_t = adv[common_t].replace(0, np.nan).dropna()
    common_t2 = dw.index.intersection(sig.index).intersection(adv_t.index)

    if len(common_t2) == 0:
        return {"date": date, "impact_cost": 0.0}

    # participation_rate = (|Δw| × AUM) / ADV
    trade_dollars = dw[common_t2] * ASSUMED_AUM
    participation = (trade_dollars / adv_t[common_t2]).clip(upper=1.0)
    impact_per_stock = 0.1 * sig[common_t2] * np.sqrt(participation)
    # Weight impact by trade size (|Δw|)
    total_impact = float((impact_per_stock * dw[common_t2]).sum())
    return {"date": date, "impact_cost": total_impact}


# ── CELL: market_impact_func ─────────────────────────────────────────────────

def compute_market_impact(
    weights_df: pd.DataFrame,
    ohlcv: pd.DataFrame,
    impact_coeff: float = 0.1,
) -> pd.Series:
    """Almgren-Chriss sqrt-law market impact: eta x sigma x sqrt(participation_rate).

    participation_rate = (|dw| x AUM) / ADV_dollars, clipped to [0, 1].
    Total impact = sum_i impact_per_stock_i x |dw_i|.

    Args:
        weights_df: (dates x tickers) portfolio weight matrix.
        ohlcv: MultiIndex DataFrame (field, ticker) daily OHLCV.
        impact_coeff: eta in the impact formula (default 0.1).

    Returns:
        Series of total market impact cost per rebalance period.
    """
    try:
        close = ohlcv.xs("Close", level=0, axis=1)
        volume = ohlcv.xs("Volume", level=0, axis=1)
    except KeyError:
        return pd.Series(0.0, index=weights_df.index[1:], name="impact_cost")

    # Daily dollar volume (30-day rolling average)
    dollar_vol = (close * volume).rolling(30).mean()

    # Daily return volatility (30-day rolling std)
    daily_ret = close.pct_change()
    vol_30d = daily_ret.rolling(30).std()

    dates = weights_df.index
    impact_list = [
        _compute_impact_one_period(weights_df, vol_30d, dollar_vol,
                                   dates[i], dates[i - 1], i)
        for i in range(1, len(dates))
    ]

    if not impact_list:
        return pd.Series(dtype=float, name="impact_cost")
    df = pd.DataFrame(impact_list).set_index("date")
    df.index = pd.DatetimeIndex(df.index)
    return df["impact_cost"]


# ── CELL: market_impact_run ──────────────────────────────────────────────────

print("Computing market impact costs from OHLCV data...")
ohlcv = load_ohlcv_data()
impact_costs = compute_market_impact(weights_df, ohlcv)

# Align with common_dates
impact_costs = impact_costs.reindex(common_dates).fillna(0.0)
mean_monthly_impact = impact_costs.mean()
print(f"Mean monthly market impact cost: {mean_monthly_impact:.4%}")


# ── CELL: equity_curve_plot ───────────────────────────────────────────────────

cum_zero = (1 + returns_zero).cumprod()
cum_fixed = (1 + returns_fixed).cumprod()
cum_tiered = (1 + returns_tiered).cumprod()

fig1, ax1 = plt.subplots(figsize=(12, 6))
ax1.plot(cum_zero.index, cum_zero.values, label="Zero TC (gross)", linewidth=2)
ax1.plot(cum_fixed.index, cum_fixed.values, label="Fixed 5 bps spread (optimistic)", linewidth=2)
ax1.plot(cum_tiered.index, cum_tiered.values,
         label="Tiered spread (10/20/30 bps by cap)", linewidth=2)
ax1.set(
    title="Transaction Cost Regimes: Cumulative Returns",
    xlabel="Date",
    ylabel="Cumulative Return ($ per $1 invested)",
)
ax1.legend()
ax1.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()


# ── CELL: tc_decomposition_bar ────────────────────────────────────────────────

# Annual average cost components (regime b: 5 bps fixed)
annual_spread_cost = (turnover * 2 * (5 / 10_000)).mean() * 12
annual_impact_cost = impact_costs.mean() * 12

fig2, ax2 = plt.subplots(figsize=(7, 5))
components = ["Spread Cost (5 bps fixed)", "Market Impact (η=0.1)"]
values = [annual_spread_cost * 100, annual_impact_cost * 100]  # in percent
bars = ax2.bar(components, values, color=["steelblue", "darkorange"])
ax2.bar_label(bars, fmt="%.2f%%")
ax2.set(
    title="Annual TC Decomposition (Average Month × 12)",
    xlabel="Cost Component",
    ylabel="Annual Cost (%)",
)
plt.tight_layout()
plt.show()


# ── CELL: participation_rate_data ────────────────────────────────────────────

# Market impact as function of participation rate: η × σ × sqrt(p_rate)
participation_rates = np.linspace(0.01, 0.20, 50)
# Use median daily volatility across the portfolio as representative σ
try:
    close_prices = ohlcv.xs("Close", level=0, axis=1)
    median_sigma = close_prices.pct_change().std().median()
except (KeyError, IndexError) as exc:
    import warnings
    warnings.warn(f"Could not extract close prices for participation chart: {exc}")
    median_sigma = 0.015  # fallback: ~1.5% daily vol

eta = 0.1
impact_curve = eta * median_sigma * np.sqrt(participation_rates)


# ── CELL: market_impact_participation ────────────────────────────────────────

fig3, ax3 = plt.subplots(figsize=(8, 5))
ax3.plot(participation_rates * 100, impact_curve * 100, color="darkorange", linewidth=2)
ax3.axvline(x=5, color="gray", linestyle="--", alpha=0.7, label="5% participation")
ax3.set(
    title=f"Market Impact vs. Participation Rate (η={eta}, σ={median_sigma:.3f})",
    xlabel="Participation Rate (%)",
    ylabel="Market Impact Cost (%)",
)
ax3.legend()
ax3.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()


# ── Save caches for downstream files ─────────────────────────────────────────

# tc_results.parquet: combines gross, net, turnover, spread/impact costs
# net_return_tiered subtracts BOTH spread AND impact costs
returns_tiered_with_impact = returns_tiered - impact_costs
tc_df = pd.DataFrame({
    "gross_return": returns_zero,
    "net_return_fixed": returns_fixed,
    "net_return_tiered": returns_tiered_with_impact,
    "turnover": turnover,
    "spread_cost_fixed": turnover * 2 * (5 / 10_000),
    "impact_cost": impact_costs,
}, index=common_dates)
tc_df.to_parquet(CACHE_DIR / "tc_results.parquet")

# ls_portfolio.parquet: weights + gross returns (already in data_setup but for downstream)
ls_portfolio_df = pd.DataFrame({
    "gross_return": returns_zero,
    "turnover": turnover,
}, index=common_dates)
ls_portfolio_df.to_parquet(CACHE_DIR / "ls_portfolio.parquet")


if __name__ == "__main__":
    from shared.backtesting import sharpe_ratio

    # ── ASSERTIONS ───────────────────────────────────────────────────────────
    # S4-1: Monthly one-way turnover ∈ [30%, 200%]
    assert 0.30 <= mean_turnover <= 2.00, (
        f"S4-1: mean monthly turnover {mean_turnover:.4f} outside [0.30, 2.00]"
    )

    # S4-2: Annualized spread drag ∈ [0.5%, 4.0%]
    assert 0.005 <= spread_drag_fixed <= 0.040, (
        f"S4-2: annualized spread drag {spread_drag_fixed:.4f} outside [0.005, 0.040]"
    )

    # S4-3: Market impact cost > 0
    assert mean_monthly_impact > 0.0, (
        f"S4-3: mean monthly impact cost {mean_monthly_impact:.8f} not > 0"
    )

    # S4-4: Net Sharpe < Gross Sharpe
    gross_sharpe = sharpe_ratio(returns_zero, periods_per_year=12)
    net_sharpe_fixed = sharpe_ratio(returns_fixed, periods_per_year=12)
    assert net_sharpe_fixed < gross_sharpe, (
        f"S4-4: net Sharpe {net_sharpe_fixed:.4f} >= gross Sharpe {gross_sharpe:.4f}"
    )

    # S4-5: Regime ordering — zero > fixed > tiered at end of period
    assert float(cum_zero.iloc[-1]) >= float(cum_fixed.iloc[-1]), (
        f"S4-5: zero cumret {cum_zero.iloc[-1]:.4f} < fixed {cum_fixed.iloc[-1]:.4f}"
    )
    assert float(cum_fixed.iloc[-1]) >= float(cum_tiered.iloc[-1]), (
        f"S4-5: fixed cumret {cum_fixed.iloc[-1]:.4f} < tiered {cum_tiered.iloc[-1]:.4f}"
    )

    # S4-6: HIGH TURNOVER warning printed (structural — already printed above if met)
    high_turnover_flag = mean_turnover > 0.50

    # S4-7: Skewness + excess kurtosis reported (structural — printed in RESULTS below)
    gross_skew = float(stats.skew(returns_zero.dropna()))
    gross_kurt = float(stats.kurtosis(returns_zero.dropna(), fisher=True))
    net_skew = float(stats.skew(returns_fixed.dropna()))
    net_kurt = float(stats.kurtosis(returns_fixed.dropna(), fisher=True))

    # ── RESULTS ──────────────────────────────────────────────────────────────
    net_sharpe_tiered = sharpe_ratio(returns_tiered, periods_per_year=12)
    gross_mdd = float(returns_zero.add(1).cumprod().pipe(
        lambda c: (c / c.cummax() - 1).min()
    ))
    net_mdd = float(returns_fixed.add(1).cumprod().pipe(
        lambda c: (c / c.cummax() - 1).min()
    ))

    print(f"══ lecture/s4_transaction_costs ════════════════════════")
    print(f"  n_periods: {len(common_dates)}")
    print(f"  mean_monthly_turnover: {mean_turnover:.4f}")
    print(f"  max_monthly_turnover: {max_turnover:.4f}")
    print(f"  spread_drag_fixed_annual: {spread_drag_fixed:.4f}")
    print(f"  mean_monthly_impact: {mean_monthly_impact:.6f}")
    print(f"  annual_spread_cost_pct: {annual_spread_cost * 100:.4f}")
    print(f"  annual_impact_cost_pct: {annual_impact_cost * 100:.4f}")
    print(f"  gross_sharpe: {gross_sharpe:.4f}")
    print(f"  net_sharpe_fixed_5bps: {net_sharpe_fixed:.4f}")
    print(f"  net_sharpe_tiered: {net_sharpe_tiered:.4f}")
    print(f"  gross_max_drawdown: {gross_mdd:.4f}")
    print(f"  net_max_drawdown: {net_mdd:.4f}")
    print(f"  gross_skewness: {gross_skew:.4f}")
    print(f"  gross_excess_kurtosis: {gross_kurt:.4f}")
    print(f"  net_skewness: {net_skew:.4f}")
    print(f"  net_excess_kurtosis: {net_kurt:.4f}")
    print(f"  cum_return_zero: {float(cum_zero.iloc[-1]):.4f}")
    print(f"  cum_return_fixed: {float(cum_fixed.iloc[-1]):.4f}")
    print(f"  cum_return_tiered: {float(cum_tiered.iloc[-1]):.4f}")
    if high_turnover_flag:
        print(f"  high_turnover_warning: TRUE")

    # ── PLOTS ─────────────────────────────────────────────────────────────────
    fig1.savefig(PLOT_DIR / "s4_equity_curves.png", dpi=150, bbox_inches="tight")
    print(f"  ── plot: s4_equity_curves.png ──")
    print(f"     type: multi-line cumulative returns (3 TC regimes)")
    print(f"     n_lines: {len(ax1.get_lines())}")
    print(f"     y_range: [{ax1.get_ylim()[0]:.2f}, {ax1.get_ylim()[1]:.2f}]")
    print(f"     title: {ax1.get_title()}")

    fig2.savefig(PLOT_DIR / "s4_tc_decomposition.png", dpi=150, bbox_inches="tight")
    print(f"  ── plot: s4_tc_decomposition.png ──")
    print(f"     type: bar chart (spread vs impact cost)")
    print(f"     n_bars: {len(ax2.patches)}")
    print(f"     title: {ax2.get_title()}")

    fig3.savefig(
        PLOT_DIR / "s4_impact_participation.png", dpi=150, bbox_inches="tight"
    )
    print(f"  ── plot: s4_impact_participation.png ──")
    print(f"     type: line chart (market impact vs participation rate)")
    print(f"     n_lines: {len(ax3.get_lines())}")
    print(f"     title: {ax3.get_title()}")

    print(f"✓ s4_transaction_costs: ALL PASSED")
