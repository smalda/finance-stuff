"""HW D2: Transaction Cost Pipeline — TransactionCostModel class.

Builds a production-grade TC model from scratch. Covers spread cost,
market impact (Almgren-Chriss sqrt law), and three spread regimes.

Three regimes:
  optimistic  — 3 bps flat (large liquid names only)
  base        — market-cap tiered: large-cap 5 bps, mid/small-cap 15 bps
  pessimistic — 25 bps flat (conservative / illiquid execution)
"""
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
from data_setup import CACHE_DIR, PLOT_DIR, load_ohlcv_data, load_mcap_tiers

SEED = 42
ASSUMED_AUM = 100_000_000  # $100M — standard research portfolio assumption
DEFAULT_DAILY_VOL = 0.01   # ~1% daily vol fallback for missing tickers


# ── CELL: tc_model_init ──────────────────────────────────────────────

class TransactionCostModel:
    """Transaction cost model: spread + market impact (Almgren-Chriss).

    Spread = sum(|dw|) * half_spread_bps / 10_000.
    Impact = eta * sigma * sqrt(participation) per stock.
    """

    def __init__(
        self,
        weights: pd.DataFrame,
        spread_bps: float | pd.Series,
        impact_coeff: float = 0.1,
    ) -> None:
        self.weights = weights.copy()
        self.spread_bps = spread_bps
        self.impact_coeff = impact_coeff
        self.daily_vol: pd.DataFrame | None = None
        self.adv: pd.DataFrame | None = None
        self.turnover: pd.Series | None = None
        self.spread_cost: pd.Series | None = None
        self.impact_cost: pd.Series | None = None
        self.net_returns: pd.Series | None = None


# ── CELL: tc_model_fit ───────────────────────────────────────────────

def fit(self, ohlcv: pd.DataFrame) -> "TransactionCostModel":
    """Precompute 30-day rolling daily vol and ADV from OHLCV."""
    close = ohlcv["Close"]
    volume = ohlcv["Volume"]

    daily_ret = close.pct_change()
    self.daily_vol = daily_ret.rolling(30, min_periods=10).std()

    dollar_vol = close * volume
    self.adv = dollar_vol.rolling(30, min_periods=10).mean()

    return self

TransactionCostModel.fit = fit


# ── CELL: tc_compute_spread_cost ─────────────────────────────────────

def _compute_spread_cost(abs_delta, half_spread):
    """Compute spread cost from absolute weight changes."""
    if np.isscalar(half_spread):
        return abs_delta.sum() * half_spread / 10_000
    spread_per_stock = (
        half_spread.reindex(abs_delta.index).fillna(half_spread.mean())
    )
    return (abs_delta * spread_per_stock).sum() / 10_000


# ── CELL: tc_compute_impact_cost ─────────────────────────────────────

def _compute_impact_cost(abs_delta, date, daily_vol, adv, impact_coeff):
    """Compute Almgren-Chriss market impact for one period."""
    if daily_vol is None or adv is None:
        return 0.0

    active = abs_delta[abs_delta > 1e-6].index
    if len(active) == 0:
        return 0.0

    vol_snap = (
        daily_vol.loc[:date].tail(1).squeeze()
        if date in daily_vol.index
        else daily_vol.iloc[-1]
    )
    vol_snap = vol_snap.reindex(active).fillna(DEFAULT_DAILY_VOL)

    adv_snap = (
        adv.loc[:date].tail(1).squeeze()
        if date in adv.index
        else adv.iloc[-1]
    )
    adv_snap = adv_snap.reindex(active).fillna(adv_snap.median())
    adv_snap = adv_snap.replace(0, np.nan).fillna(adv_snap.median())

    trade_dollars = abs_delta[active] * ASSUMED_AUM
    participation = (trade_dollars / adv_snap).clip(upper=1.0)
    impact_per_stock = impact_coeff * vol_snap * np.sqrt(participation)
    return float((impact_per_stock * abs_delta[active]).sum())


# ── CELL: tc_period_step ──────────────────────────────────────────────

def _compute_period_step(date, weights, half_spread, daily_vol, adv,
                         impact_coeff, gross_ret):
    """Compute turnover, costs, and net return for a single period."""
    prev_w = weights.loc[weights.index[weights.index.get_loc(date) - 1]]
    curr_w = weights.loc[date]
    delta_w = (curr_w - prev_w).reindex(weights.columns).fillna(0.0)
    abs_delta = delta_w.abs()
    turnover = abs_delta.sum() / 2.0

    spread_cost = _compute_spread_cost(abs_delta, half_spread)
    impact_cost = _compute_impact_cost(
        abs_delta, date, daily_vol, adv, impact_coeff,
    )
    net_ret = gross_ret - spread_cost - impact_cost
    return turnover, spread_cost, impact_cost, net_ret


# ── CELL: tc_to_series ───────────────────────────────────────────────

def _to_series(lst, col):
    """Convert list of dicts to a DatetimeIndex Series."""
    df = pd.DataFrame(lst).set_index("date")
    df.index = pd.DatetimeIndex(df.index)
    return df[col]


# ── CELL: tc_model_transform ─────────────────────────────────────────

def transform(self, gross_returns: pd.Series) -> "TransactionCostModel":
    """Compute per-period TC components and net returns."""
    if self.daily_vol is None:
        import warnings
        warnings.warn(
            "TransactionCostModel.transform() called without fit(). "
            "Market impact will be zero (no vol/ADV data)."
        )
    weights = self.weights
    trade_dates = weights.index[1:]
    common_dates = trade_dates.intersection(gross_returns.index)

    turnover_vals, spread_vals, impact_vals, net_vals = [], [], [], []

    for date in common_dates:
        to, sc, ic, nr = _compute_period_step(
            date, weights, self.spread_bps,
            self.daily_vol, self.adv, self.impact_coeff,
            gross_returns.loc[date],
        )
        turnover_vals.append({"date": date, "turnover": to})
        spread_vals.append({"date": date, "spread_cost": sc})
        impact_vals.append({"date": date, "impact_cost": ic})
        net_vals.append({"date": date, "net_return": nr})

    self.turnover = _to_series(turnover_vals, "turnover")
    self.spread_cost = _to_series(spread_vals, "spread_cost")
    self.impact_cost = _to_series(impact_vals, "impact_cost")
    self.net_returns = _to_series(net_vals, "net_return")
    return self

TransactionCostModel.transform = transform


# ── CELL: tc_model_report ────────────────────────────────────────────

def report(self, label: str = "regime") -> None:
    """Print structured TC summary; flag high-turnover periods."""
    from shared.backtesting import sharpe_ratio, max_drawdown

    mean_to = self.turnover.mean()
    ann_net_sharpe = sharpe_ratio(self.net_returns, periods_per_year=12)
    ann_net_ret = self.net_returns.mean() * 12
    mdd = max_drawdown(self.net_returns)

    print(f"  TC Report [{label}]")
    print(f"    Mean one-way monthly turnover : {mean_to:.1%}")
    if mean_to > 0.50:
        print(f"    \u26a0 HIGH TURNOVER: {mean_to:.1%} \u2014 TC drag is material")
    print(f"    Annualized net return         : {ann_net_ret:.2%}")
    print(f"    Annualized net Sharpe         : {ann_net_sharpe:.3f}")
    print(f"    Max drawdown (net)             : {mdd:.2%}")
    print(f"    Mean monthly spread cost      : {self.spread_cost.mean():.6f}")
    print(f"    Mean monthly impact cost      : {self.impact_cost.mean():.6f}")

    # Top-5 highest-cost months
    total_tc = self.spread_cost + self.impact_cost
    top5 = total_tc.nlargest(5)
    print(f"    Top-5 highest-cost months:")
    for dt, cost in top5.items():
        dom = "spread" if self.spread_cost[dt] >= self.impact_cost[dt] else "impact"
        print(f"      {dt.date()}: total={cost:.6f}  dominant={dom}")

TransactionCostModel.report = report


# ── CELL: build_spread_series ────────────────────────────────────────

def build_tiered_spread(mcap_tiers: pd.Series, tickers: list) -> pd.Series:
    """Map market-cap tier to half-spread in bps."""
    tier_map = {"large": 5.0, "mid": 15.0, "small": 15.0}
    spread = mcap_tiers.reindex(tickers).map(tier_map).fillna(15.0)
    return spread


# ── CELL: correctness_check ─────────────────────────────────────────

def correctness_check(spread_bps: float = 10.0) -> dict:
    """2-asset [1,0]->[0,1] analytical check for TransactionCostModel."""
    dates = pd.date_range("2020-01-31", periods=2, freq="ME")
    w = pd.DataFrame(
        {"ASSET_A": [1.0, 0.0], "ASSET_B": [0.0, 1.0]},
        index=dates,
    )
    gross = pd.Series([0.01], index=dates[1:])

    tcm = TransactionCostModel(weights=w, spread_bps=spread_bps)
    tcm.transform(gross)

    expected_spread = 2.0 * spread_bps / 10_000
    actual_turnover = float(tcm.turnover.iloc[0])
    actual_spread = float(tcm.spread_cost.iloc[0])
    match = abs(actual_spread - expected_spread) < 1e-10

    return {
        "turnover": actual_turnover,
        "spread_cost": actual_spread,
        "expected_spread_cost": expected_spread,
        "match": match,
    }


# ── CELL: run_three_regimes ─────────────────────────────────────────

def run_three_regimes(
    weights: pd.DataFrame,
    gross_returns: pd.Series,
    ohlcv: pd.DataFrame,
    mcap_tiers: pd.Series,
) -> dict:
    """Run TransactionCostModel across three spread regimes."""
    tickers = weights.columns.tolist()
    tiered_spread = build_tiered_spread(mcap_tiers, tickers)

    regimes = {
        "optimistic": 3.0,
        "base": tiered_spread,
        "pessimistic": 25.0,
    }

    results = {}
    for name, spread in regimes.items():
        print(f"  Running TC model [{name}]...")
        tcm = TransactionCostModel(weights=weights, spread_bps=spread)
        tcm.fit(ohlcv)
        tcm.transform(gross_returns)
        results[name] = tcm

    return results


# ── CELL: plot_equity_curves ─────────────────────────────────────────

def plot_equity_curves(
    gross_returns: pd.Series,
    regimes: dict,
) -> tuple:
    """Gross vs. net equity curves across three spread regimes."""
    fig, ax = plt.subplots(figsize=(11, 5))

    gross_cum = (1 + gross_returns).cumprod()
    ax.plot(gross_cum.index, gross_cum.values, "k-", linewidth=2, label="Gross")

    colors = {"optimistic": "#2ca02c", "base": "#1f77b4", "pessimistic": "#d62728"}
    for name, tcm in regimes.items():
        net_cum = (1 + tcm.net_returns).cumprod()
        ax.plot(net_cum.index, net_cum.values, color=colors[name],
                linewidth=1.5, label=f"Net ({name})")

    ax.set(
        title="Gross vs. Net Equity Curves \u2014 Three Spread Regimes",
        xlabel="Date",
        ylabel="Cumulative Return (starting at 1.0)",
    )
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    return fig, ax


# ── CELL: plot_tc_decomposition ──────────────────────────────────────

def plot_tc_decomposition(regime: "TransactionCostModel", label: str) -> tuple:
    """Stacked area chart: spread cost + impact cost per month."""
    fig, ax = plt.subplots(figsize=(11, 4))

    dates = regime.spread_cost.index
    spread = regime.spread_cost.values
    impact = regime.impact_cost.values

    ax.stackplot(
        dates,
        spread,
        impact,
        labels=["Spread cost", "Market impact"],
        colors=["#1f77b4", "#ff7f0e"],
        alpha=0.8,
    )
    ax.set(
        title=f"TC Decomposition Over Time \u2014 {label}",
        xlabel="Date",
        ylabel="Monthly TC drag (decimal)",
    )
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    return fig, ax


# ── CELL: plot_top5_cost_months ──────────────────────────────────────

def plot_top5_cost_months(regime: "TransactionCostModel", label: str) -> tuple:
    """Horizontal bar chart of top-5 highest-cost months."""
    total_tc = regime.spread_cost + regime.impact_cost
    top5 = total_tc.nlargest(5).sort_values()

    spread_top5 = regime.spread_cost.reindex(top5.index)
    impact_top5 = regime.impact_cost.reindex(top5.index)

    month_labels = [d.strftime("%Y-%m") for d in top5.index]

    fig, ax = plt.subplots(figsize=(8, 4))
    y_pos = range(len(top5))
    ax.barh(y_pos, spread_top5.values, color="#1f77b4", label="Spread")
    ax.barh(y_pos, impact_top5.values, left=spread_top5.values,
            color="#ff7f0e", label="Impact")
    ax.set_yticks(list(y_pos))
    ax.set_yticklabels(month_labels)
    ax.set(
        title=f"Top-5 Highest-Cost Months \u2014 {label}",
        xlabel="Monthly TC drag (decimal)",
    )
    ax.legend()
    plt.tight_layout()
    plt.show()
    return fig, ax


if __name__ == "__main__":
    from shared.backtesting import sharpe_ratio, max_drawdown
    from scipy.stats import skew, kurtosis

    print("Loading data...")
    ohlcv = load_ohlcv_data()
    mcap_tiers = load_mcap_tiers()

    # Load weights and gross returns from upstream s4 cache
    ls_pf = pd.read_parquet(CACHE_DIR / "ls_portfolio.parquet")
    weights = pd.read_parquet(CACHE_DIR / "ls_weights.parquet")
    gross_returns = ls_pf["gross_return"]

    print(f"  Weights shape: {weights.shape}")
    print(f"  Gross returns: {len(gross_returns)} months "
          f"({gross_returns.index[0].date()} to {gross_returns.index[-1].date()})")
    print(f"  Mean one-way monthly turnover: "
          f"{ls_pf['turnover'].mean():.1%}")

    if ls_pf["turnover"].mean() > 0.50:
        drag_est = ls_pf["turnover"].mean() * 2 * (10 / 10_000) * 12
        print(f"  \u26a0 HIGH TURNOVER: {ls_pf['turnover'].mean():.1%} one-way \u2014 "
              f"TC drag \u2248 {drag_est:.2%}/year at 10 bps")

    # ── CORRECTNESS CHECK ──────────────────────────────────────────
    print("\nRunning correctness check...")
    chk = correctness_check(spread_bps=10.0)
    assert chk["match"], (
        f"Correctness check FAILED: expected spread_cost="
        f"{chk['expected_spread_cost']:.8f}, "
        f"got {chk['spread_cost']:.8f}"
    )
    print(f"  Correctness check: PASS")
    print(f"    2-asset [1,0]\u2192[0,1]: turnover={chk['turnover']:.4f} "
          f"(expected 1.0), spread_cost={chk['spread_cost']:.8f} "
          f"(expected {chk['expected_spread_cost']:.8f})")

    # ── THREE-REGIME TC MODEL ──────────────────────────────────────
    print("\nRunning three-regime TC pipeline...")
    regimes = run_three_regimes(weights, gross_returns, ohlcv, mcap_tiers)

    for name, tcm in regimes.items():
        tcm.report(label=name)

    # ── ASSERTIONS ────────────────────────────────────────────────
    print("\nRunning assertions...")

    # D2-1: correctness check already passed above

    # D2-2: mean monthly one-way turnover ∈ [10%, 200%]
    mean_to = regimes["base"].turnover.mean()
    assert 0.10 <= mean_to <= 2.00, (
        f"D2-2: Mean monthly turnover {mean_to:.4f} outside [0.10, 2.00]"
    )

    # D2-3: spread cost series all ≥ 0
    for name, tcm in regimes.items():
        neg_spread = (tcm.spread_cost < -1e-12).sum()
        assert neg_spread == 0, (
            f"D2-3: Regime '{name}' has {neg_spread} negative spread cost values"
        )

    # D2-4: Sharpe ordering: pessimistic < base < optimistic
    gross_sharpe = sharpe_ratio(gross_returns, periods_per_year=12)
    sharpe_opt = sharpe_ratio(regimes["optimistic"].net_returns, periods_per_year=12)
    sharpe_base = sharpe_ratio(regimes["base"].net_returns, periods_per_year=12)
    sharpe_pess = sharpe_ratio(regimes["pessimistic"].net_returns, periods_per_year=12)

    assert sharpe_pess < sharpe_base, (
        f"D2-4: pessimistic Sharpe {sharpe_pess:.4f} should be < "
        f"base Sharpe {sharpe_base:.4f}"
    )
    assert sharpe_base < sharpe_opt, (
        f"D2-4: base Sharpe {sharpe_base:.4f} should be < "
        f"optimistic Sharpe {sharpe_opt:.4f}"
    )

    # D2-5: max drawdown net ≥ max drawdown gross (TC never reduces drawdown)
    mdd_gross = max_drawdown(gross_returns)
    mdd_base = max_drawdown(regimes["base"].net_returns)
    assert mdd_base <= mdd_gross, (
        f"D2-5: Net MDD {mdd_base:.4f} should be \u2264 gross MDD {mdd_gross:.4f}"
    )

    # ── PLOTS ─────────────────────────────────────────────────────
    fig_eq, ax_eq = plot_equity_curves(gross_returns, regimes)
    fig_eq.savefig(PLOT_DIR / "d2_equity_curves.png", dpi=150, bbox_inches="tight")

    fig_dc, ax_dc = plot_tc_decomposition(regimes["base"], label="Base")
    fig_dc.savefig(PLOT_DIR / "d2_tc_decomposition.png", dpi=150, bbox_inches="tight")

    fig_t5, ax_t5 = plot_top5_cost_months(regimes["base"], label="Base")
    fig_t5.savefig(PLOT_DIR / "d2_top5_cost_months.png", dpi=150, bbox_inches="tight")

    # ── TC MODEL RESULTS CACHE ────────────────────────────────────
    net_opt = regimes["optimistic"].net_returns
    net_base = regimes["base"].net_returns
    net_pess = regimes["pessimistic"].net_returns

    results_df = pd.DataFrame({
        "gross_return": gross_returns,
        "net_return_opt": net_opt,
        "net_return_base": net_base,
        "net_return_pess": net_pess,
        "turnover": regimes["base"].turnover,
        "spread_cost_base": regimes["base"].spread_cost,
        "impact_cost_base": regimes["base"].impact_cost,
    }).dropna()
    results_df.to_parquet(CACHE_DIR / "d2_tc_model_results.parquet")

    # ── MOMENT STATISTICS ─────────────────────────────────────────
    gross_skew = float(skew(gross_returns.dropna()))
    gross_kurt = float(kurtosis(gross_returns.dropna(), fisher=True))
    net_base_skew = float(skew(net_base.dropna()))
    net_base_kurt = float(kurtosis(net_base.dropna(), fisher=True))

    # ── RESULTS ───────────────────────────────────────────────────
    print(f"\n\u2550\u2550 hw/d2_tc_pipeline \u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550")
    print(f"  correctness_check: PASS")
    print(f"  turnover_mean_monthly: {mean_to:.4f}  ({mean_to:.1%})")
    print(f"  GROSS_SHARPE: {gross_sharpe:.4f}")
    print(f"  NET_SHARPE_OPT: {sharpe_opt:.4f}")
    print(f"  NET_SHARPE_BASE: {sharpe_base:.4f}")
    print(f"  NET_SHARPE_PESS: {sharpe_pess:.4f}")
    print(f"  mdd_gross: {mdd_gross:.4f}")
    print(f"  mdd_base_net: {mdd_base:.4f}")
    print(f"  gross_skewness: {gross_skew:.4f}")
    print(f"  gross_excess_kurtosis: {gross_kurt:.4f}")
    print(f"  net_base_skewness: {net_base_skew:.4f}")
    print(f"  net_base_excess_kurtosis: {net_base_kurt:.4f}")
    print(f"  n_months: {len(gross_returns)}")
    print(f"  spread_regimes: optimistic=3bps, base=5/15bps_tiered, pessimistic=25bps")
    print(f"  \u2500\u2500 plot: d2_equity_curves.png \u2500\u2500")
    print(f"     type: multi-line cumulative returns (gross + 3 net regimes)")
    print(f"     n_lines: {len(ax_eq.get_lines())}")
    print(f"     y_range: [{ax_eq.get_ylim()[0]:.2f}, {ax_eq.get_ylim()[1]:.2f}]")
    print(f"     title: {ax_eq.get_title()}")
    print(f"  \u2500\u2500 plot: d2_tc_decomposition.png \u2500\u2500")
    print(f"     type: stacked area (spread + impact per month)")
    print(f"     n_months: {len(regimes['base'].spread_cost)}")
    print(f"     title: {ax_dc.get_title()}")
    print(f"  \u2500\u2500 plot: d2_top5_cost_months.png \u2500\u2500")
    print(f"     type: horizontal bar chart (top-5 cost months)")
    print(f"     title: {ax_t5.get_title()}")
    print(f"\u2713 d2_tc_pipeline: ALL PASSED")
