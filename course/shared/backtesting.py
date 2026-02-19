"""Portfolio construction and backtesting utilities.

Converts cross-sectional predictions into portfolios and computes
standard performance/cost metrics: quantile sorts, long-short returns,
turnover, transaction cost drag, Sharpe ratios.

Error contract: performance metrics return np.nan on degenerate input
(< 2 observations, zero variance). Portfolio construction functions
return empty DataFrames/Series on insufficient data.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Portfolio construction
# ---------------------------------------------------------------------------

def quantile_portfolios(
    predictions: pd.Series,
    returns: pd.Series,
    n_groups: int = 10,
) -> pd.DataFrame:
    """Sort stocks into quantile portfolios each period by predicted signal.

    Args:
        predictions: Series with MultiIndex (date, ticker).
        returns: Series with MultiIndex (date, ticker) — realized returns.
        n_groups: number of groups (10=deciles, 5=quintiles).

    Returns:
        DataFrame indexed by date, columns 1..n_groups (1=bottom, n=top),
        values = equal-weighted mean return per group per period.
    """
    dates = predictions.index.get_level_values("date").unique().sort_values()
    rows = []

    for date in dates:
        if date not in returns.index.get_level_values("date"):
            continue
        pred = predictions.loc[date]
        ret = returns.loc[date]
        common = pred.index.intersection(ret.index)
        if len(common) < n_groups:
            continue

        try:
            groups = pd.qcut(pred[common], n_groups, labels=False,
                             duplicates="drop")
        except ValueError:
            continue

        row = {"date": date}
        for g in range(groups.max() + 1):
            mask = groups == g
            if mask.any():
                row[g + 1] = ret[common][mask].mean()
        rows.append(row)

    df = pd.DataFrame(rows).set_index("date")
    return df


def long_short_returns(
    predictions: pd.Series,
    returns: pd.Series,
    n_groups: int = 10,
) -> pd.Series:
    """Compute top-quantile minus bottom-quantile returns each period."""
    port = quantile_portfolios(predictions, returns, n_groups)
    if port.empty:
        return pd.Series(dtype=float, name="long_short")
    top = port.columns.max()
    return (port[top] - port[1]).rename("long_short")


# ---------------------------------------------------------------------------
# Turnover
# ---------------------------------------------------------------------------

def portfolio_turnover(
    predictions: pd.Series,
    n_groups: int = 10,
) -> pd.Series:
    """Monthly one-way turnover for an equal-weight long-short strategy.

    Turnover = fraction of the portfolio that changes each period,
    averaged across the long and short legs.

    Returns:
        Series indexed by date with one-way turnover per period.
    """
    dates = predictions.index.get_level_values("date").unique().sort_values()
    prev_long_set: set | None = None
    prev_short_set: set | None = None
    records = []

    for date in dates:
        pred = predictions.loc[date]
        try:
            groups = pd.qcut(pred, n_groups, labels=False, duplicates="drop")
        except ValueError:
            continue

        top, bottom = groups.max(), 0
        long_set = set(pred.index[groups == top])
        short_set = set(pred.index[groups == bottom])

        if not long_set or not short_set:
            continue

        if prev_long_set is not None:
            # Turnover = fraction of names that changed
            long_to = 1 - len(long_set & prev_long_set) / max(
                len(long_set | prev_long_set), 1
            )
            short_to = 1 - len(short_set & prev_short_set) / max(
                len(short_set | prev_short_set), 1
            )
            records.append({"date": date, "turnover": (long_to + short_to) / 2})

        prev_long_set = long_set
        prev_short_set = short_set

    if not records:
        return pd.Series(dtype=float, name="turnover")
    return pd.DataFrame(records).set_index("date")["turnover"]


# ---------------------------------------------------------------------------
# Performance metrics
# ---------------------------------------------------------------------------

def sharpe_ratio(returns: pd.Series, periods_per_year: int = 12) -> float:
    """Annualized Sharpe ratio (excess returns assumed — rf already subtracted)."""
    if len(returns) < 2:
        return np.nan
    std = returns.std()
    if std == 0:
        return np.nan
    return float(returns.mean() / std * np.sqrt(periods_per_year))


def cumulative_returns(returns: pd.Series) -> pd.Series:
    """Cumulative wealth index from period returns.

    Starts at 1.0. If you invested $1 at the start, this is what
    you'd have at each point.
    """
    return (1 + returns).cumprod()


def drawdown_series(returns: pd.Series) -> pd.Series:
    """Running drawdown from peak (always <= 0).

    Drawdown_t = (cumulative_t / max_cumulative_so_far) - 1
    """
    cum = cumulative_returns(returns)
    running_max = cum.cummax()
    return (cum / running_max - 1).rename("drawdown")


def max_drawdown(returns: pd.Series) -> float:
    """Maximum peak-to-trough drawdown (returned as a negative number)."""
    dd = drawdown_series(returns)
    return float(dd.min()) if len(dd) > 0 else np.nan


def var_historical(returns: pd.Series, alpha: float = 0.05) -> float:
    """Historical Value-at-Risk at confidence level (1 - alpha).

    Returns the alpha-quantile of the return distribution (a negative number
    in normal markets). Interpretation: "with (1-alpha) confidence, losses
    won't exceed |VaR| in a single period."

    Args:
        returns: period return series.
        alpha: tail probability (0.05 = 95% VaR, 0.01 = 99% VaR).

    Returns:
        Alpha-quantile of returns (negative number in normal markets).
    """
    if len(returns) < 2:
        return np.nan
    return float(np.quantile(returns, alpha))


def expected_shortfall(returns: pd.Series, alpha: float = 0.05) -> float:
    """Expected Shortfall (CVaR) — average loss beyond VaR.

    ES = E[r | r <= VaR_alpha]. Always more negative than VaR.
    Preferred over VaR because it is coherent (subadditive) and
    captures tail shape, not just a single quantile.

    Args:
        returns: period return series.
        alpha: tail probability (0.05 = 95% ES, 0.01 = 99% ES).

    Returns:
        Average loss beyond VaR (always <= VaR).
    """
    if len(returns) < 2:
        return np.nan
    threshold = np.quantile(returns, alpha)
    tail = returns[returns <= threshold]
    return float(tail.mean()) if len(tail) > 0 else float(threshold)


def sortino_ratio(returns: pd.Series, periods_per_year: int = 12) -> float:
    """Annualized Sortino ratio (penalizes downside volatility only).

    Exported from Week 05 (Backtesting & Transaction Costs).

    Uses downside deviation (sqrt of mean squared negative returns) as
    the denominator.  Preferred when the return distribution is asymmetric
    (negative skew), since Sharpe penalizes upside variance equally with
    downside.

    Args:
        returns: period returns (excess of risk-free, or total if rf ~ 0).
        periods_per_year: annualization factor (12 for monthly, 252 for daily).

    Returns:
        Annualized ratio, or np.nan if insufficient data.
    """
    if len(returns) < 2:
        return np.nan
    # Downside deviation: sqrt(E[min(r, 0)^2]) over ALL returns
    downside_diff = np.minimum(returns.values, 0.0)
    down_dev = np.sqrt(np.mean(downside_diff ** 2))
    if down_dev == 0:
        return np.nan
    return float(returns.mean() / down_dev * np.sqrt(periods_per_year))


def calmar_ratio(returns: pd.Series, periods_per_year: int = 12) -> float:
    """Calmar ratio: annualized return / |max drawdown|.

    Exported from Week 05 (Backtesting & Transaction Costs).

    Measures return per unit of worst-case drawdown risk. Higher is better.

    Args:
        returns: period returns.
        periods_per_year: annualization factor (12 for monthly).

    Returns:
        Annualized ratio, or np.nan if insufficient data.
    """
    mdd = max_drawdown(returns)
    if mdd == 0 or np.isnan(mdd):
        return np.nan
    cagr = (1 + returns.mean()) ** periods_per_year - 1
    return float(abs(cagr / mdd))


def performance_summary(
    returns: pd.Series,
    periods_per_year: int = 12,
    label: str = "",
) -> dict:
    """Comprehensive performance summary for a return series.

    Exported from Week 05 (Backtesting & Transaction Costs).

    Computes standard risk-adjusted metrics used in strategy evaluation:
    Sharpe, Sortino, Calmar, CAGR, max drawdown, plus return distribution
    moments (skewness, excess kurtosis).

    Args:
        returns: period return series (monthly assumed by default).
        periods_per_year: annualization factor (12 for monthly, 252 for daily).
        label: optional name for the return series.

    Returns:
        dict with keys:
            label: str — name of the return series.
            sharpe: float — annualized Sharpe ratio.
            sortino: float — annualized Sortino ratio.
            calmar: float — Calmar ratio (CAGR / |max drawdown|).
            cagr: float — compound annual growth rate.
            max_dd: float — maximum peak-to-trough drawdown (negative).
            skewness: float — return distribution skewness.
            excess_kurtosis: float — Fisher kurtosis (0 for normal).
            n_periods: int — number of non-NaN return observations.
    """
    from scipy import stats as _stats

    r = returns.dropna()
    n = len(r)

    sr = sharpe_ratio(r, periods_per_year)
    so = sortino_ratio(r, periods_per_year)
    mdd = max_drawdown(r)
    cagr_val = (1 + r.mean()) ** periods_per_year - 1 if n > 0 else np.nan
    cal = calmar_ratio(r, periods_per_year)
    skew = float(_stats.skew(r)) if n >= 3 else np.nan
    ekurt = float(_stats.kurtosis(r, fisher=True)) if n >= 4 else np.nan

    return dict(
        label=label,
        sharpe=sr,
        sortino=so,
        calmar=cal,
        cagr=cagr_val,
        max_dd=mdd,
        skewness=skew,
        excess_kurtosis=ekurt,
        n_periods=n,
    )


def net_returns(
    gross_returns: pd.Series,
    turnover: pd.Series,
    cost_bps: float,
) -> pd.Series:
    """Subtract transaction costs from gross returns.

    Exported from Week 04 (ML Alpha).

    Args:
        gross_returns: period gross returns.
        turnover: one-way turnover per period (aligned or alignable index).
        cost_bps: one-way cost in basis points.

    Returns:
        Series of net returns on the common index.
    """
    cost_frac = cost_bps / 10_000
    common = gross_returns.index.intersection(turnover.index)
    # Round-trip cost = 2 * one-way turnover * one-way cost
    costs = turnover[common] * 2 * cost_frac
    return (gross_returns[common] - costs).rename("net_return")


def breakeven_cost(
    gross_returns: pd.Series,
    turnover: pd.Series,
    max_bps: int = 200,
    periods_per_year: int = 12,
) -> float:
    """Find the one-way transaction cost (in bps) at which net Sharpe = 0.

    Exported from Week 04 (ML Alpha).

    Sweeps cost levels from 0 to max_bps and uses linear interpolation
    to find the exact breakeven point where the net annualized Sharpe
    ratio crosses zero.  This is a standard metric for evaluating
    whether a signal survives realistic trading costs.

    Args:
        gross_returns: period gross returns (e.g. long-short monthly).
        turnover: one-way turnover per period (aligned with gross_returns).
        max_bps: upper bound of cost sweep (default 200 bps).
        periods_per_year: annualization factor (12 for monthly).

    Returns:
        Breakeven cost in basis points.  Returns inf if the strategy
        remains profitable across the entire sweep.  Returns nan if
        gross Sharpe is already <= 0.
    """
    gross_sr = sharpe_ratio(gross_returns, periods_per_year)
    if not np.isfinite(gross_sr) or gross_sr <= 0:
        return np.nan

    prev_sr = gross_sr
    for bps in range(1, max_bps + 1):
        net_ret = net_returns(gross_returns, turnover, cost_bps=float(bps))
        sr = sharpe_ratio(net_ret, periods_per_year)
        if not np.isfinite(sr):
            continue
        if sr <= 0:
            # Linear interpolation between (bps-1, prev_sr) and (bps, sr)
            if prev_sr > 0:
                return (bps - 1) + prev_sr / (prev_sr - sr)
            return float(bps)
        prev_sr = sr

    return float("inf")
