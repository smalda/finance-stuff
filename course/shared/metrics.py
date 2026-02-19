"""Signal quality metrics for cross-sectional return prediction.

All functions operate on aligned arrays of predicted vs actual values
for a single cross-section (one date). The caller handles alignment.

Error contract: all functions return np.nan on degenerate input
(too few observations, zero variance). They never raise on bad data.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def pearson_ic(predicted: np.ndarray, actual: np.ndarray) -> float:
    """Pearson correlation between predicted and realized values.

    This is the standard Information Coefficient (IC) used in quantitative
    finance to measure cross-sectional signal quality.
    """
    predicted, actual = np.asarray(predicted), np.asarray(actual)
    if len(predicted) < 3:
        return np.nan
    corr = np.corrcoef(predicted, actual)[0, 1]
    return corr if np.isfinite(corr) else np.nan


def rank_ic(predicted: np.ndarray, actual: np.ndarray) -> float:
    """Spearman rank correlation — robust IC variant.

    More robust to outliers than Pearson IC. Preferred when the
    prediction-to-return mapping is monotonic but not necessarily linear.
    """
    predicted, actual = np.asarray(predicted), np.asarray(actual)
    if len(predicted) < 3:
        return np.nan
    corr, _ = stats.spearmanr(predicted, actual)
    return corr if np.isfinite(corr) else np.nan


def icir(ic_series: np.ndarray) -> float:
    """Information Coefficient Information Ratio: mean(IC) / std(IC).

    Measures signal consistency. An IC of 0.03 with std 0.05 (ICIR=0.6)
    is more valuable than IC of 0.05 with std 0.15 (ICIR=0.33).
    """
    ic = np.asarray(ic_series, dtype=float)
    ic = ic[np.isfinite(ic)]
    if len(ic) < 2:
        return np.nan
    s = ic.std(ddof=1)
    return ic.mean() / s if s > 0 else np.nan


def hit_rate(predicted: np.ndarray, actual: np.ndarray) -> float:
    """Fraction of correctly predicted return directions.

    A simple directional accuracy metric: what percentage of stocks
    did the model correctly rank as positive vs. negative return?
    """
    predicted, actual = np.asarray(predicted), np.asarray(actual)
    if len(predicted) < 1:
        return np.nan
    return float(np.mean(np.sign(predicted) == np.sign(actual)))


def r_squared_oos(predicted: np.ndarray, actual: np.ndarray) -> float:
    """Out-of-sample R-squared (Campbell & Thompson, 2008).

    R^2_OOS = 1 - sum((actual - predicted)^2) / sum(actual^2)

    Note: uses zero (unconditional mean) as the benchmark, which is
    standard for return prediction where the null is "returns are
    unpredictable." Values > 0 indicate predictive power.
    """
    predicted, actual = np.asarray(predicted), np.asarray(actual)
    ss_res = np.sum((actual - predicted) ** 2)
    ss_tot = np.sum(actual ** 2)
    if ss_tot == 0:
        return np.nan
    return float(1 - ss_res / ss_tot)


def deflated_sharpe_ratio(
    observed_sr: float,
    n_trials: int,
    n_obs: int,
    *,
    skew: float = 0.0,
    excess_kurt: float = 0.0,
) -> float:
    """Deflated Sharpe Ratio (Bailey & Lopez de Prado, 2014).

    Probability that the observed SR is significant after adjusting
    for multiple testing (n_trials strategies tested).

    Args:
        observed_sr: observed Sharpe ratio.
        n_trials: number of strategies/configurations tested.
        n_obs: number of return observations.
        skew: skewness of returns (0 for normal).
        excess_kurt: excess kurtosis of returns (0 for normal).

    Returns:
        Probability (0-1) that the SR is genuine. Values > 0.95
        suggest statistical significance.
    """
    from scipy.stats import norm as norm_dist

    if n_trials < 1 or n_obs < 2:
        return np.nan

    # Expected maximum SR under null hypothesis
    # When n_trials=1, E[max of 1 standard normal] = 0 (no multiple
    # testing adjustment).  The EV-theory approximation below produces
    # norm.ppf(0) = -inf for M=1, so we handle it explicitly.
    if n_trials == 1:
        sr_benchmark = 0.0
    else:
        gamma = 0.5772156649  # Euler-Mascheroni constant
        sr_std = np.sqrt(1.0 / (n_obs - 1))
        z = ((1 - gamma) * norm_dist.ppf(1 - 1.0 / n_trials)
             + gamma * norm_dist.ppf(1 - 1.0 / (n_trials * np.e)))
        sr_benchmark = sr_std * z

    # Probabilistic SR (accounting for non-normality)
    sr_var = (1 - skew * observed_sr
              + (excess_kurt / 4) * observed_sr ** 2) / (n_obs - 1)
    if sr_var <= 0:
        return np.nan

    return float(norm_dist.cdf(
        (observed_sr - sr_benchmark) / np.sqrt(sr_var)
    ))


def calibration_error(
    predicted_probs: np.ndarray,
    actual_outcomes: np.ndarray,
    n_bins: int = 10,
) -> float:
    """Expected Calibration Error for probabilistic predictions.

    Used in Week 11 (Bayesian uncertainty) to evaluate whether
    predicted probabilities match observed frequencies.

    Args:
        predicted_probs: model's predicted probabilities (0-1).
        actual_outcomes: binary outcomes (0 or 1).
        n_bins: number of calibration bins.

    Returns:
        ECE: weighted average of |predicted - observed| per bin.
    """
    predicted_probs = np.asarray(predicted_probs)
    actual_outcomes = np.asarray(actual_outcomes)
    n = len(predicted_probs)
    if n == 0:
        return np.nan

    bin_edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0

    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        mask = (predicted_probs >= lo) & (predicted_probs < hi)
        if not mask.any():
            continue
        bin_pred = predicted_probs[mask].mean()
        bin_actual = actual_outcomes[mask].mean()
        ece += mask.sum() / n * abs(bin_pred - bin_actual)

    return float(ece)


def ic_summary(ic_series: np.ndarray) -> dict:
    """Compute standard IC statistics from a time series of monthly ICs.

    Args:
        ic_series: array of per-period IC values.

    Returns:
        dict with keys:
            mean_ic: float — average IC across periods.
            std_ic: float — standard deviation of IC (ddof=1).
            icir: float — IC information ratio (mean/std).
            pct_positive: float — fraction of periods with IC > 0.
            t_stat: float — t-statistic for mean IC != 0.
            p_value: float — two-sided p-value.
            significant_5pct: bool — whether p_value < 0.05.
            n: int — number of finite IC observations used.
    """
    ic = np.asarray(ic_series, dtype=float)
    ic = ic[np.isfinite(ic)]
    n = len(ic)
    if n == 0:
        return dict(mean_ic=np.nan, std_ic=np.nan, icir=np.nan,
                    pct_positive=np.nan, t_stat=np.nan, p_value=np.nan,
                    significant_5pct=False, n=0)
    mean, std = float(ic.mean()), float(ic.std(ddof=1))
    # t-test for mean IC significantly different from zero
    if n >= 2 and std > 0:
        t_stat = float(mean / (std / np.sqrt(n)))
        p_value = float(2 * (1 - stats.t.cdf(abs(t_stat), df=n - 1)))
    else:
        t_stat = np.nan
        p_value = np.nan
    return dict(
        mean_ic=mean,
        std_ic=std,
        icir=mean / std if std > 0 else np.nan,
        pct_positive=float((ic > 0).mean()),
        t_stat=t_stat,
        p_value=p_value,
        significant_5pct=bool(np.isfinite(p_value) and p_value < 0.05),
        n=n,
    )


def prediction_quality(predicted: np.ndarray, actual: np.ndarray) -> dict:
    """Check whether model predictions are meaningfully differentiated.

    Detects degenerate models that output near-constant predictions —
    functionally predicting the unconditional mean with tiny perturbations.

    Args:
        predicted: model predictions for one cross-section.
        actual: realized values for the same cross-section.

    Returns:
        dict with:
            spread_ratio: std(predicted) / std(actual).
                ~1.0 = healthy. <<1.0 = model barely varies output.
            unique_ratio: n_unique / n_total.
                1.0 = all different. Low = clustered/degenerate.
            range_ratio: range(predicted) / range(actual).
    """
    predicted, actual = np.asarray(predicted, dtype=float), np.asarray(actual, dtype=float)
    if len(predicted) < 2:
        return dict(spread_ratio=np.nan, unique_ratio=np.nan, range_ratio=np.nan)

    std_p, std_a = predicted.std(), actual.std()
    spread_ratio = float(std_p / std_a) if std_a > 0 else np.nan

    unique_ratio = float(len(np.unique(predicted)) / len(predicted))

    range_p = predicted.max() - predicted.min()
    range_a = actual.max() - actual.min()
    range_ratio = float(range_p / range_a) if range_a > 0 else np.nan

    return dict(
        spread_ratio=spread_ratio,
        unique_ratio=unique_ratio,
        range_ratio=range_ratio,
    )


def min_track_record_length(
    observed_sr: float,
    n_trials: int = 1,
    *,
    benchmark_sr: float = 0.0,
    skew: float = 0.0,
    excess_kurt: float = 0.0,
    confidence: float = 0.95,
    max_T: int = 1200,
) -> float:
    """Minimum Track Record Length (Bailey & Lopez de Prado, 2014).

    Exported from Week 05 (Backtesting & Transaction Costs).

    Returns the minimum number of observations needed for the Deflated
    Sharpe Ratio to exceed `confidence`, given `n_trials` strategies
    tested and observed return non-normality.

    When n_trials=1 (single strategy, no multiple testing), uses the
    closed-form PSR formula.  For n_trials > 1, uses iterative search
    over T from 6 to `max_T`.

    Args:
        observed_sr: per-period (e.g. monthly) Sharpe ratio. Must match
            the observation frequency used for n_obs in DSR.
        n_trials: number of strategies/configurations tried.
        benchmark_sr: benchmark per-period Sharpe ratio to beat
            (default 0.0 = "is this SR significantly above zero?").
            Only used in the closed-form path (n_trials <= 1).
        skew: skewness of returns (0 for normal).
        excess_kurt: excess kurtosis of returns (0 for normal).
        confidence: target DSR threshold (default 0.95).
        max_T: upper bound for search (default 1200 periods).

    Returns:
        Minimum T (float) such that DSR >= confidence, or inf if not
        reached within max_T.
    """
    from scipy.stats import norm as norm_dist

    sr_excess = observed_sr - benchmark_sr
    if sr_excess <= 0:
        return float("inf")

    # Closed-form for n_trials=1: PSR with benchmark SR*.
    # Solving Phi((SR - SR*) / sqrt(Var(SR))) = confidence for T:
    #   T = 1 + z_alpha^2 * (1 - skew*SR + (kurt/4)*SR^2) / (SR - SR*)^2
    if n_trials <= 1:
        z_alpha = norm_dist.ppf(confidence)
        sr_var_factor = 1 - skew * observed_sr + (excess_kurt / 4) * observed_sr ** 2
        if sr_var_factor <= 0:
            return float("inf")
        return 1.0 + z_alpha ** 2 * sr_var_factor / sr_excess ** 2

    # Iterative search for n_trials > 1
    for T in range(6, max_T + 1):
        dsr_val = deflated_sharpe_ratio(
            observed_sr=observed_sr,
            n_trials=n_trials,
            n_obs=T,
            skew=skew,
            excess_kurt=excess_kurt,
        )
        if dsr_val is not None and np.isfinite(dsr_val) and dsr_val >= confidence:
            return float(T)
    return float("inf")


def probability_of_backtest_overfitting(
    model_ic_dict: dict,
    n_splits: int = 6,
    n_test_splits: int = 2,
    purge_gap: int = 1,
) -> tuple:
    """Probability of Backtest Overfitting via Combinatorial Purged CV.

    Exported from Week 04 (ML Alpha).

    For each CPCV path, identifies the IS-best model and checks whether
    it ranks above or below the OOS median. PBO = fraction of paths
    where the IS winner ranks below the OOS median (i.e., overfits).

    Args:
        model_ic_dict: {model_name: pd.Series of monthly IC values}.
            All series must share a common DatetimeIndex (aligned
            internally to the intersection).
        n_splits: number of CPCV groups (default 6 -> C(6,2)=15 paths).
        n_test_splits: number of test splits per path (default 2).
        purge_gap: observations to purge at train/test boundaries.

    Returns:
        (pbo, records) where:
            pbo: float in [0, 1] (fraction of paths where IS winner
                ranks below OOS median). NaN if insufficient data.
            records: list of dicts per path with keys:
                is_winner, oos_rank_winner, oos_ic_winner,
                n_models, oos_median_rank.
    """
    from .temporal import CombinatorialPurgedCV

    names = list(model_ic_dict.keys())
    ic_frames = {}
    for name, series in model_ic_dict.items():
        if series is not None and len(series) > 0:
            ic_frames[name] = series.sort_index()

    if len(ic_frames) < 2:
        return float("nan"), []

    # Align to common dates
    common_idx = None
    for s in ic_frames.values():
        common_idx = s.index if common_idx is None else common_idx.intersection(s.index)

    import pandas as pd
    ic_matrix = pd.DataFrame(
        {n: ic_frames[n].reindex(common_idx) for n in ic_frames}
    ).dropna()

    if len(ic_matrix) < n_splits + 2:
        return float("nan"), []

    cpcv = CombinatorialPurgedCV(
        n_splits=n_splits, n_test_splits=n_test_splits, purge_gap=purge_gap
    )
    X_proxy = pd.DataFrame(index=ic_matrix.index)

    paths_below_median = 0
    records = []

    for train_idx, test_idx in cpcv.split(X_proxy):
        if len(train_idx) == 0 or len(test_idx) == 0:
            continue

        is_ic = ic_matrix.iloc[train_idx].mean()
        oos_ic = ic_matrix.iloc[test_idx].mean()

        is_winner = is_ic.idxmax()
        oos_rank_of_winner = int(oos_ic.rank(ascending=False)[is_winner])
        oos_median_rank = (len(ic_frames) + 1) / 2.0

        if oos_rank_of_winner > oos_median_rank:
            paths_below_median += 1

        records.append({
            "is_winner": is_winner,
            "oos_rank_winner": oos_rank_of_winner,
            "oos_ic_winner": float(oos_ic[is_winner]),
            "n_models": len(ic_frames),
            "oos_median_rank": oos_median_rank,
        })

    n_paths = len(records)
    pbo = paths_below_median / n_paths if n_paths > 0 else float("nan")
    return pbo, records


def vs_naive_baseline(ic_model: np.ndarray, ic_naive: np.ndarray) -> dict:
    """Paired test: is the model's IC series significantly better than naive?

    Exported from Week 04 (ML Alpha).

    Uses a paired t-test on per-period IC differences. Both series must
    be aligned by date (same length, same ordering).

    Args:
        ic_model: per-period IC values for the model.
        ic_naive: per-period IC values for the naive baseline.

    Returns:
        dict with keys:
            mean_improvement: float — average IC difference (model - naive).
            t_stat: float — paired t-statistic.
            p_value: float — two-sided p-value.
            significant_5pct: bool — whether p_value < 0.05.
            n: int — number of paired observations used.
    """
    m = np.asarray(ic_model, dtype=float)
    b = np.asarray(ic_naive, dtype=float)
    mask = np.isfinite(m) & np.isfinite(b)
    m, b = m[mask], b[mask]
    n = len(m)
    if n < 2:
        return dict(mean_improvement=np.nan, t_stat=np.nan, p_value=np.nan,
                    significant_5pct=False, n=n)
    diff = m - b
    mean_diff = float(diff.mean())
    se = diff.std(ddof=1) / np.sqrt(n)
    if se == 0:
        return dict(mean_improvement=mean_diff, t_stat=np.nan, p_value=np.nan,
                    significant_5pct=False, n=n)
    t_stat = float(mean_diff / se)
    p_value = float(2 * (1 - stats.t.cdf(abs(t_stat), df=n - 1)))
    return dict(
        mean_improvement=mean_diff,
        t_stat=t_stat,
        p_value=p_value,
        significant_5pct=bool(p_value < 0.05),
        n=n,
    )


def compute_ic_series(
    predictions: pd.Series,
    actuals: pd.Series,
    method: str = "pearson",
) -> pd.Series:
    """Compute per-period cross-sectional IC from pre-computed predictions.

    Exported from Week 04 (ML Alpha).

    Given predictions and actuals with a shared MultiIndex (date, ticker),
    computes the cross-sectional correlation (IC) for each date.  This is
    the standard function for evaluating pre-computed signal quality --
    unlike evaluation.rolling_predict() which trains models on the fly,
    this operates on already-generated predictions.

    Args:
        predictions: Series with MultiIndex (date, ticker).
        actuals: Series with MultiIndex (date, ticker).
        method: 'pearson' or 'spearman' (rank IC).

    Returns:
        Series indexed by date, values = IC per period.
    """
    common = predictions.index.intersection(actuals.index)
    pred = predictions.loc[common]
    act = actuals.loc[common]

    dates = pred.index.get_level_values("date").unique().sort_values()
    ic_vals = {}

    ic_fn = rank_ic if method == "spearman" else pearson_ic
    for dt in dates:
        p, a = pred.loc[dt], act.loc[dt]
        corr = ic_fn(p.values, a.values)
        if np.isfinite(corr):
            ic_vals[dt] = corr

    return pd.Series(ic_vals, name=f"IC_{method}")
