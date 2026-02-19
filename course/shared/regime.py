"""Regime detection and statistical arbitrage utilities.

Covers Ornstein-Uhlenbeck parameter estimation, Engle-Granger cointegration,
and stubs for Hidden Markov Models and changepoint detection.

First introduced in Week 14; OU estimation also used in Week 2 (time series).

External deps: hmmlearn (imported lazily for HMM fitting).
"""
from __future__ import annotations

import numpy as np
from statsmodels.tsa.stattools import adfuller


# ---------------------------------------------------------------------------
# Ornstein-Uhlenbeck process estimation
# ---------------------------------------------------------------------------

def ou_estimate(spread: np.ndarray, dt: float = 1.0) -> dict:
    """Estimate Ornstein-Uhlenbeck parameters from a mean-reverting spread.

    Model: dX = theta * (mu - X) * dt + sigma * dW

    Uses OLS regression: X_{t+1} - X_t = a + b * X_t + epsilon
    where theta = -b/dt, mu = -a/b, sigma = std(epsilon)/sqrt(dt).

    Args:
        spread: time series of the spread (e.g. pairs trading residual).
        dt: time step size (1.0 for daily, 1/252 for trading-day fractions).

    Returns:
        dict with keys:
            theta: float — mean-reversion speed.
            mu: float — long-run mean of the process.
            sigma: float — diffusion coefficient.
            half_life: float — time to revert halfway to the mean (same units as dt).
    """
    x = np.asarray(spread, dtype=float)
    dx = np.diff(x)
    x_lag = x[:-1]

    # OLS: dx = a + b * x_lag
    A = np.column_stack([np.ones_like(x_lag), x_lag])
    coeffs, residuals, _, _ = np.linalg.lstsq(A, dx, rcond=None)
    a, b = coeffs

    theta = -b / dt
    mu = -a / b if abs(b) > 1e-10 else np.nan
    resid = dx - A @ coeffs
    sigma = resid.std() / np.sqrt(dt)
    half_life = np.log(2) / theta if theta > 0 else np.inf

    return {"theta": float(theta), "mu": float(mu),
            "sigma": float(sigma), "half_life": float(half_life)}


# ---------------------------------------------------------------------------
# Cointegration
# ---------------------------------------------------------------------------

def engle_granger_cointegration(
    y: np.ndarray,
    x: np.ndarray,
    significance: float = 0.05,
) -> dict:
    """Engle-Granger two-step cointegration test.

    Step 1: regress y on x (OLS), get residuals (the "spread").
    Step 2: ADF test on residuals — if stationary, series are cointegrated.

    Args:
        y, x: two price series of equal length.
        significance: p-value threshold for cointegration decision.

    Returns:
        dict with keys:
            intercept: float — regression intercept.
            hedge_ratio: float — OLS slope (beta) of y on x.
            spread: np.ndarray — regression residuals (the cointegrating spread).
            adf_statistic: float — ADF test statistic on the spread.
            pvalue: float — ADF test p-value.
            cointegrated: bool — True if pvalue < significance.
    """
    y, x = np.asarray(y, dtype=float), np.asarray(x, dtype=float)

    # OLS regression: y = alpha + beta * x
    A = np.column_stack([np.ones_like(x), x])
    beta = np.linalg.lstsq(A, y, rcond=None)[0]
    spread = y - A @ beta

    adf_stat, pvalue, *_ = adfuller(spread, autolag="AIC")

    return {
        "intercept": float(beta[0]),
        "hedge_ratio": float(beta[1]),
        "spread": spread,
        "adf_statistic": float(adf_stat),
        "pvalue": float(pvalue),
        "cointegrated": pvalue < significance,
    }


# ---------------------------------------------------------------------------
# Hidden Markov Models (stubs — require hmmlearn)
# ---------------------------------------------------------------------------

def fit_gaussian_hmm(
    returns: np.ndarray,
    n_states: int = 2,
    n_iter: int = 100,
    random_state: int = 42,
) -> dict:
    """Fit a Gaussian HMM to a return series for regime detection.

    Args:
        returns: 1-D array of returns.
        n_states: number of hidden states (typically 2: bull/bear).
        n_iter: EM iterations.
        random_state: for reproducibility.

    Returns:
        dict with keys:
            model: GaussianHMM — fitted hmmlearn model object.
            states: np.ndarray — decoded hidden state sequence.
            means: np.ndarray — per-state mean returns.
            variances: np.ndarray — per-state return variances.
            transition_matrix: np.ndarray — state transition probability matrix.

    Stub — implement when Week 14 blueprint is finalized.
    Requires: pip install hmmlearn
    """
    raise NotImplementedError(
        "Stub — implement when Week 14 blueprint is finalized. "
        "Expected implementation: hmmlearn.hmm.GaussianHMM("
        f"n_components={n_states}, n_iter={n_iter}).fit(returns.reshape(-1,1))"
    )


def detect_changepoints(
    series: np.ndarray,
    method: str = "pelt",
    penalty: float | None = None,
) -> list[int]:
    """Detect structural changepoints in a time series.

    Stub — implement when Week 14 blueprint is finalized.
    Expected: either ruptures library or custom CUSUM/PELT.
    """
    raise NotImplementedError(
        "Stub — implement when Week 14 blueprint is finalized."
    )
