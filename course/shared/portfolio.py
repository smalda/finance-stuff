"""Portfolio construction and optimization utilities.

Covers the major portfolio construction methods used across the course:
mean-variance (Markowitz), risk parity, hierarchical risk parity,
Black-Litterman, and Kelly sizing. Uses scipy for optimization — no
cvxpy dependency.

First introduced in Week 6; imported by downstream weeks that need
portfolio construction as infrastructure.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.spatial.distance import squareform


# ---------------------------------------------------------------------------
# Covariance estimation
# ---------------------------------------------------------------------------

def shrink_covariance(returns: pd.DataFrame, method: str = "ledoit_wolf") -> np.ndarray:
    """Shrink sample covariance toward a structured target.

    Args:
        returns: T x N DataFrame of asset returns.
        method: "ledoit_wolf" (shrink toward diagonal) or "sample" (no shrinkage).

    Returns:
        N x N shrunk covariance matrix.
    """
    if method == "sample":
        return returns.cov().values

    from sklearn.covariance import LedoitWolf
    lw = LedoitWolf().fit(returns.values)
    return lw.covariance_


# ---------------------------------------------------------------------------
# Mean-variance optimization (Markowitz)
# ---------------------------------------------------------------------------

def mean_variance_weights(
    mu: np.ndarray,
    cov: np.ndarray,
    *,
    target_vol: float | None = None,
    long_only: bool = True,
) -> np.ndarray:
    """Maximum-Sharpe (tangency) portfolio, optionally scaled to target vol.

    Args:
        mu: N-vector of expected returns.
        cov: N x N covariance matrix.
        target_vol: if set, scale weights so portfolio vol equals this.
        long_only: if True, enforce w >= 0.

    Returns:
        N-vector of portfolio weights summing to 1.
    """
    n = len(mu)

    def neg_sharpe(w):
        ret = w @ mu
        vol = np.sqrt(w @ cov @ w)
        return -ret / vol if vol > 1e-12 else 0.0

    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
    bounds = [(0.0, 1.0)] * n if long_only else [(None, None)] * n
    w0 = np.ones(n) / n

    result = minimize(neg_sharpe, w0, method="SLSQP",
                      bounds=bounds, constraints=constraints)
    w = result.x

    if target_vol is not None:
        current_vol = np.sqrt(w @ cov @ w)
        if current_vol > 1e-12:
            w *= target_vol / current_vol

    return w


def efficient_frontier(
    mu: np.ndarray,
    cov: np.ndarray,
    n_points: int = 50,
    long_only: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute the efficient frontier.

    Returns:
        (vols, rets, weights): arrays of portfolio vols, returns,
        and an (n_points x N) weight matrix.
    """
    n = len(mu)
    ret_range = np.linspace(mu.min(), mu.max(), n_points)
    vols, rets, all_w = [], [], []

    for target_ret in ret_range:
        def portfolio_vol(w):
            return np.sqrt(w @ cov @ w)

        constraints = [
            {"type": "eq", "fun": lambda w: np.sum(w) - 1.0},
            {"type": "eq", "fun": lambda w, tr=target_ret: w @ mu - tr},
        ]
        bounds = [(0.0, 1.0)] * n if long_only else [(None, None)] * n
        w0 = np.ones(n) / n

        result = minimize(portfolio_vol, w0, method="SLSQP",
                          bounds=bounds, constraints=constraints)
        if result.success:
            vols.append(np.sqrt(result.x @ cov @ result.x))
            rets.append(result.x @ mu)
            all_w.append(result.x)

    return np.array(vols), np.array(rets), np.array(all_w)


# ---------------------------------------------------------------------------
# Risk parity
# ---------------------------------------------------------------------------

def risk_parity_weights(cov: np.ndarray) -> np.ndarray:
    """Equal risk contribution portfolio weights.

    Each asset contributes equally to total portfolio variance.

    Args:
        cov: N x N covariance matrix.

    Returns:
        N-vector of weights summing to 1.
    """
    n = cov.shape[0]

    def objective(w):
        vol = np.sqrt(w @ cov @ w)
        mrc = cov @ w / vol
        rc = w * mrc
        target = vol / n
        return np.sum((rc - target) ** 2)

    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
    bounds = [(1e-4, 1.0)] * n
    w0 = np.ones(n) / n

    result = minimize(objective, w0, method="SLSQP",
                      bounds=bounds, constraints=constraints)
    return result.x


# ---------------------------------------------------------------------------
# Hierarchical Risk Parity (Lopez de Prado, 2016)
# ---------------------------------------------------------------------------

def hierarchical_risk_parity(returns: pd.DataFrame) -> pd.Series:
    """HRP portfolio weights via hierarchical clustering + inverse-variance bisection.

    Args:
        returns: T x N DataFrame of asset returns.

    Returns:
        Series of weights indexed by asset name, summing to 1.
    """
    corr = returns.corr()
    cov = returns.cov()
    n = len(corr)

    # Distance matrix from correlation
    dist = np.sqrt(0.5 * (1 - corr.values))
    np.fill_diagonal(dist, 0.0)
    link = linkage(squareform(dist), method="single")
    order = leaves_list(link)
    sorted_assets = corr.columns[order].tolist()

    # Recursive bisection
    def _bisect(assets):
        if len(assets) == 1:
            return {assets[0]: 1.0}
        mid = len(assets) // 2
        left, right = assets[:mid], assets[mid:]
        wl, wr = _bisect(left), _bisect(right)

        # Cluster variance (inverse-variance allocation)
        var_l = sum(wl[a] ** 2 * cov.loc[a, a] for a in left)
        var_r = sum(wr[a] ** 2 * cov.loc[a, a] for a in right)
        alpha = 1.0 - var_l / (var_l + var_r)

        return {**{a: w * alpha for a, w in wl.items()},
                **{a: w * (1 - alpha) for a, w in wr.items()}}

    weights = _bisect(sorted_assets)
    return pd.Series(weights).reindex(corr.columns)


# ---------------------------------------------------------------------------
# Black-Litterman
# ---------------------------------------------------------------------------

def black_litterman_posterior(
    cov: np.ndarray,
    market_caps: np.ndarray,
    P: np.ndarray,
    Q: np.ndarray,
    *,
    omega: np.ndarray | None = None,
    tau: float = 0.05,
    risk_aversion: float = 2.5,
) -> np.ndarray:
    """Black-Litterman posterior expected returns.

    Args:
        cov: N x N covariance matrix.
        market_caps: N-vector of market capitalizations.
        P: K x N pick matrix (each row is a view).
        Q: K-vector of expected view returns.
        omega: K x K view uncertainty matrix. Default: tau * diag(P @ cov @ P.T).
        tau: scaling factor for prior uncertainty.
        risk_aversion: market risk aversion coefficient (delta).

    Returns:
        N-vector of posterior expected returns.
    """
    w_mkt = market_caps / market_caps.sum()
    pi = risk_aversion * cov @ w_mkt  # implied equilibrium returns

    if omega is None:
        omega = np.diag(np.diag(tau * P @ cov @ P.T))

    tau_cov = tau * cov
    inv_tau_cov = np.linalg.inv(tau_cov)
    inv_omega = np.linalg.inv(omega)

    post_cov = np.linalg.inv(inv_tau_cov + P.T @ inv_omega @ P)
    post_mu = post_cov @ (inv_tau_cov @ pi + P.T @ inv_omega @ Q)

    return post_mu


# ---------------------------------------------------------------------------
# Kelly criterion
# ---------------------------------------------------------------------------

def kelly_fraction(mu: float, sigma_sq: float) -> float:
    """Full Kelly fraction for a single asset.

    f* = mu / sigma^2  (assuming log-utility maximization).

    Args:
        mu: expected excess return.
        sigma_sq: variance of returns.

    Returns:
        Optimal fraction of capital to allocate.
    """
    if sigma_sq <= 0:
        return np.nan
    return mu / sigma_sq


def half_kelly(mu: float, sigma_sq: float) -> float:
    """Half-Kelly: common practitioner adjustment for estimation error."""
    return kelly_fraction(mu, sigma_sq) / 2.0
