"""Derivatives pricing utilities.

Covers Black-Scholes, Greeks, implied volatility, Monte Carlo pricing,
and binomial trees. Pure numpy + scipy — no external pricing libraries.

First introduced in Week 9; imported by Week 16 (deep hedging).
"""
from __future__ import annotations

import numpy as np
from scipy.stats import norm


# ---------------------------------------------------------------------------
# Black-Scholes
# ---------------------------------------------------------------------------

def _d1d2(S: float, K: float, T: float, r: float, sigma: float) -> tuple[float, float]:
    """Compute d1 and d2 for Black-Scholes formula."""
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return d1, d2


def black_scholes_price(
    S: float, K: float, T: float, r: float, sigma: float,
    option_type: str = "call",
) -> float:
    """Closed-form Black-Scholes European option price.

    Args:
        S: current stock price.
        K: strike price.
        T: time to expiration (years).
        r: risk-free rate (continuous compounding).
        sigma: volatility (annualized).
        option_type: "call" or "put".

    Returns:
        Option price.
    """
    d1, d2 = _d1d2(S, K, T, r, sigma)

    if option_type == "call":
        return float(S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2))
    else:
        return float(K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1))


def bs_greeks(
    S: float, K: float, T: float, r: float, sigma: float,
    option_type: str = "call",
) -> dict:
    """Compute all standard Black-Scholes Greeks.

    Returns:
        dict with keys:
            delta: float — option delta (dV/dS).
            gamma: float — option gamma (d^2V/dS^2).
            vega: float — option vega per 1% vol move.
            theta: float — option theta per trading day.
            rho: float — option rho per 1% rate move.
    """
    d1, d2 = _d1d2(S, K, T, r, sigma)
    sqrt_T = np.sqrt(T)
    pdf_d1 = norm.pdf(d1)
    discount = np.exp(-r * T)

    gamma = pdf_d1 / (S * sigma * sqrt_T)
    vega = S * pdf_d1 * sqrt_T / 100  # per 1% vol move

    if option_type == "call":
        delta = norm.cdf(d1)
        theta = (-S * pdf_d1 * sigma / (2 * sqrt_T)
                 - r * K * discount * norm.cdf(d2)) / 252  # per trading day
        rho = K * T * discount * norm.cdf(d2) / 100
    else:
        delta = norm.cdf(d1) - 1
        theta = (-S * pdf_d1 * sigma / (2 * sqrt_T)
                 + r * K * discount * norm.cdf(-d2)) / 252
        rho = -K * T * discount * norm.cdf(-d2) / 100

    return {"delta": delta, "gamma": gamma, "vega": vega,
            "theta": theta, "rho": rho}


# ---------------------------------------------------------------------------
# Implied volatility
# ---------------------------------------------------------------------------

def implied_volatility(
    market_price: float, S: float, K: float, T: float, r: float,
    option_type: str = "call",
    *,
    tol: float = 1e-8,
    max_iter: int = 100,
) -> float:
    """Newton-Raphson implied volatility solver.

    Args:
        market_price: observed option price.
        S, K, T, r: BS parameters.
        option_type: "call" or "put".
        tol: convergence tolerance on price difference.
        max_iter: maximum Newton iterations.

    Returns:
        Implied volatility, or np.nan if solver fails.
    """
    sigma = 0.2  # initial guess

    for _ in range(max_iter):
        price = black_scholes_price(S, K, T, r, sigma, option_type)
        vega_raw = bs_greeks(S, K, T, r, sigma, option_type)["vega"] * 100
        if abs(vega_raw) < 1e-12:
            return np.nan
        sigma -= (price - market_price) / vega_raw
        if sigma <= 0:
            return np.nan
        if abs(price - market_price) < tol:
            return float(sigma)

    return float(sigma)


# ---------------------------------------------------------------------------
# Monte Carlo pricing
# ---------------------------------------------------------------------------

def monte_carlo_option(
    payoff_fn,
    S0: float,
    r: float,
    sigma: float,
    T: float,
    *,
    n_paths: int = 50_000,
    n_steps: int = 252,
    rng: np.random.Generator | None = None,
) -> dict:
    """Monte Carlo option pricing with geometric Brownian motion paths.

    Args:
        payoff_fn: callable(S_T) -> payoff array. Receives terminal prices.
        S0: initial stock price.
        r: risk-free rate.
        sigma: volatility.
        T: time to expiration (years).
        n_paths: simulation paths.
        n_steps: time steps per path.
        rng: numpy random generator for reproducibility.

    Returns:
        dict with keys:
            price: float — discounted expected payoff.
            std_error: float — Monte Carlo standard error of the price estimate.
            paths: np.ndarray — simulated price paths, shape (n_paths, n_steps + 1).
    """
    if rng is None:
        rng = np.random.default_rng()

    dt = T / n_steps
    Z = rng.standard_normal((n_paths, n_steps))
    log_increments = (r - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * Z
    log_paths = np.cumsum(log_increments, axis=1)
    log_paths = np.column_stack([np.zeros(n_paths), log_paths])
    paths = S0 * np.exp(log_paths)

    S_T = paths[:, -1]
    payoffs = payoff_fn(S_T)
    price = np.exp(-r * T) * payoffs.mean()
    std_error = np.exp(-r * T) * payoffs.std() / np.sqrt(n_paths)

    return {"price": float(price), "std_error": float(std_error), "paths": paths}


# ---------------------------------------------------------------------------
# Binomial tree (CRR)
# ---------------------------------------------------------------------------

def binomial_tree(
    S: float, K: float, T: float, r: float, sigma: float,
    *,
    n_steps: int = 100,
    option_type: str = "call",
    american: bool = False,
) -> float:
    """Cox-Ross-Rubinstein binomial tree option pricing.

    Args:
        S, K, T, r, sigma: standard option parameters.
        n_steps: tree depth.
        option_type: "call" or "put".
        american: if True, allow early exercise.

    Returns:
        Option price.
    """
    dt = T / n_steps
    u = np.exp(sigma * np.sqrt(dt))
    d = 1.0 / u
    p = (np.exp(r * dt) - d) / (u - d)
    disc = np.exp(-r * dt)

    # Terminal payoff
    j = np.arange(n_steps + 1)
    ST = S * u ** (n_steps - j) * d ** j
    if option_type == "call":
        V = np.maximum(ST - K, 0.0)
    else:
        V = np.maximum(K - ST, 0.0)

    # Backward induction
    for i in range(n_steps - 1, -1, -1):
        V = disc * (p * V[:-1] + (1 - p) * V[1:])
        if american:
            j_i = np.arange(i + 1)
            St = S * u ** (i - j_i) * d ** j_i
            intrinsic = np.maximum(St - K, 0.0) if option_type == "call" \
                else np.maximum(K - St, 0.0)
            V = np.maximum(V, intrinsic)

    return float(V[0])
