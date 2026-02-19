"""Market microstructure utilities.

Covers order flow imbalance (OFI), VWAP computation, bid-ask spread
metrics, and stubs for optimal execution models.

First introduced in Week 13; some spread utilities used in Week 5
(transaction cost modeling).
"""
from __future__ import annotations

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Order flow imbalance
# ---------------------------------------------------------------------------

def compute_ofi(
    bid_price: np.ndarray,
    ask_price: np.ndarray,
    bid_size: np.ndarray,
    ask_size: np.ndarray,
) -> np.ndarray:
    """Compute Order Flow Imbalance from L1 quote snapshots.

    OFI captures the net pressure from limit order book changes.
    At each timestamp, OFI = (bid-side contribution) - (ask-side contribution).

    Args:
        bid_price, ask_price: best bid/ask prices at each snapshot.
        bid_size, ask_size: best bid/ask sizes at each snapshot.

    Returns:
        OFI array (length n-1, since it uses differences).
    """
    bp, ap = np.asarray(bid_price), np.asarray(ask_price)
    bs, as_ = np.asarray(bid_size), np.asarray(ask_size)

    # Bid-side contribution
    dbp = np.diff(bp)
    dbs = np.diff(bs)
    bid_ofi = np.where(dbp > 0, bs[1:],
              np.where(dbp < 0, -bs[:-1], dbs))

    # Ask-side contribution
    dap = np.diff(ap)
    das = np.diff(as_)
    ask_ofi = np.where(dap > 0, as_[:-1],
              np.where(dap < 0, -as_[1:], -das))

    return bid_ofi + ask_ofi


# ---------------------------------------------------------------------------
# Spread and cost metrics
# ---------------------------------------------------------------------------

def quoted_spread(bid_price: np.ndarray, ask_price: np.ndarray) -> np.ndarray:
    """Quoted bid-ask spread in price units."""
    return np.asarray(ask_price) - np.asarray(bid_price)


def relative_spread(bid_price: np.ndarray, ask_price: np.ndarray) -> np.ndarray:
    """Relative spread: (ask - bid) / midpoint."""
    mid = (np.asarray(bid_price) + np.asarray(ask_price)) / 2
    return quoted_spread(bid_price, ask_price) / mid


def effective_spread(trade_price: np.ndarray, midpoint: np.ndarray,
                     side: np.ndarray) -> np.ndarray:
    """Effective spread: 2 * side * (trade_price - midpoint).

    Args:
        trade_price: execution prices.
        midpoint: prevailing mid-quote at trade time.
        side: +1 for buyer-initiated, -1 for seller-initiated.
    """
    return 2 * np.asarray(side) * (np.asarray(trade_price) - np.asarray(midpoint))


# ---------------------------------------------------------------------------
# VWAP
# ---------------------------------------------------------------------------

def vwap(prices: np.ndarray, volumes: np.ndarray) -> float:
    """Volume-weighted average price."""
    prices, volumes = np.asarray(prices), np.asarray(volumes)
    total_vol = volumes.sum()
    if total_vol == 0:
        return np.nan
    return float((prices * volumes).sum() / total_vol)


# ---------------------------------------------------------------------------
# Optimal execution (stubs)
# ---------------------------------------------------------------------------

def almgren_chriss_trajectory(
    total_shares: int,
    n_periods: int,
    sigma: float,
    eta: float,
    lam: float,
) -> np.ndarray:
    """Optimal execution trajectory (Almgren-Chriss, 2001).

    Minimizes E[cost] + lambda * Var[cost] for a linear temporary
    market impact model.

    Args:
        total_shares: total shares to execute.
        n_periods: number of trading periods.
        sigma: volatility per period.
        eta: temporary impact coefficient.
        lam: risk aversion parameter.

    Returns:
        Array of shares to trade in each period.

    Stub — implement when Week 13 blueprint is finalized.
    """
    raise NotImplementedError(
        "Stub — implement when Week 13 blueprint is finalized. "
        "Expected: closed-form solution from Almgren & Chriss (2001) "
        "using sinh/cosh schedule."
    )


def kyle_lambda(
    prices: np.ndarray,
    order_flow: np.ndarray,
) -> float:
    """Estimate Kyle's lambda (price impact coefficient).

    Stub — implement when Week 13 blueprint is finalized.
    Expected: regress price changes on signed order flow.
    """
    raise NotImplementedError(
        "Stub — implement when Week 13 blueprint is finalized."
    )
