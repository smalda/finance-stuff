"""Reinforcement learning environments for financial applications.

Stubs for gymnasium-compatible trading/execution environments.
The environment design is a core part of Week 15's educational content,
so only the interface contract is defined here.

First introduced in Week 15.

External deps: gymnasium (imported lazily).
"""
from __future__ import annotations

from typing import Any


class TradingEnv:
    """Financial trading environment for RL agents.

    Expected interface: gymnasium.Env compatible.

    Observation space: market state (prices, positions, features).
    Action space: position sizing or trade direction.
    Reward: risk-adjusted return (Sharpe-based, drawdown-penalized).

    Stub — implement when Week 15 blueprint is finalized.
    The environment design IS the educational content of Week 15,
    so this stub intentionally provides only the interface contract.

    Requires: pip install gymnasium
    """

    def __init__(self, **kwargs: Any) -> None:
        raise NotImplementedError(
            "Stub — implement when Week 15 blueprint is finalized. "
            "Expected: gymnasium.Env subclass with financial MDP formulation."
        )

    def reset(self, **kwargs: Any) -> tuple[Any, dict]:
        raise NotImplementedError

    def step(self, action: Any) -> tuple[Any, float, bool, bool, dict]:
        raise NotImplementedError


class ExecutionEnv:
    """Optimal execution environment for RL agents.

    Simulates executing a large order with market impact.
    Benchmarks against Almgren-Chriss from shared.microstructure.

    Stub — implement when Week 15 blueprint is finalized.
    """

    def __init__(self, **kwargs: Any) -> None:
        raise NotImplementedError(
            "Stub — implement when Week 15 blueprint is finalized."
        )

    def reset(self, **kwargs: Any) -> tuple[Any, dict]:
        raise NotImplementedError

    def step(self, action: Any) -> tuple[Any, float, bool, bool, dict]:
        raise NotImplementedError
