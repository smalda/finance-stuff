"""Causal inference utilities for financial applications.

Stubs for causal graph construction, treatment effect estimation,
and the "factor mirage" analysis from Week 10.

First introduced in Week 10. The heavy lifting is done by dowhy/econml;
this module provides thin wrappers with financial-domain defaults.

External deps: dowhy, econml (imported lazily).
"""
from __future__ import annotations

import pandas as pd


def build_causal_graph(edges: list[tuple[str, str]]) -> object:
    """Build a causal DAG from an edge list.

    Args:
        edges: list of (cause, effect) tuples, e.g.
            [("market_cap", "return"), ("momentum", "return")].

    Returns:
        A dowhy-compatible causal graph object.

    Stub — implement when Week 10 blueprint is finalized.
    Requires: pip install dowhy
    """
    raise NotImplementedError(
        "Stub — implement when Week 10 blueprint is finalized. "
        "Expected: dowhy.CausalModel graph construction."
    )


def estimate_ate(
    data: pd.DataFrame,
    treatment: str,
    outcome: str,
    confounders: list[str],
    *,
    method: str = "doubly_robust",
) -> dict:
    """Estimate average treatment effect with confounding adjustment.

    Args:
        data: DataFrame with treatment, outcome, and confounder columns.
        treatment: column name of the treatment variable.
        outcome: column name of the outcome variable.
        confounders: list of confounder column names.
        method: estimation method ("doubly_robust", "propensity_score",
            "linear_regression").

    Returns:
        dict with: ate (point estimate), ci_lower, ci_upper, pvalue.

    Stub — implement when Week 10 blueprint is finalized.
    Requires: pip install dowhy econml
    """
    raise NotImplementedError(
        "Stub — implement when Week 10 blueprint is finalized. "
        "Expected: dowhy identification + econml estimation."
    )


def detect_collider_bias(
    data: pd.DataFrame,
    x: str,
    y: str,
    collider: str,
) -> dict:
    """Demonstrate collider bias by conditioning on a collider variable.

    Shows how conditioning on a common effect of X and Y creates
    a spurious association — a core causal inference lesson.

    Args:
        data: DataFrame containing x, y, and collider columns.
        x: column name of the first variable.
        y: column name of the second variable.
        collider: column name of the collider (common effect of x and y).

    Returns:
        dict with keys:
            unconditional_corr: float — correlation between x and y without conditioning.
            conditional_corr: float — correlation between x and y conditioned on collider.
            bias_magnitude: float — absolute difference between the two correlations.

    Stub — implement when Week 10 blueprint is finalized.
    """
    raise NotImplementedError(
        "Stub — implement when Week 10 blueprint is finalized."
    )
