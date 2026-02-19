"""Rolling cross-sectional prediction harness.

Runs a model through walk-forward or expanding-window evaluation on a
stock-month panel, computing IC at each step. This is the single function
that replaces the 40-line rolling loop duplicated across week code files.

The caller supplies a predict_fn(x_train, y_train, x_pred) -> y_hat_array.
This works for any model family — sklearn, LightGBM, PyTorch, etc.
"""
from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd

from .metrics import pearson_ic, rank_ic, ic_summary, prediction_quality
from .temporal import walk_forward_splits, expanding_window_splits


def rolling_predict(
    features: pd.DataFrame,
    target: pd.Series,
    predict_fn: Callable,
    train_window: int = 60,
    purge_gap: int = 1,
    expanding: bool = False,
    min_train_obs: int = 100,
    min_pred_obs: int = 30,
    verbose: bool = True,
) -> tuple[pd.DataFrame, pd.Series]:
    """Run walk-forward cross-sectional prediction, return IC series.

    Args:
        features: DataFrame with MultiIndex (date, ticker), columns = features.
        target: Series with same MultiIndex — the forward return to predict.
        predict_fn: callable(x_train, y_train, x_pred) -> predictions array.
            Called once per window. Must handle numpy arrays.
        train_window: periods in training window (fixed for walk-forward,
            minimum for expanding).
        purge_gap: periods between train end and prediction date.
        expanding: if True, use expanding window instead of fixed.
        min_train_obs: skip window if fewer clean training rows.
        min_pred_obs: skip window if fewer prediction-date stocks.
        verbose: print progress every 10 windows.

    Returns:
        (ic_df, pred_series):
            ic_df: DataFrame indexed by date, columns [ic_pearson, ic_rank]
            pred_series: Series with MultiIndex (date, ticker) of all predictions
    """
    dates = features.index.get_level_values("date").unique().sort_values()

    splitter = (expanding_window_splits(dates, train_window, purge_gap)
                if expanding
                else walk_forward_splits(dates, train_window, purge_gap))

    n_total = len(dates) - (train_window + purge_gap)
    ic_records = []
    predictions = {}

    for window_num, (train_dates, pred_date) in enumerate(splitter, 1):
        if verbose and (window_num % 10 == 0 or window_num == 1):
            print(f"  Rolling predict: window {window_num}/{n_total}")

        # ---- extract training data ----
        train_mask = features.index.get_level_values("date").isin(train_dates)
        x_tr = features.loc[train_mask]
        y_tr = target.reindex(x_tr.index).dropna()
        x_tr = x_tr.loc[y_tr.index]

        # ---- extract prediction-date data ----
        if pred_date not in features.index.get_level_values("date"):
            continue
        x_pd = features.loc[pred_date]
        if pred_date not in target.index.get_level_values("date"):
            continue
        y_act = target.loc[pred_date]
        if len(y_act) < min_pred_obs:
            continue

        # ---- drop NaN rows ----
        x_tr_clean = x_tr.dropna()
        y_tr_clean = y_tr.loc[x_tr_clean.index]
        x_pd_clean = x_pd.dropna()

        if len(x_tr_clean) < min_train_obs:
            continue

        common = x_pd_clean.index.intersection(y_act.index)
        if len(common) < min_pred_obs:
            continue

        # ---- predict ----
        try:
            y_hat_arr = predict_fn(
                x_tr_clean.values, y_tr_clean.values, x_pd_clean.values
            )
        except Exception as e:
            if verbose:
                print(f"  ⚠ predict_fn failed on window {window_num}: {e}")
            continue

        y_hat = pd.Series(y_hat_arr, index=x_pd_clean.index)

        # ---- compute IC ----
        ic_p = pearson_ic(y_hat[common].values, y_act[common].values)
        ic_r = rank_ic(y_hat[common].values, y_act[common].values)
        if np.isfinite(ic_p) and np.isfinite(ic_r):
            ic_records.append(
                {"date": pred_date, "ic_pearson": ic_p, "ic_rank": ic_r}
            )

        for ticker in common:
            predictions[(pred_date, ticker)] = y_hat[ticker]

    # ---- assemble outputs ----
    ic_df = pd.DataFrame(ic_records)
    if len(ic_df) > 0:
        ic_df = ic_df.set_index("date")

    if predictions:
        pred_series = pd.Series(predictions, name="prediction")
        pred_series.index = pd.MultiIndex.from_tuples(
            pred_series.index, names=["date", "ticker"]
        )
    else:
        pred_series = pd.Series(dtype=float, name="prediction")

    # ---- signal reality warnings ----
    if verbose and len(ic_df) >= 5:
        summary = ic_summary(ic_df["ic_rank"].values)
        t, p = summary["t_stat"], summary["p_value"]
        if np.isfinite(t) and abs(t) < 1.96:
            print(f"  ⚠ WEAK SIGNAL: mean rank IC = {summary['mean_ic']:.4f}, "
                  f"t = {t:.2f}, p = {p:.3f} — "
                  f"not significantly different from zero (n={summary['n']})")

    if verbose and len(ic_df) >= 10:
        mid = len(ic_df) // 2
        ic_vals = ic_df["ic_rank"].values
        first = ic_summary(ic_vals[:mid])
        second = ic_summary(ic_vals[mid:])
        if (first["mean_ic"] > 0
                and second["mean_ic"] < first["mean_ic"] * 0.5):
            print(f"  ⚠ SIGNAL DECAY: IC 1st half = "
                  f"{first['mean_ic']:.4f} (t={first['t_stat']:.2f}), "
                  f"2nd half = {second['mean_ic']:.4f} "
                  f"(t={second['t_stat']:.2f})")

    if verbose and len(pred_series) > 0:
        # check last cross-section for prediction degeneracy
        last_date = pred_series.index.get_level_values("date").max()
        last_pred_s = pred_series.loc[last_date]
        last_act_s = target.loc[last_date].reindex(last_pred_s.index)
        common_mask = last_act_s.notna()
        if common_mask.sum() >= 10:
            pq = prediction_quality(
                last_pred_s[common_mask].values,
                last_act_s[common_mask].values,
            )
            if np.isfinite(pq["spread_ratio"]) and pq["spread_ratio"] < 0.10:
                print(f"  ⚠ DEGENERATE PREDICTIONS: spread_ratio = "
                      f"{pq['spread_ratio']:.3f} — model outputs are "
                      f"near-constant (checked on last cross-section)")

    return ic_df, pred_series
