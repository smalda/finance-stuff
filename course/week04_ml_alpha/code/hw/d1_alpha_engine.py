"""Deliverable 1: Cross-Sectional Alpha Engine — AlphaModelPipeline Class"""
import matplotlib
matplotlib.use("Agg")
import sys
import warnings
from pathlib import Path

import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LinearRegression, Ridge

sys.path.insert(0, str(Path(__file__).parent.parent))
from data_setup import (
    CACHE_DIR, PLOT_DIR, FEATURE_COLS, TRAIN_WINDOW, PURGE_GAP,
    COST_BPS, load_monthly_panel,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from shared.temporal import walk_forward_splits, PurgedWalkForwardCV
from shared.metrics import ic_summary, rank_ic, prediction_quality
from shared.backtesting import (
    quantile_portfolios, long_short_returns, portfolio_turnover,
    sharpe_ratio, net_returns, max_drawdown, cumulative_returns,
)

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)


# ── CELL: pipeline_init ──────────────────────────────────────────

class AlphaModelPipeline:
    """End-to-end walk-forward alpha model evaluation pipeline.

    Accepts any scikit-learn-compatible regressor, runs walk-forward
    out-of-sample prediction, computes signal quality metrics, and
    constructs long-short portfolios with transaction cost analysis.

    Args:
        model: sklearn-compatible regressor with fit/predict methods.
        features: DataFrame with MultiIndex (date, ticker), feature columns.
        target: Series with MultiIndex (date, ticker), forward returns.
        train_window: months in rolling training window.
        purge_gap: months between last train date and prediction date.
        cost_bps: one-way transaction cost in basis points.
        n_groups: number of quantile groups for portfolio construction.
        hp_search: if True, run HP search on first window (LightGBM only).
        hp_grid: dict of HP grid for search (LightGBM only).
        impute: if True, impute NaN with cross-sectional median per window.
    """

    def __init__(
        self,
        model,
        features: pd.DataFrame,
        target: pd.Series,
        train_window: int = 60,
        purge_gap: int = 1,
        cost_bps: float = 10.0,
        n_groups: int = 10,
        hp_search: bool = False,
        hp_grid: dict | None = None,
        impute: bool = False,
    ):
        self.model = model
        self.features = features
        self.target = target
        self.train_window = train_window
        self.purge_gap = purge_gap
        self.cost_bps = cost_bps
        self.n_groups = n_groups
        self.hp_search = hp_search
        self.hp_grid = hp_grid
        self.impute = impute

        self.dates = features.index.get_level_values("date").unique().sort_values()
        self.feature_cols = list(features.columns)

        # Populated after fit_predict()
        self.predictions_ = None
        self.ic_series_ = None
        self.rank_ic_series_ = None
        self._summary = None


# ── CELL: pipeline_impute ────────────────────────────────────────

def _impute_window(self, X_train: np.ndarray) -> np.ndarray:
    """Cross-sectional median imputation within training window."""
    X = X_train.copy()
    for j in range(X.shape[1]):
        col = X[:, j]
        mask = np.isnan(col)
        if mask.any():
            med = np.nanmedian(col)
            col[mask] = med if np.isfinite(med) else 0.0
    return X

AlphaModelPipeline._impute_window = _impute_window


# ── CELL: pipeline_hp_eval_fold ──────────────────────────────────

def _hp_eval_fold(self, X, y, tr_idx, val_idx, lr, nl):
    """Evaluate one CV fold for HP search. Returns (ic, n_iters)."""
    mdl = lgb.LGBMRegressor(
        n_estimators=500, learning_rate=lr,
        num_leaves=nl, min_child_samples=20,
        reg_alpha=0.1, reg_lambda=1.0,
        subsample=0.8, subsample_freq=1,
        colsample_bytree=0.8, random_state=42,
        verbosity=-1,
    )
    mdl.fit(
        X[tr_idx], y[tr_idx],
        eval_set=[(X[val_idx], y[val_idx])],
        callbacks=[lgb.early_stopping(50, verbose=False)],
    )
    pred = mdl.predict(X[val_idx])
    ic = np.corrcoef(pred, y[val_idx])[0, 1]
    return ic, mdl.best_iteration_

AlphaModelPipeline._hp_eval_fold = _hp_eval_fold


# ── CELL: pipeline_search_hps ────────────────────────────────────

def _search_hps(self, X, y):
    """HP search on first window for LightGBM models."""
    grid = self.hp_grid or {
        "learning_rate": [0.005, 0.01, 0.05],
        "num_leaves": [15, 31, 63],
    }
    cv = PurgedWalkForwardCV(n_splits=3, purge_gap=1)
    best_score, best_params, best_n = -np.inf, {}, 100

    for lr in grid.get("learning_rate", [0.05]):
        for nl in grid.get("num_leaves", [31]):
            scores, iters = [], []
            for tr_idx, val_idx in cv.split(X):
                ic, n_iter = self._hp_eval_fold(X, y, tr_idx, val_idx, lr, nl)
                if np.isfinite(ic):
                    scores.append(ic)
                    iters.append(n_iter)

            mean_ic = np.mean(scores) if scores else -np.inf
            if mean_ic > best_score:
                best_score = mean_ic
                best_params = {"learning_rate": lr, "num_leaves": nl}
                best_n = max(10, int(np.mean(iters)))

    return best_params, best_n

AlphaModelPipeline._search_hps = _search_hps


# ── CELL: pipeline_fit_lgbm_one ──────────────────────────────────

def _fit_lgbm_one(self, panel, features, target, train_dates,
                  searched_params, searched_n_est):
    """Fit one LightGBM model with early stopping on a single window."""
    if searched_params:
        mdl = lgb.LGBMRegressor(
            n_estimators=searched_n_est or 200,
            learning_rate=searched_params["learning_rate"],
            num_leaves=searched_params["num_leaves"],
            min_child_samples=20,
            reg_alpha=0.1, reg_lambda=1.0,
            subsample=0.8, subsample_freq=1,
            colsample_bytree=0.8, random_state=42,
            verbosity=-1,
        )
    else:
        from sklearn.base import clone
        mdl = clone(self.model)

    n_val = min(12, len(train_dates) // 5)
    val_dates = train_dates[-n_val:]
    fit_dates = train_dates[:-n_val]
    fit_mask = panel.index.get_level_values("date").isin(fit_dates)
    val_mask_inner = panel.index.get_level_values("date").isin(val_dates)
    X_fit = features.loc[fit_mask].values
    y_fit = target.loc[fit_mask].values
    X_val = features.loc[val_mask_inner].values
    y_val = target.loc[val_mask_inner].values
    if self.impute:
        X_fit = self._impute_window(X_fit)
        X_val_clean = X_val.copy()
        for j in range(X_val_clean.shape[1]):
            col = X_val_clean[:, j]
            mn = np.isnan(col)
            if mn.any():
                col[mn] = np.nanmedian(X_fit[:, j])
        X_val = X_val_clean
    mdl.fit(
        X_fit, y_fit,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(50, verbose=False)],
    )
    return mdl

AlphaModelPipeline._fit_lgbm_one = _fit_lgbm_one


# ── CELL: pipeline_predict_one ───────────────────────────────────

def _predict_one_window(self, panel, features, target, train_dates,
                        pred_date, is_lgbm, searched_params, searched_n_est):
    """Run one walk-forward window: fit model, predict OOS, return records."""
    train_mask = panel.index.get_level_values("date").isin(train_dates)
    pred_mask = panel.index.get_level_values("date") == pred_date

    X_train = features.loc[train_mask].values
    y_train = target.loc[train_mask].values
    X_pred = features.loc[pred_mask].values
    y_pred = target.loc[pred_mask].values
    tickers = panel.loc[pred_mask].index.get_level_values("ticker")

    if self.impute:
        X_train = self._impute_window(X_train)
        X_pred_clean = X_pred.copy()
        for j in range(X_pred_clean.shape[1]):
            col = X_pred_clean[:, j]
            mask_nan = np.isnan(col)
            if mask_nan.any():
                med = np.nanmedian(X_train[:, j])
                col[mask_nan] = med if np.isfinite(med) else 0.0
        X_pred = X_pred_clean

    if is_lgbm:
        mdl = self._fit_lgbm_one(panel, features, target, train_dates,
                                 searched_params, searched_n_est)
    else:
        from sklearn.base import clone
        mdl = clone(self.model)
        if self.impute:
            X_train = self._impute_window(X_train)
        mdl.fit(X_train, y_train)

    preds = mdl.predict(X_pred)

    oos_records = [{"date": pred_date, "ticker": t, "prediction": p, "actual": a}
                   for t, p, a in zip(tickers, preds, y_pred)]

    ic_p = np.corrcoef(preds, y_pred)[0, 1]
    ic_r = stats.spearmanr(preds, y_pred)[0]
    ic_rec = {
        "date": pred_date,
        "ic_pearson": ic_p if np.isfinite(ic_p) else np.nan,
        "ic_rank": ic_r if np.isfinite(ic_r) else np.nan,
    }
    return oos_records, ic_rec

AlphaModelPipeline._predict_one_window = _predict_one_window


# ── CELL: pipeline_fit_predict ───────────────────────────────────

def fit_predict(self) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run walk-forward prediction loop.

    Returns:
        ic_df: DataFrame with columns [ic_pearson, ic_rank], indexed by date.
        pred_df: DataFrame with MultiIndex (date, ticker),
                 columns [prediction, actual].
    """
    panel = self.features.join(self.target.rename("fwd_return"), how="inner")
    dates = panel.index.get_level_values("date").unique().sort_values()
    features = panel[self.feature_cols]
    target = panel["fwd_return"]

    splits = list(walk_forward_splits(dates, self.train_window, self.purge_gap))
    n_splits = len(splits)

    # HP search on first window (LightGBM only)
    searched_params, searched_n_est = None, None
    is_lgbm = isinstance(self.model, lgb.LGBMRegressor)
    if is_lgbm and self.hp_search:
        first_train = dates[:self.train_window]
        mask = panel.index.get_level_values("date").isin(first_train)
        X_first = features.loc[mask].values
        y_first = target.loc[mask].values
        if self.impute:
            X_first = self._impute_window(X_first)
        searched_params, searched_n_est = self._search_hps(X_first, y_first)
        print(f"  HP search: {searched_params}, n_est={searched_n_est}")

    oos_records, ic_records = [], []
    for i, (train_dates, pred_date) in enumerate(splits):
        if i % 10 == 0:
            print(f"  [{i+1}/{n_splits}] predicting "
                  f"{pd.Timestamp(pred_date).date()}")
        oos_rec, ic_rec = self._predict_one_window(
            panel, features, target, train_dates, pred_date,
            is_lgbm, searched_params, searched_n_est)
        oos_records.extend(oos_rec)
        ic_records.append(ic_rec)

    self.predictions_ = (
        pd.DataFrame(oos_records).set_index(["date", "ticker"])
    )
    self.ic_series_ = pd.DataFrame(ic_records).set_index("date")
    self.rank_ic_series_ = self.ic_series_["ic_rank"]
    self._summary = None  # reset cache
    print(f"  Walk-forward complete: {len(ic_records)} OOS months")

    return self.ic_series_, self.predictions_

AlphaModelPipeline.fit_predict = fit_predict


# ── CELL: pipeline_summary ───────────────────────────────────────

def summary(self) -> dict:
    """Compute comprehensive evaluation metrics.

    Returns:
        dict with: mean_ic, std_ic, icir, pct_positive, t_stat, p_value,
        mean_rank_ic, sharpe_gross, sharpe_net, mean_turnover,
        max_drawdown, n_oos_months.
    """
    if self.ic_series_ is None:
        raise RuntimeError("Call fit_predict() first.")

    if self._summary is not None:
        return self._summary

    ic_arr = self.ic_series_["ic_pearson"].dropna().values
    ic_stats = ic_summary(ic_arr)
    mean_rank = float(self.ic_series_["ic_rank"].dropna().mean())

    # Portfolio metrics
    pred_s = self.predictions_["prediction"]
    ret_s = self.predictions_["actual"]
    ls = long_short_returns(pred_s, ret_s, self.n_groups)
    to = portfolio_turnover(pred_s, self.n_groups)
    gross_sharpe = sharpe_ratio(ls)
    net_ret = net_returns(ls, to, self.cost_bps)
    net_sharpe = sharpe_ratio(net_ret)
    mdd = max_drawdown(ls)
    mean_to = float(to.mean()) if len(to) > 0 else np.nan

    self._summary = {
        "mean_ic": ic_stats["mean_ic"],
        "std_ic": ic_stats["std_ic"],
        "icir": ic_stats["icir"],
        "pct_positive": ic_stats["pct_positive"],
        "t_stat": ic_stats["t_stat"],
        "p_value": ic_stats["p_value"],
        "mean_rank_ic": mean_rank,
        "sharpe_gross": gross_sharpe,
        "sharpe_net": net_sharpe,
        "mean_turnover": mean_to,
        "max_drawdown": mdd,
        "n_oos_months": ic_stats["n"],
    }
    return self._summary

AlphaModelPipeline.summary = summary


if __name__ == "__main__":
    # ── CELL: data_quality_block ───────────────────────────────────────

    panel = load_monthly_panel()
    dates = panel.index.get_level_values("date").unique().sort_values()
    features = panel[FEATURE_COLS]
    target = panel["fwd_return"]

    n_months = len(dates)
    tickers_per = panel.groupby(level="date").size()
    nan_rate = features.isna().mean()

    print("── DATA QUALITY ──────────────────────────────────────")
    print(f"  Panel shape: {panel.shape}")
    print(f"  Feature months: {n_months}")
    print(f"  Tickers/month: {tickers_per.min()}–{tickers_per.max()}")
    print(f"  Target mean: {target.mean():.6f}, std: {target.std():.6f}")
    print(f"  Missing rates:")
    for col in FEATURE_COLS:
        print(f"    {col}: {nan_rate[col]:.2%}")
    print(f"  NOTE: LightGBM handles NaN natively; linear models use imputation")
    print(f"  SURVIVORSHIP BIAS: S&P 500 universe — results overstate")
    print()


    # ── CELL: run_default_pipeline ─────────────────────────────────────
    # Default test: LightGBM with 60-month walk-forward, HP search on first window.

    print("── ALPHAMODELPIPELINE: DEFAULT RUN (LightGBM) ────────")
    default_model = lgb.LGBMRegressor(
        n_estimators=200, learning_rate=0.05, num_leaves=31,
        min_child_samples=20, reg_alpha=0.1, reg_lambda=1.0,
        subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
        random_state=42, verbosity=-1,
    )

    pipeline = AlphaModelPipeline(
        model=default_model,
        features=features,
        target=target,
        train_window=TRAIN_WINDOW,
        purge_gap=PURGE_GAP,
        cost_bps=COST_BPS,
        n_groups=10,
        hp_search=True,
        impute=False,
    )

    ic_df, pred_df = pipeline.fit_predict()
    summary = pipeline.summary()
    print()


    # ── CELL: print_summary ───────────────────────────────────────────

    print("── PIPELINE SUMMARY ──────────────────────────────────")
    for k, v in summary.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")
    print()


    # ── CELL: verify_temporal_integrity ────────────────────────────────

    splits = list(walk_forward_splits(dates, TRAIN_WINDOW, PURGE_GAP))
    leak_count = 0
    for train_dates, pred_date in splits:
        if train_dates[-1] >= pred_date:
            leak_count += 1
        gap_months = np.searchsorted(dates, pred_date) - np.searchsorted(dates, train_dates[-1])
        if gap_months < PURGE_GAP + 1:
            leak_count += 1

    print("── TEMPORAL INTEGRITY ────────────────────────────────")
    print(f"  Walk-forward splits: {len(splits)}")
    print(f"  Purge gap: {PURGE_GAP} month(s)")
    print(f"  Leakage violations: {leak_count}")
    print()


    # ── CELL: verify_sklearn_compat ────────────────────────────────────

    print("── SKLEARN COMPATIBILITY TEST (Ridge) ────────────────")
    ridge_pipeline = AlphaModelPipeline(
        model=Ridge(alpha=1.0),
        features=features,
        target=target,
        train_window=TRAIN_WINDOW,
        purge_gap=PURGE_GAP,
        cost_bps=COST_BPS,
        impute=True,
    )
    ridge_ic, ridge_pred = ridge_pipeline.fit_predict()
    ridge_summary = ridge_pipeline.summary()
    print(f"  Ridge mean IC: {ridge_summary['mean_ic']:.4f}")
    print(f"  Ridge ICIR: {ridge_summary['icir']:.4f}")
    print(f"  Ridge Sharpe (gross): {ridge_summary['sharpe_gross']:.4f}")
    print()


    # ── CELL: portfolio_analysis ───────────────────────────────────────

    pred_s = pred_df["prediction"]
    ret_s = pred_df["actual"]
    qp = quantile_portfolios(pred_s, ret_s, 10)
    ls = long_short_returns(pred_s, ret_s, 10)
    to = portfolio_turnover(pred_s, 10)
    cum_ret = cumulative_returns(ls)
    net_ret = net_returns(ls, to, COST_BPS)
    cum_net = cumulative_returns(net_ret)

    print("── PORTFOLIO ANALYSIS ────────────────────────────────")
    print(f"  Decile months: {len(qp)}")
    print(f"  Decile groups: {len(qp.columns)}")
    print(f"  Stocks per decile: ~{panel.groupby(level='date').size().median() / 10:.0f}")
    print(f"  Long-short mean monthly: {ls.mean():.6f}")
    print(f"  Long-short std monthly: {ls.std():.6f}")
    print(f"  Gross Sharpe: {sharpe_ratio(ls):.4f}")
    print(f"  Net Sharpe ({COST_BPS} bps): {sharpe_ratio(net_ret):.4f}")
    print(f"  Mean turnover: {to.mean():.4f}")
    print(f"  Max drawdown: {max_drawdown(ls):.4f}")
    print(f"  Cumulative (gross) final: {cum_ret.iloc[-1]:.4f}")
    print(f"  Cumulative (net) final: {cum_net.iloc[-1]:.4f}")
    print()


    # ── CELL: portfolio_charts ─────────────────────────────────────────

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    decile_means = qp.mean()
    colors_dec = plt.cm.RdYlGn(np.linspace(0.2, 0.8, len(decile_means)))
    axes[0].bar(decile_means.index, decile_means.values, color=colors_dec)
    axes[0].axhline(0, color="black", linewidth=0.5)
    axes[0].set_xlabel("Decile (1=bottom, 10=top)")
    axes[0].set_ylabel("Mean Monthly Return")
    axes[0].set_title("Decile Mean Returns (GBM Signal)")

    axes[1].plot(range(len(cum_ret)), cum_ret.values,
                 label="Gross", linewidth=1.5)
    axes[1].plot(range(len(cum_net)), cum_net.values,
                 label=f"Net ({COST_BPS} bps)", linewidth=1.5, linestyle="--")
    axes[1].axhline(1.0, color="black", linewidth=0.5)
    axes[1].set_xlabel("OOS Month")
    axes[1].set_ylabel("Cumulative Return")
    axes[1].set_title("Long-Short Cumulative Returns")
    axes[1].legend()

    plt.tight_layout()
    plt.show()
    # ── ASSERTIONS ─────────────────────────────────────
    # D1-1: Class structure
    assert hasattr(AlphaModelPipeline, "__init__"), "D1-1: missing __init__"
    assert hasattr(AlphaModelPipeline, "fit_predict"), "D1-1: missing fit_predict"
    assert hasattr(AlphaModelPipeline, "summary"), "D1-1: missing summary"

    # D1-2: Accepts sklearn-compatible models (tested with LightGBM + Ridge)
    assert ridge_summary is not None, "D1-2: Ridge pipeline failed"

    # D1-3: fit_predict returns correct types
    assert isinstance(ic_df, pd.DataFrame), "D1-3: ic_df not DataFrame"
    assert "ic_pearson" in ic_df.columns, "D1-3: ic_df missing ic_pearson"
    assert "ic_rank" in ic_df.columns, "D1-3: ic_df missing ic_rank"
    assert isinstance(pred_df, pd.DataFrame), "D1-3: pred_df not DataFrame"
    assert pred_df.index.names == ["date", "ticker"], \
        f"D1-3: pred_df index names = {pred_df.index.names}"

    # D1-4: summary() has all required keys
    required_keys = [
        "mean_ic", "std_ic", "icir", "pct_positive", "t_stat", "p_value",
        "mean_rank_ic", "sharpe_gross", "sharpe_net", "mean_turnover",
        "max_drawdown", "n_oos_months",
    ]
    for k in required_keys:
        assert k in summary, f"D1-4: summary missing key '{k}'"

    # D1-5: Default run mean IC in [0.005, 0.06]
    assert 0.005 <= summary["mean_ic"] <= 0.06, \
        f"D1-5: mean IC = {summary['mean_ic']:.4f}, expected [0.005, 0.06]"

    # D1-6: Temporal integrity
    assert leak_count == 0, \
        f"D1-6: {leak_count} temporal leakage violations detected"

    # D1-7: Purge gap >= 1 month
    assert PURGE_GAP >= 1, f"D1-7: purge gap = {PURGE_GAP}, expected >= 1"

    # D1-8: DATA QUALITY block verified by presence above

    # D1-9: Handles missing values (Ridge test with impute=True passed)
    assert np.isfinite(ridge_summary["mean_ic"]), \
        "D1-9: Ridge with imputation produced non-finite IC"

    # D1-10: Long-short portfolio integrated
    assert len(ls) > 0, "D1-10: no long-short returns"
    assert len(to) > 0, "D1-10: no turnover series"
    assert np.isfinite(summary["sharpe_gross"]), "D1-10: non-finite gross Sharpe"
    assert np.isfinite(summary["sharpe_net"]), "D1-10: non-finite net Sharpe"
    assert np.isfinite(summary["max_drawdown"]), "D1-10: non-finite max drawdown"

    # ── RESULTS ────────────────────────────────────────
    print(f"══ hw/d1_alpha_engine ═══════════════════════════════")
    print(f"  n_oos_months: {summary['n_oos_months']}")
    print(f"  mean_pearson_ic: {summary['mean_ic']:.4f}")
    print(f"  mean_rank_ic: {summary['mean_rank_ic']:.4f}")
    print(f"  std_ic: {summary['std_ic']:.4f}")
    print(f"  icir: {summary['icir']:.4f}")
    print(f"  pct_positive: {summary['pct_positive']:.4f}")
    print(f"  t_stat: {summary['t_stat']:.4f}")
    print(f"  p_value: {summary['p_value']:.4f}")
    print(f"  sharpe_gross: {summary['sharpe_gross']:.4f}")
    print(f"  sharpe_net: {summary['sharpe_net']:.4f}")
    print(f"  mean_turnover: {summary['mean_turnover']:.4f}")
    print(f"  max_drawdown: {summary['max_drawdown']:.4f}")
    print(f"  ──")
    print(f"  ridge_mean_ic: {ridge_summary['mean_ic']:.4f}")
    print(f"  ridge_icir: {ridge_summary['icir']:.4f}")
    print(f"  ridge_sharpe_gross: {ridge_summary['sharpe_gross']:.4f}")

    # ── PLOT ───────────────────────────────────────────
    fig.savefig(PLOT_DIR / "d1_portfolio_summary.png",
                dpi=150, bbox_inches="tight")
    print(f"  ── plot: d1_portfolio_summary.png ──")
    print(f"     type: 2-panel (decile returns + cumulative L/S)")
    print(f"     panel_a_n_bars: {len(decile_means)}")
    print(f"     panel_a_title: {axes[0].get_title()}")
    print(f"     panel_b_y_range: [{axes[1].get_ylim()[0]:.4f}, "
          f"{axes[1].get_ylim()[1]:.4f}]")
    print(f"     panel_b_title: {axes[1].get_title()}")

    print(f"✓ d1_alpha_engine: ALL PASSED")
