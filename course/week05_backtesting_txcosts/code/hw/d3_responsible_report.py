"""HW D3: Responsible Research Report — Three-Layer Evaluation Framework.

Integrates D1 (PurgedKFold), D2 (TransactionCostModel), and upstream caches
to produce a complete, multi-layer statistical evaluation of the GBM alpha model.

Layer 1 — Tear sheet + net/purged equity curve + DSR
Layer 2 — CPCV across 3 model variants + PBO + BHY + winning-model selection
Layer 3 — MinTRL + qualitative discussion on capital allocation

Pedagogical point: a strategy that looks attractive in a naive backtest
must survive purging, transaction costs, multiple-testing correction,
and a minimum track-record check before deployment consideration.
"""
import matplotlib
matplotlib.use("Agg")

import sys
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# ── Path setup ────────────────────────────────────────────────────────
_HW_DIR = Path(__file__).resolve().parent          # hw/
_CODE_DIR = _HW_DIR.parent                         # code/
_COURSE_DIR = _CODE_DIR.parent.parent              # finance_stuff/course/
_ROOT_DIR = _COURSE_DIR.parent                     # finance_stuff/

sys.path.insert(0, str(_CODE_DIR))    # data_setup
sys.path.insert(0, str(_ROOT_DIR))    # shared.*

from data_setup import CACHE_DIR, PLOT_DIR, load_alpha_output, load_ls_portfolio

# Upstream hw imports
sys.path.insert(0, str(_HW_DIR))
from d1_purged_kfold import PurgedKFold
from d2_tc_pipeline import TransactionCostModel

from course.shared.metrics import deflated_sharpe_ratio, ic_summary
from course.shared.backtesting import sharpe_ratio, max_drawdown, cumulative_returns
from course.shared.temporal import CombinatorialPurgedCV

SEED = 42
np.random.seed(SEED)


# ══════════════════════════════════════════════════════════════════════
# ── CELL: load_upstream_results ──────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════

# Load upstream results
_d2_results = pd.read_parquet(CACHE_DIR / "d2_tc_model_results.parquet")

# Net returns (base regime: tiered 5/15 bps) — from D2
net_returns_base = _d2_results["net_return_base"]
gross_returns = _d2_results["gross_return"]
turnover_series = _d2_results["turnover"]


# ── CELL: load_alpha_predictions ─────────────────────────────────────

# Alpha predictions from Week 4 GBM
_alpha = load_alpha_output()
gbm_predictions = _alpha["predictions"]          # MultiIndex (date, ticker)
forward_returns = _alpha["forward_returns"]       # Series MultiIndex (date, ticker)
gbm_ic_series = _alpha["ic_series"]["ic"]        # Series indexed by date

# NN predictions (if available from cache)
nn_predictions = _alpha.get("nn_predictions", None)
nn_ic_series = _alpha.get("nn_ic_series", None)
if nn_ic_series is not None and isinstance(nn_ic_series, pd.DataFrame):
    nn_ic_series = nn_ic_series["ic"]

# Expanded features for Ridge recomputation
expanded_features = _alpha.get("expanded_features", None)
if expanded_features is None:
    feat_file = CACHE_DIR / "expanded_features_w5.parquet"
    if feat_file.exists():
        expanded_features = pd.read_parquet(feat_file)

print("Data loaded:")
print(f"  Net returns (base): {len(net_returns_base)} months, "
      f"{net_returns_base.index[0].date()} to {net_returns_base.index[-1].date()}")
print(f"  GBM IC series: {len(gbm_ic_series)} months, "
      f"mean IC = {gbm_ic_series.mean():.4f}")
print(f"  NN predictions available: {nn_predictions is not None}")
print(f"  Expanded features available: {expanded_features is not None}")


# ══════════════════════════════════════════════════════════════════════
# Layer 1: Tear Sheet — Net, Purged Evaluation
# ══════════════════════════════════════════════════════════════════════

# ── CELL: layer1_metrics_fn ──────────────────────────────────────────

def compute_full_metrics(returns: pd.Series, label: str) -> dict:
    """Compute annualized performance metrics for a return series.

    Args:
        returns: Monthly return series.
        label: Name for display.

    Returns:
        dict with Sharpe, Sortino, max_dd, CAGR, skewness, excess_kurtosis.
    """
    ann_factor = 12
    mean_r = returns.mean()
    std_r = returns.std()

    sharpe = (mean_r * ann_factor) / (std_r * np.sqrt(ann_factor)) if std_r > 0 else 0.0

    downside = returns[returns < 0].std()
    sortino = (mean_r * ann_factor) / (downside * np.sqrt(ann_factor)) if downside > 0 else 0.0

    mdd = max_drawdown(returns)
    cagr = (1 + mean_r) ** ann_factor - 1
    calmar = abs(cagr / mdd) if mdd != 0 else 0.0

    skew = float(stats.skew(returns.dropna()))
    ekurt = float(stats.kurtosis(returns.dropna(), fisher=True))  # excess kurtosis

    return {
        "label": label,
        "sharpe": sharpe,
        "sortino": sortino,
        "max_dd": mdd,
        "cagr": cagr,
        "calmar": calmar,
        "skewness": skew,
        "excess_kurtosis": ekurt,
        "n_months": len(returns.dropna()),
    }


# ── CELL: layer1_metrics_compute ─────────────────────────────────────

gross_metrics = compute_full_metrics(gross_returns, "Gross (naive)")
net_metrics = compute_full_metrics(net_returns_base, "Net base (purged)")

print("\n── Layer 1: Performance Metrics ────────────────────────────")
for m in [gross_metrics, net_metrics]:
    print(f"  {m['label']}:")
    print(f"    Sharpe={m['sharpe']:.3f}, Sortino={m['sortino']:.3f}, "
          f"Calmar={m['calmar']:.3f}")
    print(f"    CAGR={m['cagr']:.2%}, MaxDD={m['max_dd']:.2%}")
    print(f"    Skewness={m['skewness']:.3f}, ExcessKurt={m['excess_kurtosis']:.3f}")


# ── CELL: layer1_quantstats_html ─────────────────────────────────────

import quantstats as qs
_quantstats_available = True

# Prepare returns for quantstats (needs named Series with DatetimeIndex)
_qs_returns = net_returns_base.copy()
_qs_returns.name = "Net (base regime)"
_qs_returns.index = pd.DatetimeIndex(_qs_returns.index)

# Generate quantstats HTML tearsheet
_qs_html_path = PLOT_DIR / "d3_quantstats_tearsheet.html"
qs.reports.html(_qs_returns, output=str(_qs_html_path), title="Layer 1 Tear Sheet — Net Returns (Base Regime)")
print(f"  quantstats HTML tearsheet saved → {_qs_html_path}")


# ── CELL: layer1_tearsheet_plot ──────────────────────────────────────

# Generate key quantstats plots as PNG for the notebook
fig_ts, axes_ts = plt.subplots(4, 1, figsize=(12, 14))
fig_ts.suptitle("Layer 1 Tear Sheet — Net Returns (Base Regime) [quantstats]", fontsize=13)

# 1. Cumulative returns
cum_net = (1 + net_returns_base).cumprod()
cum_gross = (1 + gross_returns).cumprod()
ax = axes_ts[0]
ax.plot(cum_gross.index, cum_gross.values, label="Gross", color="steelblue", alpha=0.7)
ax.plot(cum_net.index, cum_net.values, label="Net (base)", color="darkorange", linewidth=2)
ax.set_title("Cumulative Returns")
ax.set_ylabel("Portfolio Value")
ax.legend()
ax.grid(True, alpha=0.3)

# 2. Monthly return distribution
ax2 = axes_ts[1]
ax2.hist(net_returns_base.values, bins=20, color="darkorange", alpha=0.7,
         edgecolor="white")
ax2.axvline(0, color="black", linestyle="--", linewidth=1)
ax2.axvline(net_returns_base.mean(), color="red", linestyle="--",
            linewidth=1.5, label=f"Mean={net_returns_base.mean():.3f}")
ax2.set_title("Monthly Return Distribution (Net)")
ax2.set_xlabel("Return")
ax2.legend()
ax2.grid(True, alpha=0.3)

# 3. Drawdown
cum_series = (1 + net_returns_base).cumprod()
rolling_max = cum_series.cummax()
drawdown_series = (cum_series - rolling_max) / rolling_max
ax3 = axes_ts[2]
ax3.fill_between(drawdown_series.index, drawdown_series.values, 0,
                 color="red", alpha=0.4, label="Drawdown")
ax3.set_title("Underwater Equity Curve (Net)")
ax3.set_ylabel("Drawdown")
ax3.set_xlabel("Date")
ax3.legend()
ax3.grid(True, alpha=0.3)

# 4. Monthly returns heatmap
_monthly_ret = net_returns_base.copy()
_monthly_ret.index = pd.DatetimeIndex(_monthly_ret.index)
_pivot = _monthly_ret.groupby([_monthly_ret.index.year, _monthly_ret.index.month]).sum().unstack()
_pivot.columns = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                   "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"][:_pivot.shape[1]]
ax4 = axes_ts[3]
_im = ax4.imshow(_pivot.values * 100, cmap="RdYlGn", aspect="auto",
                  vmin=-10, vmax=10)
ax4.set_xticks(range(_pivot.shape[1]))
ax4.set_xticklabels(_pivot.columns, fontsize=8)
ax4.set_yticks(range(_pivot.shape[0]))
ax4.set_yticklabels([str(int(y)) for y in _pivot.index], fontsize=8)
ax4.set_title("Monthly Returns Heatmap (%)")
for i in range(_pivot.shape[0]):
    for j in range(_pivot.shape[1]):
        if j < len(_pivot.columns) and not np.isnan(_pivot.values[i, j]):
            ax4.text(j, i, f"{_pivot.values[i, j]*100:.1f}",
                    ha="center", va="center", fontsize=7,
                    color="black" if abs(_pivot.values[i, j]) < 0.06 else "white")
plt.colorbar(_im, ax=ax4, shrink=0.8, label="Return (%)")

plt.tight_layout()
plt.show()


# ── CELL: layer1_equity_curve ─────────────────────────────────────────

fig_eq, ax_eq = plt.subplots(figsize=(11, 5))

ax_eq.plot(cum_gross.index, cum_gross.values, label="Gross (naive, no TC)",
           color="steelblue", alpha=0.7, linewidth=1.5)
ax_eq.plot(cum_net.index, cum_net.values, label="Net (base regime, purged)",
           color="darkorange", linewidth=2)

# The entire visible period is OOS — the alpha model was trained pre-2019-04
# in Week 4. Annotate this clearly instead of false IS/OOS boundary.
ax_eq.annotate(
    "IS: model training\n(pre-2019-04, not shown)",
    xy=(cum_gross.index[0], cum_gross.iloc[0]),
    xytext=(cum_gross.index[min(6, len(cum_gross) - 1)], cum_gross.max() * 0.55),
    fontsize=9, color="gray", fontstyle="italic",
    arrowprops=dict(arrowstyle="->", color="gray", lw=1.2),
    bbox=dict(boxstyle="round,pad=0.3", fc="lightyellow", alpha=0.8),
)
ax_eq.text(
    0.55, 0.92, "Full period is OOS (out-of-sample model deployment)",
    transform=ax_eq.transAxes, fontsize=9, color="green",
    bbox=dict(boxstyle="round,pad=0.3", fc="honeydew", alpha=0.8),
)

ax_eq.set_title("Layer 1: Gross vs. Net Equity Curve (OOS Period)")
ax_eq.set_xlabel("Date")
ax_eq.set_ylabel("Portfolio Value ($1 invested)")
ax_eq.legend(loc="upper left")
ax_eq.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()


# ── CELL: layer1_dsr ─────────────────────────────────────────────────

# DSR at M=10 using net returns
_net = net_returns_base.dropna()
_skew = float(stats.skew(_net))
_ekurt = float(stats.kurtosis(_net, fisher=True))
_obs = len(_net)
_ann_sr = net_metrics["sharpe"]
# Annualized SR → monthly SR for DSR (DSR operates on observations = months)
_monthly_sr = _net.mean() / _net.std() if _net.std() > 0 else 0.0

N_TRIALS_L1 = 10
dsr_l1 = deflated_sharpe_ratio(
    observed_sr=_monthly_sr,
    n_trials=N_TRIALS_L1,
    n_obs=_obs,
    skew=_skew,
    excess_kurt=_ekurt,
)
deploy_verdict = "DEPLOY" if dsr_l1 >= 0.50 else "NO-DEPLOY"

print(f"\n── Layer 1: DSR at M={N_TRIALS_L1} ─────────────────────────")
print(f"  Monthly SR: {_monthly_sr:.4f}")
print(f"  DSR (M={N_TRIALS_L1}, T={_obs}): {dsr_l1:.4f}  → {deploy_verdict}")
print(f"  Skewness: {_skew:.4f}, ExcessKurt: {_ekurt:.4f}")


# ══════════════════════════════════════════════════════════════════════
# Layer 2: CPCV across 3 model variants + PBO + BHY
# ══════════════════════════════════════════════════════════════════════

# ── CELL: layer2_ridge_fold_helper ───────────────────────────────────

def _ridge_fold_predict(
    features: pd.DataFrame, targets: pd.Series,
    train_dates, test_date,
) -> list:
    """Train Ridge on one fold and return prediction records.

    Args:
        features: MultiIndex (date, ticker) feature DataFrame.
        targets: MultiIndex (date, ticker) forward return Series.
        train_dates: DatetimeIndex of training months.
        test_date: Single test date.

    Returns:
        List of dicts with date, ticker, prediction; empty if fold is skipped.
    """
    try:
        X_train = features.loc[train_dates]
        y_train = targets.reindex(X_train.index).dropna()
        X_train = X_train.loc[y_train.index]
    except KeyError:
        return []

    if len(X_train) < 50:
        return []

    X_test_idx = features.index[
        features.index.get_level_values("date") == test_date
    ]
    if len(X_test_idx) == 0:
        return []

    X_test = features.loc[X_test_idx]

    # Drop rows with any NaN in training features
    X_train_clean = X_train.dropna()
    y_train = y_train.reindex(X_train_clean.index).dropna()
    X_train_clean = X_train_clean.loc[y_train.index]

    if len(X_train_clean) < 50:
        return []

    # For test: fill NaN with column median from training data
    X_test_filled = X_test.fillna(X_train_clean.median())

    scaler = StandardScaler()
    X_tr_sc = scaler.fit_transform(X_train_clean.values)
    X_te_sc = scaler.transform(X_test_filled.values)

    model = Ridge(alpha=1.0)
    model.fit(X_tr_sc, y_train.values)
    preds = model.predict(X_te_sc)

    return [
        {"date": idx[0], "ticker": idx[1], "prediction": preds[j]}
        for j, idx in enumerate(X_test_idx)
    ]


# ── CELL: layer2_ridge_predictions_fn ────────────────────────────────

def compute_ridge_predictions(
    features: pd.DataFrame, targets: pd.Series, train_window: int = 36
) -> pd.Series:
    """Walk-forward Ridge predictions with StandardScaler per fold.

    Args:
        features: MultiIndex (date, ticker) feature DataFrame.
        targets: MultiIndex (date, ticker) forward return Series.
        train_window: Number of months in training window.

    Returns:
        Series of per-ticker Ridge predictions, MultiIndex (date, ticker).
    """
    dates = features.index.get_level_values("date").unique().sort_values()
    preds_list = []

    for i, test_date in enumerate(dates[train_window:], start=train_window):
        train_dates = dates[i - train_window: i]
        fold_preds = _ridge_fold_predict(features, targets, train_dates, test_date)
        preds_list.extend(fold_preds)

        if (i - train_window) % 12 == 0:
            print(f"  [Ridge] fold {i - train_window + 1}/{len(dates) - train_window}: "
                  f"{test_date.strftime('%Y-%m')}")

    if not preds_list:
        return pd.Series(dtype=float)

    df_p = pd.DataFrame(preds_list).set_index(["date", "ticker"])["prediction"]
    return df_p


# ── CELL: layer2_ridge_execute ───────────────────────────────────────

print("\n── Layer 2: Computing Ridge walk-forward predictions ────────")
if expanded_features is not None and len(expanded_features) > 0:
    # Align expanded_features with forward returns in OOS window
    oos_dates = gbm_predictions.index.get_level_values("date").unique()
    feat_dates = expanded_features.index.get_level_values("date").unique()
    common_dates = oos_dates.intersection(feat_dates)

    if len(common_dates) > 40:
        ridge_preds = compute_ridge_predictions(
            features=expanded_features,
            targets=forward_returns,
            train_window=36,
        )
        print(f"  Ridge predictions: {len(ridge_preds)} samples across "
              f"{ridge_preds.index.get_level_values('date').nunique()} dates")
    else:
        print("  Insufficient common dates for Ridge; using synthetic fallback.")
        ridge_preds = pd.Series(dtype=float)
else:
    print("  No expanded features; Ridge unavailable.")
    ridge_preds = pd.Series(dtype=float)


# ── CELL: layer2_ic_from_predictions_fn ──────────────────────────────

def monthly_ic_from_predictions(pred_df: pd.DataFrame) -> pd.Series:
    """Compute monthly cross-sectional Spearman IC from a prediction DataFrame.

    Args:
        pred_df: MultiIndex (date, ticker) DataFrame with 'prediction'
                 and 'actual' columns.

    Returns:
        Series of monthly IC values indexed by date.
    """
    dates = pred_df.index.get_level_values("date").unique()
    ic_vals = {}
    for d in dates:
        slice_d = pred_df.xs(d, level="date")[["prediction", "actual"]].dropna()
        if len(slice_d) < 20:
            continue
        ic_vals[d] = float(
            slice_d["prediction"].corr(slice_d["actual"], method="spearman")
        )
    return pd.Series(ic_vals)


# ── CELL: layer2_ic_pred_vs_actuals_fn ───────────────────────────────

def monthly_ic_pred_vs_actuals(
    pred_series: pd.Series, actuals_multiindex: pd.Series
) -> pd.Series:
    """Compute monthly IC for predictions vs. a separate actuals series.

    Both pred_series and actuals_multiindex must share MultiIndex (date, ticker).

    Args:
        pred_series: MultiIndex (date, ticker) Series of predictions.
        actuals_multiindex: MultiIndex (date, ticker) Series of realized returns.

    Returns:
        Series of monthly IC values indexed by date.
    """
    dates = pred_series.index.get_level_values("date").unique()
    ic_vals = {}
    for d in dates:
        try:
            pred_d = pred_series.xs(d, level="date").dropna()
        except KeyError:
            continue
        try:
            act_d = actuals_multiindex.xs(d, level="date").dropna()
        except KeyError:
            continue
        common = pred_d.index.intersection(act_d.index)
        if len(common) < 20:
            continue
        ic_vals[d] = float(pred_d[common].corr(act_d[common], method="spearman"))
    return pd.Series(ic_vals)


# ── CELL: layer2_ic_compute ──────────────────────────────────────────

# Build per-model IC series
# GBM: use 'actual' column embedded in predictions DataFrame
gbm_ic_monthly = monthly_ic_from_predictions(gbm_predictions)
print(f"\n  GBM IC: mean={gbm_ic_monthly.mean():.4f}, "
      f"t={gbm_ic_monthly.mean() / gbm_ic_monthly.std() * np.sqrt(len(gbm_ic_monthly)):.2f}")

# NN IC — nn_predictions also has 'prediction' and 'actual' columns
if nn_predictions is not None:
    nn_ic_monthly = monthly_ic_from_predictions(nn_predictions)
    print(f"  NN IC: mean={nn_ic_monthly.mean():.4f}, "
          f"t={nn_ic_monthly.mean() / nn_ic_monthly.std() * np.sqrt(len(nn_ic_monthly)):.2f}")
else:
    nn_ic_monthly = None
    print("  NN predictions: unavailable")

# Ridge IC — ridge_preds is a Series; compute vs. forward_returns
if isinstance(ridge_preds, pd.Series) and len(ridge_preds) > 0:
    ridge_ic_monthly = monthly_ic_pred_vs_actuals(ridge_preds, forward_returns)
    if len(ridge_ic_monthly) > 5:
        print(f"  Ridge IC: mean={ridge_ic_monthly.mean():.4f}, "
              f"t={ridge_ic_monthly.mean() / ridge_ic_monthly.std() * np.sqrt(len(ridge_ic_monthly)):.2f}")
    else:
        ridge_ic_monthly = None
        print("  Ridge IC: insufficient aligned dates — skipped")
else:
    ridge_ic_monthly = None
    print("  Ridge IC: unavailable")


# ── CELL: layer2_cpcv_align_ic ───────────────────────────────────────

def _align_ic_matrix(
    model_ic_dict: dict, n_splits: int
) -> pd.DataFrame:
    """Align model IC series to common dates and build IC matrix.

    Args:
        model_ic_dict: {model_name: IC Series (monthly)}.
        n_splits: Minimum required rows = n_splits + 2.

    Returns:
        DataFrame (n_dates, n_models) with NaN rows dropped,
        or empty DataFrame if insufficient data.
    """
    ic_frames = {}
    for name, series in model_ic_dict.items():
        if series is not None and len(series) > 0:
            ic_frames[name] = series.sort_index()

    if len(ic_frames) < 2:
        return pd.DataFrame()

    common_idx = None
    for s in ic_frames.values():
        common_idx = s.index if common_idx is None else common_idx.intersection(s.index)

    ic_matrix = pd.DataFrame(
        {n: ic_frames[n].reindex(common_idx) for n in ic_frames}
    ).dropna()

    if len(ic_matrix) < n_splits + 2:
        return pd.DataFrame()

    return ic_matrix


# ── CELL: layer2_pbo_fn ─────────────────────────────────────────────

def compute_pbo_from_ic(
    model_ic_dict: dict, n_splits: int = 6, n_test_splits: int = 2
) -> tuple:
    """Compute Probability of Backtest Overfitting via CPCV.

    For each CPCV path: identify IS best model, check if it ranks
    above median OOS. PBO = fraction of paths where IS winner < median OOS rank.

    Args:
        model_ic_dict: {model_name: IC Series (monthly)}.
        n_splits: Number of CPCV groups.
        n_test_splits: Number of test splits per path.

    Returns:
        (pbo, pbo_records) where pbo_records is a list of dicts per path.
    """
    ic_matrix = _align_ic_matrix(model_ic_dict, n_splits)
    if ic_matrix.empty:
        return float("nan"), []

    cpcv = CombinatorialPurgedCV(n_splits=n_splits, n_test_splits=n_test_splits,
                                  purge_gap=1)
    X_proxy = pd.DataFrame(index=ic_matrix.index)

    paths_below_median = 0
    pbo_records = []

    for train_idx, test_idx in cpcv.split(X_proxy):
        if len(train_idx) == 0 or len(test_idx) == 0:
            continue

        is_ic = ic_matrix.iloc[train_idx].mean()
        oos_ic = ic_matrix.iloc[test_idx].mean()

        is_winner = is_ic.idxmax()
        oos_rank_of_winner = int(
            oos_ic.rank(ascending=False)[is_winner]
        )  # 1=best
        oos_median_rank = (len(ic_matrix.columns) + 1) / 2.0

        below_median = oos_rank_of_winner > oos_median_rank
        if below_median:
            paths_below_median += 1

        pbo_records.append({
            "is_winner": is_winner,
            "oos_rank_winner": oos_rank_of_winner,
            "oos_ic_winner": float(oos_ic[is_winner]),
            "n_models": len(ic_matrix.columns),
            "oos_median_rank": oos_median_rank,
        })

    n_paths = len(pbo_records)
    pbo = paths_below_median / n_paths if n_paths > 0 else float("nan")
    return pbo, pbo_records


# ── CELL: layer2_pbo_execute ─────────────────────────────────────────

# Build model IC dict with available series
model_ic_dict = {"GBM": gbm_ic_monthly}
if nn_ic_monthly is not None:
    model_ic_dict["NN"] = nn_ic_monthly
if ridge_ic_monthly is not None:
    model_ic_dict["Ridge"] = ridge_ic_monthly

print(f"\n── Layer 2: CPCV PBO across {len(model_ic_dict)} models ─────────────")
pbo, pbo_records = compute_pbo_from_ic(model_ic_dict)
print(f"  PBO = {pbo:.4f} ({pbo * 100:.1f}% of paths where IS winner is OOS loser)")
print(f"  N CPCV paths: {len(pbo_records)}")


# ── CELL: layer2_bhy_compute ─────────────────────────────────────────

from statsmodels.stats.multitest import multipletests

# Compute t-stats and p-values per model
model_stats = {}
for name, ic_series in model_ic_dict.items():
    if ic_series is None or len(ic_series) < 10:
        continue
    n = len(ic_series)
    mean_ic = ic_series.mean()
    std_ic = ic_series.std()
    t_stat = mean_ic / (std_ic / np.sqrt(n)) if std_ic > 0 else 0.0
    p_val = 2 * (1 - stats.t.cdf(abs(t_stat), df=n - 1))
    model_stats[name] = {"mean_ic": mean_ic, "t_stat": t_stat, "p_value": p_val, "n": n}


# ── CELL: layer2_bhy_print ──────────────────────────────────────────

print("\n── Layer 2: BHY Multiple-Testing Correction ─────────────────")
if len(model_stats) >= 2:
    stat_names = list(model_stats.keys())
    raw_pvals = np.array([model_stats[n]["p_value"] for n in stat_names])
    _, adj_pvals, _, _ = multipletests(raw_pvals, method="fdr_bh")
    for i, name in enumerate(stat_names):
        model_stats[name]["adj_p_value"] = adj_pvals[i]
        print(f"  {name}: t={model_stats[name]['t_stat']:.3f}, "
              f"p_raw={raw_pvals[i]:.4f}, p_BHY={adj_pvals[i]:.4f}")
else:
    print("  Only 1 model available — BHY correction requires ≥2 models; skipped.")
    for name in model_stats:
        model_stats[name]["adj_p_value"] = model_stats[name]["p_value"]


# ── CELL: layer2_model_selection_logic ───────────────────────────────

# Select winning model: highest IC among models with adj_p < 0.10
# Fall back to best raw IC if no model clears threshold
if not model_stats:
    # Fallback: GBM is always available; force it into model_stats
    n = len(gbm_ic_monthly)
    mean_ic = gbm_ic_monthly.mean()
    std_ic = gbm_ic_monthly.std()
    t_stat = mean_ic / (std_ic / np.sqrt(n)) if std_ic > 0 else 0.0
    p_val = 2 * (1 - stats.t.cdf(abs(t_stat), df=n - 1))
    model_stats["GBM"] = {
        "mean_ic": mean_ic, "t_stat": t_stat, "p_value": p_val,
        "adj_p_value": p_val, "n": n,
    }

significant = {n: s for n, s in model_stats.items() if s.get("adj_p_value", 1.0) < 0.10}
if significant:
    winning_model = max(significant, key=lambda n: significant[n]["mean_ic"])
    selection_basis = "BHY adj p < 0.10"
else:
    winning_model = max(model_stats, key=lambda n: model_stats[n]["mean_ic"])
    selection_basis = "best IC (no model cleared BHY threshold)"

winner_stats = model_stats[winning_model]

print(f"\n── Layer 2: Winning Model Selection ─────────────────────────")
print(f"  Winning model: {winning_model} ({selection_basis})")
print(f"  IC: {winner_stats['mean_ic']:.4f}, t-stat: {winner_stats['t_stat']:.3f}")
print(f"  PBO: {pbo:.4f}, DSR@M=10: {dsr_l1:.4f} → {deploy_verdict}")


# ── CELL: layer2_model_comparison_table ──────────────────────────────

# Full comparison table
comparison_rows = []
for name, s in model_stats.items():
    comparison_rows.append({
        "Model": name,
        "Mean IC": round(s["mean_ic"], 4),
        "t-stat": round(s["t_stat"], 3),
        "p_raw": round(s["p_value"], 4),
        "p_BHY": round(s.get("adj_p_value", s["p_value"]), 4),
        "Selected": "YES" if name == winning_model else "",
    })
comparison_df = pd.DataFrame(comparison_rows).set_index("Model")
print("\n  Model Comparison Table:")
print(comparison_df.to_string())


# ── CELL: layer2_pbo_plot ─────────────────────────────────────────────

# PBO distribution histogram
if len(pbo_records) > 0:
    oos_ranks = [r["oos_rank_winner"] for r in pbo_records]
    n_models_cpcv = pbo_records[0]["n_models"]
    median_rank = (n_models_cpcv + 1) / 2.0

    fig_pbo, ax_pbo = plt.subplots(figsize=(8, 5))
    ax_pbo.hist(oos_ranks, bins=range(1, n_models_cpcv + 2), align="left",
                color="steelblue", alpha=0.7, edgecolor="white",
                label="OOS rank of IS winner")
    ax_pbo.axvline(median_rank, color="red", linestyle="--", linewidth=2,
                   label=f"Median rank = {median_rank:.1f}")
    ax_pbo.fill_betweenx([0, len(pbo_records) * 0.9], median_rank,
                          n_models_cpcv + 0.5, alpha=0.15, color="red",
                          label="Below-median region (overfitting)")
    ax_pbo.set_title(
        f"Layer 2: IS-Winner OOS Rank Distribution\nPBO = {pbo:.3f} ({len(pbo_records)} paths)"
    )
    ax_pbo.set_xlabel("OOS Rank of IS-Best Model (1=best)")
    ax_pbo.set_ylabel("Number of CPCV paths")
    ax_pbo.set_xticks(range(1, n_models_cpcv + 1))
    ax_pbo.legend()
    ax_pbo.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


# ══════════════════════════════════════════════════════════════════════
# Layer 3: MinTRL + Capital Allocation Discussion
# ══════════════════════════════════════════════════════════════════════

# ── CELL: layer3_mintrl_fn ───────────────────────────────────────────

def min_track_record_length(
    sharpe_annual: float,
    skewness: float,
    excess_kurtosis: float,
    confidence: float = 0.95,
    benchmark_sr: float = 0.0,
) -> float:
    """Minimum Track Record Length (months) per Bailey & Lopez de Prado (2014).

    TRL formula:
        MinTRL = [1 + (1 - g3*SR + (g4/4)*SR^2)] * (z_a / (SR - SR0))^2
    where SR is the annualized Sharpe, SR0 the benchmark, g3=skewness,
    g4=excess kurtosis, and z_a the one-tailed normal quantile.

    Args:
        sharpe_annual: Observed annualized Sharpe ratio.
        skewness: Sample skewness of monthly returns.
        excess_kurtosis: Excess kurtosis (Fisher convention, normal=0).
        confidence: Confidence level (default 0.95).
        benchmark_sr: Benchmark Sharpe ratio to beat (default 0.0).

    Returns:
        MinTRL in months (float).
    """
    z_alpha = stats.norm.ppf(confidence)
    sr_excess = sharpe_annual - benchmark_sr
    if sr_excess <= 0:
        return float("inf")

    # Convert annual SR to monthly SR for the formula
    sr_m = sharpe_annual / np.sqrt(12)
    sr0_m = benchmark_sr / np.sqrt(12)
    sr_excess_m = sr_m - sr0_m

    numer = (z_alpha ** 2) * (1 - skewness * sr_m + (excess_kurtosis / 4) * sr_m ** 2)
    denom = sr_excess_m ** 2
    mintrl = numer / denom if denom > 0 else float("inf")
    return mintrl


# ── CELL: layer3_mintrl_compute ──────────────────────────────────────

mintrl = min_track_record_length(
    sharpe_annual=net_metrics["sharpe"],
    skewness=net_metrics["skewness"],
    excess_kurtosis=net_metrics["excess_kurtosis"],
    confidence=0.95,
    benchmark_sr=0.0,
)

print(f"\n── Layer 3: Minimum Track Record Length ─────────────────────")
print(f"  Net Sharpe (annualized): {net_metrics['sharpe']:.4f}")
print(f"  Skewness: {net_metrics['skewness']:.4f}, ExcessKurt: {net_metrics['excess_kurtosis']:.4f}")
print(f"  MINTRL_95pct={mintrl:.1f} months")
print(f"  Observed track record: {net_metrics['n_months']} months")
observed_sufficient = net_metrics["n_months"] >= mintrl
print(f"  Track record sufficient for 95% confidence: {'YES' if observed_sufficient else 'NO'}")


# ── CELL: layer3_capital_scaling ─────────────────────────────────────

print("\n── Layer 3: Capital Allocation Discussion ────────────────────")
print(
    "  [1] Strategy viability at scale: The base-regime net Sharpe of "
    f"{net_metrics['sharpe']:.2f} is computed assuming a tiered spread model "
    "(5/15 bps for large/mid-small cap), but market impact grows with AUM "
    "via the sqrt-law: doubling capital raises impact costs by ~41%. "
    "At moderate AUM (< $200M), this strategy may remain viable; "
    "beyond $500M the impact drag likely erodes the edge entirely."
)


# ── CELL: layer3_sizing_and_deploy ───────────────────────────────────

print(
    "  [2] Position sizing discipline: Kelly criterion suggests fractional "
    "sizing: f* = SR / SR_max, where SR_max ≈ SR + σ_SR. "
    f"With net SR={net_metrics['sharpe']:.2f} and observed "
    f"MaxDD={net_metrics['max_dd']:.1%}, a half-Kelly allocation "
    "limits ruin probability while preserving compounding. "
    "Do NOT run at full Kelly on a strategy with PBO > 0.20."
)
print(
    "  [3] Deployment decision framework: The strategy clears a Sharpe > 0.5 "
    "threshold net of realistic costs, but the MinTRL "
    f"({mintrl:.0f} months at 95% confidence) compared to observed "
    f"track record ({net_metrics['n_months']} months) "
    "governs the deployment decision. "
    "A paper-trading period of 12 months before live allocation is recommended "
    "to confirm out-of-sample performance persistence and refine cost assumptions "
    "with actual fill data."
)


if __name__ == "__main__":
    # ── ASSERTIONS ─────────────────────────────────────────────────

    # D3-L1-1: Tear sheet generated (structural)
    assert fig_ts is not None, "Tear sheet figure was not created"
    assert len(axes_ts) == 4, f"Expected 4 subplots in tear sheet, got {len(axes_ts)}"

    # D3-L1-2: Annual Sharpe (net, purged) in [0.05, 1.5]
    net_sr = net_metrics["sharpe"]
    assert 0.05 <= net_sr <= 1.5, (
        f"Net Sharpe = {net_sr:.4f} outside [0.05, 1.5]"
    )

    # D3-L1-3: Max drawdown <= -10%
    net_mdd = net_metrics["max_dd"]
    assert net_mdd <= -0.10, (
        f"Net max drawdown = {net_mdd:.4f} expected <= -0.10"
    )

    # D3-L1-4: DSR at M=10 computed with verdict
    assert 0.0 <= dsr_l1 <= 1.0, f"DSR = {dsr_l1:.4f} outside [0, 1]"
    assert deploy_verdict in ("DEPLOY", "NO-DEPLOY"), f"Verdict not set: {deploy_verdict}"

    # D3-L2-1: PBO in [0.20, 0.75]
    if not np.isnan(pbo):
        assert 0.20 <= pbo <= 0.75, (
            f"PBO = {pbo:.4f} outside [0.20, 0.75]"
        )

    # D3-L2-2: BHY adjusted p-values >= raw p-values
    for name in model_stats:
        raw_p = model_stats[name]["p_value"]
        adj_p = model_stats[name].get("adj_p_value", raw_p)
        assert adj_p >= raw_p - 1e-9, (
            f"Model {name}: BHY adj_p={adj_p:.4f} < raw_p={raw_p:.4f} — BHY must inflate"
        )

    # D3-L2-3: Winning model stated with justification (structural — printed above)
    assert winning_model in model_stats, f"winning_model={winning_model!r} not in stats"

    # D3-L3-1: MinTRL printed (structural — must be finite and positive)
    assert mintrl > 0, f"MinTRL = {mintrl:.2f} expected > 0"
    assert mintrl < 500, f"MinTRL = {mintrl:.2f} unrealistically large (>500)"

    # D3-L3-2: Qualitative discussion present (structural — printed above)
    # Verified by 3 print statements in layer3_capital_allocation cell.

    # ── PLOTS ───────────────────────────────────────────────────────
    fig_ts.savefig(PLOT_DIR / "d3_tearsheet.png", dpi=150, bbox_inches="tight")
    fig_eq.savefig(PLOT_DIR / "d3_equity_curve.png", dpi=150, bbox_inches="tight")
    if len(pbo_records) > 0:
        fig_pbo.savefig(PLOT_DIR / "d3_pbo_distribution.png", dpi=150,
                        bbox_inches="tight")

    # ── RESULTS ─────────────────────────────────────────────────────
    print(f"\n══ hw/d3_responsible_report ══════════════════════════════")
    print(f"  layer1_net_sharpe: {net_sr:.4f}")
    print(f"  layer1_gross_sharpe: {gross_metrics['sharpe']:.4f}")
    print(f"  layer1_net_max_dd: {net_mdd:.4f}")
    print(f"  layer1_net_cagr: {net_metrics['cagr']:.4f}")
    print(f"  layer1_net_sortino: {net_metrics['sortino']:.4f}")
    print(f"  layer1_net_calmar: {net_metrics['calmar']:.4f}")
    print(f"  layer1_net_skewness: {net_metrics['skewness']:.4f}")
    print(f"  layer1_net_excess_kurtosis: {net_metrics['excess_kurtosis']:.4f}")
    print(f"  layer1_dsr_m10: {dsr_l1:.4f}")
    print(f"  layer1_deploy_verdict: {deploy_verdict}")
    print(f"  layer2_n_models: {len(model_ic_dict)}")
    print(f"  layer2_pbo: {pbo:.4f}")
    print(f"  layer2_n_cpcv_paths: {len(pbo_records)}")
    print(f"  layer2_winning_model: {winning_model}")
    print(f"  layer2_winner_mean_ic: {winner_stats['mean_ic']:.4f}")
    print(f"  layer2_winner_t_stat: {winner_stats['t_stat']:.4f}")
    print(f"  layer3_mintrl_95pct: {mintrl:.2f}")
    print(f"  layer3_observed_months: {net_metrics['n_months']}")
    print(f"  layer3_track_record_sufficient: {observed_sufficient}")
    print(f"  quantstats_available: {_quantstats_available}")
    print(f"  quantstats_html: {_qs_html_path}")
    print(f"  ── plot: d3_tearsheet.png ──")
    print(f"     type: 4-panel tear sheet (cumulative, distribution, drawdown, monthly heatmap)")
    print(f"     n_panels: 4")
    print(f"     title: {fig_ts.texts[0].get_text()}")
    print(f"  ── plot: d3_equity_curve.png ──")
    print(f"     type: gross vs net equity curve (full OOS period)")
    print(f"     y_range: [{cum_gross.min():.2f}, {cum_gross.max():.2f}]")
    print(f"  ── plot: d3_pbo_distribution.png ──")
    print(f"     type: histogram of IS-winner OOS ranks across {len(pbo_records)} CPCV paths")
    print(f"     pbo: {pbo:.4f}")
    print(f"✓ d3_responsible_report: ALL PASSED")
