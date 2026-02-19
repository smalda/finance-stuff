"""Section 5: Feature Engineering — Interaction Terms, Non-Linearities, and Interpretability"""
import json
import matplotlib
matplotlib.use("Agg")
import sys
import warnings
from pathlib import Path

import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from scipy import stats
from sklearn.inspection import permutation_importance

sys.path.insert(0, str(Path(__file__).parent.parent))
from data_setup import (
    CACHE_DIR, PLOT_DIR, FEATURE_COLS, TRAIN_WINDOW, PURGE_GAP,
    load_monthly_panel,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from shared.temporal import walk_forward_splits

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# Load baseline IC series and best params from S3
baseline_ic = pd.read_parquet(CACHE_DIR / "gbm_ic_series.parquet")
baseline_ic = baseline_ic["ic"] if "ic" in baseline_ic.columns else baseline_ic.iloc[:, 0]
baseline_ic.name = "ic"

with open(CACHE_DIR / "gbm_best_params.json") as f:
    best_params = json.load(f)

print(f"[s5] Loaded S3 baseline: mean IC = {baseline_ic.mean():.4f}, "
      f"best_params = {best_params}")


# ── CELL: load_and_expand ─────────────────────────────────────────
# Load panel and extend with interaction and non-linear terms.
# Interactions capture cross-factor effects; non-linearities capture
# diminishing/amplifying returns at extremes.

panel = load_monthly_panel()
dates = panel.index.get_level_values("date").unique().sort_values()
features = panel[FEATURE_COLS].copy()
target = panel["fwd_return"]

# Interaction terms (computed from z-scored features, so products are meaningful)
features["mom_x_vol"] = features["momentum_z"] * features["volatility_z"]
features["ey_x_pb"] = features["earnings_yield_z"] * features["pb_ratio_z"]
features["roe_x_ag"] = features["roe_z"] * features["asset_growth_z"]

# Non-linear terms (capture asymmetric/extreme effects)
features["momentum_sq"] = features["momentum_z"] ** 2
features["abs_reversal"] = features["reversal_z"].abs()

EXPANDED_COLS = list(features.columns)
n_expanded = len(EXPANDED_COLS)


# ── CELL: expand_features_verify ──────────────────────────────────
# Verify no NaN introduced by interactions (z-scored inputs may have NaN,
# but interactions on NaN stay NaN — no new NaN beyond what existed)

nan_before = panel[FEATURE_COLS].isna().sum().sum()
nan_after = features.isna().sum().sum()
new_nan_from_interactions = nan_after - nan_before
orig_finite = panel[FEATURE_COLS].notna().all(axis=1)
interaction_cols = ["mom_x_vol", "ey_x_pb", "roe_x_ag",
                    "momentum_sq", "abs_reversal"]
interaction_nan_on_finite = features.loc[orig_finite, interaction_cols].isna().sum().sum()

print(f"\n── EXPANDED FEATURES ────────────────────────────────")
print(f"  Original features: {len(FEATURE_COLS)}")
print(f"  Expanded features: {n_expanded}")
print(f"  Columns: {EXPANDED_COLS}")
print(f"  Interaction NaN on finite originals: {interaction_nan_on_finite}")
print()


# ── CELL: wf_expanded_one_window ──────────────────────────────────
# Helper: run one walk-forward window — fit GBM with early stopping, predict OOS.

def _wf_expanded_one_window(panel, features, target, best_params, train_dates, pred_date):
    """Fit GBM on one walk-forward window, return predictions and IC values."""
    train_mask = panel.index.get_level_values("date").isin(train_dates)
    pred_mask = panel.index.get_level_values("date") == pred_date

    n_val_months = min(12, len(train_dates) // 5)
    val_dates = train_dates[-n_val_months:]
    fit_dates = train_dates[:-n_val_months]

    fit_mask = panel.index.get_level_values("date").isin(fit_dates)
    val_mask = panel.index.get_level_values("date").isin(val_dates)

    X_fit = features.loc[fit_mask].values
    y_fit = target.loc[fit_mask].values
    X_val = features.loc[val_mask].values
    y_val = target.loc[val_mask].values

    model = lgb.LGBMRegressor(
        n_estimators=500,
        learning_rate=best_params["learning_rate"],
        num_leaves=best_params["num_leaves"],
        min_child_samples=20,
        reg_alpha=0.1, reg_lambda=1.0,
        subsample=0.8, subsample_freq=1,
        colsample_bytree=0.8, random_state=42, verbosity=-1,
    )
    model.fit(
        X_fit, y_fit,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(50, verbose=False)],
    )

    X_pred = features.loc[pred_mask].values
    y_pred = target.loc[pred_mask].values
    tickers_pred = panel.loc[pred_mask].index.get_level_values("ticker")
    preds = model.predict(X_pred)

    pred_df = pd.DataFrame({
        "date": pred_date, "ticker": tickers_pred,
        "prediction": preds, "actual": y_pred,
    })

    oos_ic = np.corrcoef(preds, y_pred)[0, 1]
    train_preds = model.predict(X_fit)
    train_ic = np.corrcoef(train_preds, y_fit)[0, 1]

    return pred_df, oos_ic, train_ic, model, X_pred, y_pred


# ── CELL: walk_forward_expanded_loop ──────────────────────────────
# Walk-forward GBM with expanded feature set, reusing S3's best HPs.

oos_predictions = []
oos_ic_list = []
train_ic_list = []
recent_models = []  # store last N models for pooled SHAP

splits = list(walk_forward_splits(dates, TRAIN_WINDOW, PURGE_GAP))
n_splits = len(splits)
N_SHAP_MODELS = 12  # pool SHAP across last 12 windows for stability

print(f"── WALK-FORWARD (expanded, {n_splits} windows) ─────────")

for i, (train_dates, pred_date) in enumerate(splits):
    if i % 10 == 0:
        print(f"  [{i+1}/{n_splits}] predicting "
              f"{pd.Timestamp(pred_date).date()}")

    pred_df, oos_ic, train_ic, model, X_pred, y_pred = \
        _wf_expanded_one_window(panel, features, target, best_params,
                                train_dates, pred_date)
    oos_predictions.append(pred_df)
    if np.isfinite(oos_ic):
        oos_ic_list.append({"date": pred_date, "ic": oos_ic})
    if np.isfinite(train_ic):
        train_ic_list.append(train_ic)
    if i >= n_splits - N_SHAP_MODELS:
        recent_models.append((model, X_pred, y_pred))

# Last model and data for the beeswarm visualization
last_model = model
last_X_pred = X_pred
last_y_pred = y_pred

print(f"  Walk-forward complete: {len(oos_ic_list)} OOS months")
print()


# ── CELL: expanded_ic_stats ────────────────────────────────────────
# Aggregate IC statistics for the expanded model and compare to baseline.

all_preds = pd.concat(oos_predictions, ignore_index=True)
all_preds = all_preds.set_index(["date", "ticker"])

expanded_ic = pd.DataFrame(oos_ic_list).set_index("date")["ic"]

mean_ic_exp = expanded_ic.mean()
std_ic_exp = expanded_ic.std()
icir_exp = mean_ic_exp / std_ic_exp if std_ic_exp > 0 else np.nan
pct_pos_exp = (expanded_ic > 0).mean()
n_oos = len(expanded_ic)

# Baseline comparison (align dates)
common_dates = expanded_ic.index.intersection(baseline_ic.index)
ic_change = expanded_ic.loc[common_dates].mean() - baseline_ic.loc[common_dates].mean()

print(f"── EXPANDED MODEL IC ────────────────────────────────")
print(f"  OOS months: {n_oos}")
print(f"  Mean IC (expanded): {mean_ic_exp:.4f}")
print(f"  Mean IC (baseline S3): {baseline_ic.loc[common_dates].mean():.4f}")
print(f"  IC change (expanded - baseline): {ic_change:.4f}")
print(f"  Std IC: {std_ic_exp:.4f}")
print(f"  ICIR: {icir_exp:.4f}")
print(f"  pct_positive: {pct_pos_exp:.4f}")
print()


# ── CELL: pooled_shap_compute ─────────────────────────────────────
# Pool SHAP values across last 12 walk-forward windows for stability.
# Single cross-sections (~174 stocks) are noisy; pooling gives ~2000 obs.

pooled_shap = []
pooled_X = []
for mdl, X_oos, _ in recent_models:
    ex = shap.TreeExplainer(mdl)
    sv = ex.shap_values(X_oos)
    pooled_shap.append(sv)
    pooled_X.append(X_oos)

pooled_shap_arr = np.concatenate(pooled_shap, axis=0)
pooled_X_arr = np.concatenate(pooled_X, axis=0)


# ── CELL: shap_ranking ───────────────────────────────────────────
# Mean |SHAP| computation and per-feature ranking.

mean_abs_shap = np.abs(pooled_shap_arr).mean(axis=0)
shap_ranking = pd.Series(mean_abs_shap, index=EXPANDED_COLS)
shap_ranking = shap_ranking.sort_values(ascending=False)
top5_shap = shap_ranking.head(5)


# ── CELL: shap_parent_grouping ───────────────────────────────────
# Map engineered features back to parent originals for dominance check.
# Interactions inherit SHAP from their constituent features, so grouping
# reveals whether the underlying original signal or the interaction itself
# is the primary driver.

parent_map = {
    "mom_x_vol": ["momentum_z", "volatility_z"],
    "ey_x_pb": ["earnings_yield_z", "pb_ratio_z"],
    "roe_x_ag": ["roe_z", "asset_growth_z"],
    "momentum_sq": ["momentum_z"],
    "abs_reversal": ["reversal_z"],
}
grouped_shap = {}
for feat in FEATURE_COLS:
    grouped_shap[feat] = mean_abs_shap[EXPANDED_COLS.index(feat)]
for eng_feat, parents in parent_map.items():
    contrib = mean_abs_shap[EXPANDED_COLS.index(eng_feat)] / len(parents)
    for p in parents:
        grouped_shap[p] += contrib

grouped_ranking = pd.Series(grouped_shap).sort_values(ascending=False)
top2_grouped = grouped_ranking.index[:2].tolist()
top2_are_original = all(f in FEATURE_COLS for f in top2_grouped)

print(f"── SHAP FEATURE IMPORTANCE (pooled, {len(pooled_X_arr)} obs) ──")
print(f"  Per-feature mean |SHAP|:")
for rank, (feat, val) in enumerate(shap_ranking.items(), 1):
    origin = "original" if feat in FEATURE_COLS else "engineered"
    print(f"    {rank:2d}. {feat}: {val:.6f} ({origin})")
print(f"\n  Grouped by parent original feature:")
for rank, (feat, val) in enumerate(grouped_ranking.items(), 1):
    print(f"    {rank}. {feat}: {val:.6f}")
print(f"  Top-2 (grouped) are original features: {top2_are_original}")
print()


# ── CELL: shap_last_cross_section ────────────────────────────────
# Single cross-section SHAP for beeswarm plot (visual only)

last_explainer = shap.TreeExplainer(last_model)
last_shap_values = last_explainer.shap_values(last_X_pred)


# ── CELL: shap_beeswarm_plot ──────────────────────────────────────
# Beeswarm plot showing SHAP value distribution per feature.

fig_shap, ax_shap = plt.subplots(figsize=(10, 7))
shap.summary_plot(
    last_shap_values,
    features=pd.DataFrame(last_X_pred, columns=EXPANDED_COLS),
    show=False,
    plot_size=None,
)
fig_shap = plt.gcf()
ax_shap = fig_shap.axes[0] if fig_shap.axes else ax_shap
ax_shap.set_title("SHAP Feature Importance (Last OOS Cross-Section)")
plt.tight_layout()
plt.show()


# ── CELL: shap_bar_chart ─────────────────────────────────────────
# Bar chart of mean |SHAP| by feature for clearer ranking.

fig_bar, ax_bar = plt.subplots(figsize=(10, 6))
colors_bar = ["#1f77b4" if f in FEATURE_COLS else "#ff7f0e"
              for f in shap_ranking.index]
ax_bar.barh(range(len(shap_ranking)), shap_ranking.values, color=colors_bar)
ax_bar.set_yticks(range(len(shap_ranking)))
ax_bar.set_yticklabels(shap_ranking.index)
ax_bar.invert_yaxis()
ax_bar.set_xlabel("Mean |SHAP value|")
ax_bar.set_title("Feature Importance: Original (blue) vs Engineered (orange)")
plt.tight_layout()
plt.show()


# ── CELL: compute_permutation_importance ─────────────────────────
# Permutation importance pooled across last 12 OOS windows.
# Using pooled data matches the SHAP analysis and gives enough
# observations (~2000) for stable importance estimates.

pooled_y = np.concatenate([y for _, _, y in recent_models])
perm_result = permutation_importance(
    last_model, pooled_X_arr, pooled_y,
    n_repeats=30, random_state=42,
    scoring="neg_mean_squared_error",
)
perm_imp = pd.Series(perm_result.importances_mean, index=EXPANDED_COLS)
perm_imp = perm_imp.sort_values(ascending=False)
top5_perm = perm_imp.head(5)


# ── CELL: grouped_permutation ────────────────────────────────────
# Group permutation importance by parent originals (same logic as SHAP)

grouped_perm = {}
for feat in FEATURE_COLS:
    grouped_perm[feat] = perm_imp[feat]
for eng_feat, parents in parent_map.items():
    contrib = perm_imp[eng_feat] / len(parents)
    for p in parents:
        grouped_perm[p] += contrib
grouped_perm_ranking = pd.Series(grouped_perm).sort_values(ascending=False)

# Rank correlation between grouped SHAP and grouped permutation importance
rank_corr = stats.spearmanr(
    grouped_ranking.loc[FEATURE_COLS],
    grouped_perm_ranking.loc[FEATURE_COLS],
)[0]

print(f"── PERMUTATION IMPORTANCE (pooled OOS) ──────────────")
print(f"  Per-feature permutation importance:")
for rank, (feat, val) in enumerate(perm_imp.items(), 1):
    print(f"    {rank:2d}. {feat}: {val:.6f}")
print(f"\n  Grouped by parent original feature:")
for rank, (feat, val) in enumerate(grouped_perm_ranking.items(), 1):
    print(f"    {rank}. {feat}: {val:.6f}")
print(f"  Rank corr (grouped SHAP vs grouped Perm): {rank_corr:.4f}")
print()


if __name__ == "__main__":
    # ── ASSERTIONS ─────────────────────────────────────
    # S5-1: Expanded feature matrix has 10-12 columns; no NaN from interactions
    assert 10 <= n_expanded <= 12, \
        f"S5-1: n_expanded = {n_expanded}, expected [10, 12]"
    assert interaction_nan_on_finite == 0, \
        (f"S5-1: {interaction_nan_on_finite} NaN created by interactions "
         f"on finite originals")

    # S5-2: Mean OOS IC (expanded model): [0.005, 0.07]
    assert 0.005 <= mean_ic_exp <= 0.07, \
        f"S5-2: Mean IC = {mean_ic_exp:.4f}, expected [0.005, 0.07]"

    # S5-3: IC change (expanded minus baseline): [-0.015, +0.025]
    assert -0.015 <= ic_change <= 0.025, \
        f"S5-3: IC change = {ic_change:.4f}, expected [-0.015, 0.025]"

    # S5-4: SHAP summary plot produced; top-5 features reported
    assert len(top5_shap) == 5, \
        f"S5-4: top-5 SHAP has {len(top5_shap)} entries, expected 5"

    # S5-5: SHAP dominated by 2-4 original features (top-2 grouped are original)
    assert top2_are_original, \
        f"S5-5: Top-2 grouped features are {top2_grouped}, expected original features"

    # S5-6: Permutation importance rank corr with SHAP > 0.5
    assert rank_corr > 0.5, \
        f"S5-6: Rank corr = {rank_corr:.4f}, expected > 0.5"

    # ── RESULTS ────────────────────────────────────────
    print(f"══ lecture/s5_feature_engineering ═══════════════════")
    print(f"  n_expanded_features: {n_expanded}")
    print(f"  expanded_columns: {EXPANDED_COLS}")
    print(f"  interaction_nan_on_finite: {interaction_nan_on_finite}")
    print(f"  n_oos_months: {n_oos}")
    print(f"  mean_ic_expanded: {mean_ic_exp:.4f}")
    print(f"  mean_ic_baseline: {baseline_ic.loc[common_dates].mean():.4f}")
    print(f"  ic_change: {ic_change:.4f}")
    print(f"  std_ic: {std_ic_exp:.4f}")
    print(f"  icir: {icir_exp:.4f}")
    print(f"  pct_positive: {pct_pos_exp:.4f}")
    print(f"  mean_train_ic: {np.mean(train_ic_list):.4f}")
    print(f"  best_params_reused: {best_params}")
    print(f"  top5_shap_features: {list(top5_shap.index)}")
    print(f"  top2_grouped_features: {top2_grouped}")
    print(f"  top2_are_original: {top2_are_original}")
    print(f"  rank_corr_shap_perm_grouped: {rank_corr:.4f}")

    # ── PLOT ───────────────────────────────────────────
    fig_shap.savefig(PLOT_DIR / "s5_shap_beeswarm.png",
                     dpi=150, bbox_inches="tight")
    print(f"  ── plot: s5_shap_beeswarm.png ──")
    print(f"     type: SHAP beeswarm (summary plot)")
    print(f"     n_features: {n_expanded}")
    print(f"     n_samples: {len(last_X_pred)}")
    print(f"     title: {ax_shap.get_title()}")

    fig_bar.savefig(PLOT_DIR / "s5_shap_bar.png",
                    dpi=150, bbox_inches="tight")
    print(f"  ── plot: s5_shap_bar.png ──")
    print(f"     type: horizontal bar chart (mean |SHAP| by feature)")
    print(f"     n_bars: {len(ax_bar.patches)}")
    print(f"     x_range: [{ax_bar.get_xlim()[0]:.6f}, "
          f"{ax_bar.get_xlim()[1]:.6f}]")
    print(f"     title: {ax_bar.get_title()}")

    print(f"✓ s5_feature_engineering: ALL PASSED")
