"""Exercise 2: Feature Knockout — Leave-One-Out Importance via Retraining"""
import matplotlib
matplotlib.use("Agg")
import sys
import time
from pathlib import Path

import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).parent.parent))
from data_setup import (
    CACHE_DIR, PLOT_DIR, FEATURE_COLS, TRAIN_WINDOW, PURGE_GAP,
    load_monthly_panel,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from shared.temporal import walk_forward_splits

# ── S3-compatible hyperparameters (pre-set, no re-tuning per knockout) ────
GBM_PARAMS = dict(
    objective="regression",
    learning_rate=0.05,
    num_leaves=31,
    n_estimators=500,
    min_child_samples=20,
    subsample=0.8,
    colsample_bytree=0.8,
    verbosity=-1,
)
EARLY_STOPPING_ROUNDS = 50

# ── Controls (set DEBUG = True for quick validation) ──────────────────────
DEBUG = False
DEBUG_WINDOWS = 5


# ── Helpers ───────────────────────────────────────────────────────────────

def _run_walk_forward(panel: pd.DataFrame, features: list[str],
                      dates: np.ndarray, label: str) -> pd.Series:
    """Train GBM walk-forward and return monthly OOS IC (Spearman) series."""
    splits = list(walk_forward_splits(dates, TRAIN_WINDOW, PURGE_GAP))
    if DEBUG:
        splits = splits[:DEBUG_WINDOWS]

    ic_values, ic_dates = [], []
    n_splits = len(splits)
    t0 = time.time()

    for i, (train_dates, pred_date) in enumerate(splits):
        if i % 10 == 0:
            elapsed = time.time() - t0
            print(f"    [{label}] window {i+1}/{n_splits}  "
                  f"({elapsed:.0f}s elapsed)")

        train_mask = panel.index.get_level_values("date").isin(train_dates)
        pred_mask = panel.index.get_level_values("date") == pred_date

        X_train = panel.loc[train_mask, features]
        y_train = panel.loc[train_mask, "fwd_return"]
        X_pred = panel.loc[pred_mask, features]
        y_pred = panel.loc[pred_mask, "fwd_return"].values

        if len(X_pred) < 10:
            continue

        # Split last 12 months of training as early-stop validation
        unique_train = sorted(set(train_dates))
        if len(unique_train) > 12:
            val_dates = unique_train[-12:]
            val_mask = panel.index.get_level_values("date").isin(val_dates)
            fit_mask = train_mask & ~val_mask
            X_fit = panel.loc[fit_mask, features]
            y_fit = panel.loc[fit_mask, "fwd_return"]
            X_val = panel.loc[val_mask, features]
            y_val = panel.loc[val_mask, "fwd_return"]
        else:
            X_fit, y_fit = X_train.copy(), y_train.copy()
            X_val, y_val = None, None

        model = lgb.LGBMRegressor(**GBM_PARAMS)
        fit_kwargs = {}
        if X_val is not None and len(X_val) > 0:
            fit_kwargs["eval_set"] = [(X_val, y_val)]
            fit_kwargs["callbacks"] = [
                lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False),
                lgb.log_evaluation(period=-1),
            ]
        model.fit(X_fit, y_fit, **fit_kwargs)

        preds = model.predict(X_pred)
        if np.std(preds) < 1e-12:
            ic_values.append(0.0)
        else:
            ic, _ = spearmanr(preds, y_pred)
            ic_values.append(ic if np.isfinite(ic) else 0.0)
        ic_dates.append(pred_date)

    elapsed = time.time() - t0
    print(f"    [{label}] done — {n_splits} windows in {elapsed:.1f}s")
    return pd.Series(ic_values, index=pd.DatetimeIndex(ic_dates), name=label)


# ── CELL: load_data_and_baseline ──────────────────────────────────────────

panel = load_monthly_panel()
dates = panel.index.get_level_values("date").unique().sort_values()
all_features = list(FEATURE_COLS)

print(f"Panel shape: {panel.shape}")
print(f"Features: {all_features}")
print(f"Unique dates: {len(dates)}")

print("\n── Running baseline model (all 7 features) ──")
baseline_ic = _run_walk_forward(panel, all_features, dates, "baseline")
baseline_mean_ic = baseline_ic.mean()
print(f"  Baseline mean OOS IC: {baseline_mean_ic:.4f}")


# ── CELL: loo_knockout ────────────────────────────────────────────────────

print("\n── Leave-One-Out Feature Knockout ──")
loo_results = {}

for feat in all_features:
    reduced = [f for f in all_features if f != feat]
    print(f"\n  Dropping: {feat}")
    ic_series = _run_walk_forward(panel, reduced, dates, f"LOO-{feat}")
    mean_ic = ic_series.mean()
    ic_drop = baseline_mean_ic - mean_ic
    loo_results[feat] = {
        "mean_ic": mean_ic,
        "ic_drop": ic_drop,
        "ic_series": ic_series,
    }
    print(f"    Mean IC without {feat}: {mean_ic:.4f}  "
          f"(drop: {ic_drop:+.4f})")


# ── CELL: feature_correlation_matrix ──────────────────────────────────────

print("\n── Feature Correlation Matrix (7x7) ──")
# Cross-sectional correlations averaged across months
corr_by_month = []
for dt in dates:
    month_data = panel.xs(dt, level="date")[all_features]
    if len(month_data) > 10:
        corr_by_month.append(month_data.corr(method="spearman"))

avg_corr = pd.concat(corr_by_month).groupby(level=0).mean()
avg_corr = avg_corr.reindex(index=all_features, columns=all_features)

print(avg_corr.round(3).to_string())


# ── CELL: identify_top2_correlated ────────────────────────────────────────

# Find the most correlated pair (absolute value, excluding diagonal)
upper_tri = avg_corr.where(
    np.triu(np.ones(avg_corr.shape, dtype=bool), k=1)
)
max_corr_idx = upper_tri.abs().stack().idxmax()
top2_features = list(max_corr_idx)
top2_corr_val = upper_tri.loc[max_corr_idx]

print(f"\nTop-2 most correlated features: {top2_features}")
print(f"  Correlation: {top2_corr_val:.3f}")


# ── CELL: leave_two_out ──────────────────────────────────────────────────

print("\n── Leave-Two-Out: Removing top-2 correlated features jointly ──")
reduced_joint = [f for f in all_features if f not in top2_features]
print(f"  Remaining features: {reduced_joint}")

joint_ic = _run_walk_forward(panel, reduced_joint, dates, "L2O-joint")
joint_mean_ic = joint_ic.mean()
joint_drop = baseline_mean_ic - joint_mean_ic

# Individual drops for these two features
indiv_drop_sum = (loo_results[top2_features[0]]["ic_drop"]
                  + loo_results[top2_features[1]]["ic_drop"])

# Substitution ratio: joint drop / sum of individual drops
# Avoid division by zero — if individual drops sum to ~0, ratio is undefined
if abs(indiv_drop_sum) > 1e-8:
    substitution_ratio = joint_drop / indiv_drop_sum
else:
    substitution_ratio = float("nan")

print(f"  Joint mean IC: {joint_mean_ic:.4f}")
print(f"  Joint IC drop: {joint_drop:+.4f}")
print(f"  Sum of individual drops: {indiv_drop_sum:+.4f}")
print(f"  Substitution ratio: {substitution_ratio:.2f}")


# ── CELL: knockout_bar_chart ─────────────────────────────────────────────

ic_drops = {feat: loo_results[feat]["ic_drop"] for feat in all_features}
sorted_features = sorted(ic_drops, key=ic_drops.get, reverse=True)
sorted_drops = [ic_drops[f] for f in sorted_features]

fig1, ax1 = plt.subplots(figsize=(10, 5))
colors = ["#d32f2f" if d > 0 else "#1976d2" for d in sorted_drops]
bars = ax1.bar(range(len(sorted_features)), sorted_drops, color=colors)
ax1.set_xticks(range(len(sorted_features)))
ax1.set_xticklabels(sorted_features, rotation=45, ha="right")
ax1.set_ylabel("IC Drop (baseline − leave-one-out)")
ax1.set_title("Feature Knockout: IC Drop per Feature")
ax1.axhline(0, color="black", linewidth=0.8)
plt.tight_layout()
plt.show()


# ── CELL: joint_vs_individual_chart ──────────────────────────────────────

fig2, ax2 = plt.subplots(figsize=(8, 5))
labels = [f"Drop {top2_features[0]}", f"Drop {top2_features[1]}",
          "Sum of individual", "Joint drop"]
values = [
    loo_results[top2_features[0]]["ic_drop"],
    loo_results[top2_features[1]]["ic_drop"],
    indiv_drop_sum,
    joint_drop,
]
bar_colors = ["#1976d2", "#1976d2", "#ff9800", "#d32f2f"]
ax2.bar(labels, values, color=bar_colors)
ax2.set_ylabel("IC Drop")
ax2.set_title(f"Individual vs Joint Knockout (r={top2_corr_val:.2f})")
ax2.axhline(0, color="black", linewidth=0.8)
plt.tight_layout()
plt.show()


if __name__ == "__main__":
    # ── ASSERTIONS ────────────────────────────────────────────────────────

    # EX2-1: 7 LOO models, each mean OOS IC in [-0.01, 0.06]
    for feat, res in loo_results.items():
        mic = res["mean_ic"]
        assert -0.01 <= mic <= 0.06, (
            f"LOO-{feat} mean IC {mic:.4f} outside [-0.01, 0.06]"
        )

    # EX2-2: Largest single-feature IC drop in [0.001, 0.03]
    largest_drop = max(r["ic_drop"] for r in loo_results.values())
    assert 0.001 <= largest_drop <= 0.03, (
        f"Largest IC drop {largest_drop:.4f} outside [0.001, 0.03]"
    )

    # EX2-3: Top-2 correlated features removed simultaneously
    assert len(top2_features) == 2, (
        f"Expected 2 top-correlated features, got {len(top2_features)}"
    )

    # EX2-4: Substitution ratio in [1.0, 3.0]; if < 1.0 report honestly
    if np.isfinite(substitution_ratio):
        if substitution_ratio < 1.0:
            print(f"  NOTE: Substitution ratio {substitution_ratio:.2f} < 1.0 "
                  f"— features are complements, not substitutes")
        else:
            assert substitution_ratio <= 3.0, (
                f"Substitution ratio {substitution_ratio:.2f} > 3.0"
            )

    # EX2-5: Feature correlation matrix (7x7) printed
    assert avg_corr.shape == (7, 7), (
        f"Correlation matrix shape {avg_corr.shape}, expected (7, 7)"
    )

    # ── RESULTS ───────────────────────────────────────────────────────────
    print(f"\n══ seminar/ex2_feature_knockout ══════════════════════════")
    print(f"  baseline_mean_ic: {baseline_mean_ic:.4f}")
    print(f"  n_loo_models: {len(loo_results)}")
    for feat in all_features:
        r = loo_results[feat]
        print(f"  loo_{feat}: mean_ic={r['mean_ic']:.4f}  "
              f"drop={r['ic_drop']:+.4f}")
    print(f"  largest_single_drop: {largest_drop:.4f}")
    most_important = max(loo_results, key=lambda f: loo_results[f]["ic_drop"])
    print(f"  most_important_feature: {most_important}")
    print(f"  top2_correlated: {top2_features}")
    print(f"  top2_corr_value: {top2_corr_val:.3f}")
    print(f"  joint_drop: {joint_drop:+.4f}")
    print(f"  sum_individual_drops: {indiv_drop_sum:+.4f}")
    print(f"  substitution_ratio: {substitution_ratio:.2f}")
    print(f"  n_oos_months_baseline: {len(baseline_ic)}")

    # Correlation matrix
    print(f"\n  ── correlation_matrix (7x7) ──")
    for feat_row in all_features:
        row_vals = "  ".join(f"{avg_corr.loc[feat_row, c]:+.3f}"
                            for c in all_features)
        print(f"    {feat_row:20s}  {row_vals}")

    # ── PLOTS ─────────────────────────────────────────────────────────────
    fig1.savefig(PLOT_DIR / "ex2_knockout_bars.png",
                 dpi=150, bbox_inches="tight")
    print(f"\n  ── plot: ex2_knockout_bars.png ──")
    print(f"     type: bar chart of IC drop per feature")
    print(f"     n_bars: {len(ax1.patches)}")
    print(f"     y_range: [{ax1.get_ylim()[0]:.4f}, {ax1.get_ylim()[1]:.4f}]")
    print(f"     title: {ax1.get_title()}")

    fig2.savefig(PLOT_DIR / "ex2_joint_vs_individual.png",
                 dpi=150, bbox_inches="tight")
    print(f"\n  ── plot: ex2_joint_vs_individual.png ──")
    print(f"     type: grouped bar chart (individual vs joint drop)")
    print(f"     n_bars: {len(ax2.patches)}")
    print(f"     y_range: [{ax2.get_ylim()[0]:.4f}, {ax2.get_ylim()[1]:.4f}]")
    print(f"     title: {ax2.get_title()}")

    print(f"\n✓ ex2_feature_knockout: ALL PASSED")
