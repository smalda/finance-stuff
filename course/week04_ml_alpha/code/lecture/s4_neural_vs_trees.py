"""Section 4: Neural Networks vs Trees — Walk-Forward Comparison"""
import argparse
import matplotlib
matplotlib.use("Agg")
import sys
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy import stats

parser = argparse.ArgumentParser()
parser.add_argument("--device", default="cpu", choices=["cpu", "mps", "cuda"])
parser.add_argument("--debug", action="store_true",
                    help="Slash computation for code correctness check")
args = parser.parse_args()
device = args.device

sys.path.insert(0, str(Path(__file__).parent.parent))
from data_setup import (
    CACHE_DIR, PLOT_DIR, FEATURE_COLS, TRAIN_WINDOW, PURGE_GAP,
    load_monthly_panel,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from shared.temporal import walk_forward_splits, PurgedWalkForwardCV
from shared.dl_training import fit_nn, predict_nn

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)


# ── CELL: load_data_and_upstream ──────────────────────────────────
panel = load_monthly_panel()
dates = panel.index.get_level_values("date").unique().sort_values()
features = panel[FEATURE_COLS]
target = panel["fwd_return"]

# Load GBM upstream results for head-to-head comparison
gbm_ic_df = pd.read_parquet(CACHE_DIR / "gbm_ic_series.parquet")
gbm_ic_series = gbm_ic_df["ic"]
gbm_preds_df = pd.read_parquet(CACHE_DIR / "gbm_predictions.parquet")

n_features = len(FEATURE_COLS)
print("── DATA & UPSTREAM ───────────────────────────────────")
print(f"  Panel shape: {panel.shape}")
print(f"  Feature months: {len(dates)}")
print(f"  GBM IC series: {len(gbm_ic_series)} months")
print(f"  GBM predictions: {gbm_preds_df.shape}")
print(f"  Device: {device}")
print()


# ── CELL: nn_architecture ────────────────────────────────────────
# Feedforward network for cross-sectional return prediction.
# Architecture: input -> hidden1 -> ReLU -> dropout -> hidden2 -> ReLU -> dropout -> output

class AlphaNet(nn.Module):
    """Two-layer feedforward network for cross-sectional alpha prediction."""

    def __init__(self, n_input, hidden_size=32, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_input, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


# ── CELL: hp_grid_setup_nn ──────────────────────────────────────────
# Hyperparameter search on first training window using temporal CV.
# Small grid: learning_rate, hidden_size, dropout.
# Best HPs reused across all walk-forward windows.

first_train_dates = dates[:TRAIN_WINDOW]
first_train_mask = panel.index.get_level_values("date").isin(first_train_dates)
X_first = features.loc[first_train_mask].values.astype(np.float32)
y_first = target.loc[first_train_mask].values.astype(np.float32)

# Replace NaN with 0 for neural network (unlike LightGBM, NNs need clean input)
X_first = np.nan_to_num(X_first, nan=0.0)

hp_grid = {
    "lr": [1e-4, 1e-3, 1e-2],
    "hidden_size": [16, 32, 64],
    "dropout": [0.1, 0.3, 0.5],
}

if args.debug:
    hp_grid = {"lr": [1e-3], "hidden_size": [32], "dropout": [0.3]}

cv_splitter = PurgedWalkForwardCV(n_splits=3, purge_gap=1)

best_cv_ic = -np.inf
best_hp = {"lr": 1e-3, "hidden_size": 32, "dropout": 0.3}
n_combos = len(hp_grid["lr"]) * len(hp_grid["hidden_size"]) * len(hp_grid["dropout"])
combo_idx = 0

print("── HYPERPARAMETER SEARCH (NN) ───────────────────────")
print(f"  Training window: {TRAIN_WINDOW} months ({len(X_first)} obs)")
print(f"  Grid: {n_combos} combinations")
print(f"  CV: PurgedWalkForwardCV(n_splits=3)")


# ── CELL: nn_eval_one_fold ──────────────────────────────────────────
def nn_eval_one_fold(X_tr, y_tr, X_val, y_val, n_feat, hs, dp, lr, device):
    """Train one NN fold and return the IC value."""
    model_cv = AlphaNet(n_feat, hidden_size=hs, dropout=dp)
    fit_nn(
        model_cv, X_tr, y_tr,
        x_val=X_val, y_val=y_val,
        epochs=30, lr=lr, patience=10,
        batch_size=256, device=device,
    )
    preds_cv = predict_nn(model_cv, X_val, device=device)
    ic_cv = np.corrcoef(preds_cv, y_val)[0, 1]
    return ic_cv


# ── CELL: hp_search_nn_loop ────────────────────────────────────────
for lr in hp_grid["lr"]:
    for hs in hp_grid["hidden_size"]:
        for dp in hp_grid["dropout"]:
            combo_idx += 1
            fold_ics = []
            for train_idx, val_idx in cv_splitter.split(X_first):
                X_tr, y_tr = X_first[train_idx], y_first[train_idx]
                X_val, y_val = X_first[val_idx], y_first[val_idx]

                ic_cv = nn_eval_one_fold(
                    X_tr, y_tr, X_val, y_val,
                    n_features, hs, dp, lr, device,
                )
                if np.isfinite(ic_cv):
                    fold_ics.append(ic_cv)

            mean_cv = np.mean(fold_ics) if fold_ics else -np.inf
            if combo_idx % 3 == 0 or combo_idx == n_combos:
                print(f"  [{combo_idx}/{n_combos}] lr={lr}, hs={hs}, "
                      f"dp={dp} -> CV IC={mean_cv:.4f}")
            if mean_cv > best_cv_ic:
                best_cv_ic = mean_cv
                best_hp = {"lr": lr, "hidden_size": hs, "dropout": dp}

print(f"  Best: lr={best_hp['lr']}, hidden={best_hp['hidden_size']}, "
      f"dropout={best_hp['dropout']}, CV IC={best_cv_ic:.4f}")
print()


# ── CELL: nn_train_predict_one ──────────────────────────────────────
def nn_train_predict_one(panel, features, target, train_dates, pred_date,
                         best_hp, n_features, max_epochs, patience, device):
    """Train NN on one window and predict OOS month.

    Returns (pred_df, oos_ic, train_ic, final_epoch).
    """
    train_mask = panel.index.get_level_values("date").isin(train_dates)
    X_train_full = features.loc[train_mask].values.astype(np.float32)
    y_train_full = target.loc[train_mask].values.astype(np.float32)
    X_train_full = np.nan_to_num(X_train_full, nan=0.0)

    # Early stopping holdout: last 12 months of training window
    n_val_months = min(12, len(train_dates) // 5)
    val_dates = train_dates[-n_val_months:]
    fit_dates = train_dates[:-n_val_months]

    fit_mask = panel.index.get_level_values("date").isin(fit_dates)
    val_mask = panel.index.get_level_values("date").isin(val_dates)

    X_fit = np.nan_to_num(
        features.loc[fit_mask].values.astype(np.float32), nan=0.0)
    y_fit = target.loc[fit_mask].values.astype(np.float32)
    X_val = np.nan_to_num(
        features.loc[val_mask].values.astype(np.float32), nan=0.0)
    y_val = target.loc[val_mask].values.astype(np.float32)

    model = AlphaNet(
        n_features,
        hidden_size=best_hp["hidden_size"],
        dropout=best_hp["dropout"],
    )
    info = fit_nn(
        model, X_fit, y_fit,
        x_val=X_val, y_val=y_val,
        epochs=max_epochs, lr=best_hp["lr"],
        patience=patience, batch_size=256,
        device=device,
    )

    # Predict OOS month
    pred_mask = panel.index.get_level_values("date") == pred_date
    X_pred = np.nan_to_num(
        features.loc[pred_mask].values.astype(np.float32), nan=0.0)
    y_pred = target.loc[pred_mask].values
    tickers_pred = panel.loc[pred_mask].index.get_level_values("ticker")

    preds = predict_nn(model, X_pred, device=device)

    pred_df = pd.DataFrame({
        "date": pred_date,
        "ticker": tickers_pred,
        "prediction": preds,
        "actual": y_pred,
    })

    oos_ic = np.corrcoef(preds, y_pred)[0, 1]
    train_preds = predict_nn(model, X_fit, device=device)
    train_ic = np.corrcoef(train_preds, y_fit)[0, 1]

    return pred_df, oos_ic, train_ic, info["final_epoch"]


# ── CELL: nn_walk_forward_loop ──────────────────────────────────────
# Walk-forward out-of-sample predictions with the neural network.
# Each window: train on TRAIN_WINDOW months, predict next month.

MAX_EPOCHS = 50
PATIENCE = 10
if args.debug:
    MAX_EPOCHS = 5
    PATIENCE = 3

oos_predictions = []
oos_ic_list = []
train_ic_list = []
stopping_epochs = []

splits = list(walk_forward_splits(dates, TRAIN_WINDOW, PURGE_GAP))
n_splits = len(splits)

if args.debug:
    splits = splits[:5]
    n_splits = len(splits)

print(f"── WALK-FORWARD TRAINING ({n_splits} windows) ───────")

for i, (train_dates, pred_date) in enumerate(splits):
    if i % 10 == 0 or (args.debug and i % 2 == 0):
        print(f"  [{i+1}/{n_splits}] predicting "
              f"{pd.Timestamp(pred_date).date()}")

    pred_df, oos_ic, train_ic, final_epoch = nn_train_predict_one(
        panel, features, target, train_dates, pred_date,
        best_hp, n_features, MAX_EPOCHS, PATIENCE, device,
    )
    oos_predictions.append(pred_df)
    stopping_epochs.append(final_epoch)

    if np.isfinite(oos_ic):
        oos_ic_list.append({"date": pred_date, "ic": oos_ic})
    if np.isfinite(train_ic):
        train_ic_list.append(train_ic)

print(f"  Walk-forward complete: {len(oos_ic_list)} OOS months")
print()


# ── CELL: aggregate_nn_predictions ──────────────────────────────────
# Aggregate IC statistics from NN walk-forward predictions.

all_nn_preds = pd.concat(oos_predictions, ignore_index=True)
all_nn_preds = all_nn_preds.set_index(["date", "ticker"])

nn_ic_series = pd.DataFrame(oos_ic_list).set_index("date")["ic"]


# ── CELL: nn_ic_statistics ──────────────────────────────────────────
mean_nn_ic = nn_ic_series.mean()
std_nn_ic = nn_ic_series.std()
nn_icir = mean_nn_ic / std_nn_ic if std_nn_ic > 0 else np.nan
nn_pct_positive = (nn_ic_series > 0).mean()
n_nn = len(nn_ic_series)
nn_t = (mean_nn_ic / (std_nn_ic / np.sqrt(n_nn))
        if std_nn_ic > 0 and n_nn >= 2 else np.nan)
nn_p = (2 * (1 - stats.t.cdf(abs(nn_t), df=n_nn - 1))
        if np.isfinite(nn_t) else np.nan)

print("── NN OOS SIGNAL QUALITY ─────────────────────────────")
print(f"  OOS months: {n_nn}")
print(f"  Mean IC: {mean_nn_ic:.4f}")
print(f"  Std IC: {std_nn_ic:.4f}")
print(f"  ICIR: {nn_icir:.4f}")
print(f"  pct_positive: {nn_pct_positive:.4f}")
print(f"  t-stat: {nn_t:.4f}")
print(f"  p-value: {nn_p:.4f}")
print()


# ── CELL: prediction_quality_nn ──────────────────────────────────
# Check whether NN predictions have meaningful spread (non-degenerate).

spread_ratios = []
for date, group in all_nn_preds.groupby(level="date"):
    std_p = group["prediction"].std()
    std_a = group["actual"].std()
    if std_a > 0:
        spread_ratios.append(float(std_p / std_a))

median_spread = np.median(spread_ratios)

print("── NN PREDICTION QUALITY ─────────────────────────────")
print(f"  Median spread_ratio: {median_spread:.4f}")
print(f"  Min spread_ratio: {np.min(spread_ratios):.4f}")
print(f"  Max spread_ratio: {np.max(spread_ratios):.4f}")
print()


# ── CELL: training_stability ─────────────────────────────────────
# Report stopping epoch statistics to assess training stability.

mean_stop = np.mean(stopping_epochs)
std_stop = np.std(stopping_epochs)
pct_maxed = np.mean([e >= MAX_EPOCHS for e in stopping_epochs])
pct_nan_loss = 0  # tracked via NaN IC (if model diverges, IC is NaN)

print("── TRAINING STABILITY ────────────────────────────────")
print(f"  Mean stopping epoch: {mean_stop:.1f}")
print(f"  Std stopping epoch: {std_stop:.1f}")
print(f"  Pct maxed out ({MAX_EPOCHS} epochs): {pct_maxed:.2%}")
print(f"  Windows with valid IC: {n_nn}/{n_splits}")
print()


# ── CELL: overfitting_check_nn ───────────────────────────────────
# Compare in-sample IC to OOS IC.  Neural networks are prone to
# overfitting on small tabular datasets; flag if train IC > 3x OOS IC.

mean_train_ic = np.mean(train_ic_list) if train_ic_list else np.nan
oos_ratio = (mean_train_ic / mean_nn_ic
             if mean_nn_ic != 0 and np.isfinite(mean_train_ic)
             else np.inf)

print("── OVERFITTING CHECK (NN) ────────────────────────────")
print(f"  Mean train IC: {mean_train_ic:.4f}")
print(f"  Mean OOS IC: {mean_nn_ic:.4f}")
print(f"  Train/OOS ratio: {oos_ratio:.2f}")
if mean_train_ic > 3 * abs(mean_nn_ic):
    print(f"  OVERFIT: train IC ({mean_train_ic:.4f}) > "
          f"3 x |OOS IC| ({abs(mean_nn_ic):.4f})")
else:
    print(f"  No overfitting flag (ratio <= 3.0)")
print()


# ── CELL: align_ic_series ───────────────────────────────────────────
# Compare NN vs GBM IC series: absolute difference and paired t-test.

# Align IC series on common dates
common_dates = nn_ic_series.index.intersection(gbm_ic_series.index)
nn_aligned = nn_ic_series.loc[common_dates].values
gbm_aligned = gbm_ic_series.loc[common_dates].values

mean_gbm_ic = gbm_aligned.mean()
ic_diff = abs(mean_gbm_ic - mean_nn_ic)


# ── CELL: paired_test_nn_gbm ───────────────────────────────────────
# Paired t-test: is GBM IC statistically different from NN IC?
diff_series = gbm_aligned - nn_aligned
n_paired = len(diff_series)
mean_diff = diff_series.mean()
std_diff = diff_series.std()
se_diff = std_diff / np.sqrt(n_paired) if n_paired >= 2 else np.nan
paired_t = mean_diff / se_diff if se_diff > 0 else np.nan
paired_p = (2 * (1 - stats.t.cdf(abs(paired_t), df=n_paired - 1))
            if np.isfinite(paired_t) else np.nan)

print("── GBM vs NN HEAD-TO-HEAD ────────────────────────────")
print(f"  Common OOS months: {n_paired}")
print(f"  GBM mean IC: {mean_gbm_ic:.4f}")
print(f"  NN mean IC: {mean_nn_ic:.4f}")
print(f"  |GBM IC - NN IC|: {ic_diff:.4f}")
print(f"  Paired t-stat: {paired_t:.4f}")
print(f"  Paired p-value: {paired_p:.4f}")
print(f"  Mean monthly diff (GBM - NN): {mean_diff:.4f}")
print()


# ── CELL: comparison_visualization ───────────────────────────────
# Side-by-side IC statistics and overlaid monthly IC time series.

models = ["GBM", "NN"]
mean_ics = [mean_gbm_ic, mean_nn_ic]
icirs = [gbm_aligned.mean() / gbm_aligned.std() if gbm_aligned.std() > 0
         else 0, nn_icir]
pct_pos = [(gbm_aligned > 0).mean(), nn_pct_positive]

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Bar chart: Mean IC
ax0 = axes[0]
bars0 = ax0.bar(models, mean_ics, color=["#1f77b4", "#ff7f0e"], width=0.5)
ax0.set_ylabel("Mean IC")
ax0.set_title("Mean OOS IC: GBM vs NN")
ax0.axhline(0, color="black", linewidth=0.5)
for bar, val in zip(bars0, mean_ics):
    ax0.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
             f"{val:.4f}", ha="center", va="bottom", fontsize=9)

# Bar chart: ICIR
ax1 = axes[1]
bars1 = ax1.bar(models, icirs, color=["#1f77b4", "#ff7f0e"], width=0.5)
ax1.set_ylabel("ICIR")
ax1.set_title("ICIR: GBM vs NN")
ax1.axhline(0, color="black", linewidth=0.5)
for bar, val in zip(bars1, icirs):
    ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
             f"{val:.3f}", ha="center", va="bottom", fontsize=9)

# Bar chart: pct_positive
ax2 = axes[2]
bars2 = ax2.bar(models, pct_pos, color=["#1f77b4", "#ff7f0e"], width=0.5)
ax2.set_ylabel("pct_positive")
ax2.set_title("Pct Positive IC: GBM vs NN")
ax2.axhline(0.5, color="gray", linestyle="--", linewidth=0.8)
for bar, val in zip(bars2, pct_pos):
    ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
             f"{val:.3f}", ha="center", va="bottom", fontsize=9)

plt.tight_layout()
plt.show()


# ── CELL: monthly_ic_overlay ─────────────────────────────────────
# Overlaid line chart of monthly IC for both models.

fig_line, ax_line = plt.subplots(figsize=(14, 5))
x_range = range(len(common_dates))
ax_line.plot(x_range, gbm_aligned, color="#1f77b4",
             alpha=0.8, linewidth=1, label=f"GBM (mean={mean_gbm_ic:.4f})")
ax_line.plot(x_range, nn_aligned, color="#ff7f0e",
             alpha=0.8, linewidth=1, label=f"NN (mean={mean_nn_ic:.4f})")
ax_line.axhline(0, color="black", linewidth=0.5)
ax_line.set_xlabel("OOS Month")
ax_line.set_ylabel("Information Coefficient")
ax_line.set_title("Monthly OOS IC: GBM vs Neural Network")
ax_line.legend()
plt.tight_layout()
plt.show()


# ── Cache outputs ────────────────────────────────────────────────
all_nn_preds.to_parquet(CACHE_DIR / "nn_predictions.parquet")
nn_ic_series.to_frame("ic").to_parquet(CACHE_DIR / "nn_ic_series.parquet")


if __name__ == "__main__":
    # ── ASSERTIONS ─────────────────────────────────────
    # S4-1: OOS months
    assert 65 <= n_nn <= 72, \
        f"S4-1: OOS months = {n_nn}, expected [65, 72]"

    # S4-2: Mean OOS IC (NN)
    assert -0.01 <= mean_nn_ic <= 0.08, \
        f"S4-2: Mean NN IC = {mean_nn_ic:.4f}, expected [-0.01, 0.08]"

    # S4-3: Prediction quality
    assert median_spread > 0.03, \
        (f"S4-3: median spread_ratio = {median_spread:.4f}, "
         f"expected > 0.03")

    # S4-4: |GBM IC - NN IC| in [0.00, 0.03]
    assert 0.00 <= ic_diff <= 0.03, \
        f"S4-4: |GBM IC - NN IC| = {ic_diff:.4f}, expected [0.00, 0.03]"

    # S4-5: Paired t-test reported
    assert np.isfinite(paired_t), f"S4-5: paired t-stat not finite: {paired_t}"
    assert np.isfinite(paired_p), f"S4-5: paired p-value not finite: {paired_p}"

    # S4-6: Training stability (mean and std of stopping epoch)
    assert np.isfinite(mean_stop), f"S4-6: mean stopping epoch not finite"
    assert np.isfinite(std_stop), f"S4-6: std stopping epoch not finite"

    # S4-7: Overfitting check
    # (conditional — assertion is that the check is computed, not that it fires)
    assert np.isfinite(oos_ratio) or oos_ratio == np.inf, \
        f"S4-7: overfitting ratio not computed"

    # ── RESULTS ────────────────────────────────────────
    print(f"══ lecture/s4_neural_vs_trees ═══════════════════════")
    print(f"  n_oos_months: {n_nn}")
    print(f"  nn_mean_ic: {mean_nn_ic:.4f}")
    print(f"  nn_std_ic: {std_nn_ic:.4f}")
    print(f"  nn_icir: {nn_icir:.4f}")
    print(f"  nn_pct_positive: {nn_pct_positive:.4f}")
    print(f"  nn_t_stat: {nn_t:.4f}")
    print(f"  nn_p_value: {nn_p:.4f}")
    print(f"  median_spread_ratio: {median_spread:.4f}")
    print(f"  mean_stopping_epoch: {mean_stop:.1f}")
    print(f"  std_stopping_epoch: {std_stop:.1f}")
    print(f"  mean_train_ic: {mean_train_ic:.4f}")
    print(f"  train_oos_ratio: {oos_ratio:.2f}")
    print(f"  gbm_mean_ic: {mean_gbm_ic:.4f}")
    print(f"  nn_mean_ic: {mean_nn_ic:.4f}")
    print(f"  ic_diff_abs: {ic_diff:.4f}")
    print(f"  paired_t_stat: {paired_t:.4f}")
    print(f"  paired_p_value: {paired_p:.4f}")
    print(f"  best_lr: {best_hp['lr']}")
    print(f"  best_hidden_size: {best_hp['hidden_size']}")
    print(f"  best_dropout: {best_hp['dropout']}")
    print(f"  device: {device}")

    # ── PLOT ───────────────────────────────────────────
    fig.savefig(PLOT_DIR / "s4_ic_comparison_bars.png",
                dpi=150, bbox_inches="tight")
    print(f"  ── plot: s4_ic_comparison_bars.png ──")
    print(f"     type: 3-panel bar chart (mean IC, ICIR, pct_positive)")
    print(f"     n_panels: 3")
    print(f"     models: {models}")
    print(f"     title_0: {axes[0].get_title()}")

    fig_line.savefig(PLOT_DIR / "s4_monthly_ic_overlay.png",
                     dpi=150, bbox_inches="tight")
    print(f"  ── plot: s4_monthly_ic_overlay.png ──")
    print(f"     type: overlaid line chart (monthly IC, both models)")
    print(f"     n_lines: {len(ax_line.get_lines())}")
    print(f"     y_range: [{ax_line.get_ylim()[0]:.4f}, "
          f"{ax_line.get_ylim()[1]:.4f}]")
    print(f"     title: {ax_line.get_title()}")

    # ── CACHE CONFIRMATION ─────────────────────────────
    print(f"  ── cached outputs ──")
    print(f"     nn_predictions.parquet: {all_nn_preds.shape}")
    print(f"     nn_ic_series.parquet: {nn_ic_series.shape}")

    print(f"✓ s4_neural_vs_trees: ALL PASSED")
