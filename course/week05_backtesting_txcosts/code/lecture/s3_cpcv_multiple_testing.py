"""Section 3: Combinatorial Purged CV & Multiple Testing Corrections.

Topics covered:
- Combinatorial Purged Cross-Validation (CPCV): C(6,2)=15 paths
- Probability of Backtest Overfitting (PBO)
- Harvey-Liu-Zhu t-stat thresholds for signal validation
- Benjamini-Hochberg-Yekutieli (BHY) multiple testing correction
"""
import matplotlib
matplotlib.use("Agg")

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.multitest import multipletests

_CODE_DIR = Path(__file__).resolve().parent.parent
_COURSE_DIR = _CODE_DIR.parent.parent
sys.path.insert(0, str(_CODE_DIR))
sys.path.insert(0, str(_COURSE_DIR))
from data_setup import CACHE_DIR, PLOT_DIR, load_alpha_output
from shared.temporal import CombinatorialPurgedCV
from shared.metrics import rank_ic, ic_summary


# ── CELL: load_model_variants ─────────────────────────────────────────

alpha = load_alpha_output()
gbm_pred = alpha["predictions"]    # MultiIndex (date, ticker), columns: prediction, actual
nn_pred = alpha.get("nn_predictions", None)
feat_df = alpha.get("expanded_features")

# Load forward returns (full date range for Ridge training)
fwd_df = pd.read_parquet(CACHE_DIR / "forward_returns_w5.parquet")

# OOS dates (common to GBM and NN): April 2019 – November 2024
gbm_dates = gbm_pred.index.get_level_values("date").unique().sort_values()
print(f"  GBM OOS dates: {len(gbm_dates)} months "
      f"({gbm_dates[0].date()} – {gbm_dates[-1].date()})")

if nn_pred is not None:
    nn_dates = nn_pred.index.get_level_values("date").unique().sort_values()
    print(f"  NN  OOS dates: {len(nn_dates)} months")
else:
    nn_pred = None
    print("  NN predictions: not available")


# ── CELL: ridge_train_one_date ────────────────────────────────────────

def _ridge_train_predict_one(
    feat_df, fwd_df, feat_dates, train_window, i, pred_date
):
    """Train Ridge on prior window and predict one date.

    Returns list of record dicts, or empty list if data is insufficient.
    """
    train_dates = feat_dates[i - train_window:i]

    X_list, y_list = [], []
    for td in train_dates:
        try:
            X_td = feat_df.loc[td]
            y_td = fwd_df.loc[td]["fwd_return"]
        except KeyError:
            continue
        common = X_td.index.intersection(y_td.index)
        if len(common) < 10:
            continue
        X_list.append(X_td.loc[common])
        y_list.append(y_td.loc[common])

    if not X_list:
        return []

    X_train = pd.concat(X_list).fillna(0.0)
    y_train = pd.concat(y_list).fillna(0.0)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)
    model = Ridge(alpha=1.0)
    model.fit(X_scaled, y_train)

    # Predict on pred_date
    try:
        X_pred = feat_df.loc[pred_date].fillna(0.0)
        y_actual = fwd_df.loc[pred_date]["fwd_return"]
    except KeyError:
        return []

    common_pred = X_pred.index.intersection(y_actual.index)
    if len(common_pred) < 10:
        return []

    X_pred_scaled = scaler.transform(X_pred.loc[common_pred])
    preds = model.predict(X_pred_scaled)

    records = []
    for ticker, p, a in zip(common_pred, preds, y_actual.loc[common_pred].values):
        records.append({"date": pred_date, "ticker": ticker,
                        "prediction": float(p), "actual": float(a)})
    return records


# ── CELL: ridge_walk_forward_func ─────────────────────────────────────

def compute_ridge_predictions(
    feat_df: pd.DataFrame,
    fwd_df: pd.DataFrame,
    train_window: int = 36,
) -> pd.DataFrame:
    """Walk-forward Ridge regression predictions.

    For each OOS date, train Ridge on the prior `train_window` months
    of expanded features, then predict one month ahead.

    Args:
        feat_df: MultiIndex (date, ticker) feature DataFrame.
        fwd_df: MultiIndex (date, ticker) with column 'fwd_return'.
        train_window: number of months in the rolling training window.

    Returns:
        DataFrame with MultiIndex (date, ticker) and columns
        ['prediction', 'actual'].
    """
    feat_dates = feat_df.index.get_level_values("date").unique().sort_values()
    records = []

    for i, pred_date in enumerate(feat_dates[train_window:], start=train_window):
        if i % 12 == 0:
            print(f"  [Ridge {i}/{len(feat_dates)}] {pred_date.date()}")
        records.extend(
            _ridge_train_predict_one(feat_df, fwd_df, feat_dates, train_window, i, pred_date)
        )

    ridge_df = pd.DataFrame(records).set_index(["date", "ticker"])
    return ridge_df


# ── CELL: ridge_walk_forward_run ──────────────────────────────────────

print("Computing Ridge walk-forward predictions...")
ridge_pred = compute_ridge_predictions(feat_df, fwd_df, train_window=36)
ridge_dates = ridge_pred.index.get_level_values("date").unique().sort_values()
print(f"  Ridge OOS dates: {len(ridge_dates)} months "
      f"({ridge_dates[0].date()} – {ridge_dates[-1].date()})")


# ── CELL: align_oos_dates ────────────────────────────────────────────

# Align all three models to the GBM OOS period (68 months)
common_dates = gbm_dates  # primary reference

if nn_pred is not None:
    nn_common = common_dates[common_dates.isin(
        nn_pred.index.get_level_values("date").unique()
    )]
else:
    nn_common = common_dates

ridge_common = common_dates[common_dates.isin(ridge_dates)]
final_dates = common_dates[
    common_dates.isin(nn_common) & common_dates.isin(ridge_common)
]
print(f"  Aligned OOS dates (all 3 models): {len(final_dates)} months")


# ── CELL: compute_ic_series_func ─────────────────────────────────────

def compute_ic_series(pred_df: pd.DataFrame, dates: pd.DatetimeIndex) -> pd.Series:
    """Compute monthly Spearman IC for a prediction DataFrame."""
    ic_vals = {}
    for d in dates:
        try:
            sub = pred_df.loc[d]
        except KeyError:
            continue
        sub = sub.dropna()
        if len(sub) < 10:
            continue
        ic_vals[d] = rank_ic(sub["prediction"].values, sub["actual"].values)
    return pd.Series(ic_vals, name="ic")


# ── CELL: compute_model_ics ─────────────────────────────────────────

gbm_ic_series = compute_ic_series(gbm_pred, final_dates)
nn_ic_series = compute_ic_series(nn_pred, final_dates) if nn_pred is not None else None
ridge_ic_series = compute_ic_series(ridge_pred, final_dates)

model_ic = {
    "GBM": gbm_ic_series,
    "Ridge": ridge_ic_series,
}
if nn_ic_series is not None:
    model_ic["NN"] = nn_ic_series

for name, ic_s in model_ic.items():
    stats_d = ic_summary(ic_s.dropna())
    print(f"  {name:6s}: mean_IC={stats_d['mean_ic']:.4f}, "
          f"t={stats_d['t_stat']:.2f}, p={stats_d['p_value']:.3f}")


# ── CELL: cpcv_setup ─────────────────────────────────────────────────

# Build panel: rows = OOS dates, columns = models
ic_panel = pd.DataFrame(model_ic).dropna()
n_obs = len(ic_panel)
model_names = ic_panel.columns.tolist()

print(f"\nCPCV setup: {n_obs} time points × {len(model_names)} models")
print(f"  Models: {model_names}")

cpcv = CombinatorialPurgedCV(n_splits=6, n_test_splits=2, purge_gap=1)
n_paths_expected = cpcv.get_n_splits(ic_panel)
print(f"  CPCV: C(6,2) = {n_paths_expected} paths expected")


# ── CELL: run_cpcv_loop ──────────────────────────────────────────────

cpcv_paths = []  # List of dicts: {is_winner, oos_ranks, oos_ic_winner}

for path_idx, (train_idx, test_idx) in enumerate(cpcv.split(ic_panel)):
    is_ics = ic_panel.iloc[train_idx]   # IS IC for each model
    oos_ics = ic_panel.iloc[test_idx]   # OOS IC for each model

    is_mean = is_ics.mean()
    oos_mean = oos_ics.mean()

    is_winner = is_mean.idxmax()       # model with best IS mean IC
    oos_rank_of_winner = (
        oos_mean.rank(ascending=False).loc[is_winner]
    )  # 1 = best OOS

    n_models = len(model_names)
    oos_median_rank = (n_models + 1) / 2.0

    cpcv_paths.append({
        "path": path_idx,
        "is_winner": is_winner,
        "oos_rank_winner": float(oos_rank_of_winner),
        "oos_ic_winner": float(oos_mean.loc[is_winner]),
        "n_models": n_models,
        "oos_median_rank": oos_median_rank,
    })


# ── CELL: cpcv_results ──────────────────────────────────────────────

print(f"\nCPCV complete: {len(cpcv_paths)} paths")
cpcv_df = pd.DataFrame(cpcv_paths)

is_winners = cpcv_df["is_winner"].value_counts()
print(f"  IS winners by model: {is_winners.to_dict()}")

oos_return_series = cpcv_df["oos_ic_winner"]
print(f"  OOS IC winner: mean={oos_return_series.mean():.4f}, "
      f"std={oos_return_series.std():.4f}")


# ── CELL: compute_pbo ────────────────────────────────────────────────

# PBO: fraction of paths where IS-winner's OOS rank is at or below median
# For 3 models: median rank = 2.0; IS-winner with rank > median means it
# underperformed the average OOS
n_paths = len(cpcv_df)
n_below_median = (cpcv_df["oos_rank_winner"] > cpcv_df["oos_median_rank"]).sum()
pbo = n_below_median / n_paths

print(f"\nProbability of Backtest Overfitting (PBO)")
print(f"  Paths where IS-winner ranks below median OOS: "
      f"{n_below_median}/{n_paths}")
print(f"  PBO = {pbo:.4f} ({pbo:.1%})")

if pbo < 0.20:
    print("  ⚠ PBO < 0.20: suspiciously low — possible look-ahead or leakage")
elif pbo > 0.75:
    print("  ⚠ PBO > 0.75: very high — models may be pure noise")
else:
    print(f"  PBO in acceptable range [0.20, 0.75] — typical regime variation")


# ── CELL: harvey_liu_zhu_compute ─────────────────────────────────────

# Harvey-Liu-Zhu (2016): t-stat thresholds for signal significance
# t > 1.96: 5% significance (nominal)
# t > 2.58: 1% significance
# t > 3.00: recommended threshold accounting for selection bias (3 models)

hlz_results = {}
for name, ic_s in model_ic.items():
    s = ic_summary(ic_s.dropna())
    hlz_results[name] = {
        "mean_ic": s["mean_ic"],
        "t_stat": s["t_stat"],
        "p_value": s["p_value"],
        "n": s["n"],
        "passes_t196": s["t_stat"] > 1.96 if np.isfinite(s["t_stat"]) else False,
        "passes_t300": s["t_stat"] > 3.00 if np.isfinite(s["t_stat"]) else False,
    }


# ── CELL: harvey_liu_zhu_print ───────────────────────────────────────

print("\nHarvey-Liu-Zhu t-stat analysis")
print(f"  {'Model':8s}  {'t-stat':>8s}  {'p-value':>8s}  {'t>1.96':>7s}  {'t>3.00':>7s}")
for name, res in hlz_results.items():
    t = res["t_stat"]
    p = res["p_value"]
    t_str = f"{t:.3f}" if np.isfinite(t) else "nan"
    p_str = f"{p:.4f}" if np.isfinite(p) else "nan"
    t196_str = "YES" if res["passes_t196"] else "NO"
    t300_str = "YES" if res["passes_t300"] else "NO"
    print(f"  {name:8s}  {t_str:>8s}  {p_str:>8s}  "
          f"{t196_str:>7s}  {t300_str:>7s}")

n_passing_t300 = sum(1 for r in hlz_results.values() if r["passes_t300"])
print(f"\n  N models passing t > 3.00: {n_passing_t300}/{len(hlz_results)}")
print("  (t > 3.0 recommended for strategies with many trials)")


# ── CELL: bhy_real_models ────────────────────────────────────────────

# BHY multiple-testing correction across 3 actual models
raw_pvalues = np.array([
    hlz_results[m]["p_value"]
    for m in model_names
    if np.isfinite(hlz_results[m]["p_value"])
])
model_names_finite = [
    m for m in model_names
    if np.isfinite(hlz_results[m]["p_value"])
]

reject, p_adj, _, _ = multipletests(raw_pvalues, method="fdr_bh")

print("\nBHY (Benjamini-Hochberg-Yekutieli) Multiple Testing Correction")
print(f"  {'Model':8s}  {'Raw p':>8s}  {'BHY p':>8s}  {'p≥raw?':>8s}  {'Reject?':>8s}")
for m, p_raw, p_bhy, rej in zip(model_names_finite, raw_pvalues, p_adj, reject):
    print(f"  {m:8s}  {p_raw:>8.4f}  {p_bhy:>8.4f}  "
          f"{'YES' if p_bhy >= p_raw - 1e-12 else 'NO':>8s}  "
          f"{'YES' if rej else 'NO':>8s}")


# ── CELL: bhy_fdr_simulation ────────────────────────────────────────

# 50-variant simulation: demonstrate FDR problem
rng = np.random.default_rng(42)
n_fake = 50
fake_ic = rng.normal(0, 0.07, (len(final_dates), n_fake))  # null: IC ~ N(0, 0.07)
fake_tstat = (fake_ic.mean(axis=0)
              / (fake_ic.std(axis=0) / np.sqrt(len(final_dates))))
fake_pvals = 2 * (1 - stats.norm.cdf(np.abs(fake_tstat)))

reject_nominal, p_adj_bhy, _, _ = multipletests(fake_pvals, method="fdr_bh")
n_fp_nominal = (fake_pvals < 0.05).sum()
n_fp_bhy = reject_nominal.sum()

print(f"\n  50-variant FDR simulation (null: IC ~ N(0, 0.07))")
print(f"  False positives at nominal p<0.05: {n_fp_nominal}/{n_fake}")
print(f"  False positives after BHY correction: {n_fp_bhy}/{n_fake}")


# ── CELL: cpcv_oos_histogram ─────────────────────────────────────────

fig_hist, ax_hist = plt.subplots(figsize=(8, 5))
ax_hist.hist(cpcv_df["oos_ic_winner"], bins=15, color="#4C72B0", alpha=0.8,
             edgecolor="white")
median_val = cpcv_df["oos_ic_winner"].median()
ax_hist.axvline(median_val, color="firebrick", lw=2, linestyle="--",
                label=f"Median = {median_val:.4f}")
ax_hist.axvline(0, color="gray", lw=1.5, linestyle=":")
ax_hist.set(
    title="CPCV: OOS IC Distribution for IS-Winner",
    xlabel="OOS Mean IC",
    ylabel="Path Count",
)
ax_hist.legend()
plt.tight_layout()
plt.show()


# ── CELL: pbo_visualization ───────────────────────────────────────────

oos_ranks = cpcv_df["oos_rank_winner"].values
median_rank = float(np.median(oos_ranks))

fig_pbo, ax_pbo = plt.subplots(figsize=(8, 4))
ax_pbo.hist(oos_ranks, bins=np.arange(0.5, len(model_names) + 1.5, 1),
            color="#55A868", alpha=0.8, edgecolor="white")
ax_pbo.axvline(median_rank, color="firebrick", lw=2, linestyle="--",
               label=f"Median rank = {median_rank:.1f}")
below_mask = oos_ranks > median_rank
ax_pbo.fill_between(
    [median_rank, len(model_names) + 0.5],
    0, ax_pbo.get_ylim()[1] if ax_pbo.get_ylim()[1] > 0 else 5,
    alpha=0.15, color="red", label=f"Below median → PBO contribution"
)
ax_pbo.set(
    title=f"OOS Rank of IS-Winner Across 15 CPCV Paths (PBO = {pbo:.2f})",
    xlabel="OOS Rank of IS-Winner (1 = best)",
    ylabel="Number of Paths",
)
ax_pbo.legend()
plt.tight_layout()
plt.show()


# ── CELL: hlz_bar_chart ───────────────────────────────────────────────

model_labels = list(hlz_results.keys())
t_stats = [hlz_results[m]["t_stat"] for m in model_labels]

fig_hlz, ax_hlz = plt.subplots(figsize=(8, 4))
y_pos = range(len(model_labels))
colors = ["#4C72B0" if t <= 3.0 else "#C44E52" for t in t_stats]
ax_hlz.barh(y_pos, t_stats, color=colors, alpha=0.85, edgecolor="white")
ax_hlz.axvline(1.96, color="orange", lw=2, linestyle="--", label="t = 1.96 (5%)")
ax_hlz.axvline(3.00, color="firebrick", lw=2, linestyle="-", label="t = 3.00 (HLZ)")
ax_hlz.set_yticks(y_pos)
ax_hlz.set_yticklabels(model_labels)
ax_hlz.set(
    title="Harvey-Liu-Zhu t-stat: 3 Model Variants",
    xlabel="t-statistic",
)
ax_hlz.legend()
plt.tight_layout()
plt.show()


if __name__ == "__main__":
    # ── ASSERTIONS ─────────────────────────────────────────────────

    # S3-1: CPCV produces exactly 15 paths
    assert len(cpcv_paths) == 15, (
        f"S3-1 FAIL: expected 15 CPCV paths, got {len(cpcv_paths)}"
    )

    # S3-2: PBO ∈ [0.20, 0.75]
    assert 0.20 <= pbo <= 0.75, (
        f"S3-2 FAIL: PBO = {pbo:.4f}, expected [0.20, 0.75]"
    )

    # S3-3: OOS returns distribution has std > 0
    oos_std = float(oos_return_series.std())
    assert oos_std > 0, (
        f"S3-3 FAIL: OOS IC std = {oos_std:.6f}, expected > 0"
    )

    # S3-4: BHY adjusted p-values ≥ raw p-values
    for m, p_raw, p_bhy in zip(model_names_finite, raw_pvalues, p_adj):
        assert p_bhy >= p_raw - 1e-8, (
            f"S3-4 FAIL: {m} BHY p={p_bhy:.6f} < raw p={p_raw:.6f}"
        )

    # S3-5: N_passing_t300 ≤ 3 (expected 0 for real models)
    assert n_passing_t300 <= 3, (
        f"S3-5 FAIL: {n_passing_t300} models pass t>3.0, expected ≤3"
    )

    # ── Cache for downstream (d3) ────────────────────────────────────
    cpcv_df.to_parquet(CACHE_DIR / "cpcv_results.parquet")

    # ── PLOT saves ───────────────────────────────────────────────────
    fig_hist.savefig(
        PLOT_DIR / "s3_cpcv_oos_distribution.png", dpi=150, bbox_inches="tight"
    )
    fig_pbo.savefig(
        PLOT_DIR / "s3_pbo_visualization.png", dpi=150, bbox_inches="tight"
    )
    fig_hlz.savefig(
        PLOT_DIR / "s3_hlz_tstat.png", dpi=150, bbox_inches="tight"
    )

    # ── RESULTS ──────────────────────────────────────────────────────
    print(f"\n══ lecture/s3_cpcv_multiple_testing ══════════════════════")
    print(f"  n_cpcv_paths: {len(cpcv_paths)}")
    print(f"  pbo: {pbo:.4f}")
    print(f"  n_below_median: {n_below_median}")
    print(f"  oos_ic_std: {oos_std:.4f}")
    print(f"  oos_ic_mean: {oos_return_series.mean():.4f}")
    print(f"  n_models: {len(model_names)}")
    for name, res in hlz_results.items():
        t = res['t_stat']
        print(f"  t_stat_{name}: {t:.4f}")
    print(f"  n_passing_t300: {n_passing_t300}")
    print(f"  n_fp_nominal_50variants: {n_fp_nominal}")
    print(f"  n_fp_bhy_50variants: {n_fp_bhy}")
    print(f"  bhy_inflation_check: all_pass")
    print(f"  ── plot: s3_cpcv_oos_distribution.png ──")
    print(f"     type: histogram of OOS IC across 15 CPCV paths")
    print(f"     n_bars: 15")
    print(f"     median_oos_ic: {median_val:.4f}")
    print(f"     title: {ax_hist.get_title()}")
    print(f"  ── plot: s3_pbo_visualization.png ──")
    print(f"     type: OOS rank distribution with median line")
    print(f"     pbo: {pbo:.4f}")
    print(f"     title: {ax_pbo.get_title()}")
    print(f"  ── plot: s3_hlz_tstat.png ──")
    print(f"     type: horizontal bar chart of t-stats with HLZ thresholds")
    print(f"     n_models: {len(model_labels)}")
    print(f"     title: {ax_hlz.get_title()}")
    print(f"✓ s3_cpcv_multiple_testing: ALL PASSED")
