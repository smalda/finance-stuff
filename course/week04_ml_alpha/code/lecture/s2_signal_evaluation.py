"""Section 2: Signal Evaluation — IC, ICIR, and the Fundamental Law"""
import matplotlib
matplotlib.use("Agg")
import sys
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore", category=FutureWarning)

sys.path.insert(0, str(Path(__file__).parent.parent))
from data_setup import (
    load_monthly_panel, FEATURE_COLS, CACHE_DIR, PLOT_DIR,
)


# ── Data ────────────────────────────────────────────────────────────────
panel = load_monthly_panel()
dates = panel.index.get_level_values("date").unique().sort_values()
features = FEATURE_COLS
TARGET = "fwd_return"


# ── CELL: fama_macbeth_one_month ────────────────────────────────────────

def _fm_one_month(panel, features, target, dt, beta_history, min_history):
    """Run OLS for one cross-section; return (pred_series|None, beta|None)."""
    cross = panel.loc[dt].dropna(subset=features + [target])
    if len(cross) < 10:
        return None, None

    # Rank-transform features within this cross-section (percentile ranks)
    X_rank = cross[features].rank(pct=True).values
    y = cross[target].values
    y_dm = y - y.mean()

    # OLS: demeaned returns on rank-transformed features
    X_const = np.column_stack([np.ones(len(X_rank)), X_rank])
    try:
        beta = np.linalg.lstsq(X_const, y_dm, rcond=None)[0]
    except np.linalg.LinAlgError:
        return None, None

    # If enough history, predict this month using average past betas
    pred_series = None
    if len(beta_history) >= min_history:
        avg_beta = np.mean(beta_history, axis=0)
        fitted = X_const @ avg_beta
        pred_series = pd.Series(fitted, index=cross.index, name="prediction")
        pred_series = pred_series.to_frame()
        pred_series["date"] = dt
        pred_series = pred_series.set_index("date", append=True).swaplevel()
        pred_series = pred_series["prediction"]

    return pred_series, beta


# ── CELL: fama_macbeth_predict_fn ───────────────────────────────────────

def fama_macbeth_predict(panel: pd.DataFrame,
                         features: list[str],
                         target: str,
                         min_history: int = 12) -> pd.Series:
    """Expanding-window Fama-MacBeth signal: historical betas predict next month.

    For each month t, run cross-sectional OLS of demeaned returns on
    rank-transformed features, accumulate betas, then apply the expanding
    mean of historical betas to month t's ranks to form a genuinely
    out-of-sample signal.  Rank-transforming features within each
    cross-section removes outlier leverage and is standard quant practice.
    """
    predictions = []
    month_dates = panel.index.get_level_values("date").unique().sort_values()
    beta_history = []  # list of (n_features+1,) arrays

    for i, dt in enumerate(month_dates):
        if i % 20 == 0:
            print(f"  [{i+1}/{len(month_dates)}] Fama-MacBeth OLS for {dt.date()}")

        pred_series, beta = _fm_one_month(
            panel, features, target, dt, beta_history, min_history,
        )
        if beta is not None:
            beta_history.append(beta)
        if pred_series is not None:
            predictions.append(pred_series)

    return pd.concat(predictions)


# ── CELL: run_fama_macbeth ──────────────────────────────────────────────

fm_predictions = fama_macbeth_predict(panel, features, TARGET, min_history=12)
n_pred_months = fm_predictions.index.get_level_values("date").nunique()
print(f"  Fama-MacBeth predictions: {len(fm_predictions)} obs, "
      f"{n_pred_months} months (after 12-month burn-in)")


# ── CELL: compute_ic_series_fn ──────────────────────────────────────────

def _ic_one_month(pred_dt, act_dt, method):
    """Compute IC for a single month cross-section."""
    if len(pred_dt) < 10:
        return None
    if method == "spearman":
        corr, _ = stats.spearmanr(pred_dt.values, act_dt.values)
    else:
        corr = np.corrcoef(pred_dt.values, act_dt.values)[0, 1]
    return corr if np.isfinite(corr) else None


def compute_ic_series(predictions: pd.Series,
                      actuals: pd.Series,
                      method: str = "pearson") -> pd.Series:
    """Compute monthly cross-sectional IC (information coefficient).

    IC_t = corr(prediction_i, actual_i) for all stocks i at month t.

    Parameters
    ----------
    predictions : Series with MultiIndex (date, ticker)
    actuals : Series with MultiIndex (date, ticker)
    method : 'pearson' or 'spearman'

    Returns
    -------
    Series indexed by date, values = IC per month.
    """
    common = predictions.index.intersection(actuals.index)
    pred = predictions.loc[common]
    act = actuals.loc[common]

    dates_all = pred.index.get_level_values("date").unique().sort_values()
    ic_vals = {}

    for dt in dates_all:
        corr = _ic_one_month(pred.loc[dt], act.loc[dt], method)
        if corr is not None:
            ic_vals[dt] = corr

    return pd.Series(ic_vals, name=f"IC_{method}")


# ── CELL: compute_ic_series_run ─────────────────────────────────────────

actuals = panel[TARGET]
pearson_ic = compute_ic_series(fm_predictions, actuals, method="pearson")
rank_ic = compute_ic_series(fm_predictions, actuals, method="spearman")

print(f"  Pearson IC series: {len(pearson_ic)} months")
print(f"  Rank IC series:    {len(rank_ic)} months")


# ── CELL: ic_summary_fn ────────────────────────────────────────────────

def ic_summary(ic_series: pd.Series) -> dict:
    """Compute IC summary statistics from scratch.

    Returns dict with: mean_ic, std_ic, icir, t_stat, p_value, pct_positive.
    Implements the full ic_summary logic without importing from shared.
    """
    n = len(ic_series)
    mean_ic = ic_series.mean()
    std_ic = ic_series.std(ddof=1)
    icir = mean_ic / std_ic if std_ic > 0 else 0.0

    # t-test: H0: mean IC = 0
    t_stat = mean_ic / (std_ic / np.sqrt(n)) if std_ic > 0 else 0.0
    p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df=n - 1))

    pct_positive = (ic_series > 0).mean()

    return {
        "mean_ic": mean_ic,
        "std_ic": std_ic,
        "icir": icir,
        "t_stat": t_stat,
        "p_value": p_value,
        "pct_positive": pct_positive,
        "n_months": n,
    }


# ── CELL: compute_ic_summaries ──────────────────────────────────────────

pearson_summary = ic_summary(pearson_ic)
rank_summary = ic_summary(rank_ic)

print(f"\n  Pearson IC summary:")
for k, v in pearson_summary.items():
    print(f"    {k}: {v:.6f}" if isinstance(v, float) else f"    {k}: {v}")
print(f"\n  Rank IC summary:")
for k, v in rank_summary.items():
    print(f"    {k}: {v:.6f}" if isinstance(v, float) else f"    {k}: {v}")


# ── CELL: fundamental_law ──────────────────────────────────────────────

tickers_per_month = panel.groupby(level="date").size()
BR = tickers_per_month.median()  # breadth = number of independent bets
mean_ic = pearson_summary["mean_ic"]

# Fundamental Law of Active Management: IR ≈ IC × √BR
predicted_ir = mean_ic * np.sqrt(BR)
actual_ir = pearson_summary["icir"]

print(f"\n  Fundamental Law of Active Management:")
print(f"    IC  = {mean_ic:.4f}")
print(f"    BR  = {BR:.0f} (median stocks per month)")
print(f"    Predicted IR = IC × √BR = {mean_ic:.4f} × √{BR:.0f} = {predicted_ir:.4f}")
print(f"    Actual ICIR  = {actual_ir:.4f}")
print(f"    Ratio (predicted / actual) = {predicted_ir / actual_ir:.2f}"
      if actual_ir != 0 else "    Actual ICIR is zero — cannot compute ratio")


# ── CELL: ic_bar_chart ─────────────────────────────────────────────────

fig_ic, ax_ic = plt.subplots(figsize=(12, 5))
colors = ["steelblue" if v >= 0 else "salmon" for v in pearson_ic.values]
ax_ic.bar(range(len(pearson_ic)), pearson_ic.values, color=colors, width=1.0)
ax_ic.axhline(0, color="black", linewidth=0.5)
ax_ic.axhline(
    pearson_summary["mean_ic"], color="navy", linewidth=1.5,
    linestyle="--", label=f'Mean IC = {pearson_summary["mean_ic"]:.4f}',
)
ax_ic.set(
    title="Monthly Information Coefficient (Fama-MacBeth Linear Signal)",
    xlabel="Month index",
    ylabel="IC (Pearson)",
)
ax_ic.legend(loc="upper right")
plt.tight_layout()
plt.show()


# ── CELL: icir_decomposition ──────────────────────────────────────────

fig_decomp, axes = plt.subplots(1, 3, figsize=(14, 4))

# Panel 1: IC distribution
axes[0].hist(pearson_ic.values, bins=20, color="steelblue", edgecolor="white", alpha=0.8)
axes[0].axvline(pearson_summary["mean_ic"], color="navy", linestyle="--", linewidth=1.5)
axes[0].set(title="IC Distribution", xlabel="IC", ylabel="Count")

# Panel 2: Pearson vs Rank IC scatter
axes[1].scatter(pearson_ic.values, rank_ic.values, s=12, alpha=0.5, color="steelblue")
diag_min = min(pearson_ic.min(), rank_ic.min())
diag_max = max(pearson_ic.max(), rank_ic.max())
axes[1].plot([diag_min, diag_max], [diag_min, diag_max], "k--", linewidth=0.8)
axes[1].set(title="Pearson IC vs Rank IC", xlabel="Pearson IC", ylabel="Rank IC")

# Panel 3: ICIR bar comparison
ic_labels = ["Pearson", "Rank"]
ic_means = [pearson_summary["mean_ic"], rank_summary["mean_ic"]]
ic_stds = [pearson_summary["std_ic"], rank_summary["std_ic"]]
icirs = [pearson_summary["icir"], rank_summary["icir"]]

x = np.arange(len(ic_labels))
width = 0.3
axes[2].bar(x - width, ic_means, width, label="Mean IC", color="steelblue")
axes[2].bar(x, ic_stds, width, label="Std IC", color="salmon")
axes[2].bar(x + width, icirs, width, label="ICIR", color="seagreen")
axes[2].set_xticks(x)
axes[2].set_xticklabels(ic_labels)
axes[2].set(title="ICIR Decomposition")
axes[2].legend(fontsize=8)

plt.tight_layout()
plt.show()


if __name__ == "__main__":
    # ── ASSERTIONS ─────────────────────────────────────────────────────

    # S2-1: IC time series length in [60, 129]
    assert 60 <= len(pearson_ic) <= 129, \
        f"S2-1: IC series length {len(pearson_ic)}, expected [60, 129]"

    # S2-2: Mean IC (Fama-MacBeth) in [-0.01, 0.06]
    assert -0.01 <= pearson_summary["mean_ic"] <= 0.06, \
        f"S2-2: Mean IC = {pearson_summary['mean_ic']:.6f}, expected [-0.01, 0.06]"

    # S2-3: Rank IC within +/-0.02 of Pearson IC
    ic_diff = abs(rank_summary["mean_ic"] - pearson_summary["mean_ic"])
    assert ic_diff <= 0.02, \
        f"S2-3: |Rank IC - Pearson IC| = {ic_diff:.6f}, expected <= 0.02"

    # S2-4: ICIR in [0.05, 0.50]
    assert 0.05 <= pearson_summary["icir"] <= 0.50, \
        f"S2-4: ICIR = {pearson_summary['icir']:.4f}, expected [0.05, 0.50]"

    # S2-5: t-stat and p-value finite
    assert np.isfinite(pearson_summary["t_stat"]), \
        f"S2-5: t-stat is not finite: {pearson_summary['t_stat']}"
    assert np.isfinite(pearson_summary["p_value"]), \
        f"S2-5: p-value is not finite: {pearson_summary['p_value']}"

    # S2-6: Fundamental law computation present (checked by output)
    assert BR > 0, f"S2-6: BR = {BR}, expected > 0"
    assert np.isfinite(predicted_ir), \
        f"S2-6: Predicted IR not finite: {predicted_ir}"

    # S2-7: pct_positive in [0.50, 0.70]
    assert 0.50 <= pearson_summary["pct_positive"] <= 0.70, \
        f"S2-7: pct_positive = {pearson_summary['pct_positive']:.4f}, expected [0.50, 0.70]"

    # ── RESULTS ────────────────────────────────────────────────────────
    print(f"\n══ lecture/s2_signal_evaluation ══════════════════════════")
    print(f"  n_ic_months: {len(pearson_ic)}")
    print(f"  mean_pearson_ic: {pearson_summary['mean_ic']:.6f}")
    print(f"  std_pearson_ic: {pearson_summary['std_ic']:.6f}")
    print(f"  mean_rank_ic: {rank_summary['mean_ic']:.6f}")
    print(f"  std_rank_ic: {rank_summary['std_ic']:.6f}")
    print(f"  |rank_ic - pearson_ic|: {ic_diff:.6f}")
    print(f"  icir_pearson: {pearson_summary['icir']:.4f}")
    print(f"  icir_rank: {rank_summary['icir']:.4f}")
    print(f"  t_stat: {pearson_summary['t_stat']:.4f}")
    print(f"  p_value: {pearson_summary['p_value']:.6f}")
    print(f"  pct_positive: {pearson_summary['pct_positive']:.4f}")
    print(f"  breadth_BR: {BR:.0f}")
    print(f"  predicted_IR: {predicted_ir:.4f}")
    print(f"  actual_ICIR: {actual_ir:.4f}")

    # ── PLOT ───────────────────────────────────────────────────────────
    fig_ic.savefig(PLOT_DIR / "s2_ic_bar_chart.png", dpi=150, bbox_inches="tight")
    print(f"  ── plot: s2_ic_bar_chart.png ──")
    print(f"     type: bar chart (monthly IC, colored by sign)")
    print(f"     n_bars: {len(pearson_ic)}")
    print(f"     y_range: [{ax_ic.get_ylim()[0]:.4f}, {ax_ic.get_ylim()[1]:.4f}]")
    print(f"     title: {ax_ic.get_title()}")

    fig_decomp.savefig(PLOT_DIR / "s2_icir_decomposition.png", dpi=150, bbox_inches="tight")
    print(f"  ── plot: s2_icir_decomposition.png ──")
    print(f"     type: 3-panel (IC distribution, Pearson vs Rank, ICIR decomposition)")
    print(f"     titles: {[a.get_title() for a in axes]}")

    print(f"\u2713 s2_signal_evaluation: ALL PASSED")
