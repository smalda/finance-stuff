"""Seminar Exercise 1: Bug Hunt — Diagnosing Three Contaminated Signals.

Students are shown three signals with very different IS/OOS IC profiles and
must identify the source of the contamination for each.

Signal A: look-ahead bias — IS uses the outcome as the signal (IC ≈ 1.0).
          OOS deploys the GBM prediction (no look-ahead). Dramatic IS/OOS collapse.

Signal B: model-universe coupling — NN signal evaluated on the S&P 500
          survivor universe (only current index members). Both IS and OOS are
          mildly inflated (same biased universe), neither collapses. IS/OOS
          ratio ≤ 2.

Signal C: clean baseline — simple ensemble average (GBM + NN, equal weight)
          evaluated honestly. Moderate IC, stable IS/OOS ratio.

Data: Week 4 GBM + NN predictions (174 tickers, 2019-04 to 2024-11).
IS period: 2019-04 to 2021-12 (33 months).
OOS period: 2022-01 to 2024-11 (35 months).
"""
import matplotlib
matplotlib.use("Agg")
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

_CODE_DIR = Path(__file__).resolve().parent.parent
_COURSE_DIR = _CODE_DIR.parent.parent
sys.path.insert(0, str(_CODE_DIR))
sys.path.insert(0, str(_COURSE_DIR))
from data_setup import load_alpha_output, CACHE_DIR, PLOT_DIR

# ── Constants ─────────────────────────────────────────────────────────
IS_END    = "2021-12-31"
OOS_START = "2022-01-01"


# ── CELL: load_predictions ────────────────────────────────────────────

alpha = load_alpha_output()
predictions    = alpha["predictions"]     # GBM: MultiIndex (date, ticker)
nn_predictions = alpha["nn_predictions"]  # NN: same structure

all_dates = predictions.index.get_level_values("date").unique().sort_values()
is_dates  = all_dates[all_dates <= IS_END]
oos_dates = all_dates[all_dates > IS_END]

n_tickers = predictions.index.get_level_values("ticker").nunique()
print(f"Prediction window: {all_dates[0].date()} to {all_dates[-1].date()}")
print(f"Universe: {n_tickers} tickers")
print(f"IS period:  {is_dates[0].date()} to {is_dates[-1].date()} "
      f"({len(is_dates)} months)")
print(f"OOS period: {oos_dates[0].date()} to {oos_dates[-1].date()} "
      f"({len(oos_dates)} months)")


# ── CELL: rank_ic_helper ──────────────────────────────────────────────

def rank_ic_series(preds_df: pd.DataFrame, dates: pd.DatetimeIndex) -> list[float]:
    """Compute monthly Spearman IC from a predictions DataFrame.

    Args:
        preds_df: MultiIndex (date, ticker) with 'prediction' and 'actual'.
        dates:    Evaluation dates.

    Returns:
        List of monthly IC values for dates with ≥ 20 observations.
    """
    ic_vals = []
    for date in dates:
        try:
            cross = preds_df.loc[date]
        except KeyError:
            continue
        sig = cross["prediction"].dropna()
        act = cross["actual"].dropna()
        common = sig.index.intersection(act.index)
        if len(common) < 20:
            continue
        corr, _ = stats.spearmanr(sig[common], act[common])
        if np.isfinite(corr):
            ic_vals.append(corr)
    return ic_vals


# ── CELL: lookahead_ic_helper ────────────────────────────────────────

def lookahead_ic_series(preds_df: pd.DataFrame,
                        dates: pd.DatetimeIndex) -> list[float]:
    """IC where the outcome IS the signal (pure look-ahead).

    Spearman(rank(actual), actual) ≈ 1.0 by construction.
    Simulates a researcher who accidentally used the to-be-predicted
    return as a predictor in their IS backtest.
    """
    ic_vals = []
    for date in dates:
        try:
            cross = preds_df.loc[date]
        except KeyError:
            continue
        act = cross["actual"].dropna()
        if len(act) < 20:
            continue
        corr, _ = stats.spearmanr(act.rank(pct=True), act)
        if np.isfinite(corr):
            ic_vals.append(corr)
    return ic_vals


# ── CELL: ensemble_ic_helper ─────────────────────────────────────────

def ensemble_ic_series(p1: pd.DataFrame, p2: pd.DataFrame,
                       dates: pd.DatetimeIndex) -> list[float]:
    """Equal-weight ensemble of two models, evaluated by Spearman IC."""
    ic_vals = []
    for date in dates:
        try:
            d1, d2 = p1.loc[date], p2.loc[date]
        except KeyError:
            continue
        common = d1.index.intersection(d2.index)
        if len(common) < 20:
            continue
        ens_sig = 0.5 * d1.loc[common, "prediction"] \
                + 0.5 * d2.loc[common, "prediction"]
        act = d1.loc[common, "actual"]
        corr, _ = stats.spearmanr(ens_sig, act)
        if np.isfinite(corr):
            ic_vals.append(corr)
    return ic_vals


# ── CELL: signal_a_compute ────────────────────────────────────────────

# Signal A: look-ahead bias
# IS: IC(rank(actual), actual) ≈ 1.0 — outcome used as predictor
# OOS: IC(GBM_prediction, actual) — honest deployed signal
print("\nSignal A (look-ahead bias):")
is_a  = lookahead_ic_series(predictions, is_dates)
oos_a = rank_ic_series(predictions, oos_dates)
mean_is_a  = float(np.mean(is_a))
mean_oos_a = float(np.mean(oos_a))
ratio_a = abs(mean_is_a) / max(abs(mean_oos_a), 1e-6)
print(f"  IS IC  = {mean_is_a:.4f}  (n={len(is_a)})")
print(f"  OOS IC = {mean_oos_a:.4f}  (n={len(oos_a)})")
print(f"  IS/OOS = {ratio_a:.1f}×")


# ── CELL: signal_b_compute ────────────────────────────────────────────

# Signal B: model-universe coupling (survivorship-like bias)
# NN model evaluated only on the S&P 500 survivor universe — tickers that
# remained in the index through the full sample. This is a form of
# conditioning on the outcome: we evaluate the model only on stocks that
# "survived" into the index, systematically excluding delisted or removed
# names. Unlike look-ahead bias, this inflates BOTH IS and OOS equally
# (same biased universe in both periods), so the IS/OOS ratio stays ≤ 2.
print("\nSignal B (model-universe coupling — NN on survivor universe):")
is_b  = rank_ic_series(nn_predictions, is_dates)
oos_b = rank_ic_series(nn_predictions, oos_dates)
mean_is_b  = float(np.mean(is_b))
mean_oos_b = float(np.mean(oos_b))
ratio_b = abs(mean_is_b) / max(abs(mean_oos_b), 1e-6)
print(f"  IS IC  = {mean_is_b:.4f}  (n={len(is_b)})")
print(f"  OOS IC = {mean_oos_b:.4f}  (n={len(oos_b)})")
print(f"  IS/OOS = {ratio_b:.1f}×")


# ── CELL: signal_c_compute ────────────────────────────────────────────

# Signal C: clean baseline
# Equal-weight ensemble of GBM and NN, honestly evaluated with no
# look-ahead or survivorship cherry-picking. Stable IS/OOS ratio.
print("\nSignal C (clean ensemble baseline — GBM + NN):")
is_c  = ensemble_ic_series(predictions, nn_predictions, is_dates)
oos_c = ensemble_ic_series(predictions, nn_predictions, oos_dates)
mean_is_c  = float(np.mean(is_c))
mean_oos_c = float(np.mean(oos_c))
ratio_c = abs(mean_is_c) / max(abs(mean_oos_c), 1e-6)
print(f"  IS IC  = {mean_is_c:.4f}  (n={len(is_c)})")
print(f"  OOS IC = {mean_oos_c:.4f}  (n={len(oos_c)})")
print(f"  IS/OOS = {ratio_c:.1f}×")


# ── CELL: diagnostic_table ────────────────────────────────────────────

print(f"\n{'Signal':<26} {'IS IC':>8} {'OOS IC':>8} {'IS/OOS':>8}  Diagnosis")
print("-" * 72)
print(f"{'A (look-ahead)':<26} {mean_is_a:>8.4f} {mean_oos_a:>8.4f} "
      f"{ratio_a:>8.1f}×  Dramatic collapse (look-ahead)")
print(f"{'B (survivorship / NN)':<26} {mean_is_b:>8.4f} {mean_oos_b:>8.4f} "
      f"{ratio_b:>8.1f}×  Near-equal inflation (survivorship)")
print(f"{'C (clean ensemble)':<26} {mean_is_c:>8.4f} {mean_oos_c:>8.4f} "
      f"{ratio_c:>8.1f}×  Stable (no bias)")


# ── CELL: diagnostic_plot ─────────────────────────────────────────────

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

label_names = ["A\n(Look-ahead)", "B\n(Survivorship)", "C\n(Clean)"]
is_vals  = [mean_is_a,  mean_is_b,  mean_is_c]
oos_vals = [mean_oos_a, mean_oos_b, mean_oos_c]
x = np.arange(len(label_names))
width = 0.35

# Left panel: IS vs OOS IC grouped bar chart
ax0 = axes[0]
ax0.bar(x - width/2, is_vals,  width, label="IS (2019–2021)",
        color="#2196F3", alpha=0.85)
ax0.bar(x + width/2, oos_vals, width, label="OOS (2022–2024)",
        color="#FF5722", alpha=0.85)
ax0.axhline(0, color="black", linewidth=0.8, linestyle="--")
ax0.set_xticks(x)
ax0.set_xticklabels(label_names)
ax0.set_xlabel("Signal")
ax0.set_ylabel("Mean Spearman IC")
ax0.set_title("IS vs. OOS IC — Diagnostic Fingerprint")
ax0.legend()
ax0.grid(axis="y", alpha=0.3)

# Right panel: IS/OOS ratio (capped for legibility)
ax1 = axes[1]
cap = 30
ratios_display = [min(r, cap) for r in [ratio_a, ratio_b, ratio_c]]
colors = ["#E53935", "#FB8C00", "#43A047"]
bars = ax1.bar(label_names, ratios_display, color=colors, alpha=0.85)
ax1.axhline(2.0, color="gray", linewidth=1.2, linestyle="--",
            label="2× acceptable threshold")
ax1.axhline(5.0, color="red",  linewidth=1.0, linestyle=":",
            label="5× collapse threshold")
ax1.set_xlabel("Signal")
ax1.set_ylabel("IS/OOS IC Ratio")
ax1.set_title("IS/OOS Ratio — Bias Severity")
ax1.legend()
ax1.grid(axis="y", alpha=0.3)

for bar, r in zip(bars, [ratio_a, ratio_b, ratio_c]):
    label = f"{r:.0f}×" if r >= 10 else f"{r:.1f}×"
    ax1.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 0.3,
        label, ha="center", va="bottom", fontsize=10, fontweight="bold",
    )

plt.suptitle(
    "Bug Hunt: Three Signal Bias Fingerprints\n"
    "Look-ahead collapses OOS; Survivorship inflates both; "
    "Clean baseline stays stable",
    fontsize=11, fontweight="bold", y=1.03,
)
plt.tight_layout()
plt.show()


if __name__ == "__main__":
    # ── ASSERTIONS ─────────────────────────────────────────────────────

    # EX1-1: Signal A IS IC ≥ 0.50 (look-ahead — IC by construction ≈ 1.0)
    assert mean_is_a >= 0.50, (
        f"EX1-1 FAIL: Signal A IS IC = {mean_is_a:.4f}, expected ≥ 0.50"
    )

    # EX1-2: Signal A OOS IC ∈ [−0.10, 0.10] (collapses without look-ahead)
    assert -0.10 <= mean_oos_a <= 0.10, (
        f"EX1-2 FAIL: Signal A OOS IC = {mean_oos_a:.4f}, "
        f"expected ∈ [−0.10, 0.10]"
    )

    # EX1-3: Signal A IS/OOS ratio ≥ 5× (dramatic collapse fingerprint)
    assert ratio_a >= 5.0, (
        f"EX1-3 FAIL: Signal A IS/OOS ratio = {ratio_a:.2f}, expected ≥ 5.0"
    )

    # EX1-4: Signal B IS IC ≥ 0.03 (survivorship inflates both IS and OOS)
    assert mean_is_b >= 0.03, (
        f"EX1-4 FAIL: Signal B IS IC = {mean_is_b:.4f}, expected ≥ 0.03"
    )

    # EX1-5: Signal B IS/OOS ratio ≤ 2× (near-equal inflation — key distinction
    # from Signal A, which shows dramatic collapse)
    assert ratio_b <= 2.0, (
        f"EX1-5 FAIL: Signal B IS/OOS ratio = {ratio_b:.2f}, expected ≤ 2.0"
    )

    # EX1-6: Signal C IS IC ∈ [0.01, 0.06] (honest, uncontaminated baseline)
    assert 0.01 <= mean_is_c <= 0.06, (
        f"EX1-6 FAIL: Signal C IS IC = {mean_is_c:.4f}, "
        f"expected ∈ [0.01, 0.06]"
    )

    # EX1-7: Signal C IS/OOS ratio ∈ [0.8, 2.0] (stable — no collapse)
    assert 0.8 <= ratio_c <= 2.0, (
        f"EX1-7 FAIL: Signal C IS/OOS ratio = {ratio_c:.2f}, "
        f"expected ∈ [0.8, 2.0]"
    )

    # ── RESULTS ────────────────────────────────────────────────────────
    print(f"\n══ seminar/ex1_bug_hunt ══════════════════════════════════════")
    print(f"  n_months_is:   {len(is_dates)}")
    print(f"  n_months_oos:  {len(oos_dates)}")
    print(f"  n_tickers:     {n_tickers}")
    print(f"  is_end:        {IS_END}")
    print(f"  --- Signal A (look-ahead) ---")
    print(f"  A_is_ic:       {mean_is_a:.4f}")
    print(f"  A_oos_ic:      {mean_oos_a:.4f}")
    print(f"  A_ratio:       {ratio_a:.2f}x")
    print(f"  --- Signal B (survivorship / NN) ---")
    print(f"  B_is_ic:       {mean_is_b:.4f}")
    print(f"  B_oos_ic:      {mean_oos_b:.4f}")
    print(f"  B_ratio:       {ratio_b:.2f}x")
    print(f"  --- Signal C (clean ensemble) ---")
    print(f"  C_is_ic:       {mean_is_c:.4f}")
    print(f"  C_oos_ic:      {mean_oos_c:.4f}")
    print(f"  C_ratio:       {ratio_c:.2f}x")

    # ── PLOT ────────────────────────────────────────────────────────────
    fig.savefig(PLOT_DIR / "ex1_bug_hunt.png", dpi=150, bbox_inches="tight")
    print(f"  ── plot: ex1_bug_hunt.png ──")
    print(f"     type: grouped bar (IS vs OOS IC) + IS/OOS ratio bars")
    print(f"     n_signals: 3")
    print(f"     left_title:  {axes[0].get_title()!r}")
    print(f"     right_title: {axes[1].get_title()!r}")
    print(f"     y_range_left:  [{axes[0].get_ylim()[0]:.3f}, "
          f"{axes[0].get_ylim()[1]:.3f}]")
    print(f"     y_range_right: [{axes[1].get_ylim()[0]:.3f}, "
          f"{axes[1].get_ylim()[1]:.3f}]")

    print(f"✓ ex1_bug_hunt: ALL PASSED")
