"""Exercise 1: IC Autopsy — Signal Performance by VIX Regime"""
import matplotlib
matplotlib.use("Agg")
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).parent.parent))
from data_setup import CACHE_DIR, PLOT_DIR


# ── Load upstream data ──────────────────────────────────────
ic_series = pd.read_parquet(CACHE_DIR / "gbm_ic_series.parquet").squeeze()
if isinstance(ic_series, pd.DataFrame):
    ic_series = ic_series.iloc[:, 0]
ic_series.name = "ic"

vix_monthly = pd.read_parquet(CACHE_DIR / "vix_monthly.parquet").squeeze()
if isinstance(vix_monthly, pd.DataFrame):
    vix_monthly = vix_monthly.iloc[:, 0]
vix_monthly.name = "VIX"


# ── CELL: regime_classification ─────────────────────────────

# Align VIX to IC series dates
vix_aligned = vix_monthly.reindex(ic_series.index)
vix_aligned = vix_aligned.dropna()
ic_aligned = ic_series.loc[vix_aligned.index]

# Classify months by median VIX split
vix_median = vix_aligned.median()
regime = pd.Series(
    np.where(vix_aligned >= vix_median, "high_vol", "low_vol"),
    index=vix_aligned.index,
    name="regime",
)
n_high = (regime == "high_vol").sum()
n_low = (regime == "low_vol").sum()

print(f"VIX median threshold: {vix_median:.2f}")
print(f"High-vol months: {n_high}, Low-vol months: {n_low}")


# ── CELL: regime_ic_statistics ──────────────────────────────

ic_high = ic_aligned[regime == "high_vol"]
ic_low = ic_aligned[regime == "low_vol"]

mean_ic_high = ic_high.mean()
mean_ic_low = ic_low.mean()
std_ic_high = ic_high.std()
std_ic_low = ic_low.std()
regime_contrast = abs(mean_ic_high - mean_ic_low)

pct_pos_high = (ic_high > 0).mean()
pct_pos_low = (ic_low > 0).mean()

print(f"\nHigh-vol regime: mean IC = {mean_ic_high:.4f}, "
      f"std IC = {std_ic_high:.4f}, pct_positive = {pct_pos_high:.4f}")
print(f"Low-vol regime:  mean IC = {mean_ic_low:.4f}, "
      f"std IC = {std_ic_low:.4f}, pct_positive = {pct_pos_low:.4f}")
print(f"Regime contrast |IC_high - IC_low|: {regime_contrast:.4f}")


# ── CELL: two_sample_ttest ──────────────────────────────────

t_stat, p_value = stats.ttest_ind(ic_high, ic_low, equal_var=False)

print(f"\nWelch two-sample t-test (high vs low vol IC):")
print(f"  t-statistic: {t_stat:.4f}")
print(f"  p-value: {p_value:.4f}")


# ── CELL: ic_regime_plot ────────────────────────────────────

fig, ax = plt.subplots(figsize=(12, 5))

# Shade background by regime
for date in ic_aligned.index:
    color = "#FFD6D6" if regime[date] == "high_vol" else "#D6E6FF"
    idx = ic_aligned.index.get_loc(date)
    left = date - pd.Timedelta(days=15)
    right = date + pd.Timedelta(days=15)
    ax.axvspan(left, right, alpha=0.4, color=color, linewidth=0)

# Plot IC as bars
colors = ["#C0392B" if regime[d] == "high_vol" else "#2980B9"
          for d in ic_aligned.index]
ax.bar(ic_aligned.index, ic_aligned.values, width=20, color=colors, alpha=0.8)
ax.axhline(0, color="black", linewidth=0.5)
ax.axhline(mean_ic_high, color="#C0392B", linestyle="--", linewidth=1,
           label=f"Mean IC high-vol = {mean_ic_high:.3f}")
ax.axhline(mean_ic_low, color="#2980B9", linestyle="--", linewidth=1,
           label=f"Mean IC low-vol = {mean_ic_low:.3f}")
ax.set(title="GBM IC by VIX Regime",
       xlabel="Date", ylabel="Information Coefficient")
ax.legend(loc="upper left", fontsize=9)
plt.tight_layout()
plt.show()


if __name__ == "__main__":
    # ── ASSERTIONS ─────────────────────────────────
    n_total = len(ic_aligned)
    assert 60 <= n_total <= 72, \
        f"Expected 60-72 total OOS months, got {n_total}"

    assert 28 <= n_high <= 41, \
        f"High-vol count {n_high} outside [28, 41]"
    assert 28 <= n_low <= 41, \
        f"Low-vol count {n_low} outside [28, 41]"

    assert -0.05 <= mean_ic_high <= 0.10, \
        f"Mean IC high-vol {mean_ic_high:.4f} outside [-0.05, 0.10]"
    assert -0.05 <= mean_ic_low <= 0.10, \
        f"Mean IC low-vol {mean_ic_low:.4f} outside [-0.05, 0.10]"

    assert std_ic_high > std_ic_low, \
        f"Expected std_high ({std_ic_high:.4f}) > std_low ({std_ic_low:.4f})"

    assert np.isfinite(t_stat), f"t-statistic is not finite: {t_stat}"
    assert np.isfinite(p_value), f"p-value is not finite: {p_value}"

    assert 0.35 <= pct_pos_high <= 0.75, \
        f"pct_positive high-vol {pct_pos_high:.4f} outside [0.35, 0.75]"
    assert 0.35 <= pct_pos_low <= 0.75, \
        f"pct_positive low-vol {pct_pos_low:.4f} outside [0.35, 0.75]"

    # ── RESULTS ────────────────────────────────────
    print(f"\n══ seminar/ex1_ic_autopsy ══════════════════════════")
    print(f"  n_oos_months: {n_total}")
    print(f"  vix_median_threshold: {vix_median:.2f}")
    print(f"  n_high_vol: {n_high}")
    print(f"  n_low_vol: {n_low}")
    print(f"  mean_ic_high: {mean_ic_high:.4f}")
    print(f"  mean_ic_low: {mean_ic_low:.4f}")
    print(f"  std_ic_high: {std_ic_high:.4f}")
    print(f"  std_ic_low: {std_ic_low:.4f}")
    print(f"  regime_contrast: {regime_contrast:.4f}")
    print(f"  t_stat: {t_stat:.4f}")
    print(f"  p_value: {p_value:.4f}")
    print(f"  pct_positive_high: {pct_pos_high:.4f}")
    print(f"  pct_positive_low: {pct_pos_low:.4f}")

    # ── PLOT ───────────────────────────────────────
    fig.savefig(PLOT_DIR / "ex1_ic_regime.png", dpi=150, bbox_inches="tight")
    print(f"  ── plot: ex1_ic_regime.png ──")
    print(f"     type: bar chart with VIX regime shading")
    print(f"     n_bars: {len(ax.patches)}")
    print(f"     y_range: [{ax.get_ylim()[0]:.2f}, {ax.get_ylim()[1]:.2f}]")
    print(f"     title: {ax.get_title()}")

    print(f"✓ ex1_ic_autopsy: ALL PASSED")
