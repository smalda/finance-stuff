"""Section 5: Deflated Sharpe Ratio — Adjusting for Multiple Testing."""
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
from data_setup import CACHE_DIR, PLOT_DIR, load_ls_portfolio
from shared.metrics import deflated_sharpe_ratio
from shared.backtesting import sharpe_ratio, net_returns


# ── Configuration ─────────────────────────────────────────────────────
T_WINDOWS = [24, 36, 48, 60, 84, 120]   # months of track record
M_VALUES = [1, 5, 10, 20, 50]           # number of strategies tested


# ── CELL: load_portfolio ─────────────────────────────────────────────

ls = load_ls_portfolio()
gross_returns = ls["gross_returns"].dropna()
turnover = ls["turnover"].dropna()

print(f"Gross return series: {len(gross_returns)} months, "
      f"{gross_returns.index[0].date()} – {gross_returns.index[-1].date()}")


# ── CELL: return_moments ─────────────────────────────────────────────

# Compute moments over the full available history (gross returns)
gross_sr = sharpe_ratio(gross_returns, periods_per_year=12)
# Monthly (per-period) SR — this is what deflated_sharpe_ratio() expects,
# since n_obs is measured in months and the variance formula is calibrated
# to the observation frequency.
gross_monthly_sr = gross_returns.mean() / gross_returns.std()
gross_skew = float(stats.skew(gross_returns))
gross_kurt = float(stats.kurtosis(gross_returns, fisher=True))  # excess kurtosis

print(f"\nGross return moments (n={len(gross_returns)}):")
print(f"  Annualised Sharpe:  {gross_sr:.4f}")
print(f"  Monthly Sharpe:     {gross_monthly_sr:.4f}")
print(f"  Skewness:           {gross_skew:.4f}")
print(f"  Excess kurtosis:    {gross_kurt:.4f}")
if abs(gross_skew) > 1.0 or abs(gross_kurt) > 3.0:
    print("  ⚠ Heavy tails / significant skew — non-normality penalises DSR")


# ── CELL: psr_demo ────────────────────────────────────────────────────

# Probabilistic Sharpe Ratio: P(SR > SR_benchmark) for a single trial (M=1)
# This is the special case of DSR with M=1: probability the SR is genuine
# given only one strategy was tested (no multiple-testing penalty).
n_total = len(gross_returns)
psr = deflated_sharpe_ratio(
    observed_sr=gross_monthly_sr,
    n_trials=1,
    n_obs=n_total,
    skew=gross_skew,
    excess_kurt=gross_kurt,
)

print(f"\nProbabilistic Sharpe Ratio (M=1, T={n_total} months):")
print(f"  PSR = {psr:.4f}  (using monthly SR={gross_monthly_sr:.4f})")
print(f"  Interpretation: {psr*100:.1f}% probability the SR is genuine (single trial)")


# ── CELL: surface_sr_selection ───────────────────────────────────────

# Use gross monthly SR directly for the DSR surface. With monthly SR
# (typically 0.15–0.25), the surface naturally spans the full [0, 1] range
# and shows dramatic red-to-green transitions as T and M vary.
monthly_sr = gross_monthly_sr
ret_skew = gross_skew
ret_kurt = gross_kurt
sr_label = "Gross (monthly)"

print(f"\nDSR surface input ({sr_label}):")
print(f"  Monthly SR = {monthly_sr:.4f}, skew = {ret_skew:.4f}, excess kurt = {ret_kurt:.4f}")


# ── CELL: dsr_surface_loop ────────────────────────────────────────────

# Build DSR surface over T_WINDOWS × M_VALUES.
# Key design: hold observed SR constant while varying T (observations) and
# M (trials tested). This isolates the formula's statistical behavior:
# longer track records → higher DSR; more trials → lower DSR.
surface_rows = []

for T in T_WINDOWS:
    for M in M_VALUES:
        dsr = deflated_sharpe_ratio(
            observed_sr=monthly_sr,
            n_trials=M,
            n_obs=T,
            skew=ret_skew,
            excess_kurt=ret_kurt,
        )
        surface_rows.append({
            "T": T,
            "M": M,
            "dsr": dsr if (dsr is not None and np.isfinite(dsr)) else 0.0,
        })


# ── CELL: dsr_surface_pivot ──────────────────────────────────────────

dsr_df = pd.DataFrame(surface_rows)
# Pivot for heatmap (rows=M, cols=T)
dsr_pivot = dsr_df.pivot(index="M", columns="T", values="dsr")
dsr_pivot.index.name = "M (trials)"
dsr_pivot.columns.name = "T (months)"

print("\nDSR surface (rows=M, cols=T):")
print(dsr_pivot.round(3).to_string())


# ── CELL: dsr_heatmap ─────────────────────────────────────────────────

fig_heatmap, ax_hm = plt.subplots(figsize=(9, 5))

im = ax_hm.imshow(
    dsr_pivot.values,
    aspect="auto",
    cmap="RdYlGn",
    vmin=0.0,
    vmax=1.0,
    origin="lower",
)
plt.colorbar(im, ax=ax_hm, label="DSR")

ax_hm.set_xticks(range(len(T_WINDOWS)))
ax_hm.set_xticklabels([f"{t}m" for t in T_WINDOWS])
ax_hm.set_yticks(range(len(M_VALUES)))
ax_hm.set_yticklabels([str(m) for m in M_VALUES])
ax_hm.set_xlabel("Track record length T (months)")
ax_hm.set_ylabel("Number of strategies tested M")
ax_hm.set_title(f"Deflated Sharpe Ratio Surface  [monthly SR={monthly_sr:.3f}, {sr_label}]")

# Annotate cells with DSR values
for i, m in enumerate(M_VALUES):
    for j, t in enumerate(T_WINDOWS):
        val = dsr_pivot.loc[m, t]
        ax_hm.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=8,
                   color="black" if 0.25 < val < 0.85 else "white")

plt.tight_layout()
plt.show()


# ── CELL: min_trl_func ───────────────────────────────────────────────

def min_trl(
    sharpe_monthly: float,
    n_trials: int,
    skew: float = 0.0,
    excess_kurt: float = 0.0,
    confidence: float = 0.95,
) -> float:
    """Minimum Track Record Length (Bailey & López de Prado, 2014).

    Returns the minimum number of monthly observations needed for
    DSR to exceed `confidence` given `n_trials` strategies tested.
    Uses iterative search over T from 6 to 1200 months.

    Args:
        sharpe_monthly: per-period (monthly) Sharpe ratio — must match
            the observation frequency of n_obs passed to DSR.
    """
    for T in range(6, 1201):
        dsr_val = deflated_sharpe_ratio(
            observed_sr=sharpe_monthly,
            n_trials=n_trials,
            n_obs=T,
            skew=skew,
            excess_kurt=excess_kurt,
        )
        if dsr_val is not None and np.isfinite(dsr_val) and dsr_val >= confidence:
            return float(T)
    return float("inf")


# ── CELL: mintrl_compute ─────────────────────────────────────────────

# MinTRL over a range of monthly Sharpe values at M=10, 95% confidence
# Monthly SR range: 0.05–0.60 covers most realistic strategies
# (0.05 monthly ≈ 0.17 annualized, 0.60 monthly ≈ 2.08 annualized)
sharpe_range = np.linspace(0.05, 0.60, 50)
mintrl_values = [
    min_trl(sr, n_trials=10, skew=ret_skew, excess_kurt=ret_kurt)
    for sr in sharpe_range
]

# MinTRL for the observed monthly SR
obs_mintrl = min_trl(monthly_sr, n_trials=10, skew=ret_skew, excess_kurt=ret_kurt)

print(f"\nMinTRL at 95% confidence (M=10 strategies):")
print(f"  Observed monthly SR = {monthly_sr:.4f}")
print(f"  MinTRL = {obs_mintrl:.0f} months")

# Test case for assertion S5-4: monthly SR=0.23 (~annualized 0.80)
mintrl_test = min_trl(0.23, n_trials=10, skew=ret_skew, excess_kurt=ret_kurt)
print(f"  MinTRL at monthly SR=0.23 (~ann 0.80) = {mintrl_test:.0f} months")


# ── CELL: mintrl_chart ────────────────────────────────────────────────

fig_mintrl, ax_trl = plt.subplots(figsize=(8, 5))

finite_mask = np.isfinite(mintrl_values)
ax_trl.plot(sharpe_range[finite_mask], np.array(mintrl_values)[finite_mask],
            color="steelblue", linewidth=2, label="MinTRL (M=10, 95% conf.)")
ax_trl.axhline(24, color="orange", linestyle="--", alpha=0.7, label="24 months (2 years)")
ax_trl.axhline(36, color="red", linestyle="--", alpha=0.7, label="36 months (3 years)")
ax_trl.axvline(monthly_sr, color="green", linestyle=":", alpha=0.9,
               label=f"This strategy monthly SR={monthly_sr:.3f}")

ax_trl.set_xlabel("Monthly Sharpe Ratio")
ax_trl.set_ylabel("Minimum Track Record Length (months)")
ax_trl.set_title("MinTRL vs. Monthly Sharpe Ratio at 95% Confidence (M=10)")
ax_trl.legend(fontsize=9)
ax_trl.set_xlim(0.05, 0.60)
ax_trl.set_ylim(0, 600)
plt.tight_layout()
plt.show()


# ── Persist DSR surface for downstream files ──────────────────────────

dsr_df.to_parquet(CACHE_DIR / "dsr_surface.parquet")
print(f"\nDSR surface cached → {CACHE_DIR / 'dsr_surface.parquet'}")


if __name__ == "__main__":
    # ── ASSERTIONS ──────────────────────────────────────────────────

    # S5-1: PSR (computed with gross SR, M=1) must be in [0, 1]
    assert 0.0 <= psr <= 1.0, f"PSR = {psr:.4f}, expected in [0.0, 1.0]"

    # All DSR surface cells must also be in [0, 1]
    dsr_vals = dsr_pivot.values
    dsr_finite = dsr_vals[np.isfinite(dsr_vals)]
    assert len(dsr_finite) > 0, "No finite DSR values in surface"
    assert dsr_finite.min() >= 0.0, f"DSR min = {dsr_finite.min():.4f} < 0"
    assert dsr_finite.max() <= 1.0, f"DSR max = {dsr_finite.max():.4f} > 1"

    # S5-2: DSR weakly decreasing in M (more trials → lower DSR), fixed T=48
    t_test = 48 if 48 in T_WINDOWS else T_WINDOWS[len(T_WINDOWS) // 2]
    dsr_by_m = [dsr_pivot.loc[m, t_test] for m in M_VALUES]
    for i in range(len(M_VALUES) - 1):
        assert dsr_by_m[i] >= dsr_by_m[i + 1] - 1e-8, (
            f"DSR not weakly decreasing in M at T={t_test}: "
            f"DSR(M={M_VALUES[i]})={dsr_by_m[i]:.4f} < DSR(M={M_VALUES[i+1]})={dsr_by_m[i+1]:.4f}"
        )

    # S5-3: DSR weakly increasing in T (longer track record → higher DSR), fixed M=10
    m_test = 10 if 10 in M_VALUES else M_VALUES[len(M_VALUES) // 2]
    dsr_by_t = [dsr_pivot.loc[m_test, t] for t in T_WINDOWS]
    for i in range(len(T_WINDOWS) - 1):
        assert dsr_by_t[i] <= dsr_by_t[i + 1] + 1e-8, (
            f"DSR not weakly increasing in T at M={m_test}: "
            f"DSR(T={T_WINDOWS[i]})={dsr_by_t[i]:.4f} > DSR(T={T_WINDOWS[i+1]})={dsr_by_t[i+1]:.4f}"
        )

    # S5-4: MinTRL at monthly SR=0.23, M=10, 95% confidence — reasonable range
    # With excess kurtosis ~4.2, the non-normality penalty pushes MinTRL higher
    assert 6 <= mintrl_test <= 300, (
        f"MinTRL at monthly SR=0.23 = {mintrl_test:.0f} months, expected [6, 300]"
    )

    # S5-5: skewness and excess kurtosis reported (structural) — values must be finite
    assert np.isfinite(ret_skew), f"Skewness is not finite: {ret_skew}"
    assert np.isfinite(ret_kurt), f"Excess kurtosis is not finite: {ret_kurt}"

    # ── RESULTS ─────────────────────────────────────────────────────
    print(f"\n══ lecture/s5_deflated_sharpe ══════════════════════════")
    print(f"  sr_label: {sr_label}")
    print(f"  n_months_total: {n_total}")
    print(f"  gross_sr_annualized: {gross_sr:.4f}")
    print(f"  gross_sr_monthly: {gross_monthly_sr:.4f}")
    print(f"  surface_sr_monthly: {monthly_sr:.4f}")
    print(f"  skewness: {ret_skew:.4f}")
    print(f"  excess_kurtosis: {ret_kurt:.4f}")
    print(f"  psr_m1_t{n_total}: {psr:.4f}")
    print(f"  mintrl_test_m10_95pct: {mintrl_test:.0f} months")
    print(f"  mintrl_observed_sr_m10_95pct: {obs_mintrl:.0f} months")
    print(f"  dsr_t24_m1: {dsr_pivot.loc[1, 24]:.4f}")
    print(f"  dsr_t24_m50: {dsr_pivot.loc[50, 24]:.4f}")
    print(f"  dsr_t120_m50: {dsr_pivot.loc[50, 120]:.4f}")
    print(f"  dsr_range: [{dsr_finite.min():.3f}, {dsr_finite.max():.3f}]")
    print(f"  monotone_in_M_at_T={t_test}: PASS")
    print(f"  monotone_in_T_at_M={m_test}: PASS")

    # ── PLOTS ────────────────────────────────────────────────────────
    fig_heatmap.savefig(PLOT_DIR / "s5_dsr_heatmap.png", dpi=150, bbox_inches="tight")
    print(f"  ── plot: s5_dsr_heatmap.png ──")
    print(f"     type: 2D heatmap DSR surface (M × T)")
    print(f"     rows: M={M_VALUES}")
    print(f"     cols: T={T_WINDOWS}")
    print(f"     dsr_range: [{dsr_finite.min():.3f}, {dsr_finite.max():.3f}]")
    print(f"     title: {ax_hm.get_title()}")

    fig_mintrl.savefig(PLOT_DIR / "s5_mintrl_chart.png", dpi=150, bbox_inches="tight")
    print(f"  ── plot: s5_mintrl_chart.png ──")
    print(f"     type: line chart MinTRL vs. Sharpe")
    n_finite_mintrl = int(np.sum(np.isfinite(mintrl_values)))
    print(f"     n_sr_values: {n_finite_mintrl}")
    print(f"     sr_range: [{sharpe_range[0]:.2f}, {sharpe_range[-1]:.2f}]")
    print(f"     title: {ax_trl.get_title()}")

    print(f"✓ s5_deflated_sharpe: ALL PASSED")
