"""Seminar Exercise 4: DSR Calibration — How Track Record Length and Trial
Count Shape Statistical Significance.

Students explore how the Deflated Sharpe Ratio changes across a 5×5 grid of
(T, M) combinations, identifying where short track records and many backtested
strategies erode significance.
"""
import matplotlib
matplotlib.use("Agg")
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats

_CODE_DIR = Path(__file__).resolve().parent.parent
_COURSE_DIR = _CODE_DIR.parent.parent
sys.path.insert(0, str(_CODE_DIR))
sys.path.insert(0, str(_COURSE_DIR))
from data_setup import CACHE_DIR, PLOT_DIR, load_ls_portfolio
from shared.metrics import deflated_sharpe_ratio

# ── Constants ──────────────────────────────────────────────────────────────
# Fixed net annualized SR matches s5 pedagogical alternative (net SR = 0.704).
# EX4 focuses on short-track-record regime: T starts at 6 months.
FIXED_NET_SR_ANNUAL = 0.704
T_WINDOWS = [6, 12, 24, 36, 60]   # months — emphasizes extreme short-T regime
M_VALUES   = [1, 5, 10, 20, 50]   # number of strategy trials


# ── CELL: load_return_series ───────────────────────────────────────────────

ls = load_ls_portfolio()
gross_returns = ls["gross_returns"]

# Higher-moment statistics from the full OOS return series
skew_full   = stats.skew(gross_returns)
kurt_full   = stats.kurtosis(gross_returns, fisher=True)   # excess kurtosis (normal=0)

print(f"Return series: {len(gross_returns)} monthly observations")
print(f"Full-series skewness:       {skew_full:.4f}")
print(f"Full-series excess kurtosis:{kurt_full:.4f}")
print(f"Fixed net SR (annualized):  {FIXED_NET_SR_ANNUAL:.4f}")


# ── CELL: compute_dsr_grid ───────────────────────────────────────────────

# Convert annual SR to monthly before passing to the DSR formula.
# n_obs = T (monthly observations); n_trials = M.
monthly_sr = FIXED_NET_SR_ANNUAL / np.sqrt(12)

rows = []
for T in T_WINDOWS:
    # Slice the last T months for moment estimation
    ret_slice = gross_returns.iloc[-T:]
    sk_T = stats.skew(ret_slice)
    ek_T = stats.kurtosis(ret_slice, fisher=True)

    for M in M_VALUES:
        dsr = deflated_sharpe_ratio(
            monthly_sr,
            n_trials=M,
            n_obs=T,
            skew=sk_T,
            excess_kurt=ek_T,
        )
        label = "PASS" if dsr >= 0.95 else "FAIL"
        print(f"  T={T:3d}mo, M={M:3d}: SR={monthly_sr:.3f}, DSR={dsr:.4f} ({label})")
        rows.append({"T": T, "M": M, "dsr": dsr, "skew": sk_T, "excess_kurt": ek_T})

dsr_df = pd.DataFrame(rows)


# ── CELL: build_dsr_pivot ────────────────────────────────────────────────

# Pivot to 2D matrix for heatmap (rows=M, cols=T)
dsr_pivot = dsr_df.pivot(index="M", columns="T", values="dsr")
# Ensure descending M order (top=50, bottom=1) for visual clarity
dsr_pivot = dsr_pivot.loc[sorted(M_VALUES, reverse=True)]

print(f"\nDSR surface ({len(T_WINDOWS)}T × {len(M_VALUES)}M):")
print(dsr_pivot.round(4).to_string())


# ── CELL: crossover_threshold ──────────────────────────────────────────────

# For each T, find the smallest M where DSR first drops below 0.50.
# A crossover of None means DSR stays ≥ 0.50 for all M values tested.
print("\nCrossover threshold (smallest M where DSR < 0.50):")
for T in T_WINDOWS:
    t_rows = dsr_df[dsr_df["T"] == T].sort_values("M")
    crossover_M = None
    for _, row in t_rows.iterrows():
        if row["dsr"] < 0.50:
            crossover_M = int(row["M"])
            break
    if crossover_M is not None:
        print(f"  T={T:3d}mo → crossover at M={crossover_M} "
              f"(DSR={t_rows.loc[t_rows['M']==crossover_M, 'dsr'].values[0]:.4f})")
    else:
        print(f"  T={T:3d}mo → no crossover (DSR ≥ 0.50 for all M ≤ {max(M_VALUES)})")


# ── CELL: dsr_heatmap ─────────────────────────────────────────────────────

fig, ax = plt.subplots(figsize=(8, 5))

im = ax.imshow(
    dsr_pivot.values,
    aspect="auto",
    cmap="RdYlGn",
    vmin=0.0,
    vmax=1.0,
    origin="upper",
)
plt.colorbar(im, ax=ax, label="DSR")

ax.set_xticks(range(len(T_WINDOWS)))
ax.set_xticklabels([f"T={t}" for t in T_WINDOWS])
ax.set_yticks(range(len(M_VALUES)))
ax.set_yticklabels([f"M={m}" for m in sorted(M_VALUES, reverse=True)])
ax.set_xlabel("Track record length T (months)")
ax.set_ylabel("Number of trials M")
ax.set_title(
    f"Deflated Sharpe Ratio Surface\n"
    f"Net SR={FIXED_NET_SR_ANNUAL:.3f} | skew={skew_full:.2f} | excess kurt={kurt_full:.2f}"
)

# Annotate cells with DSR values
for i, M in enumerate(sorted(M_VALUES, reverse=True)):
    for j, T in enumerate(T_WINDOWS):
        val = dsr_pivot.loc[M, T]
        text_color = "black" if 0.2 < val < 0.8 else "white"
        ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                fontsize=9, color=text_color, fontweight="bold")

# Overlay DSR < 0.50 boundary line between cells
# Draw a thick contour border at DSR = 0.50
contour_data = dsr_pivot.values.copy()
# Extend the data by 0.5 cell on each side for correct contouring
T_coords = np.arange(len(T_WINDOWS))
M_coords = np.arange(len(M_VALUES))
ax.contour(T_coords, M_coords, contour_data, levels=[0.50],
           colors=["white"], linewidths=[2], linestyles=["dashed"])

plt.tight_layout()
plt.show()


if __name__ == "__main__":
    # ── ASSERTIONS ─────────────────────────────────────────────────────
    # EX4-1: All DSR cells in [0.0, 1.0]
    dsr_vals = dsr_df["dsr"].values
    assert (dsr_vals >= 0.0).all(), (
        f"EX4-1 FAIL: some DSR values below 0.0; min={dsr_vals.min():.4f}"
    )
    assert (dsr_vals <= 1.0).all(), (
        f"EX4-1 FAIL: some DSR values above 1.0; max={dsr_vals.max():.4f}"
    )

    # EX4-2: DSR decreases weakly as M increases (fixed T)
    for T in T_WINDOWS:
        t_subset = dsr_df[dsr_df["T"] == T].sort_values("M")["dsr"].values
        for k in range(len(t_subset) - 1):
            assert t_subset[k] >= t_subset[k + 1] - 1e-9, (
                f"EX4-2 FAIL: DSR not monotone at T={T}: "
                f"DSR(M={M_VALUES[k]})={t_subset[k]:.4f} < "
                f"DSR(M={M_VALUES[k+1]})={t_subset[k+1]:.4f}"
            )

    # EX4-3: DSR(T=6, M=50) ≤ 0.50
    dsr_t6_m50 = dsr_df.loc[(dsr_df["T"] == 6) & (dsr_df["M"] == 50), "dsr"].values[0]
    assert dsr_t6_m50 <= 0.50, (
        f"EX4-3 FAIL: DSR(T=6, M=50)={dsr_t6_m50:.4f}, expected ≤ 0.50"
    )

    # EX4-4: DSR(T=60, M=1) ≥ 0.90
    dsr_t60_m1 = dsr_df.loc[(dsr_df["T"] == 60) & (dsr_df["M"] == 1), "dsr"].values[0]
    assert dsr_t60_m1 >= 0.90, (
        f"EX4-4 FAIL: DSR(T=60, M=1)={dsr_t60_m1:.4f}, expected ≥ 0.90"
    )

    # EX4-5: Skewness and excess kurtosis from actual returns (non-zero)
    assert skew_full != 0.0, (
        f"EX4-5 FAIL: skewness is exactly 0.0 — returns moments not computed"
    )
    assert kurt_full != 0.0, (
        f"EX4-5 FAIL: excess kurtosis is exactly 0.0 — returns moments not computed"
    )

    # ── RESULTS ─────────────────────────────────────────────────────────
    print(f"══ seminar/ex4_dsr_calibration ══════════════════════════")
    print(f"  fixed_net_sr_annual:  {FIXED_NET_SR_ANNUAL:.4f}")
    print(f"  monthly_sr:           {monthly_sr:.4f}")
    print(f"  n_obs_total:          {len(gross_returns)}")
    print(f"  skew_full:            {skew_full:.4f}")
    print(f"  excess_kurt_full:     {kurt_full:.4f}")
    print(f"  dsr_t6_m50:           {dsr_t6_m50:.4f}")
    print(f"  dsr_t60_m1:           {dsr_t60_m1:.4f}")
    print(f"  dsr_t6_m1:            "
          f"{dsr_df.loc[(dsr_df['T']==6)&(dsr_df['M']==1),'dsr'].values[0]:.4f}")
    print(f"  dsr_t60_m50:          "
          f"{dsr_df.loc[(dsr_df['T']==60)&(dsr_df['M']==50),'dsr'].values[0]:.4f}")
    print(f"  all_cells_in_0_1:     True")
    print(f"  monotone_in_M:        True")

    # ── PLOT ─────────────────────────────────────────────────────────────
    plot_path = PLOT_DIR / "ex4_dsr_calibration.png"
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    print(f"  ── plot: ex4_dsr_calibration.png ──")
    print(f"     type: 2D DSR surface heatmap (M×T grid)")
    print(f"     grid_size: {len(M_VALUES)}×{len(T_WINDOWS)} ({len(M_VALUES)*len(T_WINDOWS)} cells)")
    print(f"     T_values: {T_WINDOWS}")
    print(f"     M_values: {M_VALUES}")
    print(f"     dsr_min: {dsr_vals.min():.4f}")
    print(f"     dsr_max: {dsr_vals.max():.4f}")
    print(f"     title: {ax.get_title()!r}")

    print(f"✓ ex4_dsr_calibration: ALL PASSED")
