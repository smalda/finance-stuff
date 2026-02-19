"""
Section 6: The Factor Zoo Problem — Why Most Factors Are Noise

Acceptance criteria (from README):
- Synthetic example: 50 random "factors" tested, at least 1 appears significant
  at 5% level by chance
- Real-world example: 6 canonical factors (MKT, SMB, HML, RMW, CMA, MOM)
  tested with IS (1963-1999) vs. OOS (2000-2023) split
- Data split: at least 10 years in-sample, at least 3 years out-of-sample
- In-sample t-stats computed for all factors
- Out-of-sample t-stats computed for all factors
- At least one factor degrades from significant IS to not significant OOS
- At least one factor remains significant out-of-sample
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats as sp_stats

sys.path.insert(0, str(Path(__file__).parent.parent))
from data_setup import load_ken_french_factors

kf = load_ken_french_factors()


# ── CELL: synthetic_factor_zoo ──────────────────────────
# Purpose: Create 50 purely random "factors" (Gaussian noise) and test each
#   with a simple t-test. Count how many appear statistically significant at
#   the 5% level. This demonstrates the multiple testing problem.
# Takeaway: With 50 random factors, we expect ~2.5 (5% of 50) to appear
#   significant by pure chance. In practice we get 2-4 false positives. This
#   is exactly the mechanism behind the factor zoo: test 316 characteristics,
#   and ~16 will look significant even if none are real. The lesson: a single
#   t-stat > 2 means nothing without out-of-sample validation.

np.random.seed(42)
n_factors = 50
n_months = 120
n_stocks = 100

fake_returns = np.random.normal(0, 0.05, (n_months, n_stocks))
fake_factors = np.random.normal(0, 0.02, (n_months, n_factors))

significant_count = 0
sig_factors = []
for j in range(n_factors):
    corrs = [np.corrcoef(fake_returns[:, i], fake_factors[:, j])[0, 1]
             for i in range(n_stocks)]
    avg_corr = np.mean(corrs)
    se = np.std(corrs) / np.sqrt(n_stocks)
    t_stat = avg_corr / se if se > 0 else 0
    if abs(t_stat) > 1.96:
        significant_count += 1
        sig_factors.append((j, t_stat))

print(f"Synthetic factor zoo: {n_factors} random factors tested")
print(f"  Significant at 5% level: {significant_count} "
      f"(expected ~{n_factors * 0.05:.1f})")
if sig_factors:
    print(f"  False positives: {[f'Factor {j} (t={t:.2f})' for j, t in sig_factors]}")


# ── CELL: real_world_is_oos ───────────────────────────
# Purpose: Test the 6 canonical factors (MKT, SMB, HML, RMW, CMA, MOM) in-
#   sample (1963-1999) vs. out-of-sample (2000-2023). Each factor is a long-
#   short portfolio aggregating thousands of stocks in Ken French's universe,
#   so t-statistics are much more powerful than single-stock characteristics.
#   The split at 2000 is historically meaningful: FF3 was published in 1993
#   using pre-1990s data, so 2000+ is genuine out-of-sample.
# Takeaway: MOM is the clearest casualty: t=5.51 IS (the strongest signal of
#   all six factors) collapses to t=0.43 OOS — momentum crashes in 2009 and
#   2020 destroyed the premium. HML degrades from t=2.85 to t=1.04 — the
#   "death of value" is a real phenomenon, not a statistical artifact. In
#   contrast, MKT (2.76→2.04), RMW (2.19→2.62), and CMA (2.79→2.11) remain
#   significant in both periods, suggesting they capture genuine, persistent
#   risk premia. SMB never reaches significance in either period (t≈1.3).
#   The lesson: even factors with overwhelming IS evidence (MOM at t=5.5!)
#   can fail OOS. This is the core problem of the factor zoo.

factor_names = ["Mkt-RF", "SMB", "HML", "RMW", "CMA", "MOM"]
kf_trimmed = kf.loc["1963":"2023", factor_names]

is_end = "1999-12-31"
oos_start = "2000-01-31"
kf_is = kf_trimmed.loc[:is_end]
kf_oos = kf_trimmed.loc[oos_start:]

print(f"\nIS:  {kf_is.index[0].date()} to {kf_is.index[-1].date()} ({len(kf_is)} months)")
print(f"OOS: {kf_oos.index[0].date()} to {kf_oos.index[-1].date()} ({len(kf_oos)} months)")

def compute_tstats(df):
    result = {}
    for col in df.columns:
        vals = df[col]
        mean = vals.mean()
        se = vals.std() / np.sqrt(len(vals))
        result[col] = mean / se if se > 0 else 0
    return result

is_tstats = compute_tstats(kf_is)
oos_tstats = compute_tstats(kf_oos)

comparison = pd.DataFrame({
    "t-stat (IS)": is_tstats,
    "t-stat (OOS)": oos_tstats,
})
comparison["Sig IS?"] = comparison["t-stat (IS)"].abs() > 1.96
comparison["Sig OOS?"] = comparison["t-stat (OOS)"].abs() > 1.96
comparison["Degraded?"] = comparison["Sig IS?"] & ~comparison["Sig OOS?"]

print("\nFactor Zoo: In-Sample vs. Out-of-Sample t-statistics:")
print(comparison.round(3))

degraded = comparison[comparison["Degraded?"]].index.tolist()
persistent = comparison[comparison["Sig OOS?"]].index.tolist()
print(f"\nSignificant IS but NOT OOS (degraded): {degraded if degraded else 'None'}")
print(f"Significant in BOTH periods (persistent): {persistent if persistent else 'None'}")


# ── CELL: plot_factor_zoo ───────────────────────────────
# Purpose: Side-by-side bar chart of IS vs. OOS t-statistics for KF factors.
# Visual: Six factor pairs. MOM shows the most dramatic collapse — tall blue
#   bar (IS=5.5) paired with a near-zero orange bar (OOS=0.4). HML degrades
#   from ~2.8 to ~1.0. MKT, RMW, CMA all maintain both bars above the red
#   dashed significance lines. SMB stays below significance in both periods.
#   The red dashed ±1.96 lines make it easy to see which factors cross the
#   threshold in each period.

fig, ax = plt.subplots(figsize=(14, 6))
x = np.arange(len(comparison))
width = 0.35
ax.bar(x - width / 2, comparison["t-stat (IS)"], width, label="In-Sample",
       alpha=0.8, color="steelblue")
ax.bar(x + width / 2, comparison["t-stat (OOS)"], width, label="Out-of-Sample",
       alpha=0.8, color="darkorange")
ax.axhline(1.96, color="red", linestyle="--", linewidth=1, alpha=0.7, label="t = ±1.96")
ax.axhline(-1.96, color="red", linestyle="--", linewidth=1, alpha=0.7)
ax.set_xticks(x)
ax.set_xticklabels(comparison.index, rotation=45, ha="right")
ax.set_ylabel("t-statistic")
ax.set_title("Factor Zoo: Do Factors Survive Out-of-Sample?\n"
             "IS (1963-1999) vs. OOS (2000-2023)",
             fontsize=13, fontweight="bold")
ax.legend()
ax.grid(True, axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig(Path(__file__).parent.parent / ".cache" / "s6_factor_zoo.png",
            dpi=150, bbox_inches="tight")
plt.close()


if __name__ == "__main__":
    # ── ASSERTIONS ─────────────────────────────────────
    assert significant_count >= 1, (
        f"Synthetic: expected >= 1 false positive, got {significant_count}")

    assert len(kf_is) >= 100, (
        f"IS period too short: {len(kf_is)} months, need >= 100")
    assert len(kf_oos) >= 36, (
        f"OOS period too short: {len(kf_oos)} months, need >= 36")

    # At least one factor should degrade (sig IS, not OOS)
    assert len(degraded) >= 1, (
        "No factor degrades from IS to OOS — expected at least one")

    # At least one factor should persist
    assert len(persistent) >= 1, (
        "No factor persists OOS — expected at least one")

    print(f"\n✓ Section 6 (Factor Zoo): All acceptance criteria passed")
    print(f"  Synthetic false positives: {significant_count}/50")
    print(f"  Degraded IS→OOS: {degraded}")
    print(f"  Persistent OOS: {persistent}")
