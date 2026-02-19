"""Exercise 4: The Factor Zoo Safari

Test 8-10 characteristics (including noise) via Fama-MacBeth,
then apply multiple testing correction.
"""
import matplotlib
matplotlib.use("Agg")
import sys
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats as scipy_stats

sys.path.insert(0, str(Path(__file__).parent.parent))
from data_setup import (
    load_equity_prices, load_monthly_returns, load_fundamentals,
    load_factor_data, CACHE_DIR,
)

warnings.filterwarnings("ignore", category=FutureWarning)

prices = load_equity_prices()
monthly_returns = load_monthly_returns()
fundamentals = load_fundamentals()
ff3 = load_factor_data("3")
bs = fundamentals["balance_sheet"]
inc = fundamentals["income_stmt"]
mcap = fundamentals["market_cap"]

common_idx = monthly_returns.index.intersection(ff3.index)
rf = ff3.loc[common_idx, "RF"]
excess_returns = monthly_returns.loc[common_idx].sub(rf, axis=0)


# ── CELL: build_characteristics ─────────────────────────────

# Build 10 characteristics: 6 real + 4 noise
np.random.seed(42)

fund_chars = {}
for ticker in monthly_returns.columns:
    chars = {}

    # 1. Book-to-market
    if ticker in bs.index.get_level_values("ticker"):
        tk_bs = bs.loc[ticker].sort_index()
        eq = tk_bs["Stockholders Equity"].dropna()
        if len(eq) > 0 and eq.iloc[-1] > 0:
            if ticker in mcap.index and mcap[ticker] > 0:
                chars["book_to_market"] = eq.iloc[-1] / mcap[ticker]

    # 2. Operating profitability
    if ticker in inc.index.get_level_values("ticker"):
        tk_inc = inc.loc[ticker].sort_index()
        oi = tk_inc.get("Operating Income", pd.Series(dtype=float)).dropna()
        if len(oi) == 0:
            oi = tk_inc.get("Net Income", pd.Series(dtype=float)).dropna()
        if len(oi) > 0 and ticker in bs.index.get_level_values("ticker"):
            tk_bs2 = bs.loc[ticker].sort_index()
            eq2 = tk_bs2["Stockholders Equity"].dropna()
            if len(eq2) > 0 and eq2.iloc[-1] > 0:
                chars["profitability"] = oi.iloc[-1] / eq2.iloc[-1]

    # 3. Asset growth
    if ticker in bs.index.get_level_values("ticker"):
        tk_bs3 = bs.loc[ticker].sort_index()
        assets = tk_bs3["Total Assets"].dropna()
        if len(assets) > 1 and assets.iloc[-2] > 0:
            chars["asset_growth"] = (assets.iloc[-1] / assets.iloc[-2]) - 1

    # 4. Earnings yield
    if ticker in inc.index.get_level_values("ticker"):
        ni = inc.loc[ticker].sort_index().get(
            "Net Income", pd.Series(dtype=float)
        ).dropna()
        if len(ni) > 0 and ticker in mcap.index and mcap[ticker] > 0:
            chars["earnings_yield"] = ni.iloc[-1] / mcap[ticker]

    fund_chars[ticker] = chars

fund_df = pd.DataFrame(fund_chars).T


# ── CELL: build_panel_with_noise ────────────────────────────

panel_records = []

for date in common_idx:
    # 5. Momentum (12-1 month)
    mom_end = date - pd.DateOffset(months=1)
    mom_start = date - pd.DateOffset(months=12)
    mask = (prices.index >= mom_start) & (prices.index <= mom_end)
    if mask.sum() < 20:
        continue
    mom_prices = prices.loc[mask]
    if len(mom_prices) < 2:
        continue
    momentum = (mom_prices.iloc[-1] / mom_prices.iloc[0]) - 1

    # 6. Short-term reversal (1-month return)
    reversal = monthly_returns.loc[date] if date in monthly_returns.index else None
    if reversal is None:
        continue

    for ticker in monthly_returns.columns:
        ret = excess_returns.loc[date, ticker] if date in excess_returns.index else np.nan
        if pd.isna(ret):
            continue

        row = {
            "date": date,
            "ticker": ticker,
            "excess_ret": ret,
            "momentum": momentum.get(ticker, np.nan),
            "reversal": reversal.get(ticker, np.nan),
        }

        # Add fundamental chars
        if ticker in fund_df.index:
            for col in fund_df.columns:
                row[col] = fund_df.loc[ticker, col]

        # 7-10. Noise characteristics
        row["noise_1"] = np.random.normal()
        row["noise_2"] = np.random.normal()
        row["noise_3"] = np.random.normal()
        row["noise_4"] = np.random.normal()

        panel_records.append(row)

panel = pd.DataFrame(panel_records)
panel = panel.set_index(["ticker", "date"]).sort_index()

all_chars = ["momentum", "book_to_market", "profitability",
             "asset_growth", "reversal", "earnings_yield",
             "noise_1", "noise_2", "noise_3", "noise_4"]
print(f"Panel shape: {panel.shape}")


# ── CELL: standardize_and_test ──────────────────────────────

# Standardize cross-sectionally
def standardize_month(group):
    """Z-score standardize within each month."""
    for col in all_chars:
        if col in group.columns:
            vals = group[col]
            mean, std = vals.mean(), vals.std()
            if std > 0:
                group[col] = (vals - mean) / std
    return group

panel_std = panel.groupby(level="date", group_keys=False).apply(standardize_month)

# Univariate Fama-MacBeth for each characteristic
fm_results = {}

for char in all_chars:
    sub = panel_std[["excess_ret", char]].dropna()
    if len(sub) < 1000:
        continue

    # Manual Fama-MacBeth
    gammas = []
    dates = sub.index.get_level_values("date").unique()
    for date in dates:
        month_data = sub.loc[sub.index.get_level_values("date") == date]
        if len(month_data) < 30:
            continue
        y = month_data["excess_ret"].values
        x = month_data[char].values
        X = sm.add_constant(x)
        model = sm.OLS(y, X).fit()
        gammas.append(model.params[1])

    gammas = np.array(gammas)
    mean_gamma = gammas.mean()
    se = gammas.std() / np.sqrt(len(gammas))
    t_stat = mean_gamma / se if se > 0 else 0
    p_value = 2 * (1 - scipy_stats.t.cdf(abs(t_stat), df=len(gammas) - 1))

    fm_results[char] = {
        "gamma": mean_gamma,
        "t_stat": t_stat,
        "p_value": p_value,
        "n_months": len(gammas),
    }

results_df = pd.DataFrame(fm_results).T
results_df = results_df.sort_values("p_value")
print("\nUnivariate Fama-MacBeth Results (sorted by p-value):")
print(results_df.round(4).to_string())


# ── CELL: multiple_testing_correction ───────────────────────

n_tests = len(results_df)

# Bonferroni correction
results_df["bonferroni_p"] = (results_df["p_value"] * n_tests).clip(upper=1.0)
results_df["bonf_sig"] = results_df["bonferroni_p"] < 0.05

# Benjamini-Hochberg
results_sorted = results_df.sort_values("p_value")
ranks = np.arange(1, n_tests + 1)
bh_threshold = ranks / n_tests * 0.05
results_sorted["bh_threshold"] = bh_threshold
results_sorted["bh_sig"] = results_sorted["p_value"] <= bh_threshold

# Harvey-Liu-Zhu threshold (t > 3.0)
results_df["hlz_sig"] = results_df["t_stat"].abs() > 3.0

naive_sig = (results_df["p_value"] < 0.05).sum()
bonf_sig = results_df["bonf_sig"].sum()
bh_sig = results_sorted["bh_sig"].sum()
hlz_sig = results_df["hlz_sig"].sum()

# Check noise factor survival
noise_chars = ["noise_1", "noise_2", "noise_3", "noise_4"]
noise_naive = sum(1 for c in noise_chars
                  if c in results_df.index and results_df.loc[c, "p_value"] < 0.05)
noise_bonf = sum(1 for c in noise_chars
                 if c in results_df.index and results_df.loc[c, "bonf_sig"])

print(f"\nMultiple Testing Correction Summary:")
print(f"  Naive (p < 0.05):         {naive_sig} / {n_tests} significant")
print(f"  Bonferroni (p < 0.005):   {bonf_sig} / {n_tests} significant")
print(f"  Benjamini-Hochberg (FDR): {bh_sig} / {n_tests} significant")
print(f"  HLZ (|t| > 3.0):         {hlz_sig} / {n_tests} significant")
print(f"  Noise factors naive sig:  {noise_naive} / {len(noise_chars)}")
print(f"  Noise factors Bonf sig:   {noise_bonf} / {len(noise_chars)}")


# ── CELL: zoo_plot ──────────────────────────────────────────

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# t-statistics bar chart
ax = axes[0]
chars_sorted = results_df.sort_values("t_stat", ascending=False)
colors = ["#2196F3" if c not in noise_chars else "#FF5252"
          for c in chars_sorted.index]
ax.barh(range(len(chars_sorted)), chars_sorted["t_stat"], color=colors,
        edgecolor="k", linewidth=0.5)
ax.axvline(2.0, color="green", ls="--", lw=1.5, label="t = 2.0")
ax.axvline(-2.0, color="green", ls="--", lw=1.5)
ax.axvline(3.0, color="red", ls="--", lw=1.5, label="t = 3.0 (HLZ)")
ax.axvline(-3.0, color="red", ls="--", lw=1.5)
ax.set_yticks(range(len(chars_sorted)))
ax.set_yticklabels(chars_sorted.index, fontsize=9)
ax.set(title="Fama-MacBeth t-Statistics by Characteristic",
       xlabel="t-statistic")
ax.legend(fontsize=8)

# Survival count comparison
ax = axes[1]
methods = ["Naive\n(t>2.0)", "Bonferroni", "BH\n(FDR=5%)", "HLZ\n(t>3.0)"]
counts = [naive_sig, bonf_sig, bh_sig, hlz_sig]
bars = ax.bar(methods, counts, color=["#66BB6A", "#42A5F5",
              "#FFA726", "#EF5350"], edgecolor="k", linewidth=0.5)
ax.set(title="How Many Factors Survive Correction?",
       ylabel="# Significant Factors")
ax.axhline(n_tests, color="gray", ls=":", alpha=0.5,
           label=f"Total tested: {n_tests}")
for bar, count in zip(bars, counts):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.2,
            str(count), ha="center", fontsize=12, fontweight="bold")
ax.legend()

plt.tight_layout()
plt.show()


if __name__ == "__main__":
    # ── ASSERTIONS ─────────────────────────────────
    # Naive: at least 2 significant
    assert naive_sig >= 2, \
        f"Only {naive_sig} significant naive (expected ≥2)"

    # After correction, fewer survive
    assert bonf_sig <= naive_sig, \
        f"Bonferroni ({bonf_sig}) should ≤ naive ({naive_sig})"

    # Noise factors should not survive Bonferroni
    assert noise_bonf == 0, \
        f"{noise_bonf} noise factors survived Bonferroni (expected 0)"

    # The drop from naive to corrected should be meaningful
    drop_pct = (naive_sig - bonf_sig) / naive_sig if naive_sig > 0 else 0
    assert drop_pct >= 0.20, \
        f"Drop from naive to Bonferroni = {drop_pct:.0%} (expected ≥20%)"

    # ── RESULTS ────────────────────────────────────
    print(f"══ seminar/ex4_factor_zoo ═══════════════════════════")
    print(f"  n_characteristics: {n_tests}")
    print(f"  n_noise: {len(noise_chars)}")
    print(f"  naive_significant: {naive_sig}")
    print(f"  bonferroni_significant: {bonf_sig}")
    print(f"  bh_significant: {bh_sig}")
    print(f"  hlz_significant: {hlz_sig}")
    print(f"  noise_naive_sig: {noise_naive}")
    print(f"  noise_bonf_sig: {noise_bonf}")
    print(f"  drop_naive_to_bonf: {drop_pct:.2f}")
    for char in all_chars:
        if char in results_df.index:
            print(f"  {char}: t={results_df.loc[char, 't_stat']:.2f}, "
                  f"p={results_df.loc[char, 'p_value']:.4f}")

    # ── PLOT ───────────────────────────────────────
    fig.savefig(CACHE_DIR / "ex4_factor_zoo.png",
                dpi=150, bbox_inches="tight")
    print(f"  ── plot: ex4_factor_zoo.png ──")
    print(f"     type: horizontal bar + comparison bar")
    print(f"     n_panels: 2")
    print(f"     title_left: {axes[0].get_title()}")
    print(f"     title_right: {axes[1].get_title()}")
    print(f"✓ ex4_factor_zoo: ALL PASSED")
