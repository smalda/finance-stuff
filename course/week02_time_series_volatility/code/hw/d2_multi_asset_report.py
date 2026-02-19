"""
Deliverable 2: A Multi-Asset Volatility Comparison Report

Acceptance criteria (from README):
- Analysis completed for all 10 tickers
- Comparison table has all specified columns, no missing values
- Panel figure has 10 subplots with matched y-axes
- At least one bond or commodity ticker shows materially different GARCH
  characteristics (lower persistence, weaker leverage) than equity tickers
- All persistence values are in [0, 1) (stationary GARCH)
- Long-run annualized vol is between 5% and 80% for all tickers
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from data_setup import load_prices, HW_TICKERS

# Import the VolatilityAnalyzer from d1
sys.path.insert(0, str(Path(__file__).parent))
from d1_volatility_analyzer import VolatilityAnalyzer


# ── CELL: run_all_analyzers ──────────────────────────────
# Purpose: Run VolatilityAnalyzer on all 10 tickers and compile a
#   comprehensive comparison table with kurtosis, skewness, ARCH effect,
#   best GARCH model, persistence, and annualized long-run volatility.
# Takeaway: EGARCH wins for 6 of 10 tickers. TLT and GLD both select
#   vanilla GARCH (no leverage). Strongest leverage: SPY γ=−0.17, weakest:
#   MSFT γ=−0.07. TSLA LR vol = 57% vs SPY = 15%. Kurtosis: BA = 18.1
#   (fattest) vs TLT = 3.5 (thinnest).

prices_df = load_prices(HW_TICKERS)
analyzers = {}
comp_rows = []

for ticker in HW_TICKERS:
    ret = prices_df[ticker].pct_change().dropna()
    va = VolatilityAnalyzer(ret, name=ticker)
    s = va.summary()
    analyzers[ticker] = va

    sf = s["stylized_facts"]
    garch_res = va.fit_garch_models()
    best_name = garch_res["best_model"]
    best_fit = garch_res["fits"].get(best_name)
    gamma = np.nan
    if best_fit is not None:
        gamma = best_fit.params.get("gamma[1]", np.nan)

    comp_rows.append({
        "Ticker": ticker,
        "Kurtosis": sf["kurtosis"],
        "Skewness": sf["skewness"],
        "ARCH": "Yes" if sf["arch_effect"] else "No",
        "Best Model": best_name,
        "Persistence": s["persistence"],
        "Ann. LR Vol": s["long_run_vol"],
        "Gamma": gamma,
    })

comp_table = pd.DataFrame(comp_rows)
print("=== Multi-Asset Volatility Comparison ===")
print(comp_table.to_string(index=False, float_format=lambda x: f"{x:.4f}"))


# ── CELL: programmatic_summary ───────────────────────────
# Purpose: Generate a programmatic summary identifying which tickers
#   have the highest persistence, strongest leverage, fattest tails,
#   and any notable differences between asset classes.
# Takeaway: Persistence ranking: TSLA (0.991) > XLE (0.986) > TLT
#   (0.981) > GLD (0.981) > BA (0.981) > ... > AAPL (0.937). Leverage:
#   all EGARCH gammas are negative (−0.17 to −0.07), BA is the only
#   GJR-GARCH (γ=+0.07). TLT, GLD, TSLA: no leverage detected by BIC.

sorted_pers = comp_table.sort_values("Persistence", ascending=False)
print("\n=== Persistence Ranking ===")
for _, row in sorted_pers.iterrows():
    print(f"  {row['Ticker']:5s}: {row['Persistence']:.4f}")

# Strongest leverage (most negative gamma for EGARCH, most positive for GJR)
sorted_gamma = comp_table.dropna(subset=["Gamma"]).sort_values("Gamma")
print("\n=== Leverage Effect (gamma) ===")
for _, row in sorted_gamma.iterrows():
    print(f"  {row['Ticker']:5s} ({row['Best Model']:10s}): γ = {row['Gamma']:.4f}")

# Fattest tails
sorted_kurt = comp_table.sort_values("Kurtosis", ascending=False)
print(f"\nFattest tails: {sorted_kurt.iloc[0]['Ticker']} (kurtosis = {sorted_kurt.iloc[0]['Kurtosis']:.1f})")
print(f"Thinnest tails: {sorted_kurt.iloc[-1]['Ticker']} (kurtosis = {sorted_kurt.iloc[-1]['Kurtosis']:.1f})")


# ── CELL: panel_figure ───────────────────────────────────
# Purpose: Produce a 2x5 panel figure showing conditional volatility from
#   the best model for each ticker, with absolute returns overlaid. All
#   panels share the same y-axis for direct cross-asset comparison.
# Visual: 10 panels in 2x5 grid with shared y-axis (0 to ~0.04 daily).
#   TSLA dominates at ~4% daily vol peaks. TLT/GLD compressed near zero.
#   SPY/QQQ/JPM show COVID spikes at ~1-2%. The shared axis makes
#   cross-asset vol hierarchy immediately visible.

fig, axes = plt.subplots(2, 5, figsize=(20, 8), sharex=True, sharey=True)
axes_flat = axes.flatten()

for ax, ticker in zip(axes_flat, HW_TICKERS):
    va = analyzers[ticker]
    cv = va.conditional_volatility()
    abs_ret = va.returns.abs()

    ax.bar(abs_ret.index, abs_ret.values, width=1, color="lightgray", alpha=0.6)
    ax.plot(cv.index, cv.values, linewidth=0.5, color="steelblue")
    best = va.fit_garch_models()["best_model"]
    ax.set_title(f"{ticker}\n({best})", fontsize=9)
    ax.tick_params(axis="both", labelsize=7)

fig.supylabel("Daily Volatility (decimal)", fontsize=11)
fig.suptitle("Conditional Volatility — 10 Assets (matched y-axis)", fontsize=13, y=1.02)
plt.tight_layout()
plt.savefig("d2_multi_asset_panel.png", dpi=120, bbox_inches="tight")
plt.close()


if __name__ == "__main__":
    # ── ASSERTIONS ─────────────────────────────────────
    # All 10 tickers present
    assert len(comp_table) == len(HW_TICKERS), (
        f"Expected {len(HW_TICKERS)} rows, got {len(comp_table)}"
    )

    # No missing values in core columns
    core_cols = ["Kurtosis", "Skewness", "Persistence", "Ann. LR Vol"]
    for col in core_cols:
        assert comp_table[col].notna().all(), f"Missing values in {col}"

    # All persistence in [0, 1)
    for _, row in comp_table.iterrows():
        assert 0 < row["Persistence"] < 1, (
            f"{row['Ticker']} persistence = {row['Persistence']:.4f}"
        )

    # All long-run vol in [5%, 80%]
    for _, row in comp_table.iterrows():
        assert 0.05 < row["Ann. LR Vol"] < 0.80, (
            f"{row['Ticker']} LR vol = {row['Ann. LR Vol']:.2%}"
        )

    # TLT or GLD shows different characteristics
    # Check: TLT or GLD has GARCH as best (no leverage) OR lower persistence
    tlt_row = comp_table[comp_table["Ticker"] == "TLT"].iloc[0]
    gld_row = comp_table[comp_table["Ticker"] == "GLD"].iloc[0]
    spy_row = comp_table[comp_table["Ticker"] == "SPY"].iloc[0]

    # At least one of TLT/GLD differs from equities in some measurable way
    tlt_different = (tlt_row["Best Model"] == "GARCH") or (tlt_row["Persistence"] < spy_row["Persistence"])
    gld_different = (gld_row["Best Model"] == "GARCH") or (gld_row["Persistence"] < spy_row["Persistence"])
    assert tlt_different or gld_different, (
        "Neither TLT nor GLD shows materially different GARCH characteristics"
    )

    print("✓ Deliverable 2: All acceptance criteria passed")
