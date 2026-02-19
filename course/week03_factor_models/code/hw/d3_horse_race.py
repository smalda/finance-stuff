"""Deliverable 3: The Factor Model Horse Race

Compare CAPM, FF3, and FF5 using Fama-MacBeth regressions.
Report which factors are priced, how R-squared changes,
and what residual alpha remains for ML to capture.
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
from linearmodels import FamaMacBeth as LMFamaMacBeth

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
mkt_rf = ff3.loc[common_idx, "Mkt-RF"]


# ── CELL: estimate_betas ────────────────────────────────────

# Rolling 60-month market betas
window = 60
beta_panel = {}

for i in range(window, len(common_idx)):
    date = common_idx[i]
    mkt_window = mkt_rf.iloc[i - window:i]
    betas = {}
    for ticker in excess_returns.columns:
        y = excess_returns[ticker].iloc[i - window:i].dropna()
        if len(y) < 36:
            continue
        x = mkt_window.loc[y.index]
        X = sm.add_constant(x)
        model = sm.OLS(y, X).fit()
        betas[ticker] = model.params["Mkt-RF"]
    beta_panel[date] = betas

beta_df = pd.DataFrame(beta_panel).T
print(f"Rolling beta panel: {beta_df.shape}")


# ── CELL: build_characteristics ─────────────────────────────

# Static characteristics
fund_chars = {}
for ticker in monthly_returns.columns:
    chars = {}

    # Log market cap (size)
    if ticker in mcap.index and mcap[ticker] > 0:
        chars["log_mcap"] = np.log(mcap[ticker])

    # Book-to-market (value)
    if ticker in bs.index.get_level_values("ticker"):
        eq = bs.loc[ticker].sort_index()["Stockholders Equity"].dropna()
        if len(eq) > 0 and eq.iloc[-1] > 0:
            if ticker in mcap.index and mcap[ticker] > 0:
                chars["bm"] = eq.iloc[-1] / mcap[ticker]

    # Operating profitability
    if ticker in inc.index.get_level_values("ticker"):
        tk_inc = inc.loc[ticker].sort_index()
        oi = tk_inc.get("Operating Income", pd.Series(dtype=float)).dropna()
        if len(oi) == 0:
            oi = tk_inc.get("Net Income", pd.Series(dtype=float)).dropna()
        if len(oi) > 0:
            if ticker in bs.index.get_level_values("ticker"):
                eq2 = bs.loc[ticker].sort_index()["Stockholders Equity"].dropna()
                if len(eq2) > 0 and eq2.iloc[-1] > 0:
                    chars["profitability"] = oi.iloc[-1] / eq2.iloc[-1]

    # Investment (asset growth)
    if ticker in bs.index.get_level_values("ticker"):
        assets = bs.loc[ticker].sort_index()["Total Assets"].dropna()
        if len(assets) > 1 and assets.iloc[-2] > 0:
            chars["investment"] = (assets.iloc[-1] / assets.iloc[-2]) - 1

    fund_chars[ticker] = chars

fund_df = pd.DataFrame(fund_chars).T
print(f"Fundamental characteristics: {fund_df.shape}")


# ── CELL: build_fm_panels ──────────────────────────────────

# Build panels for each model specification
panel_records = []

for date in beta_df.index:
    for ticker in monthly_returns.columns:
        if date not in excess_returns.index:
            continue
        ret = excess_returns.loc[date, ticker]
        if pd.isna(ret):
            continue
        beta_val = beta_df.loc[date].get(ticker, np.nan)
        if pd.isna(beta_val):
            continue

        row = {
            "date": date,
            "ticker": ticker,
            "excess_ret": ret,
            "beta": beta_val,
        }
        if ticker in fund_df.index:
            for col in fund_df.columns:
                row[col] = fund_df.loc[ticker, col]
        panel_records.append(row)

full_panel = pd.DataFrame(panel_records)
full_panel = full_panel.set_index(["ticker", "date"]).sort_index()


# ── CELL: standardize_panel ─────────────────────────────────

all_chars = ["beta", "log_mcap", "bm", "profitability", "investment"]

def standardize_month(group):
    """Z-score standardize within each month."""
    for col in all_chars:
        if col in group.columns:
            vals = group[col]
            mean, std = vals.mean(), vals.std()
            if std > 0:
                group[col] = (vals - mean) / std
    return group

panel_std = full_panel.groupby(level="date", group_keys=False).apply(
    standardize_month
)


# ── CELL: run_horse_race ────────────────────────────────────

# Three model specifications
models = {
    "CAPM": ["beta"],
    "FF3": ["beta", "log_mcap", "bm"],
    "FF5": ["beta", "log_mcap", "bm", "profitability", "investment"],
}

fm_results = {}

for model_name, char_list in models.items():
    clean = panel_std.dropna(subset=char_list + ["excess_ret"])
    if len(clean) < 1000:
        print(f"  {model_name}: insufficient data ({len(clean)} obs)")
        continue

    dep = clean[["excess_ret"]]
    indep = clean[char_list]

    fm = LMFamaMacBeth(dep, indep).fit(cov_type="kernel")

    # Compute average cross-sectional R²
    r2_list = []
    dates = clean.index.get_level_values("date").unique()
    for date in dates:
        md = clean.loc[clean.index.get_level_values("date") == date]
        if len(md) < 30:
            continue
        y = md["excess_ret"].values
        X = sm.add_constant(md[char_list].values)
        model_ols = sm.OLS(y, X).fit()
        r2_list.append(model_ols.rsquared)

    avg_r2 = np.mean(r2_list)

    # Residual alpha
    resid_abs = []
    for date in dates:
        md = clean.loc[clean.index.get_level_values("date") == date]
        if len(md) < 30:
            continue
        y = md["excess_ret"].values
        X = sm.add_constant(md[char_list].values)
        model_ols = sm.OLS(y, X).fit()
        resid_abs.extend(np.abs(model_ols.resid))

    avg_abs_resid = np.mean(resid_abs)

    fm_results[model_name] = {
        "params": fm.params,
        "tstats": fm.tstats,
        "pvalues": fm.pvalues,
        "avg_r2": avg_r2,
        "avg_abs_residual": avg_abs_resid,
        "n_obs": len(clean),
        "n_months": len(dates),
    }

    print(f"\n{model_name} ({len(clean)} obs, {len(dates)} months):")
    print(f"  Avg cross-sectional R²: {avg_r2:.4f}")
    print(f"  Avg |residual|: {avg_abs_resid:.6f}")
    for char in char_list:
        print(f"  {char}: gamma={fm.params[char]:.6f}, "
              f"NW t={fm.tstats[char]:.2f}, p={fm.pvalues[char]:.4f}")


# ── CELL: r2_progression ───────────────────────────────────

r2_values = {name: res["avg_r2"] for name, res in fm_results.items()}
residuals = {name: res["avg_abs_residual"] for name, res in fm_results.items()}

print(f"\nR² Progression:")
for name in ["CAPM", "FF3", "FF5"]:
    if name in r2_values:
        print(f"  {name}: R² = {r2_values[name]:.4f}, "
              f"|resid| = {residuals[name]:.6f}")


# ── CELL: horse_race_plot ───────────────────────────────────

fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# R² comparison
ax = axes[0]
model_names = list(fm_results.keys())
r2_vals = [fm_results[m]["avg_r2"] for m in model_names]
bars = ax.bar(model_names, r2_vals, color=["#66BB6A", "#42A5F5", "#EF5350"],
              edgecolor="k", linewidth=0.5)
for bar, val in zip(bars, r2_vals):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.001,
            f"{val:.4f}", ha="center", fontsize=10)
ax.set(title="Average Cross-Sectional R²", ylabel="R²")

# Residual alpha
ax = axes[1]
resid_vals = [fm_results[m]["avg_abs_residual"] * 100 for m in model_names]
bars = ax.bar(model_names, resid_vals, color=["#66BB6A", "#42A5F5", "#EF5350"],
              edgecolor="k", linewidth=0.5)
for bar, val in zip(bars, resid_vals):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05,
            f"{val:.2f}%", ha="center", fontsize=10)
ax.set(title="Average |Residual| (Monthly %)", ylabel="Absolute Residual (%)")

# t-statistics for FF5 factors
ax = axes[2]
if "FF5" in fm_results:
    ff5_chars = ["beta", "log_mcap", "bm", "profitability", "investment"]
    ff5_tstats = [fm_results["FF5"]["tstats"][c] for c in ff5_chars]
    colors = ["#2196F3" if abs(t) >= 2.0 else "#BDBDBD" for t in ff5_tstats]
    ax.barh(range(len(ff5_chars)), ff5_tstats, color=colors,
            edgecolor="k", linewidth=0.5)
    ax.set_yticks(range(len(ff5_chars)))
    ax.set_yticklabels(ff5_chars)
    ax.axvline(2.0, color="red", ls="--", lw=1)
    ax.axvline(-2.0, color="red", ls="--", lw=1)
    ax.set(title="FF5 Fama-MacBeth t-Statistics", xlabel="t-statistic")

plt.suptitle("Factor Model Horse Race: CAPM vs. FF3 vs. FF5", y=1.02)
plt.tight_layout()
plt.show()


if __name__ == "__main__":
    # ── ASSERTIONS ─────────────────────────────────
    # All three models ran
    assert len(fm_results) == 3, \
        f"Expected 3 models, got {len(fm_results)}"

    # R² progression: CAPM ≤ FF3 ≤ FF5
    r2_capm = fm_results["CAPM"]["avg_r2"]
    r2_ff3 = fm_results["FF3"]["avg_r2"]
    r2_ff5 = fm_results["FF5"]["avg_r2"]

    assert r2_capm <= r2_ff3 + 0.01, \
        f"R²(CAPM)={r2_capm:.4f} > R²(FF3)={r2_ff3:.4f}"
    assert r2_ff3 <= r2_ff5 + 0.01, \
        f"R²(FF3)={r2_ff3:.4f} > R²(FF5)={r2_ff5:.4f}"

    # R² ranges
    assert 0.00 <= r2_capm <= 0.15, \
        f"CAPM R² = {r2_capm:.4f}, outside [0.00, 0.15]"
    assert 0.01 <= r2_ff5 <= 0.30, \
        f"FF5 R² = {r2_ff5:.4f}, outside [0.01, 0.30]"

    # Residual alpha should remain substantial
    ff5_resid = fm_results["FF5"]["avg_abs_residual"]
    assert ff5_resid > 0.005, \
        f"FF5 avg |resid| = {ff5_resid:.6f} (expected > 0.005)"

    # At least one FF5 factor significant
    ff5_sig = sum(1 for c in ["beta", "log_mcap", "bm", "profitability",
                               "investment"]
                  if abs(fm_results["FF5"]["tstats"][c]) >= 2.0)
    assert ff5_sig >= 1, "No FF5 factors significant at t=2.0"

    # ── RESULTS ────────────────────────────────────
    print(f"══ hw/d3_horse_race ═════════════════════════════════")
    print(f"  r2_capm: {r2_capm:.4f}")
    print(f"  r2_ff3: {r2_ff3:.4f}")
    print(f"  r2_ff5: {r2_ff5:.4f}")
    print(f"  r2_improvement_capm_to_ff3: {r2_ff3 - r2_capm:.4f}")
    print(f"  r2_improvement_ff3_to_ff5: {r2_ff5 - r2_ff3:.4f}")
    print(f"  avg_abs_residual_capm: {fm_results['CAPM']['avg_abs_residual']:.6f}")
    print(f"  avg_abs_residual_ff3: {fm_results['FF3']['avg_abs_residual']:.6f}")
    print(f"  avg_abs_residual_ff5: {ff5_resid:.6f}")
    print(f"  n_ff5_significant: {ff5_sig}")

    for model_name in ["CAPM", "FF3", "FF5"]:
        chars = models[model_name]
        for c in chars:
            t = fm_results[model_name]["tstats"][c]
            p = fm_results[model_name]["pvalues"][c]
            print(f"  {model_name}_{c}: t={t:.2f}, p={p:.4f}")

    # ── PLOT ───────────────────────────────────────
    fig.savefig(CACHE_DIR / "d3_horse_race.png",
                dpi=150, bbox_inches="tight")
    print(f"  ── plot: d3_horse_race.png ──")
    print(f"     type: triple bar chart comparison")
    print(f"     n_panels: 3")
    print(f"     title: Factor Model Horse Race: CAPM vs. FF3 vs. FF5")
    print(f"✓ d3_horse_race: ALL PASSED")
