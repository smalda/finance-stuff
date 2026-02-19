"""Exercise 1: Can You Replicate Fama-French?

Construct SMB, HML, and momentum factors from raw stock data,
then compare against official Ken French returns.
"""
import matplotlib
matplotlib.use("Agg")
import sys
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from data_setup import (
    load_equity_prices, load_monthly_returns, load_fundamentals,
    load_factor_data, load_carhart_factors, CACHE_DIR,
)

warnings.filterwarnings("ignore", category=FutureWarning)

prices = load_equity_prices()
monthly_returns = load_monthly_returns()
fundamentals = load_fundamentals()
ff3_official = load_factor_data("3")
carhart = load_carhart_factors()

bs = fundamentals["balance_sheet"]
mcap = fundamentals["market_cap"]


# ── CELL: compute_book_to_market ────────────────────────────

book_equity = bs["Stockholders Equity"].dropna()
book_equity = book_equity[book_equity > 0]

bm_records = []
for ticker in book_equity.index.get_level_values("ticker").unique():
    be = book_equity.loc[ticker].sort_index().iloc[-1]
    if ticker in mcap.index and mcap[ticker] > 0:
        bm_records.append({
            "ticker": ticker,
            "book_equity": be,
            "market_cap": mcap[ticker],
            "book_to_market": be / mcap[ticker],
        })

bm_df = pd.DataFrame(bm_records).set_index("ticker")
valid_tickers = set(bm_df.index) & set(monthly_returns.columns)
bm_df = bm_df.loc[bm_df.index.isin(valid_tickers)]
print(f"Tickers with B/M: {len(bm_df)}")


# ── CELL: double_sort_smb_hml ───────────────────────────────

size_median = bm_df["market_cap"].median()
bm_30 = bm_df["book_to_market"].quantile(0.30)
bm_70 = bm_df["book_to_market"].quantile(0.70)

def assign_portfolio(row):
    """Assign to 2x3 portfolio."""
    size = "S" if row["market_cap"] < size_median else "B"
    if row["book_to_market"] <= bm_30:
        value = "L"
    elif row["book_to_market"] >= bm_70:
        value = "H"
    else:
        value = "M"
    return f"{size}/{value}"

bm_df["portfolio"] = bm_df.apply(assign_portfolio, axis=1)

smb_monthly = []
hml_monthly = []

for date in monthly_returns.index:
    rets = monthly_returns.loc[date].dropna()
    available = set(rets.index) & set(bm_df.index)
    if len(available) < 30:
        continue

    pf_rets = {}
    for pf in ["S/L", "S/M", "S/H", "B/L", "B/M", "B/H"]:
        tickers = bm_df[bm_df["portfolio"] == pf].index
        t_avail = [t for t in tickers if t in available]
        if len(t_avail) == 0:
            pf_rets[pf] = np.nan
            continue
        w = bm_df.loc[t_avail, "market_cap"]
        w = w / w.sum()
        pf_rets[pf] = (rets[t_avail] * w).sum()

    if any(np.isnan(v) for v in pf_rets.values()):
        continue

    smb = (pf_rets["S/L"] + pf_rets["S/M"] + pf_rets["S/H"]) / 3 \
        - (pf_rets["B/L"] + pf_rets["B/M"] + pf_rets["B/H"]) / 3
    hml = (pf_rets["S/H"] + pf_rets["B/H"]) / 2 \
        - (pf_rets["S/L"] + pf_rets["B/L"]) / 2

    smb_monthly.append({"date": date, "SMB_self": smb})
    hml_monthly.append({"date": date, "HML_self": hml})

smb_self = pd.DataFrame(smb_monthly).set_index("date")["SMB_self"]
hml_self = pd.DataFrame(hml_monthly).set_index("date")["HML_self"]


# ── CELL: construct_momentum ────────────────────────────────

# Momentum: 12-1 month return, top 30% vs bottom 30%
mom_monthly = []

for date in monthly_returns.index:
    mom_end = date - pd.DateOffset(months=1)
    mom_start = date - pd.DateOffset(months=12)
    mask = (prices.index >= mom_start) & (prices.index <= mom_end)
    if mask.sum() < 20:
        continue
    mom_prices = prices.loc[mask]
    if len(mom_prices) < 2:
        continue
    momentum = (mom_prices.iloc[-1] / mom_prices.iloc[0]) - 1
    momentum = momentum.dropna()

    rets = monthly_returns.loc[date].dropna()
    common = momentum.index.intersection(rets.index)
    if len(common) < 30:
        continue

    mom_sorted = momentum[common].sort_values()
    n = len(mom_sorted)
    losers = mom_sorted.iloc[:int(n * 0.3)].index
    winners = mom_sorted.iloc[int(n * 0.7):].index

    # Equal-weighted for simplicity
    mom_ret = rets[winners].mean() - rets[losers].mean()
    mom_monthly.append({"date": date, "MOM_self": mom_ret})

mom_self = pd.DataFrame(mom_monthly).set_index("date")["MOM_self"]


# ── CELL: validate_all_factors ──────────────────────────────

# Align and compare
results = {}

# SMB
common_smb = smb_self.index.intersection(ff3_official.index)
smb_corr = smb_self.loc[common_smb].corr(ff3_official.loc[common_smb, "SMB"])
smb_te = (smb_self.loc[common_smb] - ff3_official.loc[common_smb, "SMB"]).std() * np.sqrt(12)
results["SMB"] = {"corr": smb_corr, "te": smb_te, "n": len(common_smb)}

# HML
common_hml = hml_self.index.intersection(ff3_official.index)
hml_corr = hml_self.loc[common_hml].corr(ff3_official.loc[common_hml, "HML"])
hml_te = (hml_self.loc[common_hml] - ff3_official.loc[common_hml, "HML"]).std() * np.sqrt(12)
results["HML"] = {"corr": hml_corr, "te": hml_te, "n": len(common_hml)}

# Momentum
common_mom = mom_self.index.intersection(carhart.index)
mom_corr = mom_self.loc[common_mom].corr(carhart.loc[common_mom, "MOM"])
mom_te = (mom_self.loc[common_mom] - carhart.loc[common_mom, "MOM"]).std() * np.sqrt(12)
results["MOM"] = {"corr": mom_corr, "te": mom_te, "n": len(common_mom)}

print("\nFactor Replication Results:")
print(f"{'Factor':<8} {'Corr':>8} {'TE (ann)':>10} {'N months':>10}")
print("-" * 40)
for name, vals in results.items():
    print(f"{name:<8} {vals['corr']:>8.4f} {vals['te']:>10.4f} {vals['n']:>10}")


# ── CELL: comparison_plot ───────────────────────────────────

fig, axes = plt.subplots(1, 3, figsize=(16, 4))

for ax, (name, self_s, off_s, off_col) in zip(axes, [
    ("SMB", smb_self, ff3_official, "SMB"),
    ("HML", hml_self, ff3_official, "HML"),
    ("MOM", mom_self, carhart, "MOM"),
]):
    common = self_s.index.intersection(off_s.index)
    cum_self = (1 + self_s.loc[common]).cumprod()
    cum_off = (1 + off_s.loc[common, off_col]).cumprod()
    ax.plot(cum_self.index, cum_self, label="Self-built", lw=1.5)
    ax.plot(cum_off.index, cum_off, label="Ken French", lw=1.5, ls="--")
    r = self_s.loc[common].corr(off_s.loc[common, off_col])
    ax.set_title(f"{name} (r={r:.2f})")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

plt.suptitle("Self-Built vs. Official Factor Returns", y=1.02)
plt.tight_layout()
plt.show()


if __name__ == "__main__":
    # ── ASSERTIONS ─────────────────────────────────
    assert 0.05 <= smb_corr <= 0.60, \
        f"SMB corr = {smb_corr:.4f}, outside [0.05, 0.60]"
    assert 0.05 <= hml_corr <= 0.90, \
        f"HML corr = {hml_corr:.4f}, outside [0.05, 0.90]"
    assert 0.40 <= mom_corr <= 0.90, \
        f"MOM corr = {mom_corr:.4f}, outside [0.40, 0.90]"

    # Momentum should replicate better than SMB
    assert mom_corr > smb_corr, \
        f"MOM corr ({mom_corr:.4f}) should exceed SMB corr ({smb_corr:.4f})"

    # Tracking errors
    assert 0.03 <= smb_te <= 0.30, \
        f"SMB TE = {smb_te:.4f}, outside [0.03, 0.30]"
    assert 0.03 <= hml_te <= 0.30, \
        f"HML TE = {hml_te:.4f}, outside [0.03, 0.30]"
    assert 0.01 <= mom_te <= 0.20, \
        f"MOM TE = {mom_te:.4f}, outside [0.01, 0.20]"

    # ── RESULTS ────────────────────────────────────
    print(f"══ seminar/ex1_replicate_ff ═════════════════════════")
    print(f"  smb_corr: {smb_corr:.4f}")
    print(f"  hml_corr: {hml_corr:.4f}")
    print(f"  mom_corr: {mom_corr:.4f}")
    print(f"  smb_te_ann: {smb_te:.4f}")
    print(f"  hml_te_ann: {hml_te:.4f}")
    print(f"  mom_te_ann: {mom_te:.4f}")
    print(f"  n_months_smb: {results['SMB']['n']}")
    print(f"  n_months_mom: {results['MOM']['n']}")

    # ── PLOT ───────────────────────────────────────
    fig.savefig(CACHE_DIR / "ex1_factor_replication.png",
                dpi=150, bbox_inches="tight")
    print(f"  ── plot: ex1_factor_replication.png ──")
    print(f"     type: triple cumulative return comparison")
    print(f"     n_panels: 3")
    for i, name in enumerate(["SMB", "HML", "MOM"]):
        print(f"     title_{name}: {axes[i].get_title()}")
    print(f"✓ ex1_replicate_ff: ALL PASSED")
