"""Deliverable 1: The Factor Factory

A complete factor construction pipeline: download raw data, compute
fundamentals, perform portfolio double-sorts, produce monthly factor
returns, and validate against official Ken French data.
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


# ── CELL: factor_builder_class ──────────────────────────────

class FactorBuilder:
    """Construct Fama-French 5 factors + momentum from raw data.

    Attributes:
        monthly_returns: DataFrame of monthly returns (date x ticker).
        fundamentals: Dict of fundamental DataFrames from data_setup.
        factor_returns: DataFrame of monthly self-built factor returns.
        quality_report: Dict summarizing data quality issues.
    """

    def __init__(self, monthly_returns, prices, fundamentals):
        self.monthly_returns = monthly_returns
        self.prices = prices
        self.fundamentals = fundamentals
        self.factor_returns = None
        self.quality_report = {
            "missing_equity": [],
            "missing_income": [],
            "net_income_fallback": [],
            "portfolio_counts": {},
        }

    def compute_characteristics(self):
        """Compute book-to-market, profitability, and investment."""
        bs = self.fundamentals["balance_sheet"]
        inc = self.fundamentals["income_stmt"]
        mcap = self.fundamentals["market_cap"]

        records = []
        for ticker in self.monthly_returns.columns:
            row = {"ticker": ticker}

            # Book equity
            equity = np.nan
            if ticker in bs.index.get_level_values("ticker"):
                eq_vals = bs.loc[ticker].sort_index()["Stockholders Equity"].dropna()
                if len(eq_vals) > 0 and eq_vals.iloc[-1] > 0:
                    equity = eq_vals.iloc[-1]
                else:
                    self.quality_report["missing_equity"].append(ticker)
            else:
                self.quality_report["missing_equity"].append(ticker)

            # Market cap and B/M
            mk = mcap.get(ticker, np.nan)
            if not np.isnan(equity) and not np.isnan(mk) and mk > 0:
                row["book_to_market"] = equity / mk
                row["market_cap"] = mk

            # Operating profitability (OP / equity)
            if ticker in inc.index.get_level_values("ticker"):
                tk_inc = inc.loc[ticker].sort_index()
                oi = tk_inc.get("Operating Income", pd.Series(dtype=float)).dropna()
                if len(oi) == 0:
                    ni = tk_inc.get("Net Income", pd.Series(dtype=float)).dropna()
                    if len(ni) > 0:
                        oi = ni
                        self.quality_report["net_income_fallback"].append(ticker)
                if len(oi) > 0 and not np.isnan(equity) and equity > 0:
                    row["profitability"] = oi.iloc[-1] / equity
            else:
                self.quality_report["missing_income"].append(ticker)

            # Investment (asset growth)
            if ticker in bs.index.get_level_values("ticker"):
                assets = bs.loc[ticker].sort_index()["Total Assets"].dropna()
                if len(assets) > 1 and assets.iloc[-2] > 0:
                    row["investment"] = (assets.iloc[-1] / assets.iloc[-2]) - 1

            records.append(row)

        self.chars = pd.DataFrame(records).set_index("ticker")
        return self.chars

    def _double_sort(self, char_col, n_groups=3):
        """Perform 2 x n_groups sort on size and a characteristic.

        Returns dict mapping portfolio name to list of tickers.
        """
        valid = self.chars.dropna(subset=["market_cap", char_col]).copy()

        # Size split: median
        size_median = valid["market_cap"].median()
        valid["size_group"] = np.where(
            valid["market_cap"] < size_median, "S", "B"
        )

        # Characteristic split
        breakpoints = [valid[char_col].quantile(q)
                       for q in np.linspace(0, 1, n_groups + 1)]

        def assign_char_group(val):
            for i in range(n_groups):
                if val <= breakpoints[i + 1]:
                    return i
            return n_groups - 1

        valid["char_group"] = valid[char_col].apply(assign_char_group)

        portfolios = {}
        for size in ["S", "B"]:
            for cg in range(n_groups):
                mask = (valid["size_group"] == size) & (valid["char_group"] == cg)
                pf_tickers = valid[mask].index.tolist()
                pf_name = f"{size}/{cg}"
                portfolios[pf_name] = pf_tickers

        return portfolios, valid

    def _compute_vw_returns(self, portfolios, valid_chars):
        """Compute value-weighted monthly returns for each portfolio."""
        pf_returns = {}

        for date in self.monthly_returns.index:
            rets = self.monthly_returns.loc[date].dropna()
            date_rets = {}

            for pf_name, tickers in portfolios.items():
                avail = [t for t in tickers if t in rets.index]
                if len(avail) == 0:
                    date_rets[pf_name] = np.nan
                    continue
                weights = valid_chars.loc[avail, "market_cap"]
                weights = weights / weights.sum()
                date_rets[pf_name] = (rets[avail] * weights).sum()

            pf_returns[date] = date_rets

        return pd.DataFrame(pf_returns).T

    def _build_long_short(self, pf_returns, n_groups=3):
        """Build SMB and characteristic-based long-short factor."""
        # Factor = avg(S portfolios) - avg(B portfolios)
        s_cols = [c for c in pf_returns.columns if c.startswith("S/")]
        b_cols = [c for c in pf_returns.columns if c.startswith("B/")]

        smb = pf_returns[s_cols].mean(axis=1) - pf_returns[b_cols].mean(axis=1)

        # Characteristic factor: high - low
        high = n_groups - 1
        high_cols = [f"S/{high}", f"B/{high}"]
        low_cols = ["S/0", "B/0"]
        char_factor = pf_returns[high_cols].mean(axis=1) - \
                      pf_returns[low_cols].mean(axis=1)

        return smb, char_factor

    def build_all_factors(self):
        """Build all 6 factors: Mkt-RF, SMB, HML, RMW, CMA, MOM."""
        self.compute_characteristics()

        factor_series = {}

        # Market factor (value-weighted universe excess return)
        ff3 = load_factor_data("3")
        common = self.monthly_returns.index.intersection(ff3.index)
        rf = ff3.loc[common, "RF"]

        # VW market return approximation
        valid_mcap = self.chars["market_cap"].dropna()
        weights = valid_mcap / valid_mcap.sum()
        mkt_tickers = valid_mcap.index.tolist()
        mkt_ret = (self.monthly_returns[mkt_tickers] * weights).sum(axis=1)
        factor_series["Mkt-RF"] = mkt_ret.loc[common] - rf

        # HML (book-to-market: high = value, low = growth)
        pf_bm, valid_bm = self._double_sort("book_to_market", n_groups=3)
        pf_bm_rets = self._compute_vw_returns(pf_bm, valid_bm)
        smb_bm, hml = self._build_long_short(pf_bm_rets)

        self.quality_report["portfolio_counts"]["HML"] = {
            k: len(v) for k, v in pf_bm.items()
        }

        # RMW (profitability: high = robust, low = weak)
        if "profitability" in self.chars.columns:
            valid_prof = self.chars.dropna(subset=["market_cap", "profitability"])
            if len(valid_prof) >= 30:
                pf_prof, valid_p = self._double_sort("profitability", n_groups=3)
                pf_prof_rets = self._compute_vw_returns(pf_prof, valid_p)
                smb_prof, rmw = self._build_long_short(pf_prof_rets)
                self.quality_report["portfolio_counts"]["RMW"] = {
                    k: len(v) for k, v in pf_prof.items()
                }
            else:
                rmw = pd.Series(0, index=self.monthly_returns.index)
        else:
            rmw = pd.Series(0, index=self.monthly_returns.index)

        # CMA (investment: low growth = conservative, high growth = aggressive)
        # Note: CMA = Conservative - Aggressive, so we INVERT: low - high
        if "investment" in self.chars.columns:
            valid_inv = self.chars.dropna(subset=["market_cap", "investment"])
            if len(valid_inv) >= 30:
                pf_inv, valid_i = self._double_sort("investment", n_groups=3)
                pf_inv_rets = self._compute_vw_returns(pf_inv, valid_i)
                smb_inv, inv_factor = self._build_long_short(pf_inv_rets)
                cma = -inv_factor  # Invert: conservative minus aggressive
                self.quality_report["portfolio_counts"]["CMA"] = {
                    k: len(v) for k, v in pf_inv.items()
                }
            else:
                cma = pd.Series(0, index=self.monthly_returns.index)
        else:
            cma = pd.Series(0, index=self.monthly_returns.index)

        # SMB: average of SMB from all three sorts
        smb_components = [smb_bm]
        if "profitability" in self.chars.columns:
            smb_components.append(smb_prof)
        if "investment" in self.chars.columns:
            smb_components.append(smb_inv)
        smb = pd.concat(smb_components, axis=1).mean(axis=1)

        # Momentum (12-1 month return)
        mom_monthly = []
        for date in self.monthly_returns.index:
            mom_end = date - pd.DateOffset(months=1)
            mom_start = date - pd.DateOffset(months=12)
            mask = (self.prices.index >= mom_start) & \
                   (self.prices.index <= mom_end)
            if mask.sum() < 20:
                continue
            mom_prices = self.prices.loc[mask]
            if len(mom_prices) < 2:
                continue
            momentum = (mom_prices.iloc[-1] / mom_prices.iloc[0]) - 1
            momentum = momentum.dropna()

            rets = self.monthly_returns.loc[date].dropna()
            common_t = momentum.index.intersection(rets.index)
            if len(common_t) < 30:
                continue

            mom_sorted = momentum[common_t].sort_values()
            n = len(mom_sorted)
            losers = mom_sorted.iloc[:int(n * 0.3)].index
            winners = mom_sorted.iloc[int(n * 0.7):].index

            # Value-weighted
            if "market_cap" in self.chars.columns:
                w_win = self.chars.loc[
                    self.chars.index.isin(winners), "market_cap"
                ].dropna()
                w_los = self.chars.loc[
                    self.chars.index.isin(losers), "market_cap"
                ].dropna()
                if len(w_win) > 0 and len(w_los) > 0:
                    w_win = w_win / w_win.sum()
                    w_los = w_los / w_los.sum()
                    win_ret = (rets[w_win.index] * w_win).sum()
                    los_ret = (rets[w_los.index] * w_los).sum()
                else:
                    win_ret = rets[winners].mean()
                    los_ret = rets[losers].mean()
            else:
                win_ret = rets[winners].mean()
                los_ret = rets[losers].mean()

            mom_monthly.append({"date": date, "MOM": win_ret - los_ret})

        mom_df = pd.DataFrame(mom_monthly).set_index("date")["MOM"]

        # Combine all factors
        self.factor_returns = pd.DataFrame({
            "Mkt-RF": factor_series["Mkt-RF"],
            "SMB": smb,
            "HML": hml,
            "RMW": rmw,
            "CMA": cma,
            "MOM": mom_df,
        })

        return self.factor_returns

    def validate(self, official_ff5, official_carhart):
        """Compare self-built factors against official data."""
        results = {}
        factor_map = {
            "SMB": ("SMB", official_ff5),
            "HML": ("HML", official_ff5),
            "RMW": ("RMW", official_ff5),
            "CMA": ("CMA", official_ff5),
            "MOM": ("MOM", official_carhart),
        }

        for self_name, (off_name, off_df) in factor_map.items():
            if self_name not in self.factor_returns.columns:
                continue
            if off_name not in off_df.columns:
                continue

            self_s = self.factor_returns[self_name].dropna()
            common = self_s.index.intersection(off_df.index)
            if len(common) < 12:
                continue

            corr = self_s.loc[common].corr(off_df.loc[common, off_name])
            te = (self_s.loc[common] - off_df.loc[common, off_name]).std() \
                 * np.sqrt(12)

            results[self_name] = {
                "correlation": corr,
                "tracking_error_ann": te,
                "n_months": len(common),
            }

        return results


# ── CELL: run_factor_builder ────────────────────────────────

prices = load_equity_prices()
monthly_returns = load_monthly_returns()
fundamentals = load_fundamentals()

builder = FactorBuilder(monthly_returns, prices, fundamentals)
factor_returns = builder.build_all_factors()

print(f"Factor returns shape: {factor_returns.shape}")
print(f"\nFactor return statistics (monthly):")
print(factor_returns.describe().round(6).to_string())


# ── CELL: validate_factors ──────────────────────────────────

ff5_official = load_factor_data("5")
carhart_official = load_carhart_factors()
validation = builder.validate(ff5_official, carhart_official)

print("\nValidation Against Official Data:")
print(f"{'Factor':<8} {'Corr':>8} {'TE (ann)':>10} {'N months':>10}")
print("-" * 40)
for name, vals in validation.items():
    print(f"{name:<8} {vals['correlation']:>8.4f} "
          f"{vals['tracking_error_ann']:>10.4f} {vals['n_months']:>10}")


# ── CELL: quality_report ────────────────────────────────────

qr = builder.quality_report
print(f"\nQuality Report:")
print(f"  Missing equity: {len(qr['missing_equity'])} tickers")
print(f"  Missing income: {len(qr['missing_income'])} tickers")
print(f"  Net Income fallback: {len(qr['net_income_fallback'])} tickers")
print(f"  Portfolio counts:")
for factor_name, counts in qr["portfolio_counts"].items():
    print(f"    {factor_name}: {counts}")


# ── CELL: factor_cumulative_plot ────────────────────────────

fig, ax = plt.subplots(figsize=(12, 6))
for col in factor_returns.columns:
    cum = (1 + factor_returns[col].dropna()).cumprod()
    ax.plot(cum.index, cum, label=col, lw=1.5)
ax.set(title="Self-Built Factor Cumulative Returns",
       xlabel="Date", ylabel="Cumulative Return")
ax.legend(ncol=3)
ax.grid(True, alpha=0.3)
ax.axhline(1.0, color="gray", lw=0.5, ls="--")
plt.tight_layout()
plt.show()


if __name__ == "__main__":
    # ── ASSERTIONS ─────────────────────────────────
    assert factor_returns.shape[1] == 6, \
        f"Expected 6 factor columns, got {factor_returns.shape[1]}"

    # MOM should have the most months
    mom_n = factor_returns["MOM"].dropna().shape[0]
    assert mom_n >= 100, f"MOM has only {mom_n} months (expected ≥100)"

    # Fundamental factors have fewer months
    for f in ["SMB", "HML"]:
        n = factor_returns[f].dropna().shape[0]
        assert n >= 50, f"{f} has only {n} months (expected ≥50)"

    # Validation correlations
    assert 0.05 <= validation["SMB"]["correlation"] <= 0.65, \
        f"SMB corr = {validation['SMB']['correlation']:.4f}"
    assert 0.05 <= validation["HML"]["correlation"] <= 0.90, \
        f"HML corr = {validation['HML']['correlation']:.4f}"
    assert 0.40 <= validation["MOM"]["correlation"] <= 0.95, \
        f"MOM corr = {validation['MOM']['correlation']:.4f}"

    # MOM should replicate better than SMB
    assert validation["MOM"]["correlation"] > validation["SMB"]["correlation"], \
        "MOM should replicate better than SMB"

    # Portfolio counts: no empty portfolios
    for factor_name, counts in qr["portfolio_counts"].items():
        for pf, count in counts.items():
            assert count >= 10, \
                f"{factor_name} portfolio {pf} has only {count} stocks"

    # Graceful handling: at least 150 tickers processed
    n_chars = builder.chars.dropna(subset=["market_cap"]).shape[0]
    assert n_chars >= 150, \
        f"Only {n_chars} tickers with market cap (expected ≥150)"

    # ── RESULTS ────────────────────────────────────
    print(f"══ hw/d1_factor_factory ═════════════════════════════")
    print(f"  factor_returns_shape: {factor_returns.shape}")
    print(f"  n_tickers_processed: {n_chars}")
    print(f"  mom_n_months: {mom_n}")
    for name, vals in validation.items():
        print(f"  {name}_corr: {vals['correlation']:.4f}")
        print(f"  {name}_te: {vals['tracking_error_ann']:.4f}")
    print(f"  missing_equity: {len(qr['missing_equity'])}")
    print(f"  net_income_fallback: {len(qr['net_income_fallback'])}")

    # ── PLOT ───────────────────────────────────────
    fig.savefig(CACHE_DIR / "d1_factor_cumulative.png",
                dpi=150, bbox_inches="tight")
    print(f"  ── plot: d1_factor_cumulative.png ──")
    print(f"     type: multi-line cumulative returns")
    print(f"     n_lines: {len(ax.get_lines())}")
    print(f"     y_range: [{ax.get_ylim()[0]:.2f}, {ax.get_ylim()[1]:.2f}]")
    print(f"     title: {ax.get_title()}")
    print(f"✓ d1_factor_factory: ALL PASSED")
