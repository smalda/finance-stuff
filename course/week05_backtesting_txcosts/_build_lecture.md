# Week 5: Backtesting, Research Discipline & Transaction Costs

> *"A researcher testing 45 independent trading strategies on the same dataset has a greater than 50% probability of finding one that looks spectacular purely by chance."*
> -- Bailey, Borwein, Lopez de Prado & Zhu (2014), Notices of the American Mathematical Society

~~~python
import sys
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import spearmanr
from sklearn.model_selection import TimeSeriesSplit
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.multitest import multipletests

sys.path.insert(0, "code")
from data_setup import (
    CACHE_DIR, PLOT_DIR, START, END,
    load_equity_data, load_monthly_returns, load_alpha_output,
    load_ls_portfolio, load_ohlcv_data, load_mcap_tiers,
)

sys.path.insert(0, str(Path("code").resolve().parents[1]))
from shared.temporal import CombinatorialPurgedCV, PurgedWalkForwardCV
from shared.metrics import deflated_sharpe_ratio, ic_summary, rank_ic
from shared.backtesting import (
    cumulative_returns, max_drawdown, net_returns,
    sharpe_ratio,
)

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

plt.rcParams.update({
    "figure.figsize": (10, 5),
    "axes.grid": True,
    "grid.alpha": 0.3,
    "font.size": 11,
})
~~~

## The Opening Hook: When Great Backtests Go Wrong

In 2014, four mathematicians published a paper in the *Notices of the American Mathematical Society* with an extraordinary title: "Pseudo-Mathematics and Financial Charlatanism." Their argument was not about fraud. It was about honest researchers doing honest work on finite, noisy data and producing results that look spectacular by construction. Suppose you test 45 independent trading strategies on the same historical dataset. Each strategy is a genuine idea. You evaluate each one fairly. The probability that at least one strategy shows a Sharpe ratio above 1.0, purely by chance, exceeds 50%.

Now consider what this means in practice. You are an ML engineer who just arrived at a quantitative fund. You have built a gradient boosting model that predicts stock returns, run it through walk-forward cross-validation, computed a Sharpe ratio of 1.4, and prepared your presentation for the portfolio manager. The PM's first question: "What's the deflated Sharpe? Did you purge? How many variants did you try before this one? What's the net after realistic transaction costs?" You have no idea what any of that means. This week fixes that.

We are going to dismantle your backtest, piece by piece. First, we will catalog the seven canonical ways backtests fail. Then we will fix cross-validation for financial data. We will quantify the multiple-testing penalty. We will decompose transaction costs. And at the end, we will put it all together into a single responsible evaluation that reveals the truth about a strategy's viability.

---

## Section 1: The Seven Sins -- A Taxonomy of Backtest Failure

A Deutsche Bank quant team once catalogued the most common ways systematic strategies fail after a promising backtest. Every failure mode is a form of optimism that the data did not earn: look-ahead bias, survivorship bias, data snooping, ignoring transaction costs, ignoring short-selling constraints, improper benchmark comparison, and regime neglect. We will tackle the first two with code right now.

Let us start with look-ahead bias -- the equivalent of accidentally including the label as a feature. In finance, this is insidious because the label itself is constructed from future prices. The flawed signal below uses *this month's* return to "predict" this month's return. The corrected signal shifts by one month. This one-line difference is the entire distinction between a look-ahead bug and a legitimate signal.

~~~python
monthly_prices  = load_equity_data()
monthly_returns_raw = monthly_prices.pct_change()
monthly_returns = load_monthly_returns()

n_stocks  = monthly_returns.shape[1]
n_periods = monthly_returns.shape[0]
missing_pct = monthly_prices.isnull().mean().mean()
first_month_prices = monthly_prices.iloc[0]
n_survivors_strict = int(first_month_prices.notna().sum())
n_all              = n_stocks

print("── Data Quality ──────────────────────────────────────────────────")
print(f"  N_stocks:          {n_stocks}")
print(f"  N_periods:         {n_periods} monthly observations")
print(f"  Missing (prices):  {missing_pct:.4%}")
print(f"  Present at start:  {n_survivors_strict} / {n_all} tickers")
print(f"  Survivorship note: S&P 500 universe = current constituents only.")
print(f"                     Delisted stocks not in dataset — bias understated.")
print()

flawed_signal = monthly_returns.rank(axis=1, pct=True)
corrected_signal = monthly_returns.shift(1).rank(axis=1, pct=True)
~~~

We are working with the current S&P 500 universe -- roughly 450 stocks with monthly data from 2012 to 2025. Every ticker is a company that survived to today; the ones that went bankrupt or were acquired are silently absent. The **information coefficient** (IC) -- Spearman rank correlation between the predicted and actual ranking of forward returns -- measures signal quality. An IC of 0.05 sounds pathetic in ML terms, but in a universe of 450 stocks rebalanced monthly, it is enough to build a career on.

~~~python
def monthly_ic(signal: pd.DataFrame, returns: pd.DataFrame) -> pd.Series:
    """Cross-sectional Spearman IC for each date in signal's index."""
    ic_vals = {}
    for date in signal.index:
        sig = signal.loc[date].dropna()
        ret = returns.loc[date].dropna()
        common = sig.index.intersection(ret.index)
        if len(common) < 10:
            continue
        corr, _ = spearmanr(sig[common], ret[common])
        ic_vals[date] = corr
    return pd.Series(ic_vals, name="ic")

IS_START  = "2012-01-01"
IS_END    = "2017-12-31"
OOS_START = "2018-01-01"
OOS_END   = "2024-12-31"

oos_mask = (monthly_returns.index >= OOS_START) & (monthly_returns.index <= OOS_END)
oos_returns = monthly_returns.loc[oos_mask]
flawed_ic_oos    = monthly_ic(flawed_signal.loc[oos_mask],    oos_returns)
corrected_ic_oos = monthly_ic(corrected_signal.loc[oos_mask], oos_returns)
mean_flawed_ic_oos    = flawed_ic_oos.mean()
mean_corrected_ic_oos = corrected_ic_oos.mean()

print(f"  Flawed signal OOS IC:    {mean_flawed_ic_oos:.4f}  ← signal == outcome")
print(f"  Corrected signal OOS IC: {mean_corrected_ic_oos:.4f}")
~~~

IC of 1.0 for the flawed signal -- perfect foresight, because the signal literally *is* the forward return. The corrected signal shows an IC near zero. This is not a failure of the corrected signal. This is reality. But IC is abstract. What does this look like in dollar terms? Let us build a long-short portfolio -- buy the top 20 stocks by signal, short the bottom 20 -- and see the equity curves diverge.

~~~python
def long_short_monthly(
    signal: pd.DataFrame,
    returns: pd.DataFrame,
    n_leg: int = 20,
) -> pd.Series:
    """Equal-weight top-n long / bottom-n short monthly return."""
    port_rets = []
    for date in signal.index:
        sig = signal.loc[date].dropna()
        ret = returns.loc[date].dropna()
        common = sig.index.intersection(ret.index)
        if len(common) < n_leg * 2:
            continue
        ranked = sig[common].rank(ascending=True)
        n = len(ranked)
        long_ret  = ret[common][ranked > (n - n_leg)].mean()
        short_ret = ret[common][ranked <= n_leg].mean()
        port_rets.append({"date": date, "ret": long_ret - short_ret})
    return pd.DataFrame(port_rets).set_index("date")["ret"]

flawed_ret_oos    = long_short_monthly(flawed_signal.loc[oos_mask],    oos_returns)
corrected_ret_oos = long_short_monthly(corrected_signal.loc[oos_mask], oos_returns)
ann_flawed    = flawed_ret_oos.mean()    * 12
ann_corrected = corrected_ret_oos.mean() * 12
annual_gap    = ann_flawed - ann_corrected
cum_flawed    = (1 + flawed_ret_oos).cumprod()
cum_corrected = (1 + corrected_ret_oos).cumprod()

print(f"  Flawed ann. return:    {ann_flawed:.2%}")
print(f"  Corrected ann. return: {ann_corrected:.2%}")
print(f"  Annual return gap:     {annual_gap:.2%}")
~~~

An annual return gap of 400 percentage points. In production, look-ahead bugs are rarely this extreme -- realistic bugs cause inflation of 20-50% annualized -- but the mechanism is identical. Let us see this on a log-scale equity curve.

~~~python
fig_eq, ax_eq = plt.subplots(figsize=(10, 5))
ax_eq.semilogy(cum_flawed.index,    cum_flawed.values,    label="Flawed (look-ahead)", linewidth=2)
ax_eq.semilogy(cum_corrected.index, cum_corrected.values, label="Corrected (lagged signal)", linewidth=2)
ax_eq.set(
    title="Equity Curve: Flawed vs. Corrected Signal (OOS 2018–2024, log scale)",
    xlabel="Date",
    ylabel="Cumulative Return (log scale)",
)
ax_eq.legend()
ax_eq.axhline(1.0, color="gray", linestyle="--", linewidth=0.8)
plt.tight_layout()
plt.show()
~~~

On the log scale, the flawed signal rockets upward -- turning \$1 into billions by construction. The corrected signal meanders around 1.0. If you showed the flawed equity curve to a PM without context, they would either offer you a job or call security. This is why the first item on every backtesting pre-flight checklist is: *verify that no future information leaks into the signal.*

Now the second sin: survivorship bias. Every stock in our dataset exists today. Companies that went bankrupt -- Enron, WorldCom, Lehman Brothers -- are simply absent. Since our dataset only contains survivors, we simulate the bias: each year, 5% of stocks "delist" with a -50% terminal return. The difference between the survivor-only and simulated unbiased portfolios is the survivorship premium.

~~~python
oos_raw = monthly_returns_raw.loc[OOS_START:OOS_END]
ew_survivors = oos_raw.mean(axis=1)
ann_survivors = ew_survivors.mean() * 12
SEED = 42
ANNUAL_CHURN_RATE = 0.05
DELIST_RETURN = -0.50
rng = np.random.default_rng(SEED)
oos_simulated = oos_raw.copy()
n_tickers = oos_raw.shape[1]

for year in oos_raw.index.year.unique():
    year_dates = oos_raw.index[oos_raw.index.year == year]
    if len(year_dates) == 0:
        continue
    n_exit = max(1, int(n_tickers * ANNUAL_CHURN_RATE))
    exit_cols = rng.choice(n_tickers, size=n_exit, replace=False)
    exit_tickers = oos_raw.columns[exit_cols]
    delist_month = year_dates[0]
    oos_simulated.loc[delist_month, exit_tickers] = DELIST_RETURN
    if len(year_dates) > 1:
        oos_simulated.loc[year_dates[1:], exit_tickers] = 0.0

ew_all = oos_simulated.mean(axis=1)
ann_all = ew_all.mean() * 12
survivorship_premium = ann_survivors - ann_all

print(f"  Survivor EW annual return:     {ann_survivors:.4%}")
print(f"  Sim unbiased EW annual return: {ann_all:.4%}")
print(f"  Survivorship premium:          {survivorship_premium:.4%} annualized")
print(f"  (Simulates {ANNUAL_CHURN_RATE:.0%}/yr churn, "
      f"{DELIST_RETURN:.0%} delist return — consistent with historical estimates)")
~~~

A survivorship premium of approximately 3% annualized -- consistent with academic estimates (Elton, Gruber & Blake 1996 found 0.9% per year for mutual funds; Campbell Harvey's CRSP work estimates 1-4% for equities). If your backtest shows 15% annualized return, roughly 3 of those percentage points might be ghosts from companies quietly removed from the dataset.

~~~python
fig_sv, ax_sv = plt.subplots(figsize=(7, 5))
labels   = ["Unbiased\n(simulated delisting)", "Survivor-Only\n(S&P 500 universe)"]
returns_ = [ann_all, ann_survivors]
colors   = ["steelblue", "tomato"]
bars = ax_sv.bar(labels, [r * 100 for r in returns_], color=colors, alpha=0.85, width=0.5)
ax_sv.bar_label(bars, fmt="%.1f%%", padding=3)
ax_sv.set(
    title=f"Survivorship Bias: EW Annual Return OOS 2018–2024\n"
          f"(Simulated {ANNUAL_CHURN_RATE:.0%}/yr churn, {DELIST_RETURN:.0%} delist return)",
    ylabel="Annualized Return (%)",
)
ax_sv.axhline(0, color="gray", linewidth=0.8)
plt.tight_layout()
plt.show()
~~~

Over a 13-year backtest, a 3% annual premium compounds to a 46% cumulative inflation. Production quant shops use the CRSP database with 3,000+ stocks including delistings. Now that we know the failure modes, we need the tools to prevent them -- starting with cross-validation that respects financial time series structure.

---

## Section 2: Purged Cross-Validation -- Closing the Leakage Gap

Standard `TimeSeriesSplit` respects temporal order but has a blind spot unique to financial prediction. When your label is a forward-looking return, the training sample at month T-1 already "knows" almost everything the test sample at month T will reveal. **Purging** removes all training observations whose label interval overlaps with the test fold. **Embargoing** adds a buffer zone after each test fold. Without both, your cross-validated performance estimate is optimistic.

~~~python
alpha = load_alpha_output()
predictions = alpha["predictions"]
oos_dates = predictions.index.get_level_values("date").unique().sort_values()
n_obs = len(oos_dates)
K_FOLDS = 10
LABEL_DURATION = 1
EMBARGO = 1
~~~

We load the alpha model predictions from Week 4 (or a synthetic momentum fallback). The `LABEL_DURATION` of 1 month means our label looks one period ahead, and the `EMBARGO` of 1 month adds a buffer. On daily data with 21-day labels, the purge zone would be 21 times wider and the effect dramatically larger.

~~~python
class PurgedKFoldDemo:
    """Simplified purged k-fold splitter for teaching purposes.

    For each fold the test set is a contiguous block of time indices.
    Training data is all data outside the test block, minus:
      - a purge zone: the `label_duration` periods immediately before
        the test start (their labels overlap with the test period), and
      - an embargo zone: the `embargo` periods immediately after the
        test end (to guard against serial dependence).

    Args:
        n_splits:        number of folds.
        label_duration:  number of periods a label looks forward.
        embargo:         number of periods to drop after each test block.
    """

    def __init__(
        self, n_splits: int = 5, label_duration: int = 1, embargo: int = 1
    ):
        self.n_splits = n_splits
        self.label_duration = label_duration
        self.embargo = embargo
~~~

The class defines the three parameters a purged splitter needs. This is the algorithm from Lopez de Prado's Chapter 7, stripped to its essentials. The `split` method below defines four zones per fold: training (before purge and after embargo), purge (removed for label overlap), test, and embargo (safety buffer). Unlike walk-forward, training data appears on *both sides* of the test window -- more data at the cost of requiring the purging mechanism.

~~~python
def split(self, X, y=None):
    """Yield (train_indices, test_indices) for each fold.

    X must be a 1-D index or array of length n; the positional
    integers are what gets yielded (not the values).

    The fold grid starts at `label_duration + 1` to ensure every fold
    has at least one valid training observation after purging.
    """
    n = len(X)
    seed = self.label_duration + 1
    if n <= seed:
        return
    usable = n - seed
    fold_size = usable // self.n_splits

    for k in range(self.n_splits):
        test_start = seed + k * fold_size
        test_end = (
            test_start + fold_size if k < self.n_splits - 1 else n
        )

        purge_start = max(0, test_start - self.label_duration)

        embargo_end = min(n, test_end + self.embargo)

        test_idx = np.arange(test_start, test_end)

        before_purge = np.arange(0, purge_start)
        after_embargo = np.arange(embargo_end, n)
        train_idx = np.concatenate([before_purge, after_embargo])

        if len(train_idx) == 0:
            continue

        yield train_idx.copy(), test_idx.copy(), purge_start, test_end, embargo_end

PurgedKFoldDemo.split = split
~~~

We also need a function to compute IC for each fold, then we run both the walk-forward baseline and the purged version head-to-head.

~~~python
def compute_fold_ic(test_dates, predictions):
    """Compute Spearman IC across all test-period dates."""
    ic_values = []
    for date in test_dates:
        if date not in predictions.index.get_level_values("date"):
            continue
        df = predictions.loc[date].dropna(subset=["prediction", "actual"])
        if len(df) < 10:
            continue
        corr, _ = stats.spearmanr(df["prediction"], df["actual"])
        if np.isfinite(corr):
            ic_values.append(corr)
    return float(np.mean(ic_values)) if ic_values else np.nan

tss = TimeSeriesSplit(n_splits=K_FOLDS, gap=LABEL_DURATION)
wf_ic_per_fold = []
wf_splits_meta = []
for fold_idx, (train_idx, test_idx) in enumerate(tss.split(np.arange(n_obs))):
    test_dates = oos_dates[test_idx]
    fold_ic = compute_fold_ic(test_dates, predictions)
    wf_ic_per_fold.append(fold_ic)
    wf_splits_meta.append((train_idx, test_idx))
    print(f"  WF fold {fold_idx+1}/{K_FOLDS}: "
          f"test={oos_dates[test_idx[0]].date()}–{oos_dates[test_idx[-1]].date()}, "
          f"IC={fold_ic:.4f}")

wf_ic_series = pd.Series(wf_ic_per_fold, name="wf_ic")
wf_mean_ic = float(np.nanmean(wf_ic_per_fold))
print(f"  Walk-forward mean CV IC: {wf_mean_ic:.4f}")
~~~

A mean IC of approximately 0.02 -- weak by any standard, statistically insignificant at conventional levels. Now the purged version:

~~~python
pkf = PurgedKFoldDemo(
    n_splits=K_FOLDS, label_duration=LABEL_DURATION, embargo=EMBARGO
)
purged_ic_per_fold = []
purged_splits_meta = []
for fold_idx, (train_idx, test_idx, purge_start, test_end, embargo_end) in \
        enumerate(pkf.split(np.arange(n_obs))):
    test_dates = oos_dates[test_idx]
    fold_ic = compute_fold_ic(test_dates, predictions)
    purged_ic_per_fold.append(fold_ic)
    purged_splits_meta.append((train_idx, test_idx, purge_start, test_end, embargo_end))
    purge_zone_size = test_idx[0] - purge_start
    print(f"  PKF fold {fold_idx+1}/{K_FOLDS}: "
          f"train={len(train_idx)} obs, purge={purge_zone_size} obs, "
          f"test={oos_dates[test_idx[0]].date()}–{oos_dates[test_idx[-1]].date()}, "
          f"IC={fold_ic:.4f}")

purged_ic_series = pd.Series(purged_ic_per_fold, name="purged_ic")
purged_mean_ic = float(np.nanmean(purged_ic_per_fold))
ic_delta = wf_mean_ic - purged_mean_ic
print(f"  Purged KFold mean CV IC: {purged_mean_ic:.4f}")
print(f"  IC delta (WF − purged): {ic_delta:.4f}")
if ic_delta < 0.005:
    print("  NOTE: IC delta < 0.005 — purging effect modest on monthly data "
          "(label duration ≈ fold size). Effect is structural, not a bug.")
~~~

The IC delta is approximately -0.001 -- effectively zero. This is *not* because purging does not work. The label duration (1 month) equals the data frequency (monthly), compressing the contamination zone to a single observation per fold boundary. Lopez de Prado's work documents that walk-forward CV overestimates IC by 20-50% for 20-day labels on daily data. On daily data with multi-day labels -- the standard configuration at production quant shops -- the purging adjustment is material, often reducing IC estimates by 10-30%. Our monthly exercise shows the mechanism working correctly at a frequency where the effect is structurally small.

~~~python
fig_sv, ax_sv = plt.subplots(figsize=(12, 6))
n_folds_vis = len(purged_splits_meta)
date_positions = np.arange(n_obs)
for fold_idx, (train_idx, test_idx, purge_start, test_end, embargo_end) in \
        enumerate(purged_splits_meta):
    y = fold_idx
    if len(train_idx) > 0:
        ax_sv.barh(y, width=len(train_idx), left=train_idx[0],
            height=0.7, color="#4878CF", alpha=0.7, align="center")
    purge_zone_len = test_idx[0] - purge_start
    if purge_zone_len > 0:
        ax_sv.barh(y, width=purge_zone_len, left=purge_start,
            height=0.7, color="#D65F5F", alpha=0.8, align="center")
    ax_sv.barh(y, width=len(test_idx), left=test_idx[0],
        height=0.7, color="#E8A838", alpha=0.85, align="center")
    embargo_len = min(embargo_end, n_obs) - test_end
    if embargo_len > 0:
        ax_sv.barh(y, width=embargo_len, left=test_end,
            height=0.7, color="#888888", alpha=0.6, align="center")
tick_pos = np.arange(0, n_obs, max(1, n_obs // 8))
tick_labels = [str(oos_dates[i].year) for i in tick_pos]
ax_sv.set_xticks(tick_pos)
ax_sv.set_xticklabels(tick_labels, fontsize=9)
ax_sv.set_yticks(range(n_folds_vis))
ax_sv.set_yticklabels([f"Fold {k+1}" for k in range(n_folds_vis)], fontsize=9)
legend_patches = [
    mpatches.Patch(color="#4878CF", alpha=0.7, label="Train"),
    mpatches.Patch(color="#E8A838", alpha=0.85, label="Test"),
    mpatches.Patch(color="#D65F5F", alpha=0.8, label="Purged"),
    mpatches.Patch(color="#888888", alpha=0.6, label="Embargo"),
]
ax_sv.legend(handles=legend_patches, loc="lower right", fontsize=9)
ax_sv.set(
    title="Purged K-Fold Splits: Train / Test / Purged / Embargo Zones",
    xlabel="Period Index (OOS months)",
    ylabel="Fold",
)
ax_sv.invert_yaxis()
plt.tight_layout()
plt.show()
~~~

Blue is training data, orange is the test window, red is the purge zone, gray is the embargo. The purge zones are narrow because our label duration equals 1 month. On daily data, those red strips would be 21 trading days wide and visually prominent. This visualization is what a model risk analyst reviews when auditing a backtest's CV methodology.

~~~python
valid_folds = [i for i in range(len(wf_ic_per_fold))
               if np.isfinite(wf_ic_per_fold[i]) and i < len(purged_ic_per_fold)
               and np.isfinite(purged_ic_per_fold[i])]
fold_nums = [i + 1 for i in valid_folds]
wf_vals = [wf_ic_per_fold[i] for i in valid_folds]
purged_vals = [purged_ic_per_fold[i] for i in valid_folds]

fig_ic, ax_ic = plt.subplots(figsize=(10, 5))
ax_ic.plot(fold_nums, wf_vals, marker="o", linewidth=2,
           color="#4878CF", label=f"Walk-Forward (mean={wf_mean_ic:.4f})")
ax_ic.plot(fold_nums, purged_vals, marker="s", linewidth=2,
           color="#D65F5F", linestyle="--",
           label=f"Purged KFold (mean={purged_mean_ic:.4f})")
ax_ic.axhline(0, color="black", linewidth=0.8, linestyle=":")
ax_ic.set(
    title="Cross-Validation IC by Fold: Walk-Forward vs. Purged KFold",
    xlabel="Fold Index",
    ylabel="Spearman IC",
)
ax_ic.legend(fontsize=10)
ax_ic.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.show()
~~~

The two methods track each other closely. The mean IC values are within 0.001 of each other. In the seminar, students will discover that the methods disagree on *which model is best* in 6 out of 10 folds -- methodology choice affects model selection even when the aggregate IC gap is negligible. Purged CV tells us how to split correctly. But even with correct splits, we can still overfit by searching over enough variants.

---

## Section 3: Combinatorial Purged CV & the Multiple-Testing Problem

You found the best of 50 strategy variants. Walk-forward says Sharpe 1.4. But if those 50 variants were all drawn from the same noisy distribution, the expected maximum Sharpe is substantially above 1.0. Your "best" strategy may have won a beauty contest in noise. Combinatorial Purged CV generates $\binom{6}{2} = 15$ train-test paths and tests whether the in-sample winner maintains its OOS rank. The **Probability of Backtest Overfitting (PBO)** is the fraction of paths where it does not.

~~~python
alpha = load_alpha_output()
gbm_pred = alpha["predictions"]
nn_pred = alpha.get("nn_predictions", None)
feat_df = alpha.get("expanded_features")
fwd_df = pd.read_parquet(CACHE_DIR / "forward_returns_w5.parquet")

gbm_dates = gbm_pred.index.get_level_values("date").unique().sort_values()
print(f"  GBM OOS dates: {len(gbm_dates)} months "
      f"({gbm_dates[0].date()} – {gbm_dates[-1].date()})")
if nn_pred is not None:
    nn_dates = nn_pred.index.get_level_values("date").unique().sort_values()
    print(f"  NN  OOS dates: {len(nn_dates)} months")
else:
    nn_pred = None
    print("  NN predictions: not available")
~~~

We load three model variants: GBM from Week 4, a neural network if available, and a Ridge regression we train below. The Ridge training function processes one date at a time via walk-forward:

~~~python
def _ridge_train_predict_one(
    feat_df, fwd_df, feat_dates, train_window, i, pred_date
):
    """Train Ridge on prior window and predict one date.

    Returns list of record dicts, or empty list if data is insufficient.
    """
    train_dates = feat_dates[i - train_window:i]
    X_list, y_list = [], []
    for td in train_dates:
        try:
            X_td = feat_df.loc[td]
            y_td = fwd_df.loc[td]["fwd_return"]
        except KeyError:
            continue
        common = X_td.index.intersection(y_td.index)
        if len(common) < 10:
            continue
        X_list.append(X_td.loc[common])
        y_list.append(y_td.loc[common])
    if not X_list:
        return []
    X_train = pd.concat(X_list).fillna(0.0)
    y_train = pd.concat(y_list).fillna(0.0)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)
    model = Ridge(alpha=1.0)
    model.fit(X_scaled, y_train)
    try:
        X_pred = feat_df.loc[pred_date].fillna(0.0)
        y_actual = fwd_df.loc[pred_date]["fwd_return"]
    except KeyError:
        return []
    common_pred = X_pred.index.intersection(y_actual.index)
    if len(common_pred) < 10:
        return []
    X_pred_scaled = scaler.transform(X_pred.loc[common_pred])
    preds = model.predict(X_pred_scaled)
    records = []
    for ticker, p, a in zip(common_pred, preds, y_actual.loc[common_pred].values):
        records.append({"date": pred_date, "ticker": ticker,
                        "prediction": float(p), "actual": float(a)})
    return records
~~~

The walk-forward wrapper calls this for each OOS date, then we align all three models to the common period and compute IC:

~~~python
def compute_ridge_predictions(
    feat_df: pd.DataFrame,
    fwd_df: pd.DataFrame,
    train_window: int = 36,
) -> pd.DataFrame:
    """Walk-forward Ridge regression predictions."""
    feat_dates = feat_df.index.get_level_values("date").unique().sort_values()
    records = []
    for i, pred_date in enumerate(feat_dates[train_window:], start=train_window):
        if i % 12 == 0:
            print(f"  [Ridge {i}/{len(feat_dates)}] {pred_date.date()}")
        records.extend(
            _ridge_train_predict_one(feat_df, fwd_df, feat_dates, train_window, i, pred_date)
        )
    ridge_df = pd.DataFrame(records).set_index(["date", "ticker"])
    return ridge_df

print("Computing Ridge walk-forward predictions...")
ridge_pred = compute_ridge_predictions(feat_df, fwd_df, train_window=36)
ridge_dates = ridge_pred.index.get_level_values("date").unique().sort_values()
print(f"  Ridge OOS dates: {len(ridge_dates)} months "
      f"({ridge_dates[0].date()} – {ridge_dates[-1].date()})")

common_dates = gbm_dates
if nn_pred is not None:
    nn_common = common_dates[common_dates.isin(
        nn_pred.index.get_level_values("date").unique())]
else:
    nn_common = common_dates
ridge_common = common_dates[common_dates.isin(ridge_dates)]
final_dates = common_dates[
    common_dates.isin(nn_common) & common_dates.isin(ridge_common)]
print(f"  Aligned OOS dates (all 3 models): {len(final_dates)} months")
~~~

With all three models aligned, we compute monthly IC series. None achieves a t-statistic above 1.96.

~~~python
def compute_ic_series(pred_df: pd.DataFrame, dates: pd.DatetimeIndex) -> pd.Series:
    """Compute monthly Spearman IC for a prediction DataFrame."""
    ic_vals = {}
    for d in dates:
        try:
            sub = pred_df.loc[d]
        except KeyError:
            continue
        sub = sub.dropna()
        if len(sub) < 10:
            continue
        ic_vals[d] = rank_ic(sub["prediction"].values, sub["actual"].values)
    return pd.Series(ic_vals, name="ic")

gbm_ic_series = compute_ic_series(gbm_pred, final_dates)
nn_ic_series = compute_ic_series(nn_pred, final_dates) if nn_pred is not None else None
ridge_ic_series = compute_ic_series(ridge_pred, final_dates)
model_ic = {"GBM": gbm_ic_series, "Ridge": ridge_ic_series}
if nn_ic_series is not None:
    model_ic["NN"] = nn_ic_series
for name, ic_s in model_ic.items():
    stats_d = ic_summary(ic_s.dropna())
    print(f"  {name:6s}: mean_IC={stats_d['mean_ic']:.4f}, "
          f"t={stats_d['t_stat']:.2f}, p={stats_d['p_value']:.3f}")
~~~

Now let us run CPCV. For each of the 15 paths, we identify the in-sample winner and check where it ranks out-of-sample, then compute PBO.

~~~python
ic_panel = pd.DataFrame(model_ic).dropna()
n_obs = len(ic_panel)
model_names = ic_panel.columns.tolist()
print(f"\nCPCV setup: {n_obs} time points × {len(model_names)} models")
print(f"  Models: {model_names}")
cpcv = CombinatorialPurgedCV(n_splits=6, n_test_splits=2, purge_gap=1)
n_paths_expected = cpcv.get_n_splits(ic_panel)
print(f"  CPCV: C(6,2) = {n_paths_expected} paths expected")

cpcv_paths = []
for path_idx, (train_idx, test_idx) in enumerate(cpcv.split(ic_panel)):
    is_ics = ic_panel.iloc[train_idx]
    oos_ics = ic_panel.iloc[test_idx]
    is_mean = is_ics.mean()
    oos_mean = oos_ics.mean()
    is_winner = is_mean.idxmax()
    oos_rank_of_winner = oos_mean.rank(ascending=False).loc[is_winner]
    n_models = len(model_names)
    oos_median_rank = (n_models + 1) / 2.0
    cpcv_paths.append({"path": path_idx, "is_winner": is_winner,
        "oos_rank_winner": float(oos_rank_of_winner),
        "oos_ic_winner": float(oos_mean.loc[is_winner]),
        "n_models": n_models, "oos_median_rank": oos_median_rank})
print(f"\nCPCV complete: {len(cpcv_paths)} paths")
cpcv_df = pd.DataFrame(cpcv_paths)
is_winners = cpcv_df["is_winner"].value_counts()
print(f"  IS winners by model: {is_winners.to_dict()}")
oos_return_series = cpcv_df["oos_ic_winner"]
print(f"  OOS IC winner: mean={oos_return_series.mean():.4f}, "
      f"std={oos_return_series.std():.4f}")

n_paths = len(cpcv_df)
n_below_median = (cpcv_df["oos_rank_winner"] > cpcv_df["oos_median_rank"]).sum()
pbo = n_below_median / n_paths
print(f"\nProbability of Backtest Overfitting (PBO)")
print(f"  Paths where IS-winner ranks below median OOS: "
      f"{n_below_median}/{n_paths}")
print(f"  PBO = {pbo:.4f} ({pbo:.1%})")
if pbo < 0.20:
    print("  ⚠ PBO < 0.20: suspiciously low — possible look-ahead or leakage")
elif pbo > 0.75:
    print("  ⚠ PBO > 0.75: very high — models may be pure noise")
else:
    print(f"  PBO in acceptable range [0.20, 0.75] — typical regime variation")
~~~

PBO of 0.267 -- the in-sample winner maintains its ranking above the OOS median in 73% of CPCV paths. This falls in the "genuine alpha" range per Bailey et al. (2015). But PBO < 0.5 does not confirm statistical significance -- none of our three models clears even t = 1.96. PBO tells us GBM is *persistently the least-bad* of three models. That is useful for model selection, but it is not a deployment signal.

The Harvey-Liu-Zhu threshold and BHY correction tighten the screw further. Any new factor claim needs t > 3.0 to control the false discovery rate given the hundreds of factors already published.

~~~python
hlz_results = {}
for name, ic_s in model_ic.items():
    s = ic_summary(ic_s.dropna())
    hlz_results[name] = {"mean_ic": s["mean_ic"], "t_stat": s["t_stat"],
        "p_value": s["p_value"], "n": s["n"],
        "passes_t196": s["t_stat"] > 1.96 if np.isfinite(s["t_stat"]) else False,
        "passes_t300": s["t_stat"] > 3.00 if np.isfinite(s["t_stat"]) else False}
print("\nHarvey-Liu-Zhu t-stat analysis")
print(f"  {'Model':8s}  {'t-stat':>8s}  {'p-value':>8s}  {'t>1.96':>7s}  {'t>3.00':>7s}")
for name, res in hlz_results.items():
    t = res["t_stat"]
    p = res["p_value"]
    t_str = f"{t:.3f}" if np.isfinite(t) else "nan"
    p_str = f"{p:.4f}" if np.isfinite(p) else "nan"
    t196_str = "YES" if res["passes_t196"] else "NO"
    t300_str = "YES" if res["passes_t300"] else "NO"
    print(f"  {name:8s}  {t_str:>8s}  {p_str:>8s}  "
          f"{t196_str:>7s}  {t300_str:>7s}")
n_passing_t300 = sum(1 for r in hlz_results.values() if r["passes_t300"])
print(f"\n  N models passing t > 3.00: {n_passing_t300}/{len(hlz_results)}")
print("  (t > 3.0 recommended for strategies with many trials)")

raw_pvalues = np.array([hlz_results[m]["p_value"] for m in model_names
    if np.isfinite(hlz_results[m]["p_value"])])
model_names_finite = [m for m in model_names if np.isfinite(hlz_results[m]["p_value"])]
reject, p_adj, _, _ = multipletests(raw_pvalues, method="fdr_bh")
print("\nBHY (Benjamini-Hochberg-Yekutieli) Multiple Testing Correction")
print(f"  {'Model':8s}  {'Raw p':>8s}  {'BHY p':>8s}  {'p≥raw?':>8s}  {'Reject?':>8s}")
for m, p_raw, p_bhy, rej in zip(model_names_finite, raw_pvalues, p_adj, reject):
    print(f"  {m:8s}  {p_raw:>8.4f}  {p_bhy:>8.4f}  "
          f"{'YES' if p_bhy >= p_raw - 1e-12 else 'NO':>8s}  "
          f"{'YES' if rej else 'NO':>8s}")

rng = np.random.default_rng(42)
n_fake = 50
fake_ic = rng.normal(0, 0.07, (len(final_dates), n_fake))
fake_tstat = (fake_ic.mean(axis=0)
              / (fake_ic.std(axis=0) / np.sqrt(len(final_dates))))
fake_pvals = 2 * (1 - stats.norm.cdf(np.abs(fake_tstat)))
reject_nominal, p_adj_bhy, _, _ = multipletests(fake_pvals, method="fdr_bh")
n_fp_nominal = (fake_pvals < 0.05).sum()
n_fp_bhy = reject_nominal.sum()
print(f"\n  50-variant FDR simulation (null: IC ~ N(0, 0.07))")
print(f"  False positives at nominal p<0.05: {n_fp_nominal}/{n_fake}")
print(f"  False positives after BHY correction: {n_fp_bhy}/{n_fake}")
~~~

Zero out of three models clear t = 3.0. Zero clear even t = 1.96. All BHY-adjusted p-values are inflated relative to raw values -- the correction always makes it harder to reject the null. The 50-variant simulation confirms the harshness: with only 68 months of data, even random draws rarely produce spurious significance. In a larger universe with longer history, the problem becomes far worse.

~~~python
fig_hist, ax_hist = plt.subplots(figsize=(8, 5))
ax_hist.hist(cpcv_df["oos_ic_winner"], bins=15, color="#4C72B0", alpha=0.8,
             edgecolor="white")
median_val = cpcv_df["oos_ic_winner"].median()
ax_hist.axvline(median_val, color="firebrick", lw=2, linestyle="--",
                label=f"Median = {median_val:.4f}")
ax_hist.axvline(0, color="gray", lw=1.5, linestyle=":")
ax_hist.set(title="CPCV: OOS IC Distribution for IS-Winner",
    xlabel="OOS Mean IC", ylabel="Path Count")
ax_hist.legend()
plt.tight_layout()
plt.show()
~~~

The histogram shows visible spread in OOS IC across the 15 paths. Single-path walk-forward gives you one number and no sense of its variability. CPCV gives you a distribution.

~~~python
oos_ranks = cpcv_df["oos_rank_winner"].values
median_rank = float(np.median(oos_ranks))
fig_pbo, ax_pbo = plt.subplots(figsize=(8, 4))
ax_pbo.hist(oos_ranks, bins=np.arange(0.5, len(model_names) + 1.5, 1),
            color="#55A868", alpha=0.8, edgecolor="white")
ax_pbo.axvline(median_rank, color="firebrick", lw=2, linestyle="--",
               label=f"Median rank = {median_rank:.1f}")
below_mask = oos_ranks > median_rank
ax_pbo.fill_between(
    [median_rank, len(model_names) + 0.5],
    0, ax_pbo.get_ylim()[1] if ax_pbo.get_ylim()[1] > 0 else 5,
    alpha=0.15, color="red", label=f"Below median → PBO contribution")
ax_pbo.set(title=f"OOS Rank of IS-Winner Across 15 CPCV Paths (PBO = {pbo:.2f})",
    xlabel="OOS Rank of IS-Winner (1 = best)", ylabel="Number of Paths")
ax_pbo.legend()
plt.tight_layout()
plt.show()
~~~

Most mass is at rank 1. The red-shaded region represents paths where the IS winner fell below the median rank. With PBO = 0.267, this happens about 27% of the time -- a reasonably healthy result.

~~~python
model_labels = list(hlz_results.keys())
t_stats = [hlz_results[m]["t_stat"] for m in model_labels]
fig_hlz, ax_hlz = plt.subplots(figsize=(8, 4))
y_pos = range(len(model_labels))
colors = ["#4C72B0" if t <= 3.0 else "#C44E52" for t in t_stats]
ax_hlz.barh(y_pos, t_stats, color=colors, alpha=0.85, edgecolor="white")
ax_hlz.axvline(1.96, color="orange", lw=2, linestyle="--", label="t = 1.96 (5%)")
ax_hlz.axvline(3.00, color="firebrick", lw=2, linestyle="-", label="t = 3.00 (HLZ)")
ax_hlz.set_yticks(y_pos)
ax_hlz.set_yticklabels(model_labels)
ax_hlz.set(title="Harvey-Liu-Zhu t-stat: 3 Model Variants", xlabel="t-statistic")
ax_hlz.legend()
plt.tight_layout()
plt.show()
~~~

All three bars fall short of both thresholds. This is the visual punchline: the strategy that "looked promising" does not survive even the most basic significance test. Now: does it survive the cost of trading?

---

## Section 4: Transaction Cost Decomposition -- From Gross to Net Returns

Transaction costs decompose into three components: the **bid-ask spread**, **market impact** (Almgren-Chriss: $\eta \cdot \sigma \cdot \sqrt{\text{participation rate}}$), and **slippage**. The multiplier that converts per-trade costs into annual costs is **turnover** -- how much of the portfolio changes each month. High turnover amplifies every cost component linearly.

~~~python
ls = load_ls_portfolio()
gross_returns = ls["gross_returns"]
turnover = ls["turnover"]
weights_df = ls["weights"]
common_dates = gross_returns.index.intersection(turnover.index)
gross_returns = gross_returns.loc[common_dates]
turnover = turnover.loc[common_dates]

mean_turnover = turnover.mean()
max_turnover = turnover.max()
print(f"Mean one-way monthly turnover: {mean_turnover:.1%}")
print(f"Max one-way monthly turnover:  {max_turnover:.1%}")
if mean_turnover > 0.50:
    drag_monthly = mean_turnover * 2 * (5 / 10_000)
    drag_annual = drag_monthly * 12
    print(f"⚠ HIGH TURNOVER: {mean_turnover:.0%} one-way — "
          f"TC drag ≈ {drag_annual:.2%}/year at 5 bps (optimistic)")
~~~

Mean one-way monthly turnover of 140% -- nearly every position is replaced every month. Now let us build net return series under three cost regimes: zero (naive), fixed 5 bps (best-in-class execution), and tiered by market cap (10/20/30 bps for large/mid/small).

~~~python
def net_returns_from_spread(gross: pd.Series, to: pd.Series,
                             cost_bps: float) -> pd.Series:
    """Subtract round-trip spread cost from gross returns."""
    cost_frac = cost_bps / 10_000
    costs = to * 2 * cost_frac
    return (gross - costs).rename("net_return")

returns_zero = gross_returns.rename("zero_tc")
returns_fixed = net_returns_from_spread(gross_returns, turnover, cost_bps=5.0)

mcap_tiers = load_mcap_tiers()
spread_map = {"large": 10.0, "mid": 20.0, "small": 30.0}
tiered_spreads = []
for date in common_dates:
    if date not in weights_df.index:
        tiered_spreads.append(np.nan)
        continue
    w = weights_df.loc[date]
    active_tickers = w[w.abs() > 1e-8].index
    spreads = active_tickers.map(
        lambda t: spread_map.get(mcap_tiers.get(t, "mid"), 15.0))
    avg_spread = np.mean(spreads) if len(spreads) > 0 else 15.0
    tiered_spreads.append(avg_spread)
tiered_spread_series = pd.Series(tiered_spreads, index=common_dates)
tiered_costs = turnover * 2 * (tiered_spread_series / 10_000)
returns_tiered = (gross_returns - tiered_costs).rename("tiered_tc")
spread_drag_fixed = (turnover * 2 * (5 / 10_000)).mean() * 12
~~~

The formula is straightforward: round-trip cost = turnover times 2 times the half-spread. At 140% turnover and 10 bps, the monthly drag is about 0.28% or 3.4% per year. Now let us add market impact using the Almgren-Chriss square-root model.

~~~python
ASSUMED_AUM = 100_000_000

def _compute_impact_one_period(
    weights_df, vol_30d, dollar_vol, date, prev_date, i
):
    """Compute Almgren-Chriss sqrt-law impact for one rebalance period."""
    delta_w = (weights_df.iloc[i] - weights_df.iloc[i - 1]).abs()
    active = delta_w[delta_w > 1e-8].index
    if len(active) == 0:
        return {"date": date, "impact_cost": 0.0}
    available_days = vol_30d.index[vol_30d.index <= date]
    if len(available_days) == 0:
        return {"date": date, "impact_cost": 0.0}
    nearest_day = available_days[-1]
    sigma = vol_30d.loc[nearest_day, active].dropna()
    adv = dollar_vol.loc[nearest_day, active].dropna()
    common_t = sigma.index.intersection(adv.index)
    if len(common_t) == 0:
        return {"date": date, "impact_cost": 0.0}
    dw = delta_w[common_t]
    sig = sigma[common_t]
    adv_t = adv[common_t].replace(0, np.nan).dropna()
    common_t2 = dw.index.intersection(sig.index).intersection(adv_t.index)
    if len(common_t2) == 0:
        return {"date": date, "impact_cost": 0.0}
    trade_dollars = dw[common_t2] * ASSUMED_AUM
    participation = (trade_dollars / adv_t[common_t2]).clip(upper=1.0)
    impact_per_stock = 0.1 * sig[common_t2] * np.sqrt(participation)
    total_impact = float((impact_per_stock * dw[common_t2]).sum())
    return {"date": date, "impact_cost": total_impact}

def compute_market_impact(
    weights_df: pd.DataFrame, ohlcv: pd.DataFrame, impact_coeff: float = 0.1,
) -> pd.Series:
    """Almgren-Chriss sqrt-law market impact."""
    try:
        close = ohlcv.xs("Close", level=0, axis=1)
        volume = ohlcv.xs("Volume", level=0, axis=1)
    except KeyError:
        return pd.Series(0.0, index=weights_df.index[1:], name="impact_cost")
    dollar_vol = (close * volume).rolling(30).mean()
    daily_ret = close.pct_change()
    vol_30d = daily_ret.rolling(30).std()
    dates = weights_df.index
    impact_list = [
        _compute_impact_one_period(weights_df, vol_30d, dollar_vol,
                                   dates[i], dates[i - 1], i)
        for i in range(1, len(dates))]
    if not impact_list:
        return pd.Series(dtype=float, name="impact_cost")
    df = pd.DataFrame(impact_list).set_index("date")
    df.index = pd.DatetimeIndex(df.index)
    return df["impact_cost"]

print("Computing market impact costs from OHLCV data...")
ohlcv = load_ohlcv_data()
impact_costs = compute_market_impact(weights_df, ohlcv)
impact_costs = impact_costs.reindex(common_dates).fillna(0.0)
mean_monthly_impact = impact_costs.mean()
print(f"Mean monthly market impact cost: {mean_monthly_impact:.4%}")
~~~

Each rebalance period, we compute the participation rate for each stock, apply the square-root impact formula with $\eta = 0.1$ (lower end of the production range), and sum across traded stocks. The \$100M AUM assumption is standard for research-grade analysis. At institutional scale, impact costs can double or triple. Now the money plot:

~~~python
cum_zero = (1 + returns_zero).cumprod()
cum_fixed = (1 + returns_fixed).cumprod()
cum_tiered = (1 + returns_tiered).cumprod()
fig1, ax1 = plt.subplots(figsize=(12, 6))
ax1.plot(cum_zero.index, cum_zero.values, label="Zero TC (gross)", linewidth=2)
ax1.plot(cum_fixed.index, cum_fixed.values, label="Fixed 5 bps spread (optimistic)", linewidth=2)
ax1.plot(cum_tiered.index, cum_tiered.values,
         label="Tiered spread (10/20/30 bps by cap)", linewidth=2)
ax1.set(title="Transaction Cost Regimes: Cumulative Returns",
    xlabel="Date", ylabel="Cumulative Return ($ per $1 invested)")
ax1.legend()
ax1.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
~~~

The three curves diverge: gross reaches about \$2.36, optimistic 5 bps reaches \$2.15, tiered reaches \$1.79 -- a 28% Sharpe reduction. In production, firms report 40-60% live Sharpe reduction relative to backtest. Our 28% is conservative because we use $\eta = 0.1$ and \$100M AUM.

~~~python
annual_spread_cost = (turnover * 2 * (5 / 10_000)).mean() * 12
annual_impact_cost = impact_costs.mean() * 12
fig2, ax2 = plt.subplots(figsize=(7, 5))
components = ["Spread Cost (5 bps fixed)", "Market Impact (η=0.1)"]
values = [annual_spread_cost * 100, annual_impact_cost * 100]
bars = ax2.bar(components, values, color=["steelblue", "darkorange"])
ax2.bar_label(bars, fmt="%.2f%%")
ax2.set(title="Annual TC Decomposition (Average Month × 12)",
    xlabel="Cost Component", ylabel="Annual Cost (%)")
plt.tight_layout()
plt.show()
~~~

Spread costs dominate in normal markets, but market impact surges during stress periods. March 2020 -- the COVID crash -- is the highest-cost month in our dataset. When volatility spikes, the $\sigma$ term in the impact formula explodes.

~~~python
participation_rates = np.linspace(0.01, 0.20, 50)
try:
    close_prices = ohlcv.xs("Close", level=0, axis=1)
    median_sigma = close_prices.pct_change().std().median()
except (KeyError, IndexError) as exc:
    import warnings
    warnings.warn(f"Could not extract close prices for participation chart: {exc}")
    median_sigma = 0.015
eta = 0.1
impact_curve = eta * median_sigma * np.sqrt(participation_rates)

fig3, ax3 = plt.subplots(figsize=(8, 5))
ax3.plot(participation_rates * 100, impact_curve * 100, color="darkorange", linewidth=2)
ax3.axvline(x=5, color="gray", linestyle="--", alpha=0.7, label="5% participation")
ax3.set(title=f"Market Impact vs. Participation Rate (η={eta}, σ={median_sigma:.3f})",
    xlabel="Participation Rate (%)", ylabel="Market Impact Cost (%)")
ax3.legend()
ax3.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
~~~

The square-root shape: steep at low participation rates, flattening at higher ones. At 5% participation (the vertical line), most execution desks flag the trade as "large block." Transaction costs are the most common cause of the live-to-backtest gap. Now let us quantify whether the strategy's performance is statistically genuine.

---

## Section 5: The Deflated Sharpe Ratio -- When Significance Is Not Significant

Your strategy has a Sharpe ratio of 0.88. The **deflated Sharpe ratio** (DSR) adjusts for how long the backtest is, how non-normal the returns are, and how many strategies you tried. It often reveals that the "impressive" number is indistinguishable from a lucky draw from noise.

~~~python
ls = load_ls_portfolio()
gross_returns = ls["gross_returns"].dropna()
turnover = ls["turnover"].dropna()
print(f"Gross return series: {len(gross_returns)} months, "
      f"{gross_returns.index[0].date()} – {gross_returns.index[-1].date()}")
gross_sr = sharpe_ratio(gross_returns, periods_per_year=12)
gross_monthly_sr = gross_returns.mean() / gross_returns.std()
gross_skew = float(stats.skew(gross_returns))
gross_kurt = float(stats.kurtosis(gross_returns, fisher=True))
print(f"\nGross return moments (n={len(gross_returns)}):")
print(f"  Annualised Sharpe:  {gross_sr:.4f}")
print(f"  Monthly Sharpe:     {gross_monthly_sr:.4f}")
print(f"  Skewness:           {gross_skew:.4f}")
print(f"  Excess kurtosis:    {gross_kurt:.4f}")
if abs(gross_skew) > 1.0 or abs(gross_kurt) > 3.0:
    print("  ⚠ Heavy tails / significant skew — non-normality penalises DSR")
~~~

Excess kurtosis of 4.23 -- heavy tails. With kurtosis this high, the DSR formula's estimation uncertainty term is roughly 50% larger than under Gaussian assumptions, meaning you need proportionally more data for the same confidence. Let us compute the Probabilistic Sharpe Ratio (PSR, M=1) and then the full DSR surface.

~~~python
n_total = len(gross_returns)
psr = deflated_sharpe_ratio(
    observed_sr=gross_monthly_sr, n_trials=1, n_obs=n_total,
    skew=gross_skew, excess_kurt=gross_kurt)
print(f"\nProbabilistic Sharpe Ratio (M=1, T={n_total} months):")
print(f"  PSR = {psr:.4f}  (using monthly SR={gross_monthly_sr:.4f})")
print(f"  Interpretation: {psr*100:.1f}% probability the SR is genuine (single trial)")

monthly_sr = gross_monthly_sr
ret_skew = gross_skew
ret_kurt = gross_kurt
sr_label = "Gross (monthly)"
print(f"\nDSR surface input ({sr_label}):")
print(f"  Monthly SR = {monthly_sr:.4f}, skew = {ret_skew:.4f}, excess kurt = {ret_kurt:.4f}")

T_WINDOWS = [24, 36, 48, 60, 84, 120]
M_VALUES = [1, 5, 10, 20, 50]
surface_rows = []
for T in T_WINDOWS:
    for M in M_VALUES:
        dsr = deflated_sharpe_ratio(
            observed_sr=monthly_sr, n_trials=M, n_obs=T,
            skew=ret_skew, excess_kurt=ret_kurt)
        surface_rows.append({"T": T, "M": M,
            "dsr": dsr if (dsr is not None and np.isfinite(dsr)) else 0.0})
dsr_df = pd.DataFrame(surface_rows)
dsr_pivot = dsr_df.pivot(index="M", columns="T", values="dsr")
dsr_pivot.index.name = "M (trials)"
dsr_pivot.columns.name = "T (months)"
print("\nDSR surface (rows=M, cols=T):")
print(dsr_pivot.round(3).to_string())
~~~

The surface spans from 0.159 (M=50, T=24 months) to 1.000 (M=1, long track record). More trials or shorter track records push DSR toward zero. The heatmap makes this viscerally clear:

~~~python
fig_heatmap, ax_hm = plt.subplots(figsize=(9, 5))
im = ax_hm.imshow(dsr_pivot.values, aspect="auto", cmap="RdYlGn",
    vmin=0.0, vmax=1.0, origin="lower")
plt.colorbar(im, ax=ax_hm, label="DSR")
ax_hm.set_xticks(range(len(T_WINDOWS)))
ax_hm.set_xticklabels([f"{t}m" for t in T_WINDOWS])
ax_hm.set_yticks(range(len(M_VALUES)))
ax_hm.set_yticklabels([str(m) for m in M_VALUES])
ax_hm.set_xlabel("Track record length T (months)")
ax_hm.set_ylabel("Number of strategies tested M")
ax_hm.set_title(f"Deflated Sharpe Ratio Surface  [monthly SR={monthly_sr:.3f}, {sr_label}]")
for i, m in enumerate(M_VALUES):
    for j, t in enumerate(T_WINDOWS):
        val = dsr_pivot.loc[m, t]
        ax_hm.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=8,
                   color="black" if 0.25 < val < 0.85 else "white")
plt.tight_layout()
plt.show()
~~~

Red cells mean statistically indistinguishable from noise. Green cells mean it passes. The gradient from red to green as you move right (longer track record) or down (fewer trials) makes the tradeoff explicit.

Now the sobering calculation: the **Minimum Track Record Length** (MinTRL).

~~~python
def min_trl(
    sharpe_monthly: float, n_trials: int,
    skew: float = 0.0, excess_kurt: float = 0.0,
    confidence: float = 0.95,
) -> float:
    """Minimum Track Record Length (Bailey & Lopez de Prado, 2014).

    Returns minimum monthly observations needed for DSR to exceed
    `confidence` given `n_trials` strategies tested.
    """
    for T in range(6, 1201):
        dsr_val = deflated_sharpe_ratio(
            observed_sr=sharpe_monthly, n_trials=n_trials, n_obs=T,
            skew=skew, excess_kurt=excess_kurt)
        if dsr_val is not None and np.isfinite(dsr_val) and dsr_val >= confidence:
            return float(T)
    return float("inf")

sharpe_range = np.linspace(0.05, 0.60, 50)
mintrl_values = [
    min_trl(sr, n_trials=10, skew=ret_skew, excess_kurt=ret_kurt)
    for sr in sharpe_range]
obs_mintrl = min_trl(monthly_sr, n_trials=10, skew=ret_skew, excess_kurt=ret_kurt)
print(f"\nMinTRL at 95% confidence (M=10 strategies):")
print(f"  Observed monthly SR = {monthly_sr:.4f}")
print(f"  MinTRL = {obs_mintrl:.0f} months")
mintrl_test = min_trl(0.23, n_trials=10, skew=ret_skew, excess_kurt=ret_kurt)
print(f"  MinTRL at monthly SR=0.23 (~ann 0.80) = {mintrl_test:.0f} months")
~~~

MinTRL of 174 months -- 14.5 years. Under Gaussian assumptions, it would be about 10 months. The excess kurtosis of 4.23 inflated it by roughly 10x. **Non-normality is not a footnote in the DSR formula; it is the dominant term.** Most strategies have 3-5 year track records. At M=10 trials, a MinTRL of 174 months means nearly all strategies with typical track records cannot confirm their observed Sharpe ratios at 95% confidence.

~~~python
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
~~~

The MinTRL curve drops rapidly with increasing Sharpe. The horizontal lines mark typical fund evaluation horizons. Most realistic strategies require track records far longer than what funds have. We now have the full toolkit. The final section shows the responsible, integrated backtest.

---

## Section 6: The Responsible Backtest -- Putting It All Together

A responsible backtest follows this pipeline: data hygiene $\rightarrow$ purged CV $\rightarrow$ net-of-cost returns $\rightarrow$ DSR evaluation $\rightarrow$ sub-period stability $\rightarrow$ honest reporting. We run the same strategy through naive and responsible evaluations in parallel.

~~~python
_alpha = load_alpha_output()
_predictions = _alpha["predictions"]
_ic_series = _alpha["ic_series"]
_ls = load_ls_portfolio()
_gross_returns = _ls["gross_returns"]
_turnover = _ls["turnover"]
_tc = pd.read_parquet(CACHE_DIR / "tc_results.parquet")
_net_returns_tiered = _tc["net_return_tiered"]
_wf_ic_folds = pd.read_parquet(CACHE_DIR / "wf_ic.parquet")["wf_ic"]
_purged_ic_folds = pd.read_parquet(CACHE_DIR / "purged_ic.parquet")["purged_ic"]
_oos_start = _gross_returns.index[0]
_oos_end = _gross_returns.index[-1]
_sub1_end = pd.Timestamp("2021-11-30")
_sub2_start = pd.Timestamp("2021-12-31")

gross_returns = _gross_returns.copy()
naive_sharpe = sharpe_ratio(gross_returns, periods_per_year=12)
naive_ret_ann = gross_returns.mean() * 12
naive_mdd = max_drawdown(gross_returns)
naive_skew = float(stats.skew(gross_returns.dropna()))
naive_kurt = float(stats.kurtosis(gross_returns.dropna(), fisher=True))
tss_mean_ic = float(_wf_ic_folds.mean())
tss_std_ic = float(_wf_ic_folds.std())

print("── IS / OOS BOUNDARY ─────────────────────────────────────────")
print(f"  IS period:  pre-April 2019 (model training in Week 4)")
print(f"  OOS period: {_oos_start.date()} to {_oos_end.date()} ({len(gross_returns)} months)")
print()
print("── NAIVE EVALUATION (IS label; gross returns; no purging) ────")
print(f"  ann_return:      {naive_ret_ann:.4f}")
print(f"  ann_sharpe:      {naive_sharpe:.4f}")
print(f"  max_drawdown:    {naive_mdd:.4f}")
print(f"  skewness:        {naive_skew:.4f}")
print(f"  excess_kurtosis: {naive_kurt:.4f}")
print(f"  cv_mean_ic:      {tss_mean_ic:.4f}  (TimeSeriesSplit, no purging)")
~~~

Naive Sharpe of 0.877. That looks viable. Now the responsible version adds net-of-cost returns, purged CV, and DSR at M=10.

~~~python
net_ret_series = _net_returns_tiered.reindex(_gross_returns.index).dropna()
resp_sharpe = sharpe_ratio(net_ret_series, periods_per_year=12)
resp_ret_ann = net_ret_series.mean() * 12
resp_mdd = max_drawdown(net_ret_series)
resp_skew = float(stats.skew(net_ret_series.dropna()))
resp_kurt = float(stats.kurtosis(net_ret_series.dropna(), fisher=True))
purged_mean_ic = float(_purged_ic_folds.mean())
purged_std_ic = float(_purged_ic_folds.std())
n_obs = len(net_ret_series)
resp_monthly_sr = net_ret_series.mean() / net_ret_series.std() if net_ret_series.std() > 0 else 0.0
dsr_m10 = deflated_sharpe_ratio(
    observed_sr=resp_monthly_sr, n_trials=10, n_obs=n_obs,
    skew=resp_skew, excess_kurt=resp_kurt)
sharpe_gap = naive_sharpe - resp_sharpe

print()
print("── RESPONSIBLE EVALUATION (OOS label; net-tiered; purged CV) ──")
print(f"  ann_return:      {resp_ret_ann:.4f}")
print(f"  ann_sharpe:      {resp_sharpe:.4f}")
print(f"  monthly_sharpe:  {resp_monthly_sr:.4f}")
print(f"  max_drawdown:    {resp_mdd:.4f}")
print(f"  skewness:        {resp_skew:.4f}")
print(f"  excess_kurtosis: {resp_kurt:.4f}")
print(f"  cv_mean_ic:      {purged_mean_ic:.4f}  (PurgedKFold)")
print(f"  dsr_m10:         {dsr_m10:.4f}")
print(f"  sharpe_gap (naive - resp): {sharpe_gap:.4f}")
~~~

Responsible Sharpe of 0.575, down from 0.877. The gap of 0.302 -- a 34% reduction. The DSR of 0.414 triggers a NO-DEPLOY verdict. Lopez de Prado documented that the average ML fund's live Sharpe is approximately 40-60% of the backtest Sharpe. Our 65.6% retention rate is slightly better than the production benchmark, consistent with research-grade cost assumptions.

Let us check sub-period IC stability. A decaying signal is a different problem from a weak signal.

~~~python
ic_full = _ic_series["ic"]
ic_sub1 = ic_full.loc[_oos_start:_sub1_end]
ic_sub2 = ic_full.loc[_sub2_start:_oos_end]
ic_sub1_mean = float(ic_sub1.mean())
ic_sub2_mean = float(ic_sub2.mean())
ic_sub1_std = float(ic_sub1.std())
ic_sub2_std = float(ic_sub2.std())
ic_degradation = ic_sub1_mean - ic_sub2_mean

print()
print("── SUB-PERIOD IC SPLIT (OOS stability check) ─────────────────")
print(f"  Sub-period 1 (OOS: {ic_sub1.index[0].date()} to {ic_sub1.index[-1].date()}):")
print(f"    n_months: {len(ic_sub1)}")
print(f"    mean_ic:  {ic_sub1_mean:.4f}  std_ic: {ic_sub1_std:.4f}")
print(f"  Sub-period 2 (OOS: {ic_sub2.index[0].date()} to {ic_sub2.index[-1].date()}):")
print(f"    n_months: {len(ic_sub2)}")
print(f"    mean_ic:  {ic_sub2_mean:.4f}  std_ic: {ic_sub2_std:.4f}")
print(f"  IC degradation (sub1 - sub2): {ic_degradation:.4f}")
~~~

First-half IC of 0.054 dropping to 0.039 -- a 27% decline, but below the 50% threshold for a signal decay warning. The signal is weakening modestly, likely driven by the regime shift from momentum-friendly markets (2019-2021) to rate-shock markets (2022-2024). Week 14 will introduce regime-conditional backtesting.

~~~python
cum_naive = cumulative_returns(gross_returns)
net_aligned = net_ret_series.reindex(gross_returns.index).fillna(0.0)
cum_resp = cumulative_returns(net_aligned)
fig_curve, ax_curve = plt.subplots(figsize=(10, 5))
ax_curve.plot(cum_naive.index, cum_naive.values, label="Naive (gross, WF CV)", lw=2)
ax_curve.plot(cum_resp.index, cum_resp.values,
              label="Responsible (net-tiered, purged CV)", lw=2, linestyle="--")
ax_curve.axvline(x=_oos_start, color="gray", lw=1.2, linestyle=":", label="OOS start")
ax_curve.axvline(x=_sub1_end, color="lightcoral", lw=1.0, linestyle=":",
                 label="Sub-period split")
ax_curve.set(title="Naive vs. Responsible Equity Curve (IS/OOS labeled)",
    xlabel="Date", ylabel="Cumulative Return")
ax_curve.legend(fontsize=9)
plt.tight_layout()
plt.show()
~~~

The gap between the curves is the aggregate cost of honest methodology. Both curves show the same COVID drawdown in March 2020 -- costs do not help during crashes, they make them worse.

~~~python
fold_indices = np.arange(len(_wf_ic_folds))
bar_width = 0.35
fig_bar, ax_bar = plt.subplots(figsize=(10, 5))
ax_bar.bar(fold_indices - bar_width / 2, _wf_ic_folds.values,
           width=bar_width, label="Walk-Forward IC (naive)", alpha=0.8)
ax_bar.bar(fold_indices + bar_width / 2, _purged_ic_folds.values,
           width=bar_width, label="Purged CV IC (responsible)", alpha=0.8)
ax_bar.axhline(0, color="black", lw=0.8)
ax_bar.set(title="Fold-Level IC: Walk-Forward vs. Purged CV (OOS folds)",
    xlabel="Fold Index", ylabel="IC (Spearman)", xticks=fold_indices)
ax_bar.legend()
plt.tight_layout()
plt.show()
~~~

The walk-forward and purged IC bars are nearly identical -- confirming that on monthly data with 1-month labels, the CV method gap is negligible in aggregate. But the fold-level variation shows why methodology choice still affects which model you select. The gap between naive and responsible evaluation is not a problem to solve -- it is the information. The DSR of 0.414 means the strategy has only a 41% probability of being genuine at M=10. The verdict is NO-DEPLOY.

---

## Summary: The Responsible Backtesting Toolkit

| Concept | What it does | Key result |
|---------|-------------|------------|
| Look-ahead bias check | Detects future information leaking into signals | IC = 1.0 vs. -0.013 -- a 400 pp annual return gap |
| Survivorship bias | Quantifies return inflation from survivor-only universe | ~3% annual premium (simulated) |
| Purged k-fold CV | Removes label-contaminated training observations | IC delta near zero on monthly data; material on daily |
| CPCV + PBO | Tests if IS winner maintains OOS rank | PBO = 0.267 (positive signal, but weak absolute IC) |
| Harvey-Liu-Zhu threshold | Raises significance bar for multiple testing | 0/3 models clear t = 3.0 |
| TC decomposition | Spread + impact + slippage from gross returns | 28% Sharpe reduction (gross 0.877 to net 0.624) |
| Deflated Sharpe Ratio | Adjusts Sharpe for trials, length, non-normality | DSR = 0.414 at M=10 -- NO-DEPLOY |
| MinTRL | Minimum months to confirm observed Sharpe | 174 months (14.5 years) -- non-normality is dominant |

## Career Connections

**Quantitative researcher at a multi-strategy hedge fund.** The first question asked of any new strategy submission is "what's the gross and net Sharpe, and what's the DSR?" The backtest discipline pipeline you just saw -- purged CV, cost decomposition, DSR evaluation, sub-period stability -- is the exact format of a strategy evaluation memo at firms like Two Sigma, D.E. Shaw, and Citadel.

**ML engineer at a systematic fund.** You own the backtesting infrastructure. Your job is to ensure the evaluation framework *cannot* accidentally produce leaky results. The PurgedKFoldDemo class is a teaching version of the production-grade splitter you would maintain.

**Execution quant / portfolio analytics.** You run Transaction Cost Analysis -- the post-trade measurement of actual implementation shortfall versus backtest assumptions. The TC decomposition in Section 4 is the pre-trade version of what you measure post-trade.

**Model risk / validation analyst.** You audit backtests using exactly this framework. When a portfolio manager says "the model shows a Sharpe of 1.4," your job is to ask "what's the DSR, and how many variants did you test?"

## Bridge to Next Week

Week 5 established how to evaluate a strategy honestly. Week 6 asks the next question: given a strategy that has survived rigorous evaluation, how should it be sized and combined with other strategies in a portfolio? The DSR and MinTRL determine how much confidence to place in a strategy's forward Sharpe estimate, and that confidence level determines the Kelly fraction -- the optimal position size given your uncertainty. A strategy with DSR of 0.414 should receive a smaller capital allocation than one with DSR of 0.95. Week 6 makes that sizing decision quantitative.

**Forward reference -- regime conditioning (Week 14).** The Seven Sins taxonomy identifies regime neglect as a failure mode, but this week does not resolve it. Regime-conditional backtesting requires the Hidden Markov Model and change-point detection machinery from Week 14.

## Suggested Reading

- **Lopez de Prado (2018), *Advances in Financial Machine Learning*, Chapters 7-12.** The canonical text. Defines purged k-fold CV, CPCV, and the research discipline framework. Read this before any other backtest-related paper.

- **Bailey & Lopez de Prado (2014), "The Deflated Sharpe Ratio."** The DSR formula and MinTRL calculation. Short, precise, and devastating.

- **Harvey, Liu & Zhu (2016), "...And the Cross-Section of Expected Returns."** The t-stat hurdle of 3.0 for factor discovery.

- **Almgren & Chriss (2000), "Optimal Execution of Portfolio Transactions."** The canonical transaction cost decomposition. Still the industry standard 25 years later.

- **Arian, Norouzi & Seco (2024), "Backtest Overfitting in the Machine Learning Era."** The strongest current academic validation of the Lopez de Prado framework.
