"""Homework D1: PurgedKFold — Build a Leakage-Free Cross-Validator.

Students implement a production-quality PurgedKFold cross-validator from
scratch, verify correctness against a synthetic dataset, and measure the
leakage premium vs. standard TimeSeriesSplit on real GBM alpha predictions.

Downstream: hw/d3_responsible_report.py imports PurgedKFold from this file.
All class logic is importable; main execution is guarded under __name__ == '__main__'.
"""
import matplotlib
matplotlib.use("Agg")

import sys
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.model_selection import TimeSeriesSplit
from sklearn.utils.validation import check_array

warnings.filterwarnings("ignore", category=FutureWarning)

# ── Path setup ────────────────────────────────────────────────────────
# hw/ is one level below code/, which contains data_setup.py
# course/ is two levels above hw/
_CODE_DIR = Path(__file__).resolve().parent.parent
_COURSE_DIR = _CODE_DIR.parent.parent
sys.path.insert(0, str(_CODE_DIR))   # for data_setup
sys.path.insert(0, str(_COURSE_DIR)) # for shared.*
from data_setup import load_alpha_output, CACHE_DIR, PLOT_DIR

SEED = 42


# ── CELL: purged_kfold_init ──────────────────────────────────────────

class PurgedKFold:
    """K-Fold CV that purges label-overlapping training samples.

    Ref: De Prado (2018), AFML Ch. 7.
    Follows sklearn BaseCrossValidator interface.
    """

    def __init__(self, n_splits: int = 5, label_duration: int = 21,
                 embargo: int = 5) -> None:
        if n_splits < 2:
            raise ValueError(f"n_splits must be >= 2, got {n_splits}")
        if label_duration < 0:
            raise ValueError(f"label_duration must be >= 0, got {label_duration}")
        if embargo < 0:
            raise ValueError(f"embargo must be >= 0, got {embargo}")

        self.n_splits = n_splits
        self.label_duration = label_duration
        self.embargo = embargo

    def get_n_splits(self, X=None, y=None, groups=None) -> int:
        """Return the number of splitting iterations."""
        return self.n_splits


# ── CELL: purged_kfold_split ─────────────────────────────────────────

def split(self, X, y=None, groups=None):
    """Yield (train_idx, test_idx) with purging and embargo."""
    if hasattr(X, "index") and isinstance(X.index, pd.DatetimeIndex):
        yield from self._split_datetime(X)
    else:
        yield from self._split_positional(X)

PurgedKFold.split = split


# ── CELL: purged_kfold_estimate_freq ─────────────────────────────────

def _estimate_periods_per_day(self, dates: pd.DatetimeIndex) -> float:
    """Estimate sampling frequency: average periods per calendar day."""
    if len(dates) < 2:
        return 1.0
    total_days = (dates[-1] - dates[0]).days
    if total_days <= 0:
        return 1.0
    return (len(dates) - 1) / total_days

PurgedKFold._estimate_periods_per_day = _estimate_periods_per_day


# ── CELL: purged_kfold_datetime_setup ────────────────────────────────

def _datetime_setup(self, X):
    """Compute fold params for datetime splitting, or return None."""
    dates = X.index
    n = len(dates)
    if n < self.n_splits:
        raise ValueError(
            f"Cannot split {n} samples into {self.n_splits} folds."
        )

    periods_per_day = self._estimate_periods_per_day(dates)
    label_periods = max(1, round(self.label_duration * periods_per_day))

    seed = label_periods + 1
    if n <= seed:
        return None

    usable = n - seed
    fold_size = usable // self.n_splits
    label_end_dates = dates + pd.Timedelta(days=self.label_duration)
    embargo_td = pd.Timedelta(days=self.embargo)

    return dates, n, seed, fold_size, label_end_dates, embargo_td

PurgedKFold._datetime_setup = _datetime_setup


# ── CELL: purged_kfold_datetime_fold ─────────────────────────────────

def _datetime_fold_indices(dates, n, k, seed, fold_size, n_splits,
                           label_end_dates, embargo_td):
    """Compute (train_idx, test_idx) for one datetime fold, or None."""
    test_start_idx = seed + k * fold_size
    test_end_idx = test_start_idx + fold_size if k < n_splits - 1 else n
    if test_start_idx >= n:
        return None

    test_idx = np.arange(test_start_idx, min(test_end_idx, n))
    if len(test_idx) == 0:
        return None

    test_start_date = dates[test_start_idx]
    test_end_date = dates[min(test_end_idx - 1, n - 1)]
    embargo_end_date = test_end_date + embargo_td

    purge_start_idx = np.searchsorted(label_end_dates, test_start_date, side="left")
    embargo_end_idx = min(np.searchsorted(dates, embargo_end_date, side="right"), n)

    before_purge = np.arange(0, purge_start_idx)
    after_embargo = np.arange(embargo_end_idx, n)
    train_idx = np.concatenate([before_purge, after_embargo])

    if len(train_idx) == 0 or len(test_idx) == 0:
        return None
    return train_idx, test_idx


# ── CELL: purged_kfold_split_datetime ────────────────────────────────

def _split_datetime(self, X):
    """Split using DatetimeIndex for precise date-based purging."""
    setup = self._datetime_setup(X)
    if setup is None:
        return
    dates, n, seed, fold_size, label_end_dates, embargo_td = setup

    for k in range(self.n_splits):
        result = _datetime_fold_indices(
            dates, n, k, seed, fold_size, self.n_splits,
            label_end_dates, embargo_td,
        )
        if result is not None:
            yield result

PurgedKFold._split_datetime = _split_datetime


# ── CELL: purged_kfold_positional_fold ────────────────────────────────

def _positional_fold_indices(n, k, seed, fold_size, n_splits,
                             label_duration, embargo):
    """Compute (train_idx, test_idx) for one positional fold, or None."""
    test_start = seed + k * fold_size
    test_end = test_start + fold_size if k < n_splits - 1 else n
    if test_start >= n:
        return None

    test_idx = np.arange(test_start, min(test_end, n))
    if len(test_idx) == 0:
        return None

    purge_start = max(0, test_start - label_duration)
    embargo_end = min(n, test_end + embargo)

    before_purge = np.arange(0, purge_start)
    after_embargo = np.arange(embargo_end, n)
    train_idx = np.concatenate([before_purge, after_embargo])

    if len(train_idx) == 0:
        return None
    return train_idx, test_idx


# ── CELL: purged_kfold_split_positional ──────────────────────────────

def _split_positional(self, X):
    """Positional fallback: label_duration and embargo as row counts."""
    n = len(X)
    if n < self.n_splits:
        raise ValueError(
            f"Cannot split {n} samples into {self.n_splits} folds."
        )

    seed = self.label_duration + 1
    if n <= seed:
        return

    usable = n - seed
    fold_size = usable // self.n_splits

    for k in range(self.n_splits):
        result = _positional_fold_indices(
            n, k, seed, fold_size, self.n_splits,
            self.label_duration, self.embargo,
        )
        if result is not None:
            yield result

PurgedKFold._split_positional = _split_positional


# ── CELL: check_fold_leakage ────────────────────────────────────────

def _check_one_fold(train_idx, test_idx, dates, label_days, embargo_days):
    """Return (n_leaking, n_embargo_violations) for a single fold."""
    test_start_date = dates[test_idx[0]]
    test_end_date = dates[test_idx[-1]]
    embargo_end_date = test_end_date + pd.Timedelta(days=embargo_days)

    train_dates = dates[train_idx]
    label_end_dates = train_dates + pd.Timedelta(days=label_days)

    pre_test = train_dates < test_start_date
    leaking = (pre_test & (label_end_dates >= test_start_date)).sum()
    in_embargo = (train_dates > test_end_date) & (train_dates <= embargo_end_date)

    return leaking, in_embargo.sum()


# ── CELL: check_folds_aggregate ──────────────────────────────────────

def _check_folds_for_leakage(folds, dates, label_days, embargo_days):
    """Aggregate leak/embargo counts across all folds."""
    n_leaking = 0
    n_embargo_violations = 0
    n_train_total = 0

    for train_idx, test_idx in folds:
        leak, emb = _check_one_fold(
            train_idx, test_idx, dates, label_days, embargo_days,
        )
        n_leaking += leak
        n_embargo_violations += emb
        n_train_total += len(train_idx)

    return n_leaking, n_embargo_violations, n_train_total


# ── CELL: correctness_test ───────────────────────────────────────────

def run_correctness_test(n_months: int = 60, label_days: int = 21,
                         embargo_days: int = 5, k: int = 5) -> dict:
    """Verify purging and embargo correctness on synthetic data."""
    dates = pd.date_range("2015-01-31", periods=n_months, freq="ME")
    X_synth = pd.DataFrame(
        {"feature": np.random.default_rng(SEED).standard_normal(n_months)},
        index=dates,
    )

    pkf = PurgedKFold(n_splits=k, label_duration=label_days, embargo=embargo_days)
    folds = list(pkf.split(X_synth))

    n_leaking, n_embargo_violations, n_train_total = _check_folds_for_leakage(
        folds, dates, label_days, embargo_days,
    )

    return {
        "n_folds": len(folds),
        "n_train_total": n_train_total,
        "n_leaking": n_leaking,
        "n_embargo_violations": n_embargo_violations,
        "passed": n_leaking == 0 and n_embargo_violations == 0,
    }


# ── CELL: compute_fold_ics ──────────────────────────────────────────

def _compute_ic_for_date(predictions, d):
    """Compute cross-sectional Spearman IC for one date, or None."""
    if d not in predictions.index.get_level_values("date"):
        return None
    pred_d = predictions.loc[d]["prediction"].dropna()
    actual_d = predictions.loc[d]["actual"].dropna()
    common = pred_d.index.intersection(actual_d.index)
    if len(common) < 10:
        return None
    ic_val = float(pred_d[common].corr(actual_d[common], method="spearman"))
    return ic_val if np.isfinite(ic_val) else None


# ── CELL: fold_ics ───────────────────────────────────────────────────

def compute_fold_ics(predictions: pd.DataFrame, splits, dates: pd.Index
                     ) -> list:
    """Mean cross-sectional Spearman IC per fold's test period."""
    fold_ics = []
    for train_idx, test_idx in splits:
        test_dates = dates[test_idx]
        ic_vals = []
        for d in test_dates:
            ic_val = _compute_ic_for_date(predictions, d)
            if ic_val is not None:
                ic_vals.append(ic_val)
        fold_ics.append(np.mean(ic_vals) if ic_vals else np.nan)
    return fold_ics


# ── CELL: compare_cv_setup ──────────────────────────────────────────

def _run_cv_splits(predictions, k):
    """Build WF and PurgedKFold splits and compute fold ICs for each."""
    dates = predictions.index.get_level_values("date").unique().sort_values()
    n = len(dates)

    tss = TimeSeriesSplit(n_splits=k, gap=1)
    X_pos = np.arange(n).reshape(-1, 1)
    wf_splits = list(tss.split(X_pos))
    wf_ics = compute_fold_ics(predictions, wf_splits, dates)

    X_dt = pd.DataFrame({"dummy": 0}, index=dates)
    pkf = PurgedKFold(n_splits=k, label_duration=21, embargo=5)
    purged_splits = list(pkf.split(X_dt))
    purged_ics = compute_fold_ics(predictions, purged_splits, dates)

    return wf_ics, purged_ics


# ── CELL: compare_cv_methods ────────────────────────────────────────

def compare_cv_methods(predictions: pd.DataFrame, k: int = 10) -> dict:
    """Compare TimeSeriesSplit vs. PurgedKFold on IC computation."""
    wf_ics, purged_ics = _run_cv_splits(predictions, k)

    wf_clean = [v for v in wf_ics if np.isfinite(v)]
    purged_clean = [v for v in purged_ics if np.isfinite(v)]

    wf_mean = float(np.mean(wf_clean)) if wf_clean else np.nan
    purged_mean = float(np.mean(purged_clean)) if purged_clean else np.nan
    delta = wf_mean - purged_mean

    t_stat = np.nan
    if len(wf_clean) >= 2 and len(purged_clean) >= 2:
        t_stat, _ = stats.ttest_ind(wf_clean, purged_clean)

    return {
        "wf_ics": wf_ics, "purged_ics": purged_ics,
        "wf_mean": wf_mean, "purged_mean": purged_mean,
        "delta": delta,
        "t_stat": float(t_stat) if np.isfinite(t_stat) else np.nan,
        "n_wf": len(wf_clean), "n_purged": len(purged_clean),
    }


# ── CELL: split_visualization ─────────────────────────────────────────

def plot_split_timeline(n_periods: int = 260, label_days: int = 21,
                        embargo_days: int = 5, k: int = 5,
                        freq: str = "W-MON",
                        save_path=None) -> plt.Figure:
    """Visualize train / test / purged / embargo zones for k folds."""
    dates = pd.date_range("2015-01-05", periods=n_periods, freq=freq)
    X_dt = pd.DataFrame({"dummy": 0}, index=dates)
    pkf = PurgedKFold(n_splits=k, label_duration=label_days,
                      embargo=embargo_days)
    splits = list(pkf.split(X_dt))

    fig, ax = plt.subplots(figsize=(14, 4))

    colors = {
        "train": "#4C72B0",
        "test": "#DD8452",
        "purged": "#C44E52",
        "embargo": "#8C8C8C",
        "unused": "#EEEEEE",
    }

    n = len(dates)
    all_idx = np.arange(n)

    total_days = max(1, (dates[-1] - dates[0]).days)
    periods_per_day = (n - 1) / total_days
    embargo_periods = max(1, round(embargo_days * periods_per_day))

    for fold_num, (train_idx, test_idx) in enumerate(splits):
        y = fold_num

        test_set = set(test_idx)
        train_set = set(train_idx)

        test_start_idx = test_idx[0]
        test_end_idx = test_idx[-1]
        test_start_date = dates[test_start_idx]

        label_end_dates = dates + pd.Timedelta(days=label_days)

        for idx in all_idx:
            if idx in test_set:
                color = colors["test"]
            elif idx < test_start_idx and label_end_dates[idx] >= test_start_date:
                color = colors["purged"]
            elif idx > test_end_idx and idx <= test_end_idx + embargo_periods:
                color = colors["embargo"]
            elif idx in train_set:
                color = colors["train"]
            else:
                color = colors["unused"]

            ax.barh(y, width=1, left=idx, color=color, height=0.7,
                    linewidth=0, edgecolor="none")

    ax.set_yticks(range(len(splits)))
    ax.set_yticklabels([f"Fold {i + 1}" for i in range(len(splits))])
    ax.set_xlabel("Week Index")
    ax.set_title(
        f"PurgedKFold Split Visualization (k={k}, "
        f"label={label_days}d, embargo={embargo_days}d, weekly data)"
    )

    legend_patches = [
        mpatches.Patch(color=colors["train"], label="Train"),
        mpatches.Patch(color=colors["test"], label="Test"),
        mpatches.Patch(color=colors["purged"], label="Purged"),
        mpatches.Patch(color=colors["embargo"], label="Embargo"),
        mpatches.Patch(color=colors["unused"], label="Unused"),
    ]
    ax.legend(handles=legend_patches, loc="upper right", fontsize=8)
    plt.tight_layout()
    plt.show()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


# ── CELL: ic_bar_chart ────────────────────────────────────────────────

def plot_ic_comparison(results: dict, save_path=None) -> plt.Figure:
    """Bar chart comparing mean IC (WF vs. PurgedKFold) with std error bars."""
    wf_ics = [v for v in results["wf_ics"] if np.isfinite(v)]
    purged_ics = [v for v in results["purged_ics"] if np.isfinite(v)]

    means = [np.mean(wf_ics), np.mean(purged_ics)]
    stds = [np.std(wf_ics), np.std(purged_ics)]
    labels = ["TimeSeriesSplit\n(with leakage)", "PurgedKFold\n(cleaned)"]
    colors = ["#4C72B0", "#DD8452"]

    fig, ax = plt.subplots(figsize=(6, 5))
    bars = ax.bar(labels, means, yerr=stds, capsize=6, color=colors,
                  alpha=0.85, edgecolor="black", linewidth=0.7)
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_ylabel("Mean Fold IC (Spearman)")
    ax.set_title("CV Method Comparison: TimeSeriesSplit vs. PurgedKFold\n"
                 "(error bars = std across folds)")

    for bar, mean, std in zip(bars, means, stds):
        ax.text(bar.get_x() + bar.get_width() / 2, mean + std + 0.002,
                f"{mean:.4f}", ha="center", va="bottom", fontsize=10,
                fontweight="bold")

    delta = results["delta"]
    ax.text(0.5, 0.02, f"\u0394 IC (WF \u2212 Purged) = {delta:+.4f}",
            ha="center", va="bottom", transform=ax.transAxes,
            fontsize=9, color="darkred",
            bbox=dict(boxstyle="round,pad=0.3", fc="lightyellow", alpha=0.8))

    plt.tight_layout()
    plt.show()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


if __name__ == "__main__":
    # ── CORRECTNESS TEST ────────────────────────────────────────────
    print("Running correctness test...")

    corr_k5 = run_correctness_test(n_months=60, label_days=21,
                                   embargo_days=5, k=5)
    corr_k10 = run_correctness_test(n_months=60, label_days=21,
                                    embargo_days=5, k=10)

    if corr_k5["passed"] and corr_k10["passed"]:
        print("CORRECTNESS_TEST: PASS")
    else:
        n_leaking = corr_k5["n_leaking"] + corr_k10["n_leaking"]
        n_embargo = corr_k5["n_embargo_violations"] + corr_k10["n_embargo_violations"]
        print(f"CORRECTNESS_TEST: FAIL (n_leaking={n_leaking}, n_embargo_violations={n_embargo})")

    # ── FOLD COUNT CHECKS ───────────────────────────────────────────
    dates_60 = pd.date_range("2015-01-31", periods=60, freq="ME")
    X_60 = pd.DataFrame({"dummy": 0}, index=dates_60)

    pkf5 = PurgedKFold(n_splits=5, label_duration=21, embargo=5)
    pkf10 = PurgedKFold(n_splits=10, label_duration=21, embargo=5)
    folds_k5 = list(pkf5.split(X_60))
    folds_k10 = list(pkf10.split(X_60))

    print(f"  Fold count k=5: {len(folds_k5)} (expected 5)")
    print(f"  Fold count k=10: {len(folds_k10)} (expected 10)")

    # ── ASSERTIONS ──────────────────────────────────────────────────
    # D1-1: No training label overlaps test period, and no embargo violations
    assert corr_k5["n_leaking"] == 0, (
        f"D1-1 FAIL: {corr_k5['n_leaking']} leaking samples in k=5 test"
    )
    assert corr_k10["n_leaking"] == 0, (
        f"D1-1 FAIL: {corr_k10['n_leaking']} leaking samples in k=10 test"
    )
    assert corr_k5["n_embargo_violations"] == 0, (
        f"D1-1 FAIL: {corr_k5['n_embargo_violations']} embargo violations in k=5 test"
    )
    assert corr_k10["n_embargo_violations"] == 0, (
        f"D1-1 FAIL: {corr_k10['n_embargo_violations']} embargo violations in k=10 test"
    )

    # D1-2: Exact fold counts
    assert len(folds_k5) == 5, (
        f"D1-2 FAIL: expected 5 folds for k=5, got {len(folds_k5)}"
    )
    assert len(folds_k10) == 10, (
        f"D1-2 FAIL: expected 10 folds for k=10, got {len(folds_k10)}"
    )

    # D1-6: ValueError raised for invalid inputs
    err_count = 0
    for bad_kwargs in [
        {"n_splits": 1},           # < 2
        {"label_duration": -1},    # negative
        {"embargo": -3},           # negative
    ]:
        try:
            _ = PurgedKFold(**bad_kwargs)
        except ValueError:
            err_count += 1
    assert err_count == 3, (
        f"D1-6 FAIL: expected 3 ValueError raises, got {err_count}"
    )
    print("  D1-6: ValueError correctly raised for all 3 invalid input cases")

    # ── IC COMPARISON on GBM predictions ────────────────────────────
    print("\nLoading alpha predictions for IC comparison...")
    alpha = load_alpha_output()
    predictions = alpha["predictions"]
    n_oos = len(predictions.index.get_level_values("date").unique())
    print(f"  Predictions loaded: {n_oos} OOS months, "
          f"{len(predictions.index.get_level_values('ticker').unique())} tickers")

    print("Computing fold-level ICs (k=10)...")
    results = compare_cv_methods(predictions, k=10)

    print(f"\n  WF  mean IC: {results['wf_mean']:.4f}  "
          f"(n_folds={results['n_wf']})")
    print(f"  PKF mean IC: {results['purged_mean']:.4f}  "
          f"(n_folds={results['n_purged']})")
    print(f"  \u0394 IC (WF - PKF): {results['delta']:+.4f}")
    print(f"  t-stat on difference: {results['t_stat']:.3f}")

    # D1-3: PurgedKFold IC <= WF IC (direction check — flag, not fail, per spec)
    if results["purged_mean"] > results["wf_mean"]:
        print(
            f"  \u26a0 FLAG D1-3: PKF IC ({results['purged_mean']:.4f}) > WF IC "
            f"({results['wf_mean']:.4f}). On 1-month labels with monthly data, "
            f"purging effect can be near-zero or reversed. This is expected per "
            f"upstream agent notes \u2014 flagging but not failing."
        )
    else:
        print(f"  D1-3: PurgedKFold IC \u2264 WF IC \u2713")

    # D1-4: IC comparison with delta, t-stat, sample size must be printed (structural)
    # (Already printed above; confirmed below)

    # ── PLOTS ───────────────────────────────────────────────────────
    print("\nGenerating split visualization plot (k=5, 260 weeks)...")
    split_plot_path = PLOT_DIR / "d1_purged_kfold_splits.png"
    fig_splits = plot_split_timeline(
        n_periods=260, label_days=21, embargo_days=5, k=5,
        freq="W-MON", save_path=split_plot_path,
    )
    plt.close(fig_splits)

    print("Generating IC comparison bar chart...")
    ic_plot_path = PLOT_DIR / "d1_ic_comparison.png"
    fig_ic = plot_ic_comparison(results, save_path=ic_plot_path)
    plt.close(fig_ic)

    # ── RESULTS ─────────────────────────────────────────────────────
    print(f"\n\u2550\u2550 hw/d1_purged_kfold \u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550")
    print(f"  correctness_test_k5:   n_leaking={corr_k5['n_leaking']}, "
          f"n_embargo_violations={corr_k5['n_embargo_violations']}, "
          f"n_folds={corr_k5['n_folds']}, passed={corr_k5['passed']}")
    print(f"  correctness_test_k10:  n_leaking={corr_k10['n_leaking']}, "
          f"n_embargo_violations={corr_k10['n_embargo_violations']}, "
          f"n_folds={corr_k10['n_folds']}, passed={corr_k10['passed']}")
    print(f"  fold_count_k5:  {len(folds_k5)}")
    print(f"  fold_count_k10: {len(folds_k10)}")
    print(f"  wf_mean_ic:      {results['wf_mean']:.4f}")
    print(f"  purged_mean_ic:  {results['purged_mean']:.4f}")
    print(f"  ic_delta_wf_minus_pkf: {results['delta']:+.4f}")
    print(f"  ic_tstat:        {results['t_stat']:.4f}")
    print(f"  n_wf_folds:      {results['n_wf']}")
    print(f"  n_purged_folds:  {results['n_purged']}")
    print(f"  invalid_input_errors_raised: {err_count}/3")
    print(f"  \u2500\u2500 plot: d1_purged_kfold_splits.png \u2500\u2500")
    print(f"     type: horizontal timeline, train/test/purged/embargo zones")
    print(f"     n_folds: {len(folds_k5)}")
    print(f"     n_periods: 260 (weekly data)")
    print(f"     freq: W-MON")
    print(f"  \u2500\u2500 plot: d1_ic_comparison.png \u2500\u2500")
    print(f"     type: bar chart, WF vs. PurgedKFold mean IC with std bars")
    print(f"     wf_mean: {results['wf_mean']:.4f}")
    print(f"     purged_mean: {results['purged_mean']:.4f}")
    print(f"\u2713 hw/d1_purged_kfold: ALL PASSED")
