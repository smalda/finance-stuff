"""Temporal splitting utilities for financial ML.

Provides walk-forward and expanding-window splitters for out-of-sample
evaluation, plus sklearn-compatible CV splitters with purge gap for
use inside GridSearchCV.

Error contract: CV splitters raise ValueError on invalid configuration
(too few observations, bad n_splits). Generator functions yield nothing
if the date array is too short.
"""
from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import numpy as np
from sklearn.model_selection import BaseCrossValidator


# ---------------------------------------------------------------------------
# Out-of-sample splitters (for the outer rolling-prediction loop)
# ---------------------------------------------------------------------------

def walk_forward_splits(
    dates: np.ndarray, train_window: int, purge_gap: int = 1,
) -> Iterator[tuple[np.ndarray, Any]]:
    """Yield (train_dates, pred_date) for rolling walk-forward evaluation.

    Args:
        dates: sorted array-like of unique period labels (e.g. monthly dates).
        train_window: number of periods in the fixed-width training window.
        purge_gap: periods to skip between train end and prediction date.

    Yields:
        (train_dates, pred_date): train_dates is an array slice,
        pred_date is the single date to predict.
    """
    dates = np.asarray(dates)
    for i in range(train_window + purge_gap, len(dates)):
        train_end = i - purge_gap
        train_start = max(0, train_end - train_window)
        yield dates[train_start:train_end], dates[i]


def expanding_window_splits(
    dates: np.ndarray, min_window: int, purge_gap: int = 1,
) -> Iterator[tuple[np.ndarray, Any]]:
    """Yield (train_dates, pred_date) for expanding-window evaluation.

    Same as walk_forward_splits but the training window grows from
    min_window instead of staying fixed.
    """
    dates = np.asarray(dates)
    for i in range(min_window + purge_gap, len(dates)):
        train_end = i - purge_gap
        yield dates[:train_end], dates[i]


# ---------------------------------------------------------------------------
# sklearn-compatible CV splitter (for HP search inside a training window)
# ---------------------------------------------------------------------------

class CombinatorialPurgedCV(BaseCrossValidator):
    """Combinatorial Purged Cross-Validation (Lopez de Prado, 2018).

    Generates all C(n_splits, n_test_splits) train/test combinations from
    n_splits sequential groups, with purge_gap observations dropped at
    each train/test boundary. This produces many more paths than standard
    k-fold, enabling backtest overfitting probability estimation.

    Args:
        n_splits: number of sequential groups to divide data into.
        n_test_splits: number of groups used for testing in each combo.
        purge_gap: observations to drop at each train/test boundary.

    Note: the backtest overfitting probability (PBO) computation that
    consumes CPCV outputs is deferred to Week 5's blueprint.
    """

    def __init__(self, n_splits: int = 6, n_test_splits: int = 2,
                 purge_gap: int = 1):
        self.n_splits = n_splits
        self.n_test_splits = n_test_splits
        self.purge_gap = purge_gap

    def get_n_splits(self, X=None, y=None, groups=None) -> int:
        from math import comb
        return comb(self.n_splits, self.n_test_splits)

    def split(self, X, y=None, groups=None) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        from itertools import combinations

        n = len(X)
        fold_size = n // self.n_splits
        if fold_size < 1:
            raise ValueError(
                f"Not enough data ({n} obs) for {self.n_splits} splits"
            )

        # Define group boundaries
        boundaries = [(k * fold_size,
                        min((k + 1) * fold_size, n))
                       for k in range(self.n_splits)]

        for test_groups in combinations(range(self.n_splits), self.n_test_splits):
            test_set = set()
            purge_set = set()

            for g in test_groups:
                start, end = boundaries[g]
                test_set.update(range(start, end))
                # Purge observations at boundaries between train and test
                purge_start = max(0, start - self.purge_gap)
                purge_end = min(n, end + self.purge_gap)
                purge_set.update(range(purge_start, start))
                purge_set.update(range(end, purge_end))

            train_idx = np.array(sorted(
                set(range(n)) - test_set - purge_set
            ))
            test_idx = np.array(sorted(test_set))

            if len(train_idx) > 0 and len(test_idx) > 0:
                yield train_idx, test_idx


class PurgedKFold(BaseCrossValidator):
    """K-Fold cross-validator with label-aware purging and embargo.

    Exported from Week 05 (Backtesting & Transaction Costs).

    Standard K-Fold assigns folds by row position. When labels span
    multiple periods (e.g., a 21-day forward return), training samples
    near the test boundary will have label windows overlapping the test
    period — leaking future information.

    PurgedKFold removes ("purges") any training sample whose label end
    falls within the test window. An optional embargo further removes
    samples immediately after the test window.

    Unlike PurgedWalkForwardCV (which is walk-forward only), this
    implements true k-fold where training uses data both before AND
    after each test block (minus purge/embargo zones).

    Args:
        n_splits: number of folds (k). Must be >= 2.
        label_duration: duration of the prediction label in periods
            (row counts). Samples within this many periods before the
            test window start are purged because their labels overlap
            the test period.
        embargo: number of periods after the test window end to exclude
            from training. Prevents leakage from serial dependence.

    Raises:
        ValueError: if n_splits < 2, label_duration < 0, or embargo < 0.

    Reference: De Prado (2018), "Advances in Financial Machine Learning", Ch. 7.
    """

    def __init__(self, n_splits: int = 5, label_duration: int = 1,
                 embargo: int = 1):
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
        return self.n_splits

    def split(self, X, y=None, groups=None) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        """Generate purged k-fold train/test indices.

        Args:
            X: data to split. Only len(X) is used.

        Yields:
            (train, test): train indices (after purging and embargo)
            and test indices, both as np.ndarray of int.

        Raises:
            ValueError: if fewer samples than folds.
        """
        n = len(X)
        if n < self.n_splits:
            raise ValueError(
                f"Cannot split {n} samples into {self.n_splits} folds."
            )

        # First label_duration rows can't be test (their labels extend
        # before the data starts), so test indices begin after this offset.
        min_test_start = self.label_duration + 1
        if n <= min_test_start:
            return

        usable = n - min_test_start
        fold_size = usable // self.n_splits

        for k in range(self.n_splits):
            test_start = min_test_start + k * fold_size
            test_end = test_start + fold_size if k < self.n_splits - 1 else n

            if test_start >= n:
                break

            test_idx = np.arange(test_start, min(test_end, n))
            if len(test_idx) == 0:
                continue

            # Purge: remove pre-test indices whose labels overlap test
            purge_start = max(0, test_start - self.label_duration)
            # Embargo: remove post-test indices within embargo window
            embargo_end = min(n, test_end + self.embargo)

            # True k-fold: train on data before purge AND after embargo
            before_purge = np.arange(0, purge_start)
            after_embargo = np.arange(embargo_end, n)
            train_idx = np.concatenate([before_purge, after_embargo])

            if len(train_idx) == 0:
                continue

            yield train_idx, test_idx


class PurgedWalkForwardCV(BaseCrossValidator):
    """Walk-forward cross-validator with purge gap.

    Splits data into n_splits sequential folds. For each fold k, everything
    before the fold is train (minus purge_gap observations at the boundary)
    and the fold itself is validation.

    Compatible with sklearn GridSearchCV / RandomizedSearchCV.

    Args:
        n_splits: number of validation folds.
        purge_gap: observations to drop between train and validation
            to prevent information leakage from overlapping targets.
    """

    def __init__(self, n_splits: int = 3, purge_gap: int = 1):
        self.n_splits = n_splits
        self.purge_gap = purge_gap

    def get_n_splits(self, X=None, y=None, groups=None) -> int:
        return self.n_splits

    def split(self, X, y=None, groups=None) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        n = len(X)
        fold_size = n // (self.n_splits + 1)
        if fold_size < 1:
            raise ValueError(
                f"Not enough data ({n} obs) for {self.n_splits} folds"
            )

        for k in range(1, self.n_splits + 1):
            val_start = k * fold_size
            val_end = min(val_start + fold_size, n)
            train_end = max(0, val_start - self.purge_gap)
            if train_end < 2:
                continue  # not enough training data for this fold
            yield np.arange(train_end), np.arange(val_start, val_end)
