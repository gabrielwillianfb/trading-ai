"""Temporal and Walk-Forward data splitters for time-series machine learning.

Guarantees strict chronological partitioning without random shuffling or boundary leakage.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional, Tuple
import numpy as np
import pandas as pd


@dataclass
class WalkForwardFold:
    """Represents a single cross-validation fold in a Walk-Forward split."""

    fold_index: int
    train_indices: np.ndarray
    test_indices: np.ndarray
    train_start: datetime
    train_end: datetime
    test_start: datetime
    test_end: datetime
    embargo_bars: int

    def summary(self) -> str:
        return (
            f"Fold #{self.fold_index} | Train ({len(self.train_indices)} bars): {self.train_start} -> {self.train_end} | "
            f"Embargo ({self.embargo_bars} bars) | Test ({len(self.test_indices)} bars): {self.test_start} -> {self.test_end}"
        )


class TemporalSplitter:
    """Partitions time-series data chronologically into Train, Validation, and Test sets."""

    @staticmethod
    def split(
        df: pd.DataFrame,
        train_pct: float = 0.70,
        val_pct: float = 0.15,
        test_pct: float = 0.15,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Split DataFrame into non-overlapping chronological chunks: [Train | Val | Test].

        Args:
            df: Input DataFrame with DatetimeIndex.
            train_pct: Fraction for training (default: 0.70).
            val_pct: Fraction for validation (default: 0.15).
            test_pct: Fraction for out-of-sample testing (default: 0.15).

        Returns:
            Tuple of (train_df, val_df, test_df).
        """
        total_pct = train_pct + val_pct + test_pct
        if not np.isclose(total_pct, 1.0, atol=1e-4):
            raise ValueError(f"Split percentages must sum to 1.0, got: {total_pct:.4f}")

        n = len(df)
        if n < 10:
            raise ValueError(f"Dataset too small ({n} rows) to partition into 3 splits.")

        train_end_idx = int(n * train_pct)
        val_end_idx = train_end_idx + int(n * val_pct)

        train_df = df.iloc[:train_end_idx]
        val_df = df.iloc[train_end_idx:val_end_idx]
        test_df = df.iloc[val_end_idx:]

        # Verify no empty split
        if train_df.empty or val_df.empty or test_df.empty:
            raise ValueError("One of the temporal splits resulted in an empty dataset. Increase data length.")

        return train_df, val_df, test_df

    @classmethod
    def split_arrays(
        cls,
        X: pd.DataFrame,
        y: pd.Series,
        train_pct: float = 0.70,
        val_pct: float = 0.15,
        test_pct: float = 0.15,
    ) -> Tuple[
        Tuple[pd.DataFrame, pd.Series],
        Tuple[pd.DataFrame, pd.Series],
        Tuple[pd.DataFrame, pd.Series],
    ]:
        """Split feature matrix X and target y chronologically."""
        if not X.index.equals(y.index):
            raise ValueError("Indices of X and y must match exactly.")

        X_train, X_val, X_test = cls.split(X, train_pct, val_pct, test_pct)
        y_train = y.loc[X_train.index]
        y_val = y.loc[X_val.index]
        y_test = y.loc[X_test.index]

        return (X_train, y_train), (X_val, y_val), (X_test, y_test)


class WalkForwardSplitter:
    """Generates expanding or rolling window folds for temporal cross-validation with embargo."""

    @staticmethod
    def generate_folds(
        df: pd.DataFrame,
        n_splits: int = 5,
        expanding: bool = True,
        embargo_bars: int = 12,
        min_train_ratio: float = 0.40,
    ) -> List[WalkForwardFold]:
        """Generate Walk-Forward validation folds.

        Structure per fold:
        [Train Window: 0 .. T] -> [Embargo Gap: T+1 .. T+E] -> [Test Window: T+E+1 .. T+E+W]

        Args:
            df: Input dataset with DatetimeIndex.
            n_splits: Number of walk-forward folds (default: 5).
            expanding: If True, uses expanding window for train; if False, uses rolling window.
            embargo_bars: Number of buffer bars between train and test to eliminate autocorrelation.
            min_train_ratio: Minimum fraction of dataset reserved for initial training fold.

        Returns:
            List of WalkForwardFold instances.
        """
        n = len(df)
        if n < (n_splits + 1) * 20:
            raise ValueError(f"Dataset length {n} is too small for {n_splits} walk-forward splits.")

        initial_train_size = int(n * min_train_ratio)
        remaining_bars = n - initial_train_size

        test_block_size = remaining_bars // n_splits
        if test_block_size <= embargo_bars:
            raise ValueError(
                f"Test block size ({test_block_size}) must be greater than embargo_bars ({embargo_bars})."
            )

        folds: List[WalkForwardFold] = []
        timestamps = df.index

        for fold_idx in range(n_splits):
            if expanding:
                train_start_idx = 0
            else:
                train_start_idx = fold_idx * test_block_size

            train_end_idx = initial_train_size + (fold_idx * test_block_size)

            test_start_idx = train_end_idx + embargo_bars
            test_end_idx = min(train_end_idx + embargo_bars + test_block_size, n)

            if test_start_idx >= test_end_idx or test_start_idx >= n:
                break

            train_indices = np.arange(train_start_idx, train_end_idx)
            test_indices = np.arange(test_start_idx, test_end_idx)

            fold = WalkForwardFold(
                fold_index=fold_idx + 1,
                train_indices=train_indices,
                test_indices=test_indices,
                train_start=timestamps[train_start_idx].to_pydatetime(),
                train_end=timestamps[train_end_idx - 1].to_pydatetime(),
                test_start=timestamps[test_start_idx].to_pydatetime(),
                test_end=timestamps[test_end_idx - 1].to_pydatetime(),
                embargo_bars=embargo_bars,
            )
            folds.append(fold)

        return folds
