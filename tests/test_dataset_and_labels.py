"""Tests for LabelGenerator, TemporalSplitter, WalkForwardSplitter, and DatasetBuilder."""

from datetime import datetime, timedelta, timezone
import numpy as np
import pandas as pd
import pytest

from app.data.preprocessing.dataset import DatasetBuilder
from app.data.preprocessing.labeling import LabelGenerator
from app.data.preprocessing.splitters import TemporalSplitter, WalkForwardSplitter


def generate_sample_candles(n_bars: int = 120, seed: int = 42) -> pd.DataFrame:
    """Generate deterministic OHLCV test DataFrame."""
    np.random.seed(seed)
    returns = np.random.normal(loc=0.0003, scale=0.005, size=n_bars)
    price_path = 50000.0 * np.cumprod(1 + returns)

    dates = pd.date_range("2026-01-01", periods=n_bars, freq="15min", tz="UTC")

    opens = price_path
    closes = price_path * (1 + np.random.normal(0, 0.001, size=n_bars))
    highs = np.maximum(opens, closes) * 1.002
    lows = np.minimum(opens, closes) * 0.998
    volumes = np.full(n_bars, 10.0)

    return pd.DataFrame(
        {
            "open": opens,
            "high": highs,
            "low": lows,
            "close": closes,
            "volume": volumes,
        },
        index=dates,
    )


# ------------------------------------------------------------------------------
# 1. LabelGenerator Tests
# ------------------------------------------------------------------------------

def test_future_return_labels_math_and_embargo():
    """Verify future return labeling logic and trailing NaN embargo."""
    # Prices: 100, 102, 106, 108, 110 (horizon = 2, target = 5%)
    prices = pd.Series(
        [100.0, 102.0, 106.0, 108.0, 110.0],
        index=pd.date_range("2026-01-01", periods=5, freq="15min", tz="UTC"),
    )
    horizon = 2
    target = 0.05  # +5%

    labels = LabelGenerator.compute_future_return_labels(prices, horizon=horizon, target_return=target)

    # Index 0: 100 -> 106 (+6% >= 5% -> 1.0)
    assert labels.iloc[0] == 1.0
    # Index 1: 102 -> 108 (+5.88% >= 5% -> 1.0)
    assert labels.iloc[1] == 1.0
    # Index 2: 106 -> 110 (+3.77% < 5% -> 0.0)
    assert labels.iloc[2] == 0.0
    # Trailing horizon bars must be NaN
    assert pd.isna(labels.iloc[3])
    assert pd.isna(labels.iloc[4])


def test_future_peak_labels_math():
    """Verify peak label triggers 1 if any high in window hits target return."""
    dates = pd.date_range("2026-01-01", periods=6, freq="15min", tz="UTC")
    close = pd.Series([100.0, 98.0, 97.0, 96.0, 95.0, 94.0], index=dates)
    # At t=2, High spikes to 105 (+5%) even though close continues dropping
    high = pd.Series([101.0, 99.0, 106.0, 97.0, 96.0, 95.0], index=dates)

    labels = LabelGenerator.compute_future_peak_labels(
        high=high, close=close, horizon=3, target_return=0.05
    )

    # Index 0: window [99.0, 106.0, 97.0] -> max=106.0 -> (106 - 100)/100 = 6% >= 5% -> 1.0
    assert labels.iloc[0] == 1.0
    # Trailing 3 bars must be NaN
    assert pd.isna(labels.iloc[3])
    assert pd.isna(labels.iloc[4])
    assert pd.isna(labels.iloc[5])


def test_triple_barrier_labels():
    """Verify triple-barrier logic (Take Profit vs Stop Loss)."""
    dates = pd.date_range("2026-01-01", periods=5, freq="15min", tz="UTC")
    close = pd.Series([100.0, 100.0, 100.0, 100.0, 100.0], index=dates)
    high = pd.Series([100.5, 102.5, 100.5, 100.5, 100.5], index=dates)  # Hits TP (+2%) at t=1
    low = pd.Series([99.5, 99.5, 99.5, 99.5, 99.5], index=dates)

    labels = LabelGenerator.compute_triple_barrier_labels(
        high=high, low=low, close=close, horizon=2, take_profit=0.02, stop_loss=0.01
    )

    assert labels.iloc[0] == 1.0
    assert pd.isna(labels.iloc[-1])
    assert pd.isna(labels.iloc[-2])


# ------------------------------------------------------------------------------
# 2. TemporalSplitter Tests
# ------------------------------------------------------------------------------

def test_temporal_splitter_chronological_integrity():
    """Verify strict chronological monotonicity across Train, Val, and Test."""
    df = generate_sample_candles(100)
    train_df, val_df, test_df = TemporalSplitter.split(df, train_pct=0.70, val_pct=0.15, test_pct=0.15)

    assert len(train_df) == 70
    assert len(val_df) == 15
    assert len(test_df) == 15

    # CRITICAL: Verify NO overlap and strict time ordering
    assert train_df.index.max() < val_df.index.min()
    assert val_df.index.max() < test_df.index.min()


def test_temporal_splitter_invalid_percentages():
    """Verify invalid percentage sums raise ValueError."""
    df = generate_sample_candles(50)
    with pytest.raises(ValueError):
        TemporalSplitter.split(df, train_pct=0.80, val_pct=0.20, test_pct=0.20)  # sums to 1.20


# ------------------------------------------------------------------------------
# 3. WalkForwardSplitter Tests
# ------------------------------------------------------------------------------

def test_walk_forward_splitter_folds_and_embargo():
    """Verify walk-forward fold generation with non-overlapping embargo gap."""
    df = generate_sample_candles(120)
    embargo = 6
    folds = WalkForwardSplitter.generate_folds(
        df, n_splits=3, expanding=True, embargo_bars=embargo, min_train_ratio=0.50
    )

    assert len(folds) == 3

    for fold in folds:
        assert len(fold.train_indices) > 0
        assert len(fold.test_indices) > 0
        # Verify embargo gap between train end and test start
        train_max_idx = fold.train_indices.max()
        test_min_idx = fold.test_indices.min()
        assert test_min_idx - train_max_idx > embargo
        assert fold.train_end < fold.test_start

    # Expanding window property: each fold train size grows
    assert len(folds[1].train_indices) > len(folds[0].train_indices)
    assert len(folds[2].train_indices) > len(folds[1].train_indices)


# ------------------------------------------------------------------------------
# 4. DatasetBuilder Tests
# ------------------------------------------------------------------------------

def test_dataset_builder_build_and_split():
    """Verify DatasetBuilder creates aligned dataset and splits correctly."""
    df = generate_sample_candles(120)
    dataset = DatasetBuilder.build(
        candles_df=df,
        symbol="BTC/USDT",
        timeframe="15m",
        horizon=6,
        target_return=0.003,
        label_method="peak",
        drop_na=True,
    )

    assert dataset.total_samples > 0
    assert len(dataset.X) == len(dataset.y)
    assert len(dataset.dataset_hash) == 64  # SHA256 hex string
    assert 0 in dataset.class_distribution or 1 in dataset.class_distribution

    # Verify deterministic hash
    dataset_again = DatasetBuilder.build(
        candles_df=df,
        symbol="BTC/USDT",
        timeframe="15m",
        horizon=6,
        target_return=0.003,
        label_method="peak",
        drop_na=True,
    )
    assert dataset.dataset_hash == dataset_again.dataset_hash

    # Test splitting
    splits = DatasetBuilder.split_dataset(dataset, train_pct=0.70, val_pct=0.15, test_pct=0.15)
    assert len(splits.X_train) == len(splits.y_train)
    assert len(splits.X_val) == len(splits.y_val)
    assert len(splits.X_test) == len(splits.y_test)
    assert splits.X_train.index.max() < splits.X_val.index.min()
    assert splits.X_val.index.max() < splits.X_test.index.min()
