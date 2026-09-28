"""Dataset constructor combining feature matrices, target labels, and cryptographic hashing."""

import hashlib
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Literal, Optional
import numpy as np
import pandas as pd

from app.data.preprocessing.labeling import LabelGenerator
from app.data.preprocessing.splitters import TemporalSplitter
from app.features.pipeline import FeaturePipeline


@dataclass
class Dataset:
    """Complete aligned dataset containing feature matrix X, target y, and metadata."""

    X: pd.DataFrame
    y: pd.Series
    symbol: str
    timeframe: str
    target_name: str
    feature_names: List[str]
    dataset_hash: str
    start_date: datetime
    end_date: datetime
    total_samples: int
    class_distribution: Dict[int, int] = field(default_factory=dict)

    def summary(self) -> str:
        pos_ratio = (
            self.class_distribution.get(1, 0) / max(self.total_samples, 1)
        ) * 100.0
        return (
            f"Dataset: {self.symbol} [{self.timeframe}] | Samples: {self.total_samples} | "
            f"Features: {len(self.feature_names)} | Hash: {self.dataset_hash[:12]}...\n"
            f"Date Range: {self.start_date.isoformat()} -> {self.end_date.isoformat()}\n"
            f"Class Distribution: {self.class_distribution} (Positive Rate: {pos_ratio:.1f}%)"
        )


@dataclass
class DatasetSplits:
    """Container holding chronological Train, Validation, and Test partitions."""

    X_train: pd.DataFrame
    y_train: pd.Series
    X_val: pd.DataFrame
    y_val: pd.Series
    X_test: pd.DataFrame
    y_test: pd.Series
    dataset: Dataset

    def summary(self) -> str:
        return (
            f"Train ({len(self.X_train)} samples): {self.X_train.index.min()} -> {self.X_train.index.max()} "
            f"(Pos: {(self.y_train == 1).mean()*100:.1f}%)\n"
            f"Val   ({len(self.X_val)} samples): {self.X_val.index.min()} -> {self.X_val.index.max()} "
            f"(Pos: {(self.y_val == 1).mean()*100:.1f}%)\n"
            f"Test  ({len(self.X_test)} samples): {self.X_test.index.min()} -> {self.X_test.index.max()} "
            f"(Pos: {(self.y_test == 1).mean()*100:.1f}%)"
        )


LabelMethodType = Literal["peak", "return", "triple_barrier", "dynamic_atr_tb"]


class DatasetBuilder:
    """Orchestrates building clean aligned datasets ready for ML training."""

    @staticmethod
    def _compute_dataset_hash(X: pd.DataFrame, y: pd.Series) -> str:
        """Compute SHA-256 hash of dataset arrays for versioning and auditing."""
        hasher = hashlib.sha256()
        hasher.update(X.values.tobytes())
        hasher.update(y.values.tobytes())
        return hasher.hexdigest()

    @classmethod
    def build(
        cls,
        candles_df: pd.DataFrame,
        features_df: Optional[pd.DataFrame] = None,
        symbol: str = "BTC/USDT",
        timeframe: str = "15m",
        horizon: int = 12,
        target_return: float = 0.005,
        label_method: LabelMethodType = "peak",
        take_profit: float = 0.01,
        stop_loss: float = 0.005,
        tp_atr_mult: float = 2.0,
        sl_atr_mult: float = 1.0,
        feature_subset: Optional[List[str]] = None,
        drop_na: bool = True,
    ) -> Dataset:
        """Construct an aligned Dataset instance from candles and features.

        Args:
            candles_df: Raw OHLCV DataFrame with DatetimeIndex.
            features_df: Precomputed features DataFrame (optional, calculated if None).
            symbol: Trading pair.
            timeframe: Candle interval.
            horizon: Prediction horizon in bars forward.
            target_return: Return threshold for positive label.
            label_method: 'peak', 'return', 'triple_barrier', or 'dynamic_atr_tb'.
            take_profit: Take profit threshold for triple_barrier.
            stop_loss: Stop loss threshold for triple_barrier.
            tp_atr_mult: Take profit ATR multiplier for dynamic_atr_tb.
            sl_atr_mult: Stop loss ATR multiplier for dynamic_atr_tb.
            feature_subset: Optional subset of feature column names to retain.
            drop_na: Whether to drop rows containing NaNs (warmup + future horizon).

        Returns:
            Dataset instance.
        """
        if candles_df.empty:
            raise ValueError("Cannot build dataset from empty candles DataFrame.")

        # 1. Compute features if not supplied
        if features_df is None:
            features_df = FeaturePipeline.compute_features(candles_df, drop_na=False)

        if feature_subset is not None:
            avail_cols = [c for c in feature_subset if c in features_df.columns]
            if avail_cols:
                features_df = features_df[avail_cols]

        # 2. Compute labels
        if label_method == "peak":
            y = LabelGenerator.compute_future_peak_labels(
                high=candles_df["high"],
                close=candles_df["close"],
                horizon=horizon,
                target_return=target_return,
            )
        elif label_method == "return":
            y = LabelGenerator.compute_future_return_labels(
                close=candles_df["close"],
                horizon=horizon,
                target_return=target_return,
            )
        elif label_method == "triple_barrier":
            y = LabelGenerator.compute_triple_barrier_labels(
                high=candles_df["high"],
                low=candles_df["low"],
                close=candles_df["close"],
                horizon=horizon,
                take_profit=take_profit,
                stop_loss=stop_loss,
            )
        elif label_method == "dynamic_atr_tb":
            atr = FeaturePipeline.compute_features(candles_df, drop_na=False).get("atr_14")
            if atr is None:
                from app.features.indicators import compute_atr
                atr = compute_atr(candles_df["high"], candles_df["low"], candles_df["close"], period=14)
            y = LabelGenerator.compute_dynamic_atr_triple_barrier_labels(
                high=candles_df["high"],
                low=candles_df["low"],
                close=candles_df["close"],
                atr=atr,
                horizon=horizon,
                tp_atr_mult=tp_atr_mult,
                sl_atr_mult=sl_atr_mult,
            )
        else:
            raise ValueError(f"Unknown label_method: {label_method}")

        # 3. Align features X and target y on common index
        common_idx = features_df.index.intersection(y.index)
        X_aligned = features_df.loc[common_idx]
        y_aligned = y.loc[common_idx]

        if drop_na:
            valid_mask = (~X_aligned.isnull().any(axis=1)) & (~y_aligned.isnull())
            X_clean = X_aligned[valid_mask]
            y_clean = y_aligned[valid_mask].astype(int)
        else:
            X_clean = X_aligned
            y_clean = y_aligned

        if X_clean.empty:
            raise ValueError("Dataset is empty after dropping NaN values. Increase input history length.")

        # 4. Metadata and hashing
        dataset_hash = cls._compute_dataset_hash(X_clean, y_clean)
        class_dist = y_clean.value_counts().to_dict()

        return Dataset(
            X=X_clean,
            y=y_clean,
            symbol=symbol,
            timeframe=timeframe,
            target_name=y_clean.name,
            feature_names=list(X_clean.columns),
            dataset_hash=dataset_hash,
            start_date=X_clean.index.min().to_pydatetime(),
            end_date=X_clean.index.max().to_pydatetime(),
            total_samples=len(X_clean),
            class_distribution=class_dist,
        )

    @staticmethod
    def split_dataset(
        dataset: Dataset,
        train_pct: float = 0.70,
        val_pct: float = 0.15,
        test_pct: float = 0.15,
    ) -> DatasetSplits:
        """Split a Dataset into chronological Train, Validation, and Test sets."""
        (X_train, y_train), (X_val, y_val), (X_test, y_test) = TemporalSplitter.split_arrays(
            dataset.X, dataset.y, train_pct=train_pct, val_pct=val_pct, test_pct=test_pct
        )

        return DatasetSplits(
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            X_test=X_test,
            y_test=y_test,
            dataset=dataset,
        )
