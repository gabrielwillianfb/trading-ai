"""Comprehensive tests for feature engineering, mathematical indicators, and anti-lookahead guarantees."""

from datetime import datetime, timedelta, timezone
import numpy as np
import pandas as pd
import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from app.features.indicators import (
    compute_atr,
    compute_bollinger_bands,
    compute_ema,
    compute_ema_distances,
    compute_macd,
    compute_returns,
    compute_rsi,
    compute_sma,
    compute_volatility,
    compute_volume_features,
)
from app.features.pipeline import FeaturePipeline
from app.features.repository import FeatureRepository


def generate_synthetic_ohlcv(
    n_bars: int = 150,
    start_price: float = 50000.0,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate synthetic realistic random-walk OHLCV candlestick DataFrame."""
    np.random.seed(seed)
    returns = np.random.normal(loc=0.0002, scale=0.008, size=n_bars)
    price_path = start_price * np.cumprod(1 + returns)

    dates = pd.date_range("2026-01-01", periods=n_bars, freq="15min", tz="UTC")

    opens = price_path
    closes = price_path * (1 + np.random.normal(0, 0.002, size=n_bars))
    highs = np.maximum(opens, closes) * (1 + np.abs(np.random.normal(0, 0.003, size=n_bars)))
    lows = np.minimum(opens, closes) * (1 - np.abs(np.random.normal(0, 0.003, size=n_bars)))
    volumes = np.random.uniform(5.0, 50.0, size=n_bars)

    df = pd.DataFrame(
        {
            "open": opens,
            "high": highs,
            "low": lows,
            "close": closes,
            "volume": volumes,
        },
        index=dates,
    )
    return df


# ------------------------------------------------------------------------------
# 1. Mathematical Indicator Unit Tests
# ------------------------------------------------------------------------------

def test_compute_returns():
    """Verify multi-period returns calculation."""
    prices = pd.Series([100.0, 105.0, 110.0, 108.0, 120.0])
    returns_df = compute_returns(prices, periods=[1, 2])

    assert np.isclose(returns_df["return_1"].iloc[1], 0.05)  # (105 - 100) / 100
    assert np.isclose(returns_df["return_1"].iloc[2], 110.0 / 105.0 - 1.0)
    assert np.isclose(returns_df["return_2"].iloc[2], 0.10)  # (110 - 100) / 100


def test_compute_rsi_bounds():
    """Verify RSI produces bounded outputs between 0 and 100."""
    df = generate_synthetic_ohlcv(100)
    rsi = compute_rsi(df["close"], period=14)

    valid_rsi = rsi.dropna()
    assert len(valid_rsi) > 0
    assert (valid_rsi >= 0.0).all()
    assert (valid_rsi <= 100.0).all()


def test_compute_macd_consistency():
    """Verify MACD histogram equality: macd_hist = macd - macd_signal."""
    df = generate_synthetic_ohlcv(100)
    macd_df = compute_macd(df["close"], fast=12, slow=26, signal=9)

    valid = macd_df.dropna()
    diff = valid["macd"] - valid["macd_signal"]
    assert np.allclose(valid["macd_hist"], diff)


def test_compute_bollinger_bands_ordering():
    """Verify Upper Band >= Middle Band >= Lower Band."""
    df = generate_synthetic_ohlcv(100)
    bb = compute_bollinger_bands(df["close"], window=20, num_std=2.0).dropna()

    assert (bb["bb_upper"] >= bb["bb_middle"]).all()
    assert (bb["bb_middle"] >= bb["bb_lower"]).all()
    assert (bb["bb_width"] >= 0.0).all()


def test_compute_atr_positive():
    """Verify ATR values are strictly positive."""
    df = generate_synthetic_ohlcv(100)
    atr = compute_atr(df["high"], df["low"], df["close"], period=14).dropna()

    assert (atr > 0.0).all()


def test_compute_volume_features():
    """Verify volume ratio math."""
    vol = pd.Series([10.0] * 30)
    vol_feat = compute_volume_features(vol, window=20).dropna()
    assert np.allclose(vol_feat["volume_ratio"], 1.0)


# ------------------------------------------------------------------------------
# 2. CRITICAL ANTI-LOOK-AHEAD BIAS TESTS
# ------------------------------------------------------------------------------

def test_zero_look_ahead_bias_window_expansion():
    """CRITICAL TEST 1: Verify that features at timestamp T are identical

    whether computed on data up to T or on a larger dataset up to T + N.
    """
    df_full = generate_synthetic_ohlcv(n_bars=120, seed=123)
    target_cutoff = 70  # Point-in-time T

    df_past = df_full.iloc[: target_cutoff + 1]  # Data up to T only

    # Compute features on past dataset
    features_past = FeaturePipeline.compute_features(df_past, drop_na=False)

    # Compute features on full dataset (including future bars)
    features_full = FeaturePipeline.compute_features(df_full, drop_na=False)

    # The feature vector at the target_cutoff timestamp must be IDENTICAL in both
    past_row = features_past.iloc[target_cutoff]
    full_row = features_full.iloc[target_cutoff]

    for col in FeaturePipeline.get_feature_names():
        val_past = past_row[col]
        val_full = full_row[col]
        if pd.isna(val_past):
            assert pd.isna(val_full), f"Lookahead mismatch on NaN for feature '{col}'"
        else:
            assert np.isclose(
                val_past, val_full, rtol=1e-8, atol=1e-8
            ), f"Lookahead bias detected on feature '{col}'! Past={val_past}, Full={val_full}"


def test_zero_look_ahead_bias_future_data_mutation():
    """CRITICAL TEST 2: Verify that mutating future candles (T+1 .. T+N)

    has ZERO effect on features computed for any timestamp t <= T.
    """
    df_original = generate_synthetic_ohlcv(n_bars=100, seed=999)
    features_orig = FeaturePipeline.compute_features(df_original, drop_na=False)

    # Mutate the future aggressively (at and after index 60)
    df_mutated = df_original.copy()
    df_mutated.iloc[60:, df_mutated.columns.get_loc("close")] *= 5.0
    df_mutated.iloc[60:, df_mutated.columns.get_loc("high")] *= 5.0
    df_mutated.iloc[60:, df_mutated.columns.get_loc("volume")] *= 10.0

    features_mutated = FeaturePipeline.compute_features(df_mutated, drop_na=False)

    # Verify that for all rows 0..59, features are 100% identical
    for idx in range(60):
        for col in FeaturePipeline.get_feature_names():
            val_orig = features_orig.iloc[idx][col]
            val_mut = features_mutated.iloc[idx][col]
            if pd.isna(val_orig):
                assert pd.isna(val_mut), f"Mutation affected NaNs at index {idx} in '{col}'"
            else:
                assert np.isclose(
                    val_orig, val_mut, rtol=1e-8, atol=1e-8
                ), f"Future mutation leaked backward into index {idx} for feature '{col}'!"


# ------------------------------------------------------------------------------
# 3. Pipeline & Repository Integration Tests
# ------------------------------------------------------------------------------

def test_feature_pipeline_output_structure():
    """Verify FeaturePipeline returns the expected schema and no NaNs when drop_na=True."""
    df = generate_synthetic_ohlcv(100)
    features_df = FeaturePipeline.compute_features(df, drop_na=True)

    expected_cols = FeaturePipeline.get_feature_names()
    assert list(features_df.columns) == expected_cols
    assert not features_df.isnull().any().any()
    assert len(features_df) > 0


@pytest.mark.asyncio
async def test_feature_repository_save_and_query(db_session: AsyncSession):
    """Verify persisting features to database and querying with point-in-time barrier."""
    df = generate_synthetic_ohlcv(80)
    features_df = FeaturePipeline.compute_features(df, drop_na=True)

    symbol = "BTC/USDT"
    timeframe = "15m"

    # 1. Save features
    saved = await FeatureRepository.save_features(features_df, symbol, timeframe, db_session)
    assert saved == len(features_df)

    # 2. Idempotent re-save
    saved_again = await FeatureRepository.save_features(features_df, symbol, timeframe, db_session)
    assert saved_again == len(features_df)

    # 3. Query all features
    queried_df = await FeatureRepository.get_features_dataframe(symbol, timeframe, session=db_session)
    assert len(queried_df) == len(features_df)
    assert "rsi_14" in queried_df.columns
    assert "macd" in queried_df.columns

    # 4. Query with as_of_time barrier (Point-in-time test)
    mid_index = len(features_df) // 2
    as_of = features_df.index[mid_index]

    as_of_df = await FeatureRepository.get_features_dataframe(
        symbol, timeframe, as_of_time=as_of, session=db_session
    )
    assert len(as_of_df) == mid_index + 1
    assert as_of_df.index.max() == as_of
