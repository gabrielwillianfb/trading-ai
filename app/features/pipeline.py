"""Feature engineering pipeline for transforming raw OHLCV candlestick data into ML features."""

import logging
from typing import List, Optional
import numpy as np
import pandas as pd

from app.features.indicators import (
    compute_adx_dmi,
    compute_atr,
    compute_bollinger_bands,
    compute_cci,
    compute_chop_index,
    compute_cmf,
    compute_donchian_channels,
    compute_ema,
    compute_ema_distances,
    compute_linear_reg_slope,
    compute_macd,
    compute_mfi,
    compute_obv_slope,
    compute_parkinson_volatility,
    compute_returns,
    compute_rsi,
    compute_stochastic,
    compute_volatility,
    compute_volume_features,
    compute_williams_r,
)
from app.features.regime import MarketRegimeDetector

logger = logging.getLogger(__name__)


class FeaturePipeline:
    """Standardized quantitative feature extraction pipeline with regime awareness."""

    FEATURE_NAMES: List[str] = [
        "return_1",
        "return_5",
        "return_15",
        "return_30",
        "ema_9",
        "ema_20",
        "ema_50",
        "ema_200",
        "dist_ema_9",
        "dist_ema_20",
        "dist_ema_50",
        "dist_ema_200",
        "rsi_14",
        "rsi_7",
        "rsi_21",
        "stoch_k",
        "stoch_d",
        "cci_20",
        "williams_r_14",
        "macd",
        "macd_signal",
        "macd_hist",
        "adx_14",
        "dmi_plus_14",
        "dmi_minus_14",
        "atr_14",
        "atr_pct_14",
        "bb_upper",
        "bb_middle",
        "bb_lower",
        "bb_width",
        "bb_percent",
        "donchian_width_20",
        "donchian_pos_20",
        "volatility_20",
        "parkinson_vol_20",
        "volume_sma_20",
        "volume_ratio",
        "mfi_14",
        "cmf_20",
        "obv_slope_10",
        "hl_spread",
        "oc_spread",
        "linear_slope_14",
        "chop_index_14",
        "regime_trend_score",
        "regime_ema_ratio",
        "regime_price_to_ema200",
        "regime_volatility_state",
        "regime_rsi_regime",
    ]

    @classmethod
    def get_feature_names(cls) -> List[str]:
        """Return the canonical ordered list of feature names produced by this pipeline."""
        return cls.FEATURE_NAMES.copy()

    @classmethod
    def compute_features(
        cls,
        df: pd.DataFrame,
        drop_na: bool = True,
        return_periods: Optional[List[int]] = None,
    ) -> pd.DataFrame:
        """Transform an OHLCV DataFrame into an engineered feature matrix.

        All calculations are strictly backward-looking to guarantee ZERO look-ahead bias.

        Args:
            df: DataFrame containing columns 'open', 'high', 'low', 'close', 'volume'
                with a DatetimeIndex in UTC.
            drop_na: Whether to drop initial warmup rows containing NaNs.
            return_periods: Lag periods for returns (default: [1, 5, 15, 30]).

        Returns:
            pd.DataFrame indexed by timestamp containing all calculated feature columns.
        """
        if df.empty:
            return pd.DataFrame(columns=cls.FEATURE_NAMES)

        required_cols = {"open", "high", "low", "close", "volume"}
        missing_cols = required_cols - set(df.columns)
        if missing_cols:
            raise ValueError(f"Input DataFrame is missing required OHLCV columns: {missing_cols}")

        close = df["close"].astype(float)
        high = df["high"].astype(float)
        low = df["low"].astype(float)
        open_ = df["open"].astype(float)
        volume = df["volume"].astype(float)

        feature_frames: List[pd.DataFrame] = []

        # 1. Multi-period Returns
        p_returns = return_periods or [1, 5, 15, 30]
        returns_df = compute_returns(close, periods=p_returns)
        feature_frames.append(returns_df)

        # 2. Exponential Moving Averages (EMAs)
        ema_spans = [9, 20, 50, 200]
        ema_dict = {}
        for span in ema_spans:
            ema_series = compute_ema(close, span=span).rename(f"ema_{span}")
            ema_dict[f"ema_{span}"] = ema_series

        ema_df = pd.DataFrame(ema_dict, index=df.index)
        feature_frames.append(ema_df)

        # 3. Distance from Close to EMAs
        dist_df = compute_ema_distances(close, ema_dict)
        feature_frames.append(dist_df)

        # 4. Multi-scale RSI
        rsi_14 = compute_rsi(close, period=14).to_frame("rsi_14")
        rsi_7 = compute_rsi(close, period=7).to_frame("rsi_7")
        rsi_21 = compute_rsi(close, period=21).to_frame("rsi_21")
        feature_frames.extend([rsi_14, rsi_7, rsi_21])

        # 5. Stochastic Oscillator
        stoch_df = compute_stochastic(high=high, low=low, close=close, k_period=14, d_period=3)
        feature_frames.append(stoch_df)

        # 6. CCI & Williams %R
        cci_20 = compute_cci(high=high, low=low, close=close, period=20).to_frame("cci_20")
        williams_14 = compute_williams_r(high=high, low=low, close=close, period=14).to_frame("williams_r_14")
        feature_frames.extend([cci_20, williams_14])

        # 7. Trend/Momentum: MACD & ADX/DMI
        macd_df = compute_macd(close, fast=12, slow=26, signal=9)
        adx_df = compute_adx_dmi(high=high, low=low, close=close, period=14)
        feature_frames.extend([macd_df, adx_df])

        # 8. Volatility: ATR, ATR%, Bollinger Bands, Donchian Channels
        atr_14_series = compute_atr(high=high, low=low, close=close, period=14)
        atr_14 = atr_14_series.to_frame("atr_14")
        atr_pct_14 = ((atr_14_series / close) * 100.0).rename("atr_pct_14").to_frame("atr_pct_14")
        bb_df = compute_bollinger_bands(close, window=20, num_std=2.0)
        donchian_df = compute_donchian_channels(high=high, low=low, close=close, window=20)
        feature_frames.extend([atr_14, atr_pct_14, bb_df, donchian_df])

        # 9. Realized Volatility & Parkinson Extreme Value Volatility
        vol_df = compute_volatility(close, window=20).to_frame("volatility_20")
        p_vol_df = compute_parkinson_volatility(high=high, low=low, window=20).to_frame("parkinson_vol_20")
        feature_frames.extend([vol_df, p_vol_df])

        # 10. Volume Flow: SMA, Ratio, MFI, CMF, OBV Slope
        vol_features_df = compute_volume_features(volume, window=20)
        mfi_14 = compute_mfi(high=high, low=low, close=close, volume=volume, period=14).to_frame("mfi_14")
        cmf_20 = compute_cmf(high=high, low=low, close=close, volume=volume, period=20).to_frame("cmf_20")
        obv_slope_10 = compute_obv_slope(close=close, volume=volume, window=10).to_frame("obv_slope_10")
        feature_frames.extend([vol_features_df, mfi_14, cmf_20, obv_slope_10])

        # 11. Intra-bar Price Action Spreads & Linear Regression Slope
        hl_spread = ((high - low) / close).rename("hl_spread")
        oc_spread = ((close - open_) / open_).rename("oc_spread")
        linear_slope = compute_linear_reg_slope(close, window=14).to_frame("linear_slope_14")
        chop_idx = compute_chop_index(high=high, low=low, close=close, window=14).to_frame("chop_index_14")
        spreads_df = pd.DataFrame({"hl_spread": hl_spread, "oc_spread": oc_spread}, index=df.index)
        feature_frames.extend([spreads_df, linear_slope, chop_idx])

        # 12. Market Regime & Trend Confirmation
        regime_df = MarketRegimeDetector.compute_regime_features(df)
        feature_frames.append(regime_df)

        # Combine all features into single DataFrame
        features_df = pd.concat(feature_frames, axis=1)

        # Reorder to canonical column order
        canonical_cols = [c for c in cls.FEATURE_NAMES if c in features_df.columns]
        features_df = features_df[canonical_cols]

        if drop_na:
            features_df = features_df.dropna()

        return features_df
