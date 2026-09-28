"""Market regime detection and trend confirmation indicators."""

from typing import Dict, Optional
import numpy as np
import pandas as pd


class MarketRegimeDetector:
    """Calculates macro trend regimes and volatility states to filter trading signals."""

    @staticmethod
    def compute_regime_features(df: pd.DataFrame) -> pd.DataFrame:
        """Compute regime identification features from OHLCV series.

        Features:
            - `regime_trend_score`: +1.0 (Strong Bull), 0.0 (Neutral), -1.0 (Strong Bear).
            - `regime_ema_ratio`: Ratio of EMA 50 to EMA 200.
            - `regime_price_to_ema200`: Percentage distance from close to EMA 200.
            - `regime_volatility_state`: 1 (High Volatility), 0 (Normal), -1 (Compression).
            - `regime_rsi_regime`: 1 (Bull zone > 50), -1 (Bear zone < 50).

        Args:
            df: OHLCV DataFrame with DatetimeIndex in UTC.

        Returns:
            DataFrame with regime features aligned to the original index.
        """
        close = df["close"].astype(float)
        high = df["high"].astype(float)
        low = df["low"].astype(float)

        ema_50 = close.ewm(span=50, adjust=False).mean()
        ema_200 = close.ewm(span=200, adjust=False).mean()

        # Trend Score: +1 if Close > EMA 200 and EMA 50 > EMA 200; -1 if below; 0 otherwise
        bull_condition = (close > ema_200) & (ema_50 > ema_200)
        bear_condition = (close < ema_200) & (ema_50 < ema_200)

        trend_score = np.where(bull_condition, 1.0, np.where(bear_condition, -1.0, 0.0))
        ema_ratio = (ema_50 / ema_200) - 1.0
        price_to_ema200 = (close / ema_200) - 1.0

        # Volatility regime using rolling ATR normalized by close
        prev_close = close.shift(1)
        tr = pd.concat(
            [
                high - low,
                (high - prev_close).abs(),
                (low - prev_close).abs(),
            ],
            axis=1,
        ).max(axis=1)
        atr_14 = tr.rolling(window=14, min_periods=5).mean()
        norm_atr = (atr_14 / close) * 100.0

        atr_q75 = norm_atr.rolling(window=200, min_periods=5).quantile(0.75)
        atr_q25 = norm_atr.rolling(window=200, min_periods=5).quantile(0.25)

        vol_state = np.where(norm_atr > atr_q75, 1.0, np.where(norm_atr < atr_q25, -1.0, 0.0))

        # RSI calculation for regime momentum
        delta = close.diff()
        gain = delta.clip(lower=0.0)
        loss = -delta.clip(upper=0.0)
        avg_gain = gain.rolling(window=14, min_periods=5).mean()
        avg_loss = loss.rolling(window=14, min_periods=5).mean()
        rs = avg_gain / avg_loss.replace(0.0, np.nan)
        rsi = 100.0 - (100.0 / (1.0 + rs))
        rsi_regime = np.where(rsi >= 50.0, 1.0, -1.0)

        return pd.DataFrame(
            {
                "regime_trend_score": trend_score,
                "regime_ema_ratio": ema_ratio,
                "regime_price_to_ema200": price_to_ema200,
                "regime_volatility_state": vol_state,
                "regime_rsi_regime": rsi_regime,
            },
            index=df.index,
        )
