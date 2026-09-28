"""Label generation algorithms for financial machine learning classification."""

import logging
from typing import Optional
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class LabelGenerator:
    """Generates binary and multi-class target labels for time-series forecasting."""

    @staticmethod
    def compute_future_return_labels(
        close: pd.Series,
        horizon: int = 12,
        target_return: float = 0.005,
    ) -> pd.Series:
        """Compute binary label based on exact price return at horizon bar t + H.

        Formula:
            FutureReturn = (Close_{t+H} - Close_t) / Close_t
            Label = 1 if FutureReturn >= target_return else 0

        The last `horizon` bars are assigned NaN since future outcome is not yet realized.

        Args:
            close: Price series with DatetimeIndex.
            horizon: Number of bars into the future (default: 12).
            target_return: Minimum return threshold (e.g. 0.005 = 0.5%).

        Returns:
            pd.Series with integer labels (0 or 1) and NaN for the last `horizon` bars.
        """
        if len(close) < horizon + 1:
            return pd.Series(np.nan, index=close.index, name=f"label_ret_{horizon}")

        # Shift close backwards by `horizon` bars (look into the future)
        future_close = close.shift(-horizon)
        future_return = (future_close - close) / close

        # Create binary label
        labels = (future_return >= target_return).astype(float)

        # Force last `horizon` elements to NaN (embargo/unrealized future)
        labels.iloc[-horizon:] = np.nan

        return labels.rename(f"label_ret_h{horizon}_t{int(target_return * 10000)}bps")

    @staticmethod
    def compute_future_peak_labels(
        high: pd.Series,
        close: pd.Series,
        horizon: int = 12,
        target_return: float = 0.005,
    ) -> pd.Series:
        """Compute binary label based on maximum price appreciation within window [t+1 .. t+H].

        Formula:
            MaxHigh = max(High_{t+1}, High_{t+2}, ..., High_{t+H})
            MaxReturn = (MaxHigh - Close_t) / Close_t
            Label = 1 if MaxReturn >= target_return else 0

        Args:
            high: High price series.
            close: Close price series.
            horizon: Number of bars into the future (default: 12).
            target_return: Minimum peak return threshold.

        Returns:
            pd.Series of binary labels (0 or 1) with NaNs at the end.
        """
        n = len(close)
        if n < horizon + 1:
            return pd.Series(np.nan, index=close.index, name=f"label_peak_{horizon}")

        high_vals = high.values
        close_vals = close.values
        labels = np.full(n, np.nan)

        for i in range(n - horizon):
            window_max_high = np.max(high_vals[i + 1 : i + 1 + horizon])
            max_return = (window_max_high - close_vals[i]) / close_vals[i]
            labels[i] = 1.0 if max_return >= target_return else 0.0

        return pd.Series(
            labels,
            index=close.index,
            name=f"label_peak_h{horizon}_t{int(target_return * 10000)}bps",
        )

    @staticmethod
    def compute_triple_barrier_labels(
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        horizon: int = 12,
        take_profit: float = 0.01,
        stop_loss: float = 0.005,
    ) -> pd.Series:
        """Compute triple-barrier label (Take Profit vs Stop Loss vs Time Limit).

        Label = 1 if Take Profit is touched first.
        Label = 0 if Stop Loss is touched first or time limit expires without touching TP.

        Args:
            high: High price series.
            low: Low price series.
            close: Close price series.
            horizon: Max holding period in bars.
            take_profit: Take profit percentage (e.g. 0.01 = +1.0%).
            stop_loss: Stop loss percentage (e.g. 0.005 = -0.5%).

        Returns:
            pd.Series of binary labels.
        """
        n = len(close)
        if n < horizon + 1:
            return pd.Series(np.nan, index=close.index, name="label_tb")

        high_vals = high.values
        low_vals = low.values
        close_vals = close.values
        labels = np.full(n, np.nan)

        for i in range(n - horizon):
            entry_price = close_vals[i]
            tp_price = entry_price * (1.0 + take_profit)
            sl_price = entry_price * (1.0 - stop_loss)

            label = 0.0
            for step in range(1, horizon + 1):
                idx = i + step
                h = high_vals[idx]
                l = low_vals[idx]

                # Check if TP was touched
                if h >= tp_price:
                    label = 1.0
                    break
                # Check if SL was touched
                if l <= sl_price:
                    label = 0.0
                    break

            labels[i] = label

        return pd.Series(
            labels,
            index=close.index,
            name=f"label_tb_h{horizon}_tp{int(take_profit*10000)}_sl{int(stop_loss*10000)}",
        )

    @staticmethod
    def compute_dynamic_atr_triple_barrier_labels(
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        atr: pd.Series,
        horizon: int = 12,
        tp_atr_mult: float = 2.0,
        sl_atr_mult: float = 1.0,
    ) -> pd.Series:
        """Compute dynamic volatility-adjusted triple-barrier labels using instantaneous ATR.

        Upper Barrier (Take Profit): Entry + (tp_atr_mult * ATR_t)
        Lower Barrier (Stop Loss):   Entry - (sl_atr_mult * ATR_t)

        Label = 1.0 if Upper Barrier is touched first within horizon.
        Label = 0.0 if Lower Barrier is touched first or time expires.

        Args:
            high: High price series.
            low: Low price series.
            close: Close price series.
            atr: Instantaneous backward-looking ATR series.
            horizon: Maximum holding bars.
            tp_atr_mult: Multiple of ATR for Take Profit (e.g. 2.0).
            sl_atr_mult: Multiple of ATR for Stop Loss (e.g. 1.0).

        Returns:
            pd.Series of binary labels with NaNs at the end.
        """
        n = len(close)
        if n < horizon + 1:
            return pd.Series(np.nan, index=close.index, name="label_dyn_atr_tb")

        high_vals = high.values
        low_vals = low.values
        close_vals = close.values
        atr_vals = atr.values
        labels = np.full(n, np.nan)

        for i in range(n - horizon):
            entry_price = close_vals[i]
            cur_atr = atr_vals[i]
            if np.isnan(cur_atr) or cur_atr <= 0:
                continue

            tp_price = entry_price + (tp_atr_mult * cur_atr)
            sl_price = entry_price - (sl_atr_mult * cur_atr)

            label = 0.0
            for step in range(1, horizon + 1):
                idx = i + step
                h = high_vals[idx]
                l = low_vals[idx]

                if h >= tp_price:
                    label = 1.0
                    break
                if l <= sl_price:
                    label = 0.0
                    break

            labels[i] = label

        return pd.Series(
            labels,
            index=close.index,
            name=f"label_dyn_atr_h{horizon}_tp{tp_atr_mult:.1f}x_sl{sl_atr_mult:.1f}x",
        )

