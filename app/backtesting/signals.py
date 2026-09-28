"""Trading signal generation from model probabilities and quantitative rules."""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional
import pandas as pd

from app.models.inference.predictor import PredictionResult


class SignalAction(str, Enum):
    """Trading action directive."""

    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"


@dataclass
class TradingSignal:
    """Discrete trading order recommendation generated at the close of candle t."""

    timestamp: datetime
    symbol: str
    action: SignalAction
    probability: float
    reason: str
    metadata: Dict[str, Any] = field(default_factory=dict)


class SignalGenerator:
    """Generates discrete actionable trading signals from model probability outputs."""

    @staticmethod
    def generate_signal_from_prediction(
        prediction: PredictionResult,
        buy_threshold: float = 0.60,
        sell_threshold: float = 0.40,
        require_trend_filter: bool = False,
        require_chop_filter: bool = False,
        max_entry_rsi: Optional[float] = None,
    ) -> TradingSignal:
        """Map a single model prediction probability to a BUY, SELL, or HOLD signal.

        Args:
            prediction: PredictionResult instance with probability_up.
            buy_threshold: Minimum probability_up required to trigger BUY (default: 0.60).
            sell_threshold: Maximum probability_up before triggering SELL / Exit (default: 0.40).
            require_trend_filter: If True, only BUY if regime_trend_score >= 0 or price above EMA50.
            require_chop_filter: If True, only BUY if chop_index <= 61.8 (not choppy consolidation).
            max_entry_rsi: Optional maximum RSI limit for BUY (e.g. 70.0 to prevent buying top of spike).

        Returns:
            TradingSignal instance.
        """
        prob_up = prediction.probability_up
        raw = prediction.raw_features or {}

        # Check Quantitative Filters
        filter_failed = False
        filter_reason = ""

        if require_trend_filter:
            trend_score = raw.get("regime_trend_score", 0.0)
            dist_ema50 = raw.get("dist_ema_50", 0.0)
            if trend_score < 0 and dist_ema50 < -0.005:
                filter_failed = True
                filter_reason = f"Trend filter active: Bearish regime ({trend_score:.1f}, dist={dist_ema50:.3f})"

        if not filter_failed and require_chop_filter:
            chop = raw.get("chop_index_14", 50.0)
            if chop > 61.8:
                filter_failed = True
                filter_reason = f"Chop filter active: Market is choppy ({chop:.1f} > 61.8)"

        if not filter_failed and max_entry_rsi is not None:
            rsi = raw.get("rsi_14", 50.0)
            if rsi > max_entry_rsi:
                filter_failed = True
                filter_reason = f"RSI filter active: Overbought ({rsi:.1f} > {max_entry_rsi:.1f})"

        if prob_up >= buy_threshold and not filter_failed:
            action = SignalAction.BUY
            reason = f"Probability UP ({prob_up:.3f}) >= Buy Threshold ({buy_threshold:.3f})"
        elif prob_up >= buy_threshold and filter_failed:
            action = SignalAction.HOLD
            reason = f"BUY rejected by filter: {filter_reason}"
        elif prob_up <= sell_threshold:
            action = SignalAction.SELL
            reason = f"Probability UP ({prob_up:.3f}) <= Sell Threshold ({sell_threshold:.3f})"
        else:
            action = SignalAction.HOLD
            reason = f"Probability UP ({prob_up:.3f}) in neutral zone [{sell_threshold:.3f}, {buy_threshold:.3f}]"

        return TradingSignal(
            timestamp=prediction.timestamp,
            symbol=prediction.symbol,
            action=action,
            probability=prob_up,
            reason=reason,
            metadata={
                "model_version": prediction.model_version,
                "horizon": prediction.horizon,
                "probability_down": prediction.probability_down,
            },
        )

    @classmethod
    def generate_signals(
        cls,
        predictions: List[PredictionResult],
        buy_threshold: float = 0.60,
        sell_threshold: float = 0.40,
        require_trend_filter: bool = False,
        require_chop_filter: bool = False,
        max_entry_rsi: Optional[float] = None,
    ) -> List[TradingSignal]:
        """Convert a list of predictions into a chronological series of TradingSignals."""
        return [
            cls.generate_signal_from_prediction(
                p,
                buy_threshold=buy_threshold,
                sell_threshold=sell_threshold,
                require_trend_filter=require_trend_filter,
                require_chop_filter=require_chop_filter,
                max_entry_rsi=max_entry_rsi,
            )
            for p in predictions
        ]
