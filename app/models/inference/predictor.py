"""Model inference engine producing continuous probability predictions and tracking audits."""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd
import xgboost as xgb
from sqlalchemy.ext.asyncio import AsyncSession

from app.data.storage.models import Prediction

logger = logging.getLogger(__name__)


@dataclass
class PredictionResult:
    """Standardized output structure for model probability forecasts."""

    timestamp: datetime
    symbol: str
    timeframe: str
    probability_up: float
    probability_down: float
    predicted_class: int
    model_version: str
    model_version_id: Optional[int] = None
    horizon: int = 12
    raw_features: Optional[Dict[str, Any]] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Format prediction as JSON-compatible dictionary."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "symbol": self.symbol,
            "timeframe": self.timeframe,
            "probability_up": round(self.probability_up, 4),
            "probability_down": round(self.probability_down, 4),
            "predicted_class": self.predicted_class,
            "model_version": self.model_version,
            "model_version_id": self.model_version_id,
            "horizon": self.horizon,
        }


class ModelPredictor:
    """Executes probability forecasting and validates feature alignment for live and batch inference."""

    def __init__(
        self,
        model: Any,
        feature_names: List[str],
        model_version: str,
        model_version_id: Optional[int] = None,
        horizon: int = 12,
    ):
        self.model = model
        self.feature_names = feature_names
        self.model_version = model_version
        self.model_version_id = model_version_id
        self.horizon = horizon

    def _align_feature_row(self, features: Dict[str, float]) -> pd.DataFrame:
        """Ensure input feature dictionary matches the expected feature column order."""
        missing = set(self.feature_names) - set(features.keys())
        if missing:
            raise ValueError(f"Input features missing required model columns: {missing}")

        row = [float(features[name]) for name in self.feature_names]
        return pd.DataFrame([row], columns=self.feature_names)

    def predict_single(
        self,
        features: Dict[str, float],
        timestamp: Optional[datetime] = None,
        symbol: str = "BTC/USDT",
        timeframe: str = "15m",
    ) -> PredictionResult:
        """Generate continuous probability predictions for a single bar feature snapshot.

        Args:
            features: Dictionary mapping feature names to numerical values.
            timestamp: Timestamp of the bar (defaults to UTC now).
            symbol: Trading pair.
            timeframe: Candle interval.

        Returns:
            PredictionResult instance with probability_up and probability_down.
        """
        ts = timestamp or datetime.now(timezone.utc)
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)

        X = self._align_feature_row(features)
        proba = self.model.predict_proba(X)[0]

        # Probability for Class 0 (Down/Neutral) and Class 1 (Up)
        prob_down = float(proba[0])
        prob_up = float(proba[1])
        predicted_class = 1 if prob_up >= 0.50 else 0

        return PredictionResult(
            timestamp=ts,
            symbol=symbol,
            timeframe=timeframe,
            probability_up=prob_up,
            probability_down=prob_down,
            predicted_class=predicted_class,
            model_version=self.model_version,
            model_version_id=self.model_version_id,
            horizon=self.horizon,
            raw_features=features,
        )

    def predict_batch(
        self,
        features_df: pd.DataFrame,
        symbol: str = "BTC/USDT",
        timeframe: str = "15m",
    ) -> List[PredictionResult]:
        """Execute probability predictions over a historical or multi-row DataFrame.

        Args:
            features_df: DataFrame with feature columns and DatetimeIndex.
            symbol: Trading pair.
            timeframe: Candle interval.

        Returns:
            List of PredictionResult instances.
        """
        if features_df.empty:
            return []

        # Validate columns
        missing = set(self.feature_names) - set(features_df.columns)
        if missing:
            raise ValueError(f"Features DataFrame missing required model columns: {missing}")

        X = features_df[self.feature_names]
        probas = self.model.predict_proba(X)

        results: List[PredictionResult] = []
        for i, (ts, row) in enumerate(features_df.iterrows()):
            dt = ts.to_pydatetime() if hasattr(ts, "to_pydatetime") else ts
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)

            p_down = float(probas[i, 0])
            p_up = float(probas[i, 1])
            pred_cls = 1 if p_up >= 0.50 else 0

            results.append(
                PredictionResult(
                    timestamp=dt,
                    symbol=symbol,
                    timeframe=timeframe,
                    probability_up=p_up,
                    probability_down=p_down,
                    predicted_class=pred_cls,
                    model_version=self.model_version,
                    model_version_id=self.model_version_id,
                    horizon=self.horizon,
                    raw_features=row.to_dict(),
                )
            )

        return results

    @staticmethod
    async def save_predictions(
        predictions: List[PredictionResult],
        session: AsyncSession,
    ) -> int:
        """Persist prediction audit records into the predictions table in database.

        Args:
            predictions: List of PredictionResult instances.
            session: Active AsyncSession.

        Returns:
            Number of prediction records saved.
        """
        if not predictions:
            return 0

        db_records = []
        for p in predictions:
            if p.model_version_id is None:
                logger.warning(
                    f"Prediction for {p.symbol} at {p.timestamp} has no model_version_id. Skipping DB save."
                )
                continue

            db_records.append(
                Prediction(
                    timestamp=p.timestamp,
                    symbol=p.symbol,
                    timeframe=p.timeframe,
                    model_version_id=p.model_version_id,
                    probability_up=p.probability_up,
                    probability_down=p.probability_down,
                    predicted_class=p.predicted_class,
                    horizon=p.horizon,
                    raw_features=p.raw_features,
                )
            )

        if not db_records:
            return 0

        session.add_all(db_records)
        await session.commit()
        logger.info(f"Recorded {len(db_records)} predictions into 'predictions' table.")
        return len(db_records)
