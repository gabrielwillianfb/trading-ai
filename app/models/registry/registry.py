"""Model Registry for versioning, artifact serialization, and deployment governance."""

import logging
import os
import re
from pathlib import Path
from typing import List, Optional
import xgboost as xgb
from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.data.storage.models import ModelVersion
from app.models.training.trainer import TrainingResult

logger = logging.getLogger(__name__)


class ModelRegistry:
    """Manages model artifacts, version metadata, and active deployment flags."""

    @staticmethod
    async def get_next_version(session: AsyncSession) -> str:
        """Generate the next sequential model version string (e.g. 'model_v001', 'model_v002')."""
        query = select(ModelVersion.version).order_by(ModelVersion.id.desc())
        result = await session.execute(query)
        versions = result.scalars().all()

        max_num = 0
        for v in versions:
            match = re.search(r"model_v(\d+)", v)
            if match:
                num = int(match.group(1))
                if num > max_num:
                    max_num = num

        next_num = max_num + 1
        return f"model_v{next_num:03d}"

    @classmethod
    async def register_model(
        cls,
        training_result: TrainingResult,
        session: AsyncSession,
        version: Optional[str] = None,
        symbol: Optional[str] = None,
        timeframe: Optional[str] = None,
        set_active: bool = False,
        model_dir: str = "models",
    ) -> ModelVersion:
        """Save model binary artifact to disk and persist full metadata to database.

        Args:
            training_result: Output from ModelTrainer.
            session: Active AsyncSession.
            version: Optional custom version tag (e.g. 'model_v001'). Auto-generated if None.
            symbol: Target trading pair (e.g. 'BTC/USDT').
            timeframe: Target candle interval (e.g. '15m').
            set_active: Whether to mark this model as the active inference model.
            model_dir: Directory path for artifact storage.

        Returns:
            Created ModelVersion database record.
        """
        # Ensure model storage directory exists
        os.makedirs(model_dir, exist_ok=True)

        if not version:
            version = await cls.get_next_version(session)

        # Select file format based on model type
        if training_result.model_type == "xgboost_classifier" and hasattr(training_result.model, "save_model"):
            artifact_filename = f"{version}.ubj"
            artifact_path = str(Path(model_dir) / artifact_filename)
            logger.info(f"Serializing XGBoost model artifact to {artifact_path}...")
            training_result.model.save_model(artifact_path)
        else:
            import joblib
            artifact_filename = f"{version}.joblib"
            artifact_path = str(Path(model_dir) / artifact_filename)
            logger.info(f"Serializing {training_result.model_type} model artifact via joblib to {artifact_path}...")
            joblib.dump(training_result.model, artifact_path)

        # Create database record
        model_record = ModelVersion(
            version=version,
            model_type=training_result.model_type,
            training_start_date=training_result.start_date,
            training_end_date=training_result.end_date,
            feature_names=training_result.feature_names,
            hyperparameters=training_result.hyperparameters,
            validation_metrics=training_result.validation_metrics,
            test_metrics=training_result.test_metrics,
            dataset_hash=training_result.dataset_hash,
            file_path=artifact_path,
            is_active=False,
        )
        session.add(model_record)
        await session.commit()
        await session.refresh(model_record)

        if set_active:
            await cls.set_active_model(version, session)
            await session.refresh(model_record)

        logger.info(
            f"Successfully registered {version} (Active: {model_record.is_active}) with {len(training_result.feature_names)} features."
        )
        return model_record

    @staticmethod
    async def get_active_model(session: AsyncSession) -> Optional[ModelVersion]:
        """Fetch the currently active model version for live/testnet inference.

        Returns:
            Active ModelVersion instance or None if no model is marked active.
        """
        query = (
            select(ModelVersion)
            .where(ModelVersion.is_active == True)  # noqa: E712
            .order_by(ModelVersion.trained_at.desc())
            .limit(1)
        )
        result = await session.execute(query)
        return result.scalar_one_or_none()

    @staticmethod
    async def get_model_by_version(version: str, session: AsyncSession) -> Optional[ModelVersion]:
        """Look up a model version by its unique tag (e.g. 'model_v001')."""
        query = select(ModelVersion).where(ModelVersion.version == version)
        result = await session.execute(query)
        return result.scalar_one_or_none()

    @classmethod
    async def set_active_model(cls, version: str, session: AsyncSession) -> ModelVersion:
        """Promote a specific model version to active status and deactivate all others.

        Prevents silent model replacement by requiring explicit activation.
        """
        target = await cls.get_model_by_version(version, session)
        if not target:
            raise ValueError(f"Cannot activate non-existent model version: '{version}'")

        # Deactivate all models
        await session.execute(update(ModelVersion).values(is_active=False))

        # Activate target model
        target.is_active = True
        await session.commit()
        await session.refresh(target)

        logger.info(f"Promoted model '{version}' to ACTIVE status in Model Registry.")
        return target

    @staticmethod
    async def list_models(session: AsyncSession) -> List[ModelVersion]:
        """List all registered models ordered chronologically by training date."""
        query = select(ModelVersion).order_by(ModelVersion.trained_at.desc())
        result = await session.execute(query)
        return list(result.scalars().all())

    @staticmethod
    def load_model_artifact(model_version: ModelVersion):
        """Load the serialized binary model artifact into memory for inference.

        Args:
            model_version: ModelVersion database record.

        Returns:
            Fitted estimator or ensemble instance ready for predict_proba.
        """
        if not os.path.exists(model_version.file_path):
            raise FileNotFoundError(
                f"Model artifact file not found at: {model_version.file_path} for version {model_version.version}"
            )

        if model_version.file_path.endswith(".joblib") or model_version.file_path.endswith(".pkl"):
            import joblib
            return joblib.load(model_version.file_path)
        else:
            model = xgb.XGBClassifier()
            model.load_model(model_version.file_path)
            return model
