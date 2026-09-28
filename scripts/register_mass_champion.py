"""Register the 507-trade Continuous Portfolio Multi-Model in Registry."""

import asyncio
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import logging
from app.config.logging import setup_logging
from app.config.settings import get_settings
from app.data.storage.db import get_engine, get_session_factory
from app.data.storage.repository import CandleRepository
from app.data.preprocessing.dataset import DatasetBuilder
from app.features.pipeline import FeaturePipeline
from app.models.training.multi_model_trainer import MultiModelTrainer
from app.models.registry.registry import ModelRegistry


async def main() -> None:
    settings = get_settings()
    setup_logging(settings.LOG_LEVEL)
    logger = logging.getLogger("scripts.register_mass_champion")

    engine = get_engine(settings.DATABASE_URL)
    session_factory = get_session_factory(settings.DATABASE_URL)

    async with session_factory() as session:
        df_btc = await CandleRepository.get_as_dataframe("BTC/USDT", "1h", session=session)
        if df_btc.empty:
            logger.error("No candle data available for BTC/USDT [1h]. Run download_candles.py first.")
            return

        feats = FeaturePipeline.compute_features(df_btc, drop_na=False)
        ds = DatasetBuilder.build(
            candles_df=df_btc,
            features_df=feats,
            symbol="BTC/USDT",
            timeframe="1h",
            horizon=16,
            target_return=0.008,
            label_method="peak",
            drop_na=True,
        )
        splits = DatasetBuilder.split_dataset(ds, train_pct=0.70, val_pct=0.15, test_pct=0.15)
        tr = MultiModelTrainer.train_model(
            "extra_trees",
            splits,
            hyperparameters={"n_estimators": 120, "max_depth": 5, "min_samples_split": 8, "min_samples_leaf": 4, "class_weight": "balanced"}
        )

        model_ver = await ModelRegistry.register_model(
            training_result=tr,
            symbol="BTC/USDT",
            timeframe="1h",
            session=session,
            set_active=True,
            model_dir="models",
        )
        logger.info(f"Registered Active Champion Model: {model_ver.version} (Type: {model_ver.model_type})")

    await engine.dispose()

if __name__ == "__main__":
    asyncio.run(main())
