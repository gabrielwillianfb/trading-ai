"""CLI script to train, evaluate, and register XGBoost models in the Model Registry."""

import argparse
import asyncio
import logging
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.config.logging import setup_logging
from app.config.settings import get_settings
from app.data.preprocessing.dataset import DatasetBuilder
from app.data.storage.db import get_engine, get_session_factory, init_db
from app.data.storage.repository import CandleRepository
from app.models.registry.registry import ModelRegistry
from app.models.training.trainer import ModelTrainer


async def main() -> None:
    parser = argparse.ArgumentParser(description="Train and register XGBoost ML model.")
    parser.add_argument("--symbol", type=str, default="BTC/USDT", help="Trading symbol.")
    parser.add_argument("--timeframe", type=str, default="15m", help="Candle timeframe.")
    parser.add_argument("--horizon", type=int, default=12, help="Prediction horizon in bars.")
    parser.add_argument(
        "--target-return", type=float, default=0.005, help="Target return threshold (e.g. 0.005 for 0.5%%)."
    )
    parser.add_argument(
        "--label-method",
        type=str,
        choices=["peak", "return", "triple_barrier"],
        default="peak",
        help="Labeling method.",
    )
    parser.add_argument("--n-estimators", type=int, default=300, help="Number of boosting trees.")
    parser.add_argument("--max-depth", type=int, default=4, help="Max tree depth.")
    parser.add_argument("--learning-rate", type=float, default=0.03, help="Learning rate (eta).")
    parser.add_argument(
        "--set-active", action="store_true", help="Set this model as active for live inference."
    )
    parser.add_argument(
        "--db-url", type=str, default=None, help="Database connection URL."
    )
    args = parser.parse_args()

    settings = get_settings()
    setup_logging(settings.LOG_LEVEL)
    logger = logging.getLogger("scripts.train_model")

    db_url = args.db_url or settings.DATABASE_URL
    engine = get_engine(db_url)
    await init_db(engine)
    session_factory = get_session_factory(db_url)

    logger.info("=" * 60)
    logger.info("XGBOOST MODEL TRAINING PIPELINE")
    logger.info(f"Symbol:       {args.symbol} [{args.timeframe}]")
    logger.info(f"Target:       Horizon {args.horizon} bars | Return {args.target_return*100:.2f}% ({args.label_method})")
    logger.info(f"Parameters:   n_estimators={args.n_estimators}, max_depth={args.max_depth}, lr={args.learning_rate}")
    logger.info(f"Set Active:   {args.set_active}")
    logger.info("=" * 60)

    try:
        async with session_factory() as session:
            # 1. Fetch candles from database
            df_candles = await CandleRepository.get_as_dataframe(args.symbol, args.timeframe, session=session)
            if df_candles.empty:
                logger.error(
                    f"No candles found for {args.symbol} [{args.timeframe}]. Please run download_candles.py first."
                )
                return

            logger.info(f"Loaded {len(df_candles)} candles from DB.")

            # 2. Build aligned dataset
            dataset = DatasetBuilder.build(
                candles_df=df_candles,
                symbol=args.symbol,
                timeframe=args.timeframe,
                horizon=args.horizon,
                target_return=args.target_return,
                label_method=args.label_method,
                drop_na=True,
            )
            logger.info(f"Dataset constructed: {dataset.total_samples} samples, 27 features.")

            # 3. Create temporal train / validation / test splits
            splits = DatasetBuilder.split_dataset(dataset, train_pct=0.70, val_pct=0.15, test_pct=0.15)
            logger.info(
                f"Splits: Train={len(splits.X_train)}, Val={len(splits.X_val)}, Test={len(splits.X_test)}"
            )

            # 4. Train XGBoost model with early stopping
            hyperparams = {
                "n_estimators": args.n_estimators,
                "max_depth": args.max_depth,
                "learning_rate": args.learning_rate,
            }
            training_result = ModelTrainer.train(splits, hyperparameters=hyperparams, early_stopping_rounds=25)

            # 5. Display detailed training report
            logger.info("\n" + training_result.summary() + "\n")

            # 6. Register model in Model Registry
            registered_model = await ModelRegistry.register_model(
                training_result=training_result,
                symbol=args.symbol,
                timeframe=args.timeframe,
                session=session,
                set_active=args.set_active,
            )

            logger.info(
                f"SUCCESS: Model registered as '{registered_model.version}' "
                f"(ID: {registered_model.id}, Active: {registered_model.is_active})."
            )
            logger.info(f"Artifact saved at: {registered_model.file_path}")

    finally:
        await engine.dispose()


if __name__ == "__main__":
    asyncio.run(main())
