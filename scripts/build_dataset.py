"""CLI script to build and inspect datasets with temporal splits and label statistics."""

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
from app.data.preprocessing.splitters import WalkForwardSplitter
from app.data.storage.db import get_engine, get_session_factory
from app.data.storage.repository import CandleRepository


async def main() -> None:
    parser = argparse.ArgumentParser(description="Construct and validate ML dataset.")
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
    parser.add_argument(
        "--db-url", type=str, default=None, help="Database connection URL."
    )
    args = parser.parse_args()

    settings = get_settings()
    setup_logging(settings.LOG_LEVEL)
    logger = logging.getLogger("scripts.build_dataset")

    db_url = args.db_url or settings.DATABASE_URL
    engine = get_engine(db_url)
    session_factory = get_session_factory(db_url)

    logger.info("=" * 60)
    logger.info("DATASET PREPARATION & TEMPORAL PARTITIONING")
    logger.info(f"Target:       {args.symbol} [{args.timeframe}]")
    logger.info(f"Horizon:      {args.horizon} bars | Target Return: {args.target_return*100:.2f}%")
    logger.info(f"Label Method: {args.label_method.upper()}")
    logger.info("=" * 60)

    try:
        async with session_factory() as session:
            df_candles = await CandleRepository.get_as_dataframe(args.symbol, args.timeframe, session=session)
            if df_candles.empty:
                logger.error(f"No candles found for {args.symbol} [{args.timeframe}]. Run download_candles.py first.")
                return

            logger.info(f"Loaded {len(df_candles)} candles from database.")

            dataset = DatasetBuilder.build(
                candles_df=df_candles,
                symbol=args.symbol,
                timeframe=args.timeframe,
                horizon=args.horizon,
                target_return=args.target_return,
                label_method=args.label_method,
                drop_na=True,
            )

            logger.info("\n" + dataset.summary() + "\n")

            # Perform temporal train/val/test split
            splits = DatasetBuilder.split_dataset(dataset, train_pct=0.70, val_pct=0.15, test_pct=0.15)
            logger.info("--- TEMPORAL SPLITS (Sequential) ---")
            logger.info(splits.summary())

            # Perform Walk-Forward splits demonstration
            logger.info("\n--- WALK-FORWARD VALIDATION FOLDS ---")
            if len(dataset.X) >= 100:
                folds = WalkForwardSplitter.generate_folds(
                    dataset.X, n_splits=4, expanding=True, embargo_bars=args.horizon
                )
                for f in folds:
                    logger.info(f.summary())
            else:
                logger.info("Dataset too small for multi-fold walk-forward demonstration.")

    finally:
        await engine.dispose()


if __name__ == "__main__":
    asyncio.run(main())
