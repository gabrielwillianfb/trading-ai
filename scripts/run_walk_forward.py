"""CLI script to execute multi-period Walk-Forward validation and statistical stability tests."""

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
from app.data.storage.db import get_engine, get_session_factory
from app.data.storage.repository import CandleRepository
from app.models.training.walk_forward import WalkForwardValidator


async def main() -> None:
    parser = argparse.ArgumentParser(description="Run Walk-Forward temporal validation.")
    parser.add_argument("--symbol", type=str, default="BTC/USDT", help="Trading symbol.")
    parser.add_argument("--timeframe", type=str, default="15m", help="Candle timeframe.")
    parser.add_argument("--splits", type=int, default=4, help="Number of walk-forward folds.")
    parser.add_argument("--horizon", type=int, default=12, help="Prediction horizon in bars.")
    parser.add_argument(
        "--target-return", type=float, default=0.005, help="Target return threshold (0.005 for 0.5%%)."
    )
    parser.add_argument(
        "--label-method",
        type=str,
        choices=["peak", "return", "triple_barrier"],
        default="peak",
        help="Labeling method.",
    )
    parser.add_argument("--n-estimators", type=int, default=200, help="Boosting trees.")
    parser.add_argument("--max-depth", type=int, default=4, help="Max tree depth.")
    parser.add_argument("--learning-rate", type=float, default=0.03, help="Learning rate.")
    parser.add_argument("--db-url", type=str, default=None, help="Database connection URL.")
    args = parser.parse_args()

    settings = get_settings()
    setup_logging(settings.LOG_LEVEL)
    logger = logging.getLogger("scripts.run_walk_forward")

    db_url = args.db_url or settings.DATABASE_URL
    engine = get_engine(db_url)
    session_factory = get_session_factory(db_url)

    logger.info("=" * 70)
    logger.info("WALK-FORWARD TEMPORAL VALIDATION EXPERIMENT")
    logger.info(f"Target:       {args.symbol} [{args.timeframe}] | Folds: {args.splits}")
    logger.info(f"Label:        Horizon {args.horizon} bars | Return {args.target_return*100:.2f}% ({args.label_method})")
    logger.info("=" * 70)

    try:
        async with session_factory() as session:
            df_candles = await CandleRepository.get_as_dataframe(args.symbol, args.timeframe, session=session)
            if df_candles.empty:
                logger.error(f"No candles found for {args.symbol} [{args.timeframe}]. Run download_candles.py first.")
                return

            logger.info(f"Loaded {len(df_candles)} candles from DB.")

            dataset = DatasetBuilder.build(
                candles_df=df_candles,
                symbol=args.symbol,
                timeframe=args.timeframe,
                horizon=args.horizon,
                target_return=args.target_return,
                label_method=args.label_method,
                drop_na=True,
            )
            logger.info(f"Aligned dataset: {dataset.total_samples} samples, 27 features.")

            hyperparams = {
                "n_estimators": args.n_estimators,
                "max_depth": args.max_depth,
                "learning_rate": args.learning_rate,
            }

            report = WalkForwardValidator.run_validation(
                dataset=dataset,
                n_splits=args.splits,
                expanding=True,
                embargo_bars=args.horizon,
                hyperparameters=hyperparams,
                early_stopping_rounds=20,
            )

            logger.info("\n" + report.summary() + "\n")

    finally:
        await engine.dispose()


if __name__ == "__main__":
    asyncio.run(main())
