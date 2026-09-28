"""CLI script to generate and store quantitative features for stored market candles."""

import argparse
import asyncio
import logging
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.config.logging import setup_logging
from app.config.settings import get_settings
from app.data.storage.db import get_engine, get_session_factory, init_db
from app.data.storage.repository import CandleRepository
from app.features.pipeline import FeaturePipeline
from app.features.repository import FeatureRepository


async def main() -> None:
    parser = argparse.ArgumentParser(description="Compute quantitative features for stored candles.")
    parser.add_argument(
        "--symbols",
        nargs="+",
        default=None,
        help="Target symbols (e.g. BTC/USDT ETH/USDT). Default: settings.SYMBOLS",
    )
    parser.add_argument(
        "--timeframes",
        nargs="+",
        default=None,
        help="Target timeframes (e.g. 5m 15m 1h 4h). Default: settings.TIMEFRAMES",
    )
    parser.add_argument(
        "--db-url",
        type=str,
        default=None,
        help="Override database connection URL.",
    )
    args = parser.parse_args()

    settings = get_settings()
    setup_logging(settings.LOG_LEVEL)
    logger = logging.getLogger("scripts.build_features")

    symbols = args.symbols or settings.SYMBOLS
    timeframes = args.timeframes or settings.TIMEFRAMES
    db_url = args.db_url or settings.DATABASE_URL

    logger.info("=" * 60)
    logger.info("FEATURE ENGINEERING PIPELINE")
    logger.info(f"Target Symbols:    {symbols}")
    logger.info(f"Target Timeframes: {timeframes}")
    logger.info(f"Database URL:      {db_url}")
    logger.info("=" * 60)

    engine = get_engine(db_url)
    await init_db(engine)
    session_factory = get_session_factory(db_url)

    total_features_saved = 0

    try:
        for symbol in symbols:
            for tf in timeframes:
                logger.info(f"\nProcessing features for {symbol} [{tf}]...")

                async with session_factory() as session:
                    # 1. Fetch candles from database
                    df_candles = await CandleRepository.get_as_dataframe(symbol, tf, session=session)
                    if df_candles.empty:
                        logger.warning(f"No candles found in database for {symbol} [{tf}]. Skipping.")
                        continue

                    logger.info(f"Loaded {len(df_candles)} candles from DB.")

                    # 2. Compute features
                    features_df = FeaturePipeline.compute_features(df_candles, drop_na=True)
                    logger.info(
                        f"Computed {len(features_df)} feature vectors with {features_df.shape[1]} features each."
                    )

                    # 3. Save features to database
                    saved = await FeatureRepository.save_features(features_df, symbol, tf, session)
                    total_features_saved += saved
                    logger.info(f"Persisted {saved} feature records to 'features' table.")

        logger.info("\n" + "=" * 60)
        logger.info(f"FEATURE ENGINEERING COMPLETED! Total records stored: {total_features_saved}")
        logger.info(f"Feature columns: {FeaturePipeline.get_feature_names()}")
        logger.info("=" * 60)

    finally:
        await engine.dispose()


if __name__ == "__main__":
    asyncio.run(main())
