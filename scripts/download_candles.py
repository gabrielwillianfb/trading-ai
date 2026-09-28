"""CLI script to download historical candlestick data and persist to PostgreSQL."""

import argparse
import asyncio
import logging
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.config.logging import setup_logging
from app.config.settings import get_settings
from app.data.collectors.binance import BinanceClient
from app.data.collectors.historical import HistoricalDataCollector
from app.data.preprocessing.validator import DataValidator
from app.data.storage.db import get_engine, get_session_factory, init_db
from app.data.storage.repository import CandleRepository


async def main() -> None:
    parser = argparse.ArgumentParser(description="Download historical candlestick market data.")
    parser.add_argument(
        "--symbols",
        nargs="+",
        default=None,
        help="Symbols to download (e.g. BTC/USDT ETH/USDT). Default: configured symbols in .env",
    )
    parser.add_argument(
        "--timeframes",
        nargs="+",
        default=None,
        help="Timeframes to download (e.g. 5m 15m 1h 4h). Default: configured timeframes in .env",
    )
    parser.add_argument(
        "--days",
        type=int,
        default=30,
        help="Number of days of history to download (default: 30).",
    )
    parser.add_argument(
        "--start-date",
        type=str,
        default=None,
        help="Custom start date in ISO format (YYYY-MM-DD or YYYY-MM-DDTHH:MM:SSZ).",
    )
    parser.add_argument(
        "--end-date",
        type=str,
        default=None,
        help="Custom end date in ISO format (default: now).",
    )
    parser.add_argument(
        "--db-url",
        type=str,
        default=None,
        help="Override database connection URL (e.g. sqlite+aiosqlite:///data/trading_ai.db).",
    )
    args = parser.parse_args()

    settings = get_settings()
    setup_logging(settings.LOG_LEVEL)
    logger = logging.getLogger("scripts.download_candles")

    symbols = args.symbols or settings.SYMBOLS
    timeframes = args.timeframes or settings.TIMEFRAMES

    now_utc = datetime.now(timezone.utc)
    if args.start_date:
        start_time = datetime.fromisoformat(args.start_date.replace("Z", "+00:00"))
        if start_time.tzinfo is None:
            start_time = start_time.replace(tzinfo=timezone.utc)
    else:
        start_time = now_utc - timedelta(days=args.days)

    if args.end_date:
        end_time = datetime.fromisoformat(args.end_date.replace("Z", "+00:00"))
        if end_time.tzinfo is None:
            end_time = end_time.replace(tzinfo=timezone.utc)
    else:
        end_time = now_utc

    logger.info("=" * 60)
    logger.info(f"HISTORICAL MARKET DATA DOWNLOAD")
    logger.info(f"Target Symbols:    {symbols}")
    logger.info(f"Target Timeframes: {timeframes}")
    logger.info(f"Date Range:        {start_time.isoformat()} -> {end_time.isoformat()}")
    logger.info("=" * 60)

    # Initialize database if needed
    db_url = args.db_url or settings.DATABASE_URL
    engine = get_engine(db_url)
    await init_db(engine)
    session_factory = get_session_factory(db_url)

    # For downloading public historical market data, Binance public endpoints can be used
    client = BinanceClient(base_url="https://api.binance.com/api")
    collector = HistoricalDataCollector(client)

    grand_total = 0

    try:
        for symbol in symbols:
            for tf in timeframes:
                logger.info(f"\nFetching {symbol} [{tf}]...")
                candles = await collector.collect_range(
                    symbol=symbol,
                    timeframe=tf,
                    start_time=start_time,
                    end_time=end_time,
                )

                if not candles:
                    logger.warning(f"No candles retrieved for {symbol} [{tf}].")
                    continue

                async with session_factory() as session:
                    saved = await CandleRepository.save_candles(candles, session)
                    grand_total += saved

                    # Validate dataset in DB
                    df = await CandleRepository.get_as_dataframe(symbol, tf, session=session)
                    report = DataValidator.validate_dataframe(df, tf)

                    logger.info(
                        f"[{symbol} | {tf}] Stored {saved} candles. DB Total: {len(df)}. "
                        f"Status: {'HEALTHY' if report.is_valid else 'WARNING (gaps/anomalies)'}"
                    )
                    if report.gaps_count > 0:
                        logger.warning(f"Detected {report.gaps_count} gap(s) in {symbol} [{tf}].")

        logger.info("\n" + "=" * 60)
        logger.info(f"ALL DOWNLOADS COMPLETED! Total records persisted: {grand_total}")
        logger.info("=" * 60)

    finally:
        await client.close()
        await engine.dispose()


if __name__ == "__main__":
    asyncio.run(main())
