"""CLI script to initialize the PostgreSQL database schema and verify connectivity."""

import asyncio
import logging
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.config.logging import setup_logging
from app.config.settings import get_settings
from app.data.storage.db import check_db_health, get_engine, init_db


async def main() -> None:
    settings = get_settings()
    setup_logging(settings.LOG_LEVEL)
    logger = logging.getLogger("scripts.init_db")

    logger.info("Connecting to database...")
    logger.info(f"Target host: {settings.POSTGRES_HOST}:{settings.POSTGRES_PORT}, DB: {settings.POSTGRES_DB}")

    engine = get_engine()
    healthy = await check_db_health(engine)
    if not healthy:
        logger.error(
            "Could not connect to PostgreSQL. "
            "Please ensure the container is running: `docker compose up -d postgres`."
        )
        sys.exit(1)

    logger.info("Database connection verified! Initializing tables...")
    await init_db(engine)
    logger.info("All 11 tables successfully created/verified in PostgreSQL:")
    logger.info(
        " - market_candles\n"
        " - features\n"
        " - model_versions\n"
        " - predictions\n"
        " - signals\n"
        " - orders\n"
        " - fills\n"
        " - positions\n"
        " - portfolio_snapshots\n"
        " - risk_events\n"
        " - trades"
    )
    await engine.dispose()
    logger.info("Initialization completed successfully.")


if __name__ == "__main__":
    asyncio.run(main())
