"""CLI script to run simulated Paper Trading sessions using historical feeds or simulated streams."""

import argparse
import asyncio
import logging
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.config.logging import setup_logging
from app.config.settings import get_settings
from app.data.collectors.base import CandleData
from app.data.storage.db import get_engine, get_session_factory, init_db
from app.data.storage.models import MarketCandle
from app.execution.paper.account import VirtualAccount
from app.execution.paper.engine import PaperTradingEngine
from app.risk.manager import RiskManager
from sqlalchemy import select


async def main() -> None:
    parser = argparse.ArgumentParser(description="Run simulated Paper Trading session.")
    parser.add_argument("--symbol", type=str, default="BTC/USDT", help="Trading symbol.")
    parser.add_argument("--timeframe", type=str, default="15m", help="Candle timeframe.")
    parser.add_argument("--initial-cash", "--capital", type=float, default=400.0, dest="initial_cash", help="Virtual starting capital ($).")
    parser.add_argument("--steps", type=int, default=50, help="Number of latest candles to process.")
    parser.add_argument("--db-url", type=str, default=None, help="Database connection URL.")
    args = parser.parse_args()

    settings = get_settings()
    setup_logging(settings.LOG_LEVEL)
    logger = logging.getLogger("scripts.run_paper_trading")

    db_url = args.db_url or settings.DATABASE_URL
    engine = get_engine(db_url)
    await init_db(engine)
    session_factory = get_session_factory(db_url)

    logger.info("=" * 70)
    logger.info("PAPER TRADING EXECUTION ENGINE")
    logger.info(f"Target:       {args.symbol} [{args.timeframe}]")
    logger.info(f"Capital:      ${args.initial_cash:,.2f} USDT")
    logger.info(f"Steps:        {args.steps} bars")
    logger.info("=" * 70)

    account = VirtualAccount(initial_cash=args.initial_cash)
    risk_manager = RiskManager(settings=settings)
    paper_engine = PaperTradingEngine(account=account, risk_manager=risk_manager)

    try:
        async with session_factory() as session:
            # Initialize engine with active model
            initialized = await paper_engine.initialize(session)
            if not initialized:
                logger.error("No active model found. Run train_model.py first.")
                return

            # Fetch candles from database
            query = (
                select(MarketCandle)
                .where(
                    MarketCandle.symbol == args.symbol,
                    MarketCandle.timeframe == args.timeframe,
                )
                .order_by(MarketCandle.timestamp.asc())
            )
            result = await session.execute(query)
            db_candles = result.scalars().all()

            if not db_candles:
                logger.error(f"No candles found in DB for {args.symbol} [{args.timeframe}].")
                return

            candles_to_process = db_candles[-args.steps :]
            logger.info(f"Loaded {len(candles_to_process)} candles to process in Paper Trading...")

            for i, c in enumerate(candles_to_process, 1):
                candle_data = CandleData(
                    symbol=c.symbol,
                    timeframe=c.timeframe,
                    timestamp=c.timestamp,
                    open=c.open,
                    high=c.high,
                    low=c.low,
                    close=c.close,
                    volume=c.volume,
                    is_closed=c.is_closed,
                )

                step_res = await paper_engine.process_candle(candle_data, session)
                if step_res["order_filled"]:
                    logger.info(f"Step #{i}: Order Fill -> {step_res['order_filled']}")
                if step_res["intrabar_exit"]:
                    logger.info(f"Step #{i}: Intrabar Exit -> {step_res['intrabar_exit']}")

            final_equity = account.get_total_equity()
            logger.info("\n" + "=" * 70)
            logger.info("PAPER TRADING SESSION SUMMARY:")
            logger.info(f"  - Initial Cash:     ${account.initial_cash:,.2f}")
            logger.info(f"  - Final Cash:       ${account.cash:,.2f}")
            logger.info(f"  - Final Equity:     ${final_equity:,.2f}")
            logger.info(f"  - Net Realized PnL: ${account.realized_pnl:+,.2f}")
            logger.info(f"  - Unrealized PnL:   ${account.get_unrealized_pnl():+,.2f}")
            logger.info(f"  - Total Return:     {((final_equity - account.initial_cash)/account.initial_cash)*100:+.2f}%")
            logger.info(f"  - Commissions Paid: ${account.total_commissions_paid:,.2f}")
            logger.info("=" * 70)

    finally:
        await engine.dispose()


if __name__ == "__main__":
    asyncio.run(main())
