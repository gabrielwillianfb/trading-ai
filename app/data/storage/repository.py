"""Candle repository for database operations, idempotency, and anti-look-ahead queries."""

import logging
from datetime import datetime, timezone
from typing import List, Optional

import pandas as pd
from sqlalchemy import func, select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.ext.asyncio import AsyncSession

from app.data.collectors.base import CandleData
from app.data.storage.models import MarketCandle

logger = logging.getLogger(__name__)


class CandleRepository:
    """Handles persistence, retrieval, and time-travel querying for market candles."""

    @staticmethod
    async def save_candles(candles: List[CandleData], session: AsyncSession) -> int:
        """Batch upsert/insert candles into the database idempotently.

        Args:
            candles: List of CandleData instances to persist.
            session: Active AsyncSession.

        Returns:
            Number of candles processed and committed.
        """
        if not candles:
            return 0

        # Determine dialect (PostgreSQL vs SQLite)
        bind = session.bind
        dialect_name = bind.dialect.name if bind else "postgresql"

        records = [
            {
                "symbol": c.symbol,
                "timeframe": c.timeframe,
                "timestamp": c.timestamp if c.timestamp.tzinfo else c.timestamp.replace(tzinfo=timezone.utc),
                "open": c.open,
                "high": c.high,
                "low": c.low,
                "close": c.close,
                "volume": c.volume,
                "quote_volume": c.quote_volume,
                "trades_count": c.trades_count,
                "is_closed": c.is_closed,
            }
            for c in candles
        ]

        # Chunk insertion to avoid huge parameter limits
        chunk_size = 500
        total_saved = 0

        for i in range(0, len(records), chunk_size):
            chunk = records[i : i + chunk_size]

            if dialect_name == "postgresql":
                stmt = pg_insert(MarketCandle).values(chunk)
                stmt = stmt.on_conflict_do_update(
                    constraint="uq_candle_symbol_tf_time",
                    set_={
                        "open": stmt.excluded.open,
                        "high": stmt.excluded.high,
                        "low": stmt.excluded.low,
                        "close": stmt.excluded.close,
                        "volume": stmt.excluded.volume,
                        "quote_volume": stmt.excluded.quote_volume,
                        "trades_count": stmt.excluded.trades_count,
                        "is_closed": stmt.excluded.is_closed,
                    },
                )
                await session.execute(stmt)
            elif dialect_name == "sqlite":
                stmt = sqlite_insert(MarketCandle).values(chunk)
                stmt = stmt.on_conflict_do_update(
                    index_elements=["symbol", "timeframe", "timestamp"],
                    set_={
                        "open": stmt.excluded.open,
                        "high": stmt.excluded.high,
                        "low": stmt.excluded.low,
                        "close": stmt.excluded.close,
                        "volume": stmt.excluded.volume,
                        "quote_volume": stmt.excluded.quote_volume,
                        "trades_count": stmt.excluded.trades_count,
                        "is_closed": stmt.excluded.is_closed,
                    },
                )
                await session.execute(stmt)
            else:
                # Generic fallback using merge
                for rec in chunk:
                    candle_obj = MarketCandle(**rec)
                    await session.merge(candle_obj)

            total_saved += len(chunk)

        await session.commit()
        logger.info(f"Saved {total_saved} candles to database successfully.")
        return total_saved

    @staticmethod
    async def get_candles(
        symbol: str,
        timeframe: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: Optional[int] = None,
        session: Optional[AsyncSession] = None,
    ) -> List[MarketCandle]:
        """Fetch candles for a symbol and timeframe chronologically.

        Args:
            symbol: Trading pair (e.g. 'BTC/USDT').
            timeframe: Interval (e.g. '15m').
            start_time: Start date (inclusive).
            end_time: End date (inclusive).
            limit: Max records to return.
            session: Active AsyncSession.

        Returns:
            List of MarketCandle records ordered ascending by timestamp.
        """
        if session is None:
            raise ValueError("AsyncSession is required.")

        query = select(MarketCandle).where(
            MarketCandle.symbol == symbol,
            MarketCandle.timeframe == timeframe,
        )

        if start_time is not None:
            if start_time.tzinfo is None:
                start_time = start_time.replace(tzinfo=timezone.utc)
            query = query.where(MarketCandle.timestamp >= start_time)

        if end_time is not None:
            if end_time.tzinfo is None:
                end_time = end_time.replace(tzinfo=timezone.utc)
            query = query.where(MarketCandle.timestamp <= end_time)

        query = query.order_by(MarketCandle.timestamp.asc())

        if limit is not None:
            query = query.limit(limit)

        result = await session.execute(query)
        candles = list(result.scalars().all())
        for c in candles:
            if c.timestamp is not None and c.timestamp.tzinfo is None:
                c.timestamp = c.timestamp.replace(tzinfo=timezone.utc)
        return candles

    @staticmethod
    async def get_latest_candle_time(
        symbol: str,
        timeframe: str,
        session: AsyncSession,
    ) -> Optional[datetime]:
        """Get the latest recorded candle timestamp for a symbol and timeframe.

        Args:
            symbol: Trading pair (e.g. 'BTC/USDT').
            timeframe: Interval (e.g. '15m').
            session: Active AsyncSession.

        Returns:
            Latest UTC datetime or None if no candles exist.
        """
        query = select(func.max(MarketCandle.timestamp)).where(
            MarketCandle.symbol == symbol,
            MarketCandle.timeframe == timeframe,
        )
        result = await session.execute(query)
        val = result.scalar_one_or_none()
        if val is not None and val.tzinfo is None:
            val = val.replace(tzinfo=timezone.utc)
        return val

    @staticmethod
    async def get_candle_count(
        symbol: str,
        timeframe: str,
        session: AsyncSession,
    ) -> int:
        """Count total stored candles for a symbol and timeframe."""
        query = select(func.count(MarketCandle.id)).where(
            MarketCandle.symbol == symbol,
            MarketCandle.timeframe == timeframe,
        )
        result = await session.execute(query)
        return result.scalar_one() or 0

    @classmethod
    async def get_as_dataframe(
        cls,
        symbol: str,
        timeframe: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        as_of_time: Optional[datetime] = None,
        session: Optional[AsyncSession] = None,
    ) -> pd.DataFrame:
        """Retrieve candles as a pandas DataFrame with strict anti-look-ahead guarantees.

        Args:
            symbol: Trading pair (e.g. 'BTC/USDT').
            timeframe: Interval (e.g. '15m').
            start_time: Starting filter timestamp (inclusive).
            end_time: Ending filter timestamp (inclusive).
            as_of_time: Point-in-time barrier. Zero data beyond this timestamp will be included.
            session: Active AsyncSession.

        Returns:
            pd.DataFrame indexed by UTC timestamp with OHLCV columns.
        """
        effective_end = end_time
        if as_of_time is not None:
            if as_of_time.tzinfo is None:
                as_of_time = as_of_time.replace(tzinfo=timezone.utc)
            if effective_end is None or as_of_time < effective_end:
                effective_end = as_of_time

        candles = await cls.get_candles(
            symbol=symbol,
            timeframe=timeframe,
            start_time=start_time,
            end_time=effective_end,
            session=session,
        )

        if not candles:
            return pd.DataFrame(
                columns=["open", "high", "low", "close", "volume", "quote_volume", "trades_count"]
            )

        data = [
            {
                "timestamp": c.timestamp,
                "open": c.open,
                "high": c.high,
                "low": c.low,
                "close": c.close,
                "volume": c.volume,
                "quote_volume": c.quote_volume,
                "trades_count": c.trades_count,
            }
            for c in candles
        ]

        df = pd.DataFrame(data)
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        df.set_index("timestamp", inplace=True)
        df.sort_index(inplace=True)
        return df
