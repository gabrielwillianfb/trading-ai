"""Repository for persisting and querying computed technical features."""

import logging
from datetime import datetime, timezone
from typing import List, Optional
import pandas as pd
from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.ext.asyncio import AsyncSession

from app.data.storage.models import MarketFeature

logger = logging.getLogger(__name__)


class FeatureRepository:
    """Handles storage, retrieval, and time-travel querying for engineered features."""

    @staticmethod
    async def save_features(
        features_df: pd.DataFrame,
        symbol: str,
        timeframe: str,
        session: AsyncSession,
    ) -> int:
        """Persist a feature DataFrame into the database idempotently.

        Args:
            features_df: DataFrame with DatetimeIndex and float feature columns.
            symbol: Trading pair (e.g. 'BTC/USDT').
            timeframe: Candle interval (e.g. '15m').
            session: Active AsyncSession.

        Returns:
            Number of feature records saved.
        """
        if features_df.empty:
            return 0

        bind = session.bind
        dialect_name = bind.dialect.name if bind else "postgresql"

        records = []
        for ts, row in features_df.iterrows():
            # Convert timestamp to UTC
            dt = ts.to_pydatetime() if hasattr(ts, "to_pydatetime") else ts
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            else:
                dt = dt.astimezone(timezone.utc)

            # Clean NaN / inf to float or None for JSON serialization
            vals = {}
            for col, val in row.items():
                if pd.isna(val):
                    vals[str(col)] = None
                else:
                    vals[str(col)] = float(val)

            records.append(
                {
                    "symbol": symbol,
                    "timeframe": timeframe,
                    "timestamp": dt,
                    "values": vals,
                }
            )

        chunk_size = 500
        total_saved = 0

        for i in range(0, len(records), chunk_size):
            chunk = records[i : i + chunk_size]

            if dialect_name == "postgresql":
                stmt = pg_insert(MarketFeature).values(chunk)
                stmt = stmt.on_conflict_do_update(
                    constraint="uq_feature_symbol_tf_time",
                    set_={"values": stmt.excluded["values"]},
                )
                await session.execute(stmt)
            elif dialect_name == "sqlite":
                stmt = sqlite_insert(MarketFeature).values(chunk)
                stmt = stmt.on_conflict_do_update(
                    index_elements=["symbol", "timeframe", "timestamp"],
                    set_={"values": stmt.excluded["values"]},
                )
                await session.execute(stmt)
            else:
                for rec in chunk:
                    feat_obj = MarketFeature(**rec)
                    await session.merge(feat_obj)

            total_saved += len(chunk)

        await session.commit()
        logger.info(f"Persisted {total_saved} feature records for {symbol} [{timeframe}].")
        return total_saved

    @staticmethod
    async def get_features_dataframe(
        symbol: str,
        timeframe: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        as_of_time: Optional[datetime] = None,
        session: Optional[AsyncSession] = None,
    ) -> pd.DataFrame:
        """Fetch features from database as a structured DataFrame.

        Guarantees point-in-time consistency with as_of_time.

        Args:
            symbol: Trading pair (e.g. 'BTC/USDT').
            timeframe: Candle interval (e.g. '15m').
            start_time: Filter starting timestamp (inclusive).
            end_time: Filter ending timestamp (inclusive).
            as_of_time: Point-in-time barrier. Zero future data beyond this barrier will be returned.
            session: Active AsyncSession.

        Returns:
            pd.DataFrame indexed by UTC timestamp with feature columns.
        """
        if session is None:
            raise ValueError("AsyncSession is required.")

        effective_end = end_time
        if as_of_time is not None:
            if as_of_time.tzinfo is None:
                as_of_time = as_of_time.replace(tzinfo=timezone.utc)
            if effective_end is None or as_of_time < effective_end:
                effective_end = as_of_time

        query = select(MarketFeature).where(
            MarketFeature.symbol == symbol,
            MarketFeature.timeframe == timeframe,
        )

        if start_time is not None:
            if start_time.tzinfo is None:
                start_time = start_time.replace(tzinfo=timezone.utc)
            query = query.where(MarketFeature.timestamp >= start_time)

        if effective_end is not None:
            query = query.where(MarketFeature.timestamp <= effective_end)

        query = query.order_by(MarketFeature.timestamp.asc())
        result = await session.execute(query)
        rows = result.scalars().all()

        if not rows:
            return pd.DataFrame()

        records = []
        timestamps = []
        for r in rows:
            ts = r.timestamp
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=timezone.utc)
            timestamps.append(ts)
            records.append(r.values)

        df = pd.DataFrame(records, index=pd.to_datetime(timestamps, utc=True))
        df.sort_index(inplace=True)
        return df
