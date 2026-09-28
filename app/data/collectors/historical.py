"""Historical candlestick data collector with automated pagination and integrity checks."""

import asyncio
import logging
from datetime import datetime, timezone
from typing import List, Optional

from app.data.collectors.base import CandleData, ExchangeClient

logger = logging.getLogger(__name__)


TIMEFRAME_SECONDS = {
    "1m": 60,
    "3m": 180,
    "5m": 300,
    "15m": 900,
    "30m": 1800,
    "1h": 3600,
    "2h": 7200,
    "4h": 14400,
    "6h": 21600,
    "8h": 28800,
    "12h": 43200,
    "1d": 86400,
    "3d": 259200,
    "1w": 604800,
}


def timeframe_to_seconds(timeframe: str) -> int:
    """Convert timeframe string (e.g. '15m') to duration in seconds."""
    tf = timeframe.lower()
    if tf not in TIMEFRAME_SECONDS:
        raise ValueError(f"Unknown timeframe '{timeframe}'. Supported: {list(TIMEFRAME_SECONDS.keys())}")
    return TIMEFRAME_SECONDS[tf]


def timeframe_to_milliseconds(timeframe: str) -> int:
    """Convert timeframe string to duration in milliseconds."""
    return timeframe_to_seconds(timeframe) * 1000


class HistoricalDataCollector:
    """Coordinates fetching and paginating historical candles from any ExchangeClient."""

    def __init__(self, exchange_client: ExchangeClient):
        self.exchange = exchange_client

    async def collect_range(
        self,
        symbol: str,
        timeframe: str,
        start_time: datetime,
        end_time: Optional[datetime] = None,
        chunk_limit: int = 1000,
        request_delay: float = 0.1,
    ) -> List[CandleData]:
        """Fetch historical candles across an arbitrary date range using sequential pagination.

        Args:
            symbol: Standardized symbol (e.g. 'BTC/USDT').
            timeframe: Candle interval (e.g. '15m').
            start_time: Starting timestamp (UTC).
            end_time: Ending timestamp (UTC). Defaults to current time.
            chunk_limit: Max candles per request (default 1000).
            request_delay: Pause between consecutive requests to respect rate limits.

        Returns:
            List of chronologically ordered, deduplicated CandleData instances.
        """
        # Normalize timezone to UTC
        if start_time.tzinfo is None:
            start_time = start_time.replace(tzinfo=timezone.utc)
        else:
            start_time = start_time.astimezone(timezone.utc)

        if end_time is None:
            end_time = datetime.now(timezone.utc)
        elif end_time.tzinfo is None:
            end_time = end_time.replace(tzinfo=timezone.utc)
        else:
            end_time = end_time.astimezone(timezone.utc)

        start_ms = int(start_time.timestamp() * 1000)
        end_ms = int(end_time.timestamp() * 1000)
        step_ms = timeframe_to_milliseconds(timeframe)

        if start_ms >= end_ms:
            raise ValueError(f"start_time ({start_time}) must be earlier than end_time ({end_time})")

        logger.info(
            f"Starting collection for {symbol} [{timeframe}] from {start_time.isoformat()} to {end_time.isoformat()}"
        )

        all_candles: List[CandleData] = []
        current_start_ms = start_ms
        batch_num = 1

        while current_start_ms < end_ms:
            logger.debug(f"Fetching batch #{batch_num} starting at {current_start_ms}...")
            batch = await self.exchange.get_klines(
                symbol=symbol,
                timeframe=timeframe,
                start_time=current_start_ms,
                end_time=end_ms,
                limit=chunk_limit,
            )

            if not batch:
                logger.info(f"No more candles returned by exchange at {current_start_ms}. Ending pagination.")
                break

            all_candles.extend(batch)
            last_candle = batch[-1]
            last_candle_ms = int(last_candle.timestamp.timestamp() * 1000)

            # Advance current_start_ms past the last received candle
            next_start_ms = last_candle_ms + step_ms

            if next_start_ms <= current_start_ms:
                # Protection against infinite loops if exchange repeats timestamps
                logger.warning("Pagination did not advance. Breaking to avoid infinite loop.")
                break

            current_start_ms = next_start_ms
            batch_num += 1

            if len(batch) < chunk_limit:
                # Reached the end of available historical data up to end_time
                break

            if request_delay > 0:
                await asyncio.sleep(request_delay)

        # Deduplicate by timestamp and sort ascending
        seen_timestamps = set()
        deduped_candles: List[CandleData] = []
        for candle in sorted(all_candles, key=lambda c: c.timestamp):
            ts = candle.timestamp
            if ts not in seen_timestamps:
                seen_timestamps.add(ts)
                deduped_candles.append(candle)

        logger.info(
            f"Completed collection for {symbol} [{timeframe}]: {len(deduped_candles)} total valid candles retrieved."
        )
        return deduped_candles
