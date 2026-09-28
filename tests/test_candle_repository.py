"""Unit and integration tests for CandleRepository and anti-lookahead querying."""

from datetime import datetime, timedelta, timezone
from typing import List
import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from app.data.collectors.base import CandleData
from app.data.storage.repository import CandleRepository


def generate_sample_candles(
    symbol: str = "BTC/USDT",
    timeframe: str = "15m",
    count: int = 10,
    start_time: datetime = datetime(2026, 1, 1, 0, 0, tzinfo=timezone.utc),
) -> List[CandleData]:
    """Generate deterministic sample candles."""
    candles = []
    step = timedelta(minutes=15)
    for i in range(count):
        ts = start_time + (step * i)
        candles.append(
            CandleData(
                symbol=symbol,
                timeframe=timeframe,
                timestamp=ts,
                open=50000.0 + (i * 10),
                high=50100.0 + (i * 10),
                low=49900.0 + (i * 10),
                close=50050.0 + (i * 10),
                volume=10.0 + i,
                quote_volume=500000.0,
                trades_count=100 + i,
                is_closed=True,
            )
        )
    return candles


@pytest.mark.asyncio
async def test_save_candles_idempotency(db_session: AsyncSession):
    """Verify that saving candles twice is idempotent and does not create duplicate rows."""
    candles = generate_sample_candles(count=5)

    # First save
    saved_1 = await CandleRepository.save_candles(candles, db_session)
    assert saved_1 == 5

    count_1 = await CandleRepository.get_candle_count("BTC/USDT", "15m", db_session)
    assert count_1 == 5

    # Second save with exact same records (upsert)
    saved_2 = await CandleRepository.save_candles(candles, db_session)
    assert saved_2 == 5

    count_2 = await CandleRepository.get_candle_count("BTC/USDT", "15m", db_session)
    assert count_2 == 5  # No duplicates created


@pytest.mark.asyncio
async def test_get_candles_with_date_filters(db_session: AsyncSession):
    """Verify date range filtering on get_candles."""
    start = datetime(2026, 1, 1, 0, 0, tzinfo=timezone.utc)
    candles = generate_sample_candles(count=10, start_time=start)
    await CandleRepository.save_candles(candles, db_session)

    # Filter from index 2 to 6 (5 candles)
    filter_start = start + timedelta(minutes=15 * 2)
    filter_end = start + timedelta(minutes=15 * 6)

    filtered = await CandleRepository.get_candles(
        symbol="BTC/USDT",
        timeframe="15m",
        start_time=filter_start,
        end_time=filter_end,
        session=db_session,
    )
    assert len(filtered) == 5
    assert filtered[0].timestamp == filter_start
    assert filtered[-1].timestamp == filter_end


@pytest.mark.asyncio
async def test_get_latest_candle_time(db_session: AsyncSession):
    """Verify retrieval of latest recorded candle timestamp."""
    start = datetime(2026, 1, 1, 0, 0, tzinfo=timezone.utc)
    candles = generate_sample_candles(count=8, start_time=start)
    await CandleRepository.save_candles(candles, db_session)

    latest = await CandleRepository.get_latest_candle_time("BTC/USDT", "15m", db_session)
    expected_latest = start + timedelta(minutes=15 * 7)
    assert latest == expected_latest


@pytest.mark.asyncio
async def test_as_of_time_prevents_look_ahead_bias(db_session: AsyncSession):
    """CRITICAL TEST: Verify that as_of_time strictly blocks future candles."""
    start = datetime(2026, 1, 1, 0, 0, tzinfo=timezone.utc)
    # Generate 20 candles: from 00:00 to 04:45
    candles = generate_sample_candles(count=20, start_time=start)
    await CandleRepository.save_candles(candles, db_session)

    # Simulate point-in-time T at candle 10 (02:30)
    point_in_time_t = start + timedelta(minutes=15 * 10)

    df = await CandleRepository.get_as_dataframe(
        symbol="BTC/USDT",
        timeframe="15m",
        as_of_time=point_in_time_t,
        session=db_session,
    )

    # Must contain exactly 11 candles (index 0 to 10) and NEVER candles past point_in_time_t
    assert len(df) == 11
    assert df.index.max() == point_in_time_t
    assert all(ts <= point_in_time_t for ts in df.index)
