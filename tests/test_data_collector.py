"""Unit tests for HistoricalDataCollector and DataValidator."""

from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional
import pandas as pd
import pytest

from app.data.collectors.base import CandleData, ExchangeClient
from app.data.collectors.historical import (
    HistoricalDataCollector,
    timeframe_to_milliseconds,
    timeframe_to_seconds,
)
from app.data.preprocessing.validator import DataValidator


class DummyExchangeClient(ExchangeClient):
    """Mock exchange returning generated candles in chunks."""

    def __init__(self, step_minutes: int = 15):
        self.step_minutes = step_minutes

    async def get_ticker(self, symbol: str) -> Dict[str, Any]:
        return {"symbol": symbol, "last_price": 50000.0}

    async def get_klines(
        self,
        symbol: str,
        timeframe: str,
        start_time: Optional[int] = None,
        end_time: Optional[int] = None,
        limit: int = 1000,
    ) -> List[CandleData]:
        candles: List[CandleData] = []
        curr_ms = start_time or 0
        step_ms = self.step_minutes * 60 * 1000
        count = 0

        while count < limit and (end_time is None or curr_ms <= end_time):
            ts = datetime.fromtimestamp(curr_ms / 1000.0, tz=timezone.utc)
            candles.append(
                CandleData(
                    symbol=symbol,
                    timeframe=timeframe,
                    timestamp=ts,
                    open=50000.0,
                    high=50500.0,
                    low=49500.0,
                    close=50200.0,
                    volume=10.0,
                )
            )
            curr_ms += step_ms
            count += 1

        return candles

    async def get_balance(self) -> Dict[str, float]:
        return {"USDT": 10000.0}

    async def get_positions(self) -> List[Dict[str, Any]]:
        return []

    async def create_order(self, *args, **kwargs) -> Dict[str, Any]:
        return {}

    async def cancel_order(self, *args, **kwargs) -> Dict[str, Any]:
        return {}


def test_timeframe_helpers():
    """Test timeframe duration conversion helpers."""
    assert timeframe_to_seconds("5m") == 300
    assert timeframe_to_seconds("15m") == 900
    assert timeframe_to_seconds("1h") == 3600
    assert timeframe_to_seconds("4h") == 14400
    assert timeframe_to_milliseconds("15m") == 900000

    with pytest.raises(ValueError):
        timeframe_to_seconds("99x")


@pytest.mark.asyncio
async def test_historical_data_collector_pagination():
    """Test pagination across multiple chunks in HistoricalDataCollector."""
    mock_exchange = DummyExchangeClient(step_minutes=15)
    collector = HistoricalDataCollector(mock_exchange)

    start_date = datetime(2026, 1, 1, 0, 0, tzinfo=timezone.utc)
    # 2500 candles = ~26 days of 15m data (requires 3 paginated requests of limit=1000)
    end_date = start_date + timedelta(minutes=15 * 2500)

    candles = await collector.collect_range(
        symbol="BTC/USDT",
        timeframe="15m",
        start_time=start_date,
        end_time=end_date,
        chunk_limit=1000,
        request_delay=0.0,
    )

    assert len(candles) > 2000
    # Verify chronological ascending order
    for i in range(1, len(candles)):
        assert candles[i].timestamp > candles[i - 1].timestamp


@pytest.mark.asyncio
async def test_historical_data_collector_invalid_range():
    """Test exception when start_time >= end_time."""
    collector = HistoricalDataCollector(DummyExchangeClient())
    now = datetime.now(timezone.utc)

    with pytest.raises(ValueError):
        await collector.collect_range(
            symbol="BTC/USDT",
            timeframe="15m",
            start_time=now,
            end_time=now - timedelta(hours=1),
        )


def test_data_validator_clean_series():
    """Test DataValidator passes on clean candle DataFrame."""
    dates = pd.date_range("2026-01-01", periods=100, freq="15min", tz="UTC")
    df = pd.DataFrame(
        {
            "open": [50000.0] * 100,
            "high": [50500.0] * 100,
            "low": [49500.0] * 100,
            "close": [50200.0] * 100,
            "volume": [10.0] * 100,
        },
        index=dates,
    )

    report = DataValidator.validate_dataframe(df, "15m")
    assert report.is_valid is True
    assert report.total_candles == 100
    assert report.gaps_count == 0
    assert report.anomalies_count == 0


def test_data_validator_detects_gaps():
    """Test DataValidator accurately detects missing periods."""
    dates_part1 = pd.date_range("2026-01-01 00:00", periods=10, freq="15min", tz="UTC")
    # Introduce a 1-hour gap (skipping 4 bars)
    dates_part2 = pd.date_range("2026-01-01 03:30", periods=10, freq="15min", tz="UTC")
    dates = dates_part1.append(dates_part2)

    df = pd.DataFrame(
        {
            "open": [50000.0] * 20,
            "high": [50500.0] * 20,
            "low": [49500.0] * 20,
            "close": [50200.0] * 20,
            "volume": [10.0] * 20,
        },
        index=dates,
    )

    report = DataValidator.validate_dataframe(df, "15m")
    assert report.is_valid is False
    assert report.gaps_count == 1
    assert report.gaps_details[0]["missing_candles"] > 0


def test_data_validator_detects_anomalies():
    """Test DataValidator detects corrupted price and volume records."""
    dates = pd.date_range("2026-01-01", periods=5, freq="15min", tz="UTC")
    df = pd.DataFrame(
        {
            "open": [50000.0, 50000.0, 50000.0, 50000.0, 50000.0],
            "high": [50500.0, 48000.0, 50500.0, 50500.0, 50500.0],  # High < Low at index 1
            "low": [49500.0, 49500.0, 49500.0, 49500.0, 49500.0],
            "close": [50200.0, 50200.0, 50200.0, 50200.0, 50200.0],
            "volume": [10.0, 10.0, -5.0, 10.0, 10.0],             # Negative volume at index 2
        },
        index=dates,
    )

    report = DataValidator.validate_dataframe(df, "15m")
    assert report.is_valid is False
    assert report.anomalies_count >= 2
