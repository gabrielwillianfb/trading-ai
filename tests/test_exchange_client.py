"""Unit tests for ExchangeClient and BinanceClient with mock HTTP transport."""

import pytest
import httpx
from app.data.collectors.base import (
    AuthenticationError,
    CandleData,
    ExchangeError,
    RateLimitError,
)
from app.data.collectors.binance import BinanceClient


def test_symbol_normalization():
    """Verify symbol conversions between exchange format and internal format."""
    assert BinanceClient.to_exchange_symbol("BTC/USDT") == "BTCUSDT"
    assert BinanceClient.to_exchange_symbol("ETH-USDT") == "ETHUSDT"
    assert BinanceClient.from_exchange_symbol("BTCUSDT") == "BTC/USDT"
    assert BinanceClient.from_exchange_symbol("ETHUSDC") == "ETH/USDC"


def test_hmac_signature_generation():
    """Verify HMAC-SHA256 signature generation."""
    client = BinanceClient(api_key="test_key", api_secret="test_secret")
    params = {"symbol": "BTCUSDT", "side": "BUY", "quantity": "1.0", "timestamp": 1600000000000}
    sig = client._sign_payload(params)
    assert isinstance(sig, str)
    assert len(sig) == 64  # SHA256 hex digest length


@pytest.mark.asyncio
async def test_binance_get_klines_parsing():
    """Test parsing Binance raw kline array to normalized CandleData."""
    # Mock Binance Kline response payload
    mock_raw_kline = [
        [
            1700000000000,      # Open time ms
            "50000.00",         # Open
            "50500.00",         # High
            "49800.00",         # Low
            "50200.00",         # Close
            "15.50000000",      # Volume
            1700000899999,      # Close time ms
            "778100.00000000",  # Quote asset volume
            120,                # Number of trades
            "8.00000000",       # Taker buy base asset volume
            "401600.00000000",  # Taker buy quote asset volume
            "0",
        ]
    ]

    def mock_handler(request: httpx.Request) -> httpx.Response:
        assert "/v3/klines" in str(request.url)
        assert "symbol=BTCUSDT" in str(request.url)
        assert "interval=15m" in str(request.url)
        return httpx.Response(200, json=mock_raw_kline)

    transport = httpx.MockTransport(mock_handler)
    client = BinanceClient(api_key="test_key", api_secret="test_secret")
    client._http_client = httpx.AsyncClient(transport=transport)

    try:
        candles = await client.get_klines(symbol="BTC/USDT", timeframe="15m", limit=1)
        assert len(candles) == 1
        c = candles[0]
        assert isinstance(c, CandleData)
        assert c.symbol == "BTC/USDT"
        assert c.timeframe == "15m"
        assert c.open == 50000.0
        assert c.high == 50500.0
        assert c.low == 49800.0
        assert c.close == 50200.0
        assert c.volume == 15.5
        assert c.quote_volume == 778100.0
        assert c.trades_count == 120
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_binance_get_ticker():
    """Test get_ticker endpoint parsing."""
    mock_ticker_resp = {
        "symbol": "BTCUSDT",
        "lastPrice": "51200.50",
        "bidPrice": "51200.00",
        "askPrice": "51201.00",
        "volume": "1420.50",
        "highPrice": "52000.00",
        "lowPrice": "49500.00",
        "priceChangePercent": "2.35",
        "closeTime": 1700001000000,
    }

    def mock_handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=mock_ticker_resp)

    transport = httpx.MockTransport(mock_handler)
    client = BinanceClient()
    client._http_client = httpx.AsyncClient(transport=transport)

    try:
        ticker = await client.get_ticker("BTC/USDT")
        assert ticker["symbol"] == "BTC/USDT"
        assert ticker["last_price"] == 51200.50
        assert ticker["bid"] == 51200.00
        assert ticker["ask"] == 51201.00
        assert ticker["price_change_pct"] == 2.35
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_binance_rate_limiting_error():
    """Test that HTTP 429 raises RateLimitError after retries."""
    def mock_handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(429, text="Too Many Requests", headers={"Retry-After": "0"})

    transport = httpx.MockTransport(mock_handler)
    client = BinanceClient(max_retries=1)
    client._http_client = httpx.AsyncClient(transport=transport)

    try:
        with pytest.raises(RateLimitError):
            await client.get_ticker("BTC/USDT")
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_binance_authentication_error():
    """Test that HTTP 401 raises AuthenticationError."""
    def mock_handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, text="Unauthorized: Invalid API-key")

    transport = httpx.MockTransport(mock_handler)
    client = BinanceClient(api_key="bad_key", api_secret="bad_secret")
    client._http_client = httpx.AsyncClient(transport=transport)

    try:
        with pytest.raises(AuthenticationError):
            await client.get_balance()
    finally:
        await client.close()
