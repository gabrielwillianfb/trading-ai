"""Tests for Binance Testnet execution, SymbolFilters, and TestnetOrderManager."""

from datetime import datetime, timezone
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.data.collectors.binance import BinanceClient
from app.data.storage.models import Fill, Order, Position, Trade
from app.execution.testnet.engine import TestnetTradingEngine
from app.execution.testnet.filters import SymbolFilters
from app.execution.testnet.order_manager import TestnetOrderManager
from app.risk.state import OrderRequest


# ------------------------------------------------------------------------------
# 1. SymbolFilters Tests
# ------------------------------------------------------------------------------

def test_symbol_filters_formatting():
    """Verify SymbolFilters formats quantity, price, and checks minNotional correctly."""
    filters = SymbolFilters(
        symbol="BTCUSDT",
        min_qty=0.00001,
        max_qty=100.0,
        step_size=0.00001,
        min_price=0.01,
        max_price=1000000.0,
        tick_size=0.01,
        min_notional=5.0,
    )

    # 1. Quantity formatting
    assert filters.format_quantity(0.12345678) == 0.12345
    assert filters.format_quantity(0.000005) == 0.0  # Below min_qty

    # 2. Price formatting
    assert filters.format_price(50123.456) == 50123.46

    # 3. Notional validation
    assert filters.validate_notional(0.001, 50000.0) is True  # $50 >= $5
    assert filters.validate_notional(0.00001, 100.0) is False  # $0.001 < $5


def test_symbol_filters_from_exchange_info():
    """Verify parsing SymbolFilters from Binance exchangeInfo dictionary."""
    raw_info = {
        "symbol": "ETHUSDT",
        "filters": [
            {"filterType": "PRICE_FILTER", "minPrice": "0.01", "maxPrice": "100000.00", "tickSize": "0.01"},
            {"filterType": "LOT_SIZE", "minQty": "0.0001", "maxQty": "9000.0", "stepSize": "0.0001"},
            {"filterType": "NOTIONAL", "minNotional": "5.0"},
        ],
    }

    filters = SymbolFilters.from_exchange_info(raw_info)
    assert filters.symbol == "ETHUSDT"
    assert filters.step_size == 0.0001
    assert filters.tick_size == 0.01
    assert filters.min_notional == 5.0


# ------------------------------------------------------------------------------
# 2. Testnet Order Manager Tests (with Mocked BinanceClient)
# ------------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_testnet_order_manager_buy_and_sell_flow(db_session: AsyncSession):
    """Verify TestnetOrderManager interacts with BinanceClient and persists records into database."""
    mock_client = MagicMock(spec=BinanceClient)
    mock_client.base_url = "https://testnet.binance.vision/api"

    # Mock get_exchange_info
    mock_client.get_exchange_info = AsyncMock(
        return_value={
            "symbols": [
                {
                    "symbol": "BTCUSDT",
                    "filters": [
                        {"filterType": "LOT_SIZE", "minQty": "0.00001", "stepSize": "0.00001"},
                        {"filterType": "NOTIONAL", "minNotional": "5.0"},
                    ],
                }
            ]
        }
    )

    # Mock get_account_info
    mock_client.get_account_info = AsyncMock(
        return_value={
            "canTrade": True,
            "balances": [
                {"asset": "USDT", "free": "10000.00", "locked": "0.00"},
                {"asset": "BTC", "free": "0.00", "locked": "0.00"},
            ],
        }
    )

    # Mock create_order BUY
    mock_client.create_order = AsyncMock(
        return_value={
            "orderId": 12345678,
            "status": "FILLED",
            "executedQty": "0.02000",
            "cummulativeQuoteQty": "1000.00",
        }
    )

    manager = TestnetOrderManager(client=mock_client)

    # 1. Fetch testnet portfolio state
    state = await manager.fetch_testnet_portfolio_state("BTC/USDT", current_price=50000.0)
    assert state.cash == 10000.0
    assert state.equity == 10000.0

    # 2. Execute Testnet BUY
    order_req = OrderRequest(
        symbol="BTC/USDT",
        side="BUY",
        quantity=0.02,
        price=50000.0,
        stop_loss_price=49000.0,
        take_profit_price=52000.0,
    )

    order, fill, pos = await manager.execute_testnet_buy_order(
        order_req=order_req,
        approved_qty=0.02,
        session=db_session,
    )

    assert order.mode == "testnet"
    assert order.status == "FILLED"
    assert order.exchange_order_id == "12345678"
    assert fill is not None
    assert fill.quantity == 0.02
    assert pos.is_open is True

    # 3. Execute Testnet SELL
    mock_client.create_order = AsyncMock(
        return_value={
            "orderId": 87654321,
            "status": "FILLED",
            "executedQty": "0.02000",
            "cummulativeQuoteQty": "1040.00",  # Sold @ $52,000 -> +$40 PnL
        }
    )

    sell_order, sell_fill, trade = await manager.execute_testnet_sell_order(
        symbol="BTC/USDT",
        quantity=0.02,
        market_price=52000.0,
        exit_reason="take_profit",
        session=db_session,
    )

    assert sell_order.status == "FILLED"
    assert sell_order.exchange_order_id == "87654321"
    assert sell_fill is not None
    assert trade is not None
    assert trade.pnl > 0
    assert trade.mode == "testnet"

    # Verify DB position is closed
    pos_res = await db_session.execute(
        select(Position).where(Position.symbol == "BTC/USDT", Position.mode == "testnet")
    )
    db_pos = pos_res.scalar_one_or_none()
    assert db_pos.is_open is False


# ------------------------------------------------------------------------------
# 3. Security Assertion Tests
# ------------------------------------------------------------------------------

def test_engine_rejects_production_urls():
    """Verify TestnetTradingEngine rejects any client configured with a production URL."""
    prod_client = MagicMock(spec=BinanceClient)
    prod_client.base_url = "https://api.binance.com/api"

    with pytest.raises(ValueError, match="SECURITY VIOLATION"):
        TestnetTradingEngine(client=prod_client)
