"""Tests for VirtualAccount, PaperOrderManager, PaperTradingEngine, and database persistence."""

from datetime import datetime, timedelta, timezone
import numpy as np
import pandas as pd
import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.data.collectors.base import CandleData
from app.data.preprocessing.dataset import DatasetBuilder
from app.data.storage.models import (
    Fill,
    MarketCandle,
    MarketFeature,
    Order,
    PortfolioSnapshot,
    Position,
    Prediction,
    Signal,
    Trade,
)
from app.execution.paper.account import VirtualAccount
from app.execution.paper.engine import PaperTradingEngine
from app.execution.paper.order_manager import PaperOrderManager
from app.models.registry.registry import ModelRegistry
from app.models.training.trainer import ModelTrainer
from app.risk.manager import RiskManager
from app.risk.state import OrderRequest


def generate_sample_candles_for_paper(n_bars: int = 50, start_date="2026-01-01") -> list[CandleData]:
    """Generate deterministic sequence of CandleData objects."""
    dates = pd.date_range(start_date, periods=n_bars, freq="15min", tz="UTC")
    candles = []
    price = 50000.0

    for dt in dates:
        price *= 1.001
        c = CandleData(
            symbol="BTC/USDT",
            timeframe="15m",
            timestamp=dt.to_pydatetime(),
            open=price * 0.999,
            high=price * 1.002,
            low=price * 0.998,
            close=price,
            volume=25.0,
            is_closed=True,
        )
        candles.append(c)
    return candles


# ------------------------------------------------------------------------------
# 1. VirtualAccount Tests
# ------------------------------------------------------------------------------

def test_virtual_account_buy_sell_lifecycle():
    """Verify VirtualAccount cash, equity, fees, and PnL accounting."""
    account = VirtualAccount(initial_cash=10000.0, commission_rate=0.001, slippage_rate=0.0005)

    assert account.cash == 10000.0
    assert account.get_total_equity() == 10000.0

    # Execute BUY 0.1 BTC @ $50,000
    buy_res = account.execute_buy(
        symbol="BTC/USDT",
        quantity=0.1,
        market_price=50000.0,
        stop_loss_price=49000.0,
        take_profit_price=52000.0,
    )

    # Buy fill price = 50000 * 1.0005 = $50,025.00
    # Gross cost = 0.1 * 50025 = $5002.50
    # Fee = 5002.50 * 0.001 = $5.0025
    # Total debited = $5007.5025
    assert np.isclose(buy_res["price"], 50025.0)
    assert np.isclose(account.cash, 10000.0 - 5007.5025)
    assert "BTC/USDT" in account.positions

    # Update price to $51,000
    account.update_price("BTC/USDT", 51000.0)
    assert account.get_unrealized_pnl() > 0
    assert account.get_total_equity() > 10000.0

    # Execute SELL 0.1 BTC @ $51,000
    sell_res = account.execute_sell(
        symbol="BTC/USDT",
        quantity=0.1,
        market_price=51000.0,
        exit_reason="take_profit",
    )

    assert "BTC/USDT" not in account.positions
    assert sell_res["net_pnl"] > 0
    assert account.realized_pnl > 0
    assert account.consecutive_losses == 0


# ------------------------------------------------------------------------------
# 2. PaperOrderManager Tests
# ------------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_order_manager_buy_and_intrabar_exit(db_session: AsyncSession):
    """Verify PaperOrderManager creates orders, fills, positions, and handles intrabar SL exit."""
    account = VirtualAccount(initial_cash=10000.0)
    order_req = OrderRequest(
        symbol="ETH/USDT",
        side="BUY",
        quantity=1.0,
        price=3000.0,
        stop_loss_price=2950.0,  # SL at $2950
        take_profit_price=3100.0,
    )

    # 1. Create and fill BUY
    order, fill, pos = await PaperOrderManager.create_and_fill_buy_order(
        account=account,
        order_req=order_req,
        approved_qty=1.0,
        session=db_session,
    )

    assert order.status == "FILLED"
    assert fill.price > 3000.0  # Slippage applied
    assert pos.is_open is True

    # 2. Check Intrabar SL exit when Low drops to $2900
    trade = await PaperOrderManager.check_intrabar_exits(
        account=account,
        symbol="ETH/USDT",
        high=3020.0,
        low=2900.0,  # Below SL of $2950
        open_price=2980.0,
        timestamp=datetime.now(timezone.utc),
        session=db_session,
    )

    assert trade is not None
    assert trade.exit_reason == "stop_loss"
    assert trade.pnl < 0
    assert account.consecutive_losses == 1

    # Verify DB position is closed
    pos_res = await db_session.execute(select(Position).where(Position.symbol == "ETH/USDT"))
    db_pos = pos_res.scalar_one_or_none()
    assert db_pos.is_open is False


# ------------------------------------------------------------------------------
# 3. PaperTradingEngine End-to-End Test
# ------------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_paper_trading_engine_end_to_end_lifecycle(
    db_session: AsyncSession, tmp_path
):
    """Verify complete PaperTradingEngine processing candles, running inference, risk checks, and persisting DB audit records."""
    # 1. Prepare and register active model
    df_raw = pd.DataFrame(
        {
            "open": np.linspace(40000, 42000, 120),
            "high": np.linspace(40100, 42100, 120),
            "low": np.linspace(39900, 41900, 120),
            "close": np.linspace(40050, 42050, 120),
            "volume": np.full(120, 20.0),
        },
        index=pd.date_range("2026-01-01", periods=120, freq="15min", tz="UTC"),
    )
    dataset = DatasetBuilder.build(candles_df=df_raw, symbol="BTC/USDT", timeframe="15m", horizon=6, drop_na=True)
    splits = DatasetBuilder.split_dataset(dataset)
    train_res = ModelTrainer.train(splits, hyperparameters={"n_estimators": 20}, early_stopping_rounds=5)

    model_dir = str(tmp_path / "models")
    await ModelRegistry.register_model(
        training_result=train_res,
        symbol="BTC/USDT",
        timeframe="15m",
        session=db_session,
        set_active=True,
        model_dir=model_dir,
    )

    # 2. Initialize PaperTradingEngine
    account = VirtualAccount(initial_cash=10000.0)
    risk_manager = RiskManager(max_position_size_pct=0.20, max_daily_loss_pct=0.05)
    engine = PaperTradingEngine(
        account=account,
        risk_manager=risk_manager,
        buy_threshold=0.40,  # Lower threshold to trigger trades in test
        sell_threshold=0.30,
    )
    init_success = await engine.initialize(db_session)
    assert init_success is True

    # 3. Process stream of candles
    candles = generate_sample_candles_for_paper(n_bars=35, start_date="2026-02-01")
    for c in candles:
        step_res = await engine.process_candle(c, db_session)
        assert step_res["equity"] is not None

    # 4. Verify database persistence in all transactional tables
    candles_count = (await db_session.execute(select(MarketCandle))).scalars().all()
    assert len(candles_count) >= 35

    snapshots_count = (await db_session.execute(select(PortfolioSnapshot))).scalars().all()
    assert len(snapshots_count) == 35

    predictions = (await db_session.execute(select(Prediction))).scalars().all()
    assert len(predictions) > 0

    signals = (await db_session.execute(select(Signal))).scalars().all()
    assert len(signals) > 0

    orders = (await db_session.execute(select(Order))).scalars().all()
    fills = (await db_session.execute(select(Fill))).scalars().all()
    assert len(orders) == len(fills)
