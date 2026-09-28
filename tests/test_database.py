"""Tests for SQLAlchemy database models, constraints, and queries."""

from datetime import datetime, timezone
import pytest
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from app.data.storage.models import (
    Fill,
    MarketCandle,
    MarketFeature,
    ModelVersion,
    Order,
    PortfolioSnapshot,
    Position,
    Prediction,
    RiskEvent,
    Signal,
    Trade,
    utc_now,
)


@pytest.mark.asyncio
async def test_create_and_query_market_candle(db_session: AsyncSession):
    """Test inserting and retrieving OHLCV candle."""
    now = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)
    candle = MarketCandle(
        symbol="BTC/USDT",
        timeframe="15m",
        timestamp=now,
        open=50000.0,
        high=50500.0,
        low=49800.0,
        close=50200.0,
        volume=12.5,
        is_closed=True,
    )
    db_session.add(candle)
    await db_session.commit()

    result = await db_session.execute(
        select(MarketCandle).where(MarketCandle.symbol == "BTC/USDT")
    )
    retrieved = result.scalar_one()
    assert retrieved.id is not None
    assert retrieved.close == 50200.0
    assert retrieved.symbol == "BTC/USDT"


@pytest.mark.asyncio
async def test_market_candle_unique_constraint(db_session: AsyncSession):
    """Test that duplicate (symbol, timeframe, timestamp) violates unique constraint."""
    now = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)
    candle1 = MarketCandle(
        symbol="ETH/USDT",
        timeframe="1h",
        timestamp=now,
        open=3000.0,
        high=3050.0,
        low=2980.0,
        close=3020.0,
        volume=100.0,
    )
    candle2 = MarketCandle(
        symbol="ETH/USDT",
        timeframe="1h",
        timestamp=now,
        open=3020.0,
        high=3060.0,
        low=3010.0,
        close=3040.0,
        volume=50.0,
    )
    db_session.add(candle1)
    await db_session.commit()

    db_session.add(candle2)
    with pytest.raises(IntegrityError):
        await db_session.commit()
    await db_session.rollback()


@pytest.mark.asyncio
async def test_create_and_query_market_feature(db_session: AsyncSession):
    """Test inserting and retrieving feature metrics."""
    now = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)
    feature = MarketFeature(
        symbol="BTC/USDT",
        timeframe="15m",
        timestamp=now,
        values={
            "rsi": 55.4,
            "macd": 12.3,
            "macd_signal": 10.1,
            "ema_20": 50100.0,
            "return_1": 0.004,
        },
    )
    db_session.add(feature)
    await db_session.commit()

    result = await db_session.execute(
        select(MarketFeature).where(MarketFeature.symbol == "BTC/USDT")
    )
    retrieved = result.scalar_one()
    assert retrieved.values["rsi"] == 55.4
    assert retrieved.values["ema_20"] == 50100.0


@pytest.mark.asyncio
async def test_model_version_and_predictions_relationship(db_session: AsyncSession):
    """Test ModelVersion creation and cascade prediction association."""
    now = utc_now()
    model = ModelVersion(
        version="model_v001",
        model_type="xgboost_classifier",
        training_start_date=datetime(2025, 1, 1, tzinfo=timezone.utc),
        training_end_date=datetime(2025, 12, 31, tzinfo=timezone.utc),
        feature_names=["rsi", "macd", "ema_20", "return_1"],
        hyperparameters={"max_depth": 4, "n_estimators": 100, "learning_rate": 0.05},
        validation_metrics={"accuracy": 0.58, "precision": 0.61, "f1": 0.59},
        test_metrics={"accuracy": 0.56, "precision": 0.59, "f1": 0.57},
        dataset_hash="a1b2c3d4e5f6",
        file_path="models/model_v001.ubj",
        is_active=True,
    )
    db_session.add(model)
    await db_session.commit()

    pred = Prediction(
        timestamp=now,
        symbol="BTC/USDT",
        timeframe="15m",
        model_version_id=model.id,
        probability_up=0.72,
        probability_down=0.28,
        predicted_class=1,
        horizon=12,
        raw_features={"rsi": 62.0},
    )
    db_session.add(pred)
    await db_session.commit()

    result = await db_session.execute(
        select(ModelVersion).where(ModelVersion.version == "model_v001")
    )
    retrieved_model = result.scalar_one()
    assert retrieved_model.is_active is True
    assert len(retrieved_model.predictions) == 1
    assert retrieved_model.predictions[0].probability_up == 0.72


@pytest.mark.asyncio
async def test_order_fill_and_position_flow(db_session: AsyncSession):
    """Test creating Order, associated Fill, and Position."""
    now = utc_now()
    order = Order(
        client_order_id="test_ord_001",
        symbol="BTC/USDT",
        side="BUY",
        order_type="MARKET",
        quantity=0.1,
        price=50000.0,
        stop_loss=49500.0,
        take_profit=51000.0,
        status="FILLED",
        mode="paper",
    )
    db_session.add(order)
    await db_session.commit()

    fill = Fill(
        order_id=order.id,
        client_order_id=order.client_order_id,
        symbol="BTC/USDT",
        side="BUY",
        price=50020.0,  # includes slight slippage
        quantity=0.1,
        fee=5.002,
        fee_asset="USDT",
        slippage_pct=0.0004,
        timestamp=now,
    )
    pos = Position(
        symbol="BTC/USDT",
        side="LONG",
        quantity=0.1,
        entry_price=50020.0,
        current_price=50020.0,
        stop_loss=49500.0,
        take_profit=51000.0,
        is_open=True,
        mode="paper",
    )
    db_session.add_all([fill, pos])
    await db_session.commit()

    res_order = await db_session.execute(
        select(Order).where(Order.client_order_id == "test_ord_001")
    )
    retrieved_order = res_order.scalar_one()
    assert len(retrieved_order.fills) == 1
    assert retrieved_order.fills[0].price == 50020.0


@pytest.mark.asyncio
async def test_portfolio_snapshot_and_risk_events(db_session: AsyncSession):
    """Test PortfolioSnapshot, RiskEvent, and completed Trade records."""
    now = utc_now()
    snapshot = PortfolioSnapshot(
        timestamp=now,
        mode="paper",
        total_equity=10250.0,
        cash_balance=9000.0,
        unrealized_pnl=250.0,
        realized_pnl=0.0,
        peak_equity=10250.0,
        current_drawdown_pct=0.0,
        daily_pnl=250.0,
        daily_return_pct=0.025,
        open_positions_count=1,
    )
    risk_event = RiskEvent(
        timestamp=now,
        event_type="LIMIT_EXCEEDED",
        symbol="ETH/USDT",
        reason="Max risk per trade exceeded threshold",
        details={"risk_pct": 0.008, "max_allowed": 0.005},
        kill_switch_active=False,
    )
    trade = Trade(
        symbol="BTC/USDT",
        side="LONG",
        mode="paper",
        entry_time=datetime(2026, 1, 1, 10, 0, tzinfo=timezone.utc),
        exit_time=datetime(2026, 1, 1, 14, 0, tzinfo=timezone.utc),
        entry_price=50000.0,
        exit_price=51000.0,
        quantity=0.1,
        pnl=100.0,
        pnl_pct=0.02,
        fees=10.0,
        duration_seconds=14400,
        exit_reason="TAKE_PROFIT",
    )
    db_session.add_all([snapshot, risk_event, trade])
    await db_session.commit()

    res_trade = await db_session.execute(select(Trade).where(Trade.symbol == "BTC/USDT"))
    retrieved_trade = res_trade.scalar_one()
    assert retrieved_trade.pnl == 100.0
    assert retrieved_trade.exit_reason == "TAKE_PROFIT"
