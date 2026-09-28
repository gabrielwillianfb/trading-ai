"""Tests for FastAPI REST API v1 endpoints (Status, Models, Portfolio, Signals, Orders, Trades, Risk)."""

from datetime import datetime, timezone
import pytest
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession

from app.data.storage.db import get_session
from app.data.storage.models import (
    Fill,
    ModelVersion,
    Order,
    PortfolioSnapshot,
    Position,
    Prediction,
    RiskEvent,
    Signal,
    Trade,
)
from app.main import app


@pytest.fixture
async def client(db_session: AsyncSession):
    """Create async test client overriding the database session dependency."""
    async def override_get_session():
        yield db_session

    app.dependency_overrides[get_session] = override_get_session
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac
    app.dependency_overrides.clear()


# ------------------------------------------------------------------------------
# 1. Status & Models Endpoints
# ------------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_get_system_status(client: AsyncClient):
    """Verify GET /api/v1/status returns 200 with operational flags."""
    response = await client.get("/api/v1/status")
    assert response.status_code == 200
    data = response.json()

    assert data["app_name"] == "TradingAI"
    assert data["trading_mode"] in ("paper", "testnet")
    assert data["is_live_trading_forbidden"] is True
    assert data["database_connected"] is True
    assert "server_time_utc" in data


@pytest.mark.asyncio
async def test_get_models_list(client: AsyncClient, db_session: AsyncSession):
    """Verify GET /api/v1/models returns list of registered model versions."""
    now = datetime.now(timezone.utc)
    model = ModelVersion(
        version="model_v099",
        model_type="xgboost",
        file_path="models/model_v099.ubj",
        is_active=True,
        training_start_date=now,
        training_end_date=now,
        trained_at=now,
        hyperparameters={"n_estimators": 50},
        validation_metrics={"f1": 0.65},
        test_metrics={"f1": 0.62, "accuracy": 0.60},
        feature_names=["rsi_14", "macd"],
        created_at=now,
    )
    db_session.add(model)
    await db_session.commit()

    response = await client.get("/api/v1/models")
    assert response.status_code == 200
    data = response.json()
    assert len(data) >= 1
    versions = [m["version"] for m in data]
    assert "model_v099" in versions


# ------------------------------------------------------------------------------
# 2. Portfolio Endpoint
# ------------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_get_portfolio_snapshot_and_positions(client: AsyncClient, db_session: AsyncSession):
    """Verify GET /api/v1/portfolio returns latest snapshot and open positions."""
    now = datetime.now(timezone.utc)
    snap = PortfolioSnapshot(
        timestamp=now,
        mode="paper",
        total_equity=10500.0,
        cash_balance=9000.0,
        unrealized_pnl=500.0,
        realized_pnl=200.0,
        peak_equity=10600.0,
        current_drawdown_pct=0.94,
        daily_pnl=150.0,
        daily_return_pct=1.5,
        open_positions_count=1,
    )
    pos = Position(
        symbol="BTC/USDT",
        side="LONG",
        quantity=0.03,
        entry_price=50000.0,
        current_price=51000.0,
        unrealized_pnl=30.0,
        realized_pnl=0.0,
        is_open=True,
        mode="paper",
        opened_at=now,
    )
    db_session.add(snap)
    db_session.add(pos)
    await db_session.commit()

    response = await client.get("/api/v1/portfolio")
    assert response.status_code == 200
    data = response.json()

    assert data["total_equity"] == 10500.0
    assert data["cash_balance"] == 9000.0
    assert len(data["open_positions"]) >= 1
    assert data["open_positions"][0]["symbol"] == "BTC/USDT"


# ------------------------------------------------------------------------------
# 3. Signals, Orders, and Trades Endpoints
# ------------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_get_signals(client: AsyncClient, db_session: AsyncSession):
    """Verify GET /api/v1/signals returns historical decision signals."""
    now = datetime.now(timezone.utc)
    sig = Signal(
        timestamp=now,
        symbol="ETH/USDT",
        timeframe="15m",
        signal_type="BUY",
        probability_up=0.72,
        probability_down=0.28,
        confidence=0.44,
        threshold_used=0.65,
        is_processed=True,
    )
    db_session.add(sig)
    await db_session.commit()

    response = await client.get("/api/v1/signals?symbol=ETH/USDT")
    assert response.status_code == 200
    data = response.json()
    assert len(data) >= 1
    assert data[0]["symbol"] == "ETH/USDT"
    assert data[0]["signal_type"] == "BUY"


@pytest.mark.asyncio
async def test_get_orders_and_fills(client: AsyncClient, db_session: AsyncSession):
    """Verify GET /api/v1/orders returns orders with nested fills."""
    now = datetime.now(timezone.utc)
    order = Order(
        client_order_id="test_ord_001",
        exchange_order_id="ex_001",
        symbol="BTC/USDT",
        side="BUY",
        order_type="MARKET",
        quantity=0.01,
        price=50000.0,
        status="FILLED",
        mode="paper",
        created_at=now,
        updated_at=now,
    )
    db_session.add(order)
    await db_session.flush()

    fill = Fill(
        order_id=order.id,
        client_order_id="test_ord_001",
        symbol="BTC/USDT",
        side="BUY",
        price=50025.0,
        quantity=0.01,
        fee=0.50,
        fee_asset="USDT",
        slippage_pct=0.0005,
        timestamp=now,
    )
    db_session.add(fill)
    await db_session.commit()

    response = await client.get("/api/v1/orders?symbol=BTC/USDT")
    assert response.status_code == 200
    data = response.json()
    assert len(data) >= 1
    assert data[0]["client_order_id"] == "test_ord_001"
    assert len(data[0]["fills"]) == 1
    assert data[0]["fills"][0]["price"] == 50025.0


@pytest.mark.asyncio
async def test_get_trades_summary(client: AsyncClient, db_session: AsyncSession):
    """Verify GET /api/v1/trades aggregates performance metrics across closed trades."""
    now = datetime.now(timezone.utc)
    t1 = Trade(
        symbol="BTC/USDT",
        side="LONG",
        mode="paper",
        entry_time=now,
        exit_time=now,
        entry_price=50000.0,
        exit_price=51000.0,
        quantity=0.1,
        pnl=95.0,
        pnl_pct=1.9,
        fees=5.0,
        duration_seconds=900,
        exit_reason="take_profit",
    )
    t2 = Trade(
        symbol="BTC/USDT",
        side="LONG",
        mode="paper",
        entry_time=now,
        exit_time=now,
        entry_price=50000.0,
        exit_price=49500.0,
        quantity=0.1,
        pnl=-55.0,
        pnl_pct=-1.1,
        fees=5.0,
        duration_seconds=900,
        exit_reason="stop_loss",
    )
    db_session.add(t1)
    db_session.add(t2)
    await db_session.commit()

    response = await client.get("/api/v1/trades")
    assert response.status_code == 200
    data = response.json()

    assert data["total_trades"] >= 2
    assert data["winning_trades"] >= 1
    assert data["losing_trades"] >= 1
    assert data["win_rate_pct"] > 0
    assert len(data["trades"]) >= 2


# ------------------------------------------------------------------------------
# 4. Risk Events & Kill Switch Endpoints
# ------------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_get_risk_events(client: AsyncClient, db_session: AsyncSession):
    """Verify GET /api/v1/risk/events returns audit trail."""
    now = datetime.now(timezone.utc)
    ev = RiskEvent(
        timestamp=now,
        event_type="MAX_DAILY_LOSS_EXCEEDED",
        symbol="BTC/USDT",
        reason="Daily loss 3.2% exceeded 3.0% threshold",
        details={"loss_pct": 3.2},
        kill_switch_active=False,
    )
    db_session.add(ev)
    await db_session.commit()

    response = await client.get("/api/v1/risk/events")
    assert response.status_code == 200
    data = response.json()
    assert len(data) >= 1
    assert data[0]["event_type"] == "MAX_DAILY_LOSS_EXCEEDED"


@pytest.mark.asyncio
async def test_post_kill_switch_trigger_and_reset(client: AsyncClient):
    """Verify POST /api/v1/risk/kill-switch manually activates and resets circuit breaker."""
    # 1. Trigger
    res_trig = await client.post(
        "/api/v1/risk/kill-switch",
        json={"action": "trigger", "reason": "Severe market anomaly observed"},
    )
    assert res_trig.status_code == 200
    data_trig = res_trig.json()
    assert data_trig["success"] is True
    assert data_trig["circuit_breaker_active"] is True

    # 2. Reset
    res_reset = await client.post(
        "/api/v1/risk/kill-switch",
        json={"action": "reset", "reason": "Anomaly resolved by operator"},
    )
    assert res_reset.status_code == 200
    data_reset = res_reset.json()
    assert data_reset["success"] is True
    assert data_reset["circuit_breaker_active"] is False

    # 3. Invalid action
    res_inv = await client.post(
        "/api/v1/risk/kill-switch",
        json={"action": "invalid_action", "reason": "test"},
    )
    assert res_inv.status_code == 400
