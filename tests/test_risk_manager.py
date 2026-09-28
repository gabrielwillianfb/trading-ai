"""Tests for RiskManager, risk rules, position limits, and circuit breakers."""

from datetime import datetime, timezone
import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from app.risk.manager import RiskManager
from app.risk.state import (
    OrderRequest,
    PortfolioState,
    RiskAction,
    RiskSeverity,
)


@pytest.fixture
def risk_manager() -> RiskManager:
    """Create standard RiskManager instance for unit tests."""
    return RiskManager(
        max_position_size_pct=0.10,  # Max 10% position size
        max_daily_loss_pct=0.03,  # Max 3% daily loss
        max_drawdown_pct=0.05,  # Max 5% drawdown
        max_consecutive_losses=4,  # Max 4 consecutive losses
        require_stop_loss=True,
    )


def test_order_validation_approved(risk_manager: RiskManager):
    """Verify normal compliant order is approved without modifications."""
    portfolio = PortfolioState(
        cash=10000.0,
        equity=10000.0,
        peak_equity=10000.0,
        daily_pnl=0.0,
        consecutive_losses=0,
    )
    # Order value = 0.01 * 50000 = $500 (5% of equity <= 10% limit)
    order = OrderRequest(
        symbol="BTC/USDT",
        side="BUY",
        quantity=0.01,
        price=50000.0,
        stop_loss_price=49500.0,  # -1.0%
        take_profit_price=51000.0,  # +2.0%
    )

    result = risk_manager.validate_order(order, portfolio)
    assert result.allowed is True
    assert result.action == RiskAction.APPROVED
    assert result.approved_quantity == 0.01


def test_position_size_capping_and_adjustment(risk_manager: RiskManager):
    """Verify order exceeding max position size is downsized to 10% cap."""
    portfolio = PortfolioState(
        cash=10000.0,
        equity=10000.0,
        peak_equity=10000.0,
    )
    # Proposed order value = 0.05 * 50000 = $2500 (25% of equity > 10% limit)
    order = OrderRequest(
        symbol="BTC/USDT",
        side="BUY",
        quantity=0.05,
        price=50000.0,
        stop_loss_price=49500.0,
    )

    # With allow_partial_fill=True -> Downsize to $1000 / 50000 = 0.02 BTC
    result = risk_manager.validate_order(order, portfolio, allow_partial_fill=True)
    assert result.allowed is True
    assert result.action == RiskAction.ADJUSTED
    assert result.approved_quantity == 0.02
    assert result.original_quantity == 0.05

    # With allow_partial_fill=False -> REJECT
    result_no_part = risk_manager.validate_order(order, portfolio, allow_partial_fill=False)
    assert result_no_part.allowed is False
    assert result_no_part.action == RiskAction.REJECTED
    assert result_no_part.rule_triggered == "MAX_POSITION_SIZE_EXCEEDED"


def test_mandatory_stop_loss_rejection(risk_manager: RiskManager):
    """Verify BUY order without valid stop loss is rejected."""
    portfolio = PortfolioState(cash=10000.0, equity=10000.0, peak_equity=10000.0)

    # 1. Missing stop loss
    order_none = OrderRequest(
        symbol="BTC/USDT",
        side="BUY",
        quantity=0.01,
        price=50000.0,
        stop_loss_price=None,
    )
    res_none = risk_manager.validate_order(order_none, portfolio)
    assert res_none.allowed is False
    assert res_none.rule_triggered == "MANDATORY_STOP_LOSS_REQUIRED"

    # 2. Invalid stop loss (above entry price)
    order_invalid = OrderRequest(
        symbol="BTC/USDT",
        side="BUY",
        quantity=0.01,
        price=50000.0,
        stop_loss_price=51000.0,
    )
    res_inv = risk_manager.validate_order(order_invalid, portfolio)
    assert res_inv.allowed is False
    assert res_inv.rule_triggered == "MANDATORY_STOP_LOSS_REQUIRED"


def test_daily_loss_limit_rejection(risk_manager: RiskManager):
    """Verify new BUY orders are rejected if daily loss exceeds 3%."""
    # Equity = $10,000, Daily PnL = -$350 (-3.5% > 3% limit)
    portfolio = PortfolioState(
        cash=9650.0,
        equity=9650.0,
        peak_equity=10000.0,
        daily_pnl=-350.0,
    )
    order = OrderRequest(
        symbol="BTC/USDT",
        side="BUY",
        quantity=0.01,
        price=50000.0,
        stop_loss_price=49500.0,
    )

    result = risk_manager.validate_order(order, portfolio)
    assert result.allowed is False
    assert result.action == RiskAction.REJECTED
    assert result.rule_triggered == "MAX_DAILY_LOSS_EXCEEDED"

    # However, a SELL order to close positions must still be allowed
    sell_order = OrderRequest(
        symbol="BTC/USDT",
        side="SELL",
        quantity=0.01,
        price=50000.0,
    )
    sell_res = risk_manager.validate_order(sell_order, portfolio)
    assert sell_res.allowed is True
    assert sell_res.action == RiskAction.APPROVED


def test_max_drawdown_trips_circuit_breaker(risk_manager: RiskManager):
    """Verify drawdown >= 5% trips the circuit breaker and blocks trading."""
    # Peak = $10,000, Current Equity = $9,400 (Drawdown = 6% > 5% limit)
    portfolio = PortfolioState(
        cash=9400.0,
        equity=9400.0,
        peak_equity=10000.0,
    )
    order = OrderRequest(
        symbol="BTC/USDT",
        side="BUY",
        quantity=0.01,
        price=50000.0,
        stop_loss_price=49500.0,
    )

    result = risk_manager.validate_order(order, portfolio)
    assert result.allowed is False
    assert result.action == RiskAction.CIRCUIT_BREAKER_TRIGGERED
    assert risk_manager.is_circuit_breaker_active is True

    # Subsequent orders must also be rejected due to active circuit breaker
    subsequent_order = OrderRequest(symbol="BTC/USDT", side="BUY", quantity=0.001, price=50000.0, stop_loss_price=49000.0)
    sub_res = risk_manager.validate_order(subsequent_order, portfolio)
    assert sub_res.allowed is False
    assert sub_res.rule_triggered == "CIRCUIT_BREAKER_ACTIVE"


def test_consecutive_losses_trips_circuit_breaker(risk_manager: RiskManager):
    """Verify 4 consecutive losses trips the circuit breaker."""
    portfolio = PortfolioState(
        cash=9800.0,
        equity=9800.0,
        peak_equity=10000.0,
        consecutive_losses=4,  # Reached limit of 4
    )
    order = OrderRequest(
        symbol="BTC/USDT",
        side="BUY",
        quantity=0.01,
        price=50000.0,
        stop_loss_price=49500.0,
    )

    result = risk_manager.validate_order(order, portfolio)
    assert result.allowed is False
    assert result.action == RiskAction.CIRCUIT_BREAKER_TRIGGERED
    assert result.rule_triggered == "MAX_CONSECUTIVE_LOSSES_EXCEEDED"
    assert risk_manager.is_circuit_breaker_active is True


def test_circuit_breaker_manual_trip_and_reset(risk_manager: RiskManager):
    """Verify manual emergency stop and reset functionality."""
    assert risk_manager.is_circuit_breaker_active is False

    # Manual emergency trigger
    risk_manager.trigger_circuit_breaker("Emergency market volatility spike")
    assert risk_manager.is_circuit_breaker_active is True
    assert "Emergency market volatility" in risk_manager.circuit_breaker_reason

    # Manual reset
    risk_manager.reset_circuit_breaker("Risk supervisor reviewed market conditions")
    assert risk_manager.is_circuit_breaker_active is False
    assert risk_manager.circuit_breaker_reason is None


@pytest.mark.asyncio
async def test_risk_event_db_persistence(db_session: AsyncSession):
    """Verify audit logging of risk events into risk_events database table."""
    event = await RiskManager.record_risk_event(
        event_type="CIRCUIT_BREAKER",
        reason="Max drawdown exceeded 5%",
        severity="CRITICAL",
        details={"peak_equity": 10000.0, "current_equity": 9400.0, "drawdown_pct": 6.0},
        kill_switch_active=True,
        session=db_session,
    )

    assert event is not None
    assert event.id is not None
    assert event.event_type == "CIRCUIT_BREAKER"
    assert event.reason == "Max drawdown exceeded 5%"
    assert event.kill_switch_active is True
    assert event.details["severity"] == "CRITICAL"
    assert event.details["drawdown_pct"] == 6.0
