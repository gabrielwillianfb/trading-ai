"""Independent Risk Manager governing all order authorizations, portfolio safety, and circuit breakers."""

import logging
from datetime import datetime, timezone
from typing import Any, Dict, Optional
from sqlalchemy.ext.asyncio import AsyncSession

from app.config.settings import Settings, get_settings
from app.data.storage.models import RiskEvent
from app.risk.state import (
    OrderRequest,
    PortfolioState,
    RiskAction,
    RiskCheckResult,
    RiskSeverity,
)

logger = logging.getLogger(__name__)


class RiskManager:
    """Autonomous risk control engine enforcing mandatory trading bounds and capital protection."""

    def __init__(
        self,
        max_position_size_pct: float = 0.10,
        max_daily_loss_pct: float = 0.03,
        max_drawdown_pct: float = 0.05,
        max_consecutive_losses: int = 4,
        require_stop_loss: bool = True,
        settings: Optional[Settings] = None,
    ):
        if settings is not None:
            self.max_position_size_pct = settings.MAX_POSITION_SIZE_PCT
            self.max_daily_loss_pct = settings.MAX_DAILY_LOSS_PCT
            self.max_drawdown_pct = settings.MAX_DRAWDOWN_PCT
            self.max_consecutive_losses = settings.MAX_CONSECUTIVE_LOSSES
        else:
            self.max_position_size_pct = max_position_size_pct
            self.max_daily_loss_pct = max_daily_loss_pct
            self.max_drawdown_pct = max_drawdown_pct
            self.max_consecutive_losses = max_consecutive_losses

        self.require_stop_loss = require_stop_loss

        # Circuit breaker internal state
        self.is_circuit_breaker_active: bool = False
        self.circuit_breaker_reason: Optional[str] = None
        self.circuit_breaker_triggered_at: Optional[datetime] = None

    def trigger_circuit_breaker(self, reason: str) -> None:
        """Trip emergency stop circuit breaker and halt all new order authorizations."""
        self.is_circuit_breaker_active = True
        self.circuit_breaker_reason = reason
        self.circuit_breaker_triggered_at = datetime.now(timezone.utc)
        logger.critical(f"CIRCUIT BREAKER TRIGGERED! Reason: {reason}")

    def reset_circuit_breaker(self, reason: str) -> None:
        """Manually clear circuit breaker status with audit reasoning."""
        prev_reason = self.circuit_breaker_reason
        self.is_circuit_breaker_active = False
        self.circuit_breaker_reason = None
        self.circuit_breaker_triggered_at = None
        logger.info(f"Circuit Breaker reset: '{reason}' (Previous reason was: '{prev_reason}')")

    def validate_order(
        self,
        order: OrderRequest,
        portfolio: PortfolioState,
        allow_partial_fill: bool = True,
    ) -> RiskCheckResult:
        """Evaluate an incoming order request against all risk parameters.

        Args:
            order: Proposed OrderRequest from trading strategy.
            portfolio: Current financial state of account.
            allow_partial_fill: Whether to downsize orders exceeding limits instead of outright rejection.

        Returns:
            RiskCheckResult with authorization decision and approved size.
        """
        # 1. Circuit Breaker Check
        if self.is_circuit_breaker_active:
            return RiskCheckResult(
                allowed=False,
                action=RiskAction.REJECTED,
                original_quantity=order.quantity,
                approved_quantity=0.0,
                reasons=[f"Circuit Breaker is ACTIVE: {self.circuit_breaker_reason}"],
                rule_triggered="CIRCUIT_BREAKER_ACTIVE",
                severity=RiskSeverity.CRITICAL,
            )

        # 2. Mandatory Stop Loss Check (for BUY orders)
        if order.side.upper() == "BUY" and self.require_stop_loss:
            if order.stop_loss_price is None or order.stop_loss_price >= order.price or order.stop_loss_price <= 0:
                return RiskCheckResult(
                    allowed=False,
                    action=RiskAction.REJECTED,
                    original_quantity=order.quantity,
                    approved_quantity=0.0,
                    reasons=[
                        f"Missing or invalid Stop Loss price: {order.stop_loss_price} for entry price {order.price}"
                    ],
                    rule_triggered="MANDATORY_STOP_LOSS_REQUIRED",
                    severity=RiskSeverity.WARNING,
                )

        # 3. Consecutive Losses Check (Kill Switch)
        if portfolio.consecutive_losses >= self.max_consecutive_losses:
            reason = (
                f"Consecutive losses limit reached: {portfolio.consecutive_losses} >= {self.max_consecutive_losses}"
            )
            self.trigger_circuit_breaker(reason)
            return RiskCheckResult(
                allowed=False,
                action=RiskAction.CIRCUIT_BREAKER_TRIGGERED,
                original_quantity=order.quantity,
                approved_quantity=0.0,
                reasons=[reason],
                rule_triggered="MAX_CONSECUTIVE_LOSSES_EXCEEDED",
                severity=RiskSeverity.CRITICAL,
            )

        # 4. Maximum Drawdown Check
        max_dd_limit_pct = self.max_drawdown_pct * 100.0
        if portfolio.current_drawdown_pct >= max_dd_limit_pct:
            reason = (
                f"Maximum Drawdown limit breached: {portfolio.current_drawdown_pct:.2f}% >= {max_dd_limit_pct:.2f}%"
            )
            self.trigger_circuit_breaker(reason)
            return RiskCheckResult(
                allowed=False,
                action=RiskAction.CIRCUIT_BREAKER_TRIGGERED,
                original_quantity=order.quantity,
                approved_quantity=0.0,
                reasons=[reason],
                rule_triggered="MAX_DRAWDOWN_EXCEEDED",
                severity=RiskSeverity.CRITICAL,
            )

        # 5. Maximum Daily Loss Check (for BUY orders)
        if order.side.upper() == "BUY":
            max_daily_limit_pct = self.max_daily_loss_pct * 100.0
            if portfolio.daily_loss_pct >= max_daily_limit_pct:
                return RiskCheckResult(
                    allowed=False,
                    action=RiskAction.REJECTED,
                    original_quantity=order.quantity,
                    approved_quantity=0.0,
                    reasons=[
                        f"Daily Loss limit breached: {portfolio.daily_loss_pct:.2f}% >= {max_daily_limit_pct:.2f}%"
                    ],
                    rule_triggered="MAX_DAILY_LOSS_EXCEEDED",
                    severity=RiskSeverity.WARNING,
                )

        # SELL orders (closing positions) are always authorized if above checks pass
        if order.side.upper() == "SELL":
            return RiskCheckResult(
                allowed=True,
                action=RiskAction.APPROVED,
                original_quantity=order.quantity,
                approved_quantity=order.quantity,
                reasons=["Sell / position reduction order authorized."],
                severity=RiskSeverity.INFO,
            )

        # 6. Position Sizing Check (MAX_POSITION_SIZE_PCT)
        max_position_capital = portfolio.equity * self.max_position_size_pct
        proposed_cost = order.quantity * order.price
        approved_qty = order.quantity
        adjusted = False
        reasons = []

        if proposed_cost > max_position_capital:
            if not allow_partial_fill:
                return RiskCheckResult(
                    allowed=False,
                    action=RiskAction.REJECTED,
                    original_quantity=order.quantity,
                    approved_quantity=0.0,
                    reasons=[
                        f"Proposed order value ${proposed_cost:.2f} exceeds position cap of ${max_position_capital:.2f}"
                    ],
                    rule_triggered="MAX_POSITION_SIZE_EXCEEDED",
                    severity=RiskSeverity.WARNING,
                )
            else:
                approved_qty = max_position_capital / order.price
                adjusted = True
                reasons.append(
                    f"Position size adjusted down from {order.quantity:.6f} to {approved_qty:.6f} "
                    f"to respect MAX_POSITION_SIZE_PCT ({self.max_position_size_pct*100:.1f}%)"
                )

        # 7. Available Cash Balance Check
        estimated_fee = (approved_qty * order.price) * 0.001
        total_required_cash = (approved_qty * order.price) + estimated_fee

        if total_required_cash > portfolio.cash:
            if not allow_partial_fill:
                return RiskCheckResult(
                    allowed=False,
                    action=RiskAction.REJECTED,
                    original_quantity=order.quantity,
                    approved_quantity=0.0,
                    reasons=[
                        f"Insufficient cash: required ${total_required_cash:.2f}, available ${portfolio.cash:.2f}"
                    ],
                    rule_triggered="INSUFFICIENT_CASH_BALANCE",
                    severity=RiskSeverity.WARNING,
                )
            else:
                max_affordable_qty = portfolio.cash / (order.price * 1.001)
                if max_affordable_qty <= 0:
                    return RiskCheckResult(
                        allowed=False,
                        action=RiskAction.REJECTED,
                        original_quantity=order.quantity,
                        approved_quantity=0.0,
                        reasons=[f"Zero available cash to open position: ${portfolio.cash:.2f}"],
                        rule_triggered="INSUFFICIENT_CASH_BALANCE",
                        severity=RiskSeverity.WARNING,
                    )
                approved_qty = min(approved_qty, max_affordable_qty)
                adjusted = True
                reasons.append(f"Adjusted for available cash: {approved_qty:.6f}")

        # Final authorization
        action = RiskAction.ADJUSTED if adjusted else RiskAction.APPROVED
        return RiskCheckResult(
            allowed=True,
            action=action,
            original_quantity=order.quantity,
            approved_quantity=approved_qty,
            reasons=reasons if reasons else ["Order within all risk parameters."],
            severity=RiskSeverity.INFO if not adjusted else RiskSeverity.WARNING,
            metadata={
                "max_position_capital": max_position_capital,
                "equity": portfolio.equity,
                "cash": portfolio.cash,
            },
        )

    @staticmethod
    async def record_risk_event(
        event_type: str,
        reason: str,
        symbol: Optional[str] = None,
        severity: str = "WARNING",
        details: Optional[Dict[str, Any]] = None,
        kill_switch_active: bool = False,
        session: Optional[AsyncSession] = None,
    ) -> Optional[RiskEvent]:
        """Persist an audit trail entry in the risk_events database table."""
        if session is None:
            logger.warning(f"Risk event '{event_type}': {reason} (No active DB session provided)")
            return None

        event_details = details or {}
        event_details["severity"] = severity

        event = RiskEvent(
            timestamp=datetime.now(timezone.utc),
            event_type=event_type,
            symbol=symbol,
            reason=reason,
            details=event_details,
            kill_switch_active=kill_switch_active,
        )
        session.add(event)
        await session.commit()
        await session.refresh(event)
        logger.info(f"Persisted RiskEvent #{event.id} [{event_type} - {severity}]: {reason}")
        return event
