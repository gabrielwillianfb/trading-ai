"""Risk management data models, portfolio state representations, and check results."""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional


class RiskAction(str, Enum):
    """Result action determined by the Risk Manager."""

    APPROVED = "APPROVED"
    ADJUSTED = "ADJUSTED"
    REJECTED = "REJECTED"
    CIRCUIT_BREAKER_TRIGGERED = "CIRCUIT_BREAKER_TRIGGERED"


class RiskSeverity(str, Enum):
    """Severity level of risk events and check failures."""

    INFO = "INFO"
    WARNING = "WARNING"
    CRITICAL = "CRITICAL"


@dataclass
class OrderRequest:
    """Standardized order request submitted by trading strategies for risk validation."""

    symbol: str
    side: str  # "BUY" or "SELL"
    quantity: float
    price: float
    stop_loss_price: Optional[float] = None
    take_profit_price: Optional[float] = None
    model_version: Optional[str] = None
    signal_id: Optional[int] = None
    timestamp: Optional[datetime] = None


@dataclass
class PortfolioState:
    """Current point-in-time financial state of the trading account."""

    cash: float
    equity: float
    peak_equity: float
    daily_pnl: float = 0.0
    consecutive_losses: int = 0
    open_positions: Dict[str, float] = field(default_factory=dict)

    @property
    def current_drawdown_pct(self) -> float:
        """Calculate current drawdown percentage from peak equity."""
        if self.peak_equity <= 0:
            return 0.0
        return max(0.0, ((self.peak_equity - self.equity) / self.peak_equity) * 100.0)

    @property
    def daily_loss_pct(self) -> float:
        """Calculate current daily loss percentage relative to total equity."""
        if self.equity <= 0:
            return 0.0
        if self.daily_pnl < 0:
            return (abs(self.daily_pnl) / self.equity) * 100.0
        return 0.0


@dataclass
class RiskCheckResult:
    """Formal audit decision output produced by RiskManager.validate_order."""

    allowed: bool
    action: RiskAction
    original_quantity: float
    approved_quantity: float
    reasons: List[str] = field(default_factory=list)
    rule_triggered: Optional[str] = None
    severity: RiskSeverity = RiskSeverity.INFO
    metadata: Dict[str, Any] = field(default_factory=dict)

    def summary(self) -> str:
        reasons_str = "; ".join(self.reasons) if self.reasons else "All risk checks passed."
        return (
            f"Risk Check: {self.action.value} | Allowed: {self.allowed} | "
            f"Qty: {self.original_quantity} -> {self.approved_quantity} | "
            f"Rule: {self.rule_triggered} | Severity: {self.severity.value} | Details: {reasons_str}"
        )
