"""Pydantic request and response schemas for FastAPI REST endpoints."""

from datetime import datetime
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class SystemStatusResponse(BaseModel):
    """System operational status schema."""

    app_name: str
    app_env: str
    trading_mode: str
    is_live_trading_forbidden: bool = True
    database_connected: bool
    active_model_version: Optional[str] = None
    circuit_breaker_active: bool
    circuit_breaker_reason: Optional[str] = None
    server_time_utc: datetime


class ModelVersionResponse(BaseModel):
    """Registered ML model version metadata schema."""

    id: int
    version: str
    model_type: str
    is_active: bool
    feature_names: List[str]
    hyperparameters: Dict[str, Any]
    validation_metrics: Dict[str, Any]
    test_metrics: Dict[str, Any]
    file_path: str
    trained_at: datetime
    created_at: datetime


class PositionResponse(BaseModel):
    """Active or closed position schema."""

    id: int
    symbol: str
    side: str
    quantity: float
    entry_price: float
    current_price: float
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    unrealized_pnl: float
    realized_pnl: float
    is_open: bool
    mode: str
    opened_at: datetime
    closed_at: Optional[datetime] = None


class PortfolioResponse(BaseModel):
    """Current portfolio financial snapshot schema."""

    timestamp: datetime
    mode: str
    total_equity: float
    cash_balance: float
    unrealized_pnl: float
    realized_pnl: float
    peak_equity: float
    current_drawdown_pct: float
    daily_pnl: float
    daily_return_pct: float
    open_positions_count: int
    open_positions: List[PositionResponse] = []


class SignalResponse(BaseModel):
    """Generated trading signal schema."""

    id: int
    timestamp: datetime
    symbol: str
    timeframe: str
    signal_type: str
    probability_up: float
    probability_down: float
    confidence: float
    threshold_used: float
    prediction_id: Optional[int] = None
    is_processed: bool


class FillResponse(BaseModel):
    """Order fill execution schema."""

    id: int
    price: float
    quantity: float
    fee: float
    fee_asset: str
    slippage_pct: float
    timestamp: datetime


class OrderResponse(BaseModel):
    """Order lifecycle record schema."""

    id: int
    client_order_id: str
    exchange_order_id: Optional[str] = None
    symbol: str
    side: str
    order_type: str
    quantity: float
    price: Optional[float] = None
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    status: str
    mode: str
    rejection_reason: Optional[str] = None
    signal_id: Optional[int] = None
    created_at: datetime
    updated_at: datetime
    fills: List[FillResponse] = []


class TradeResponse(BaseModel):
    """Completed round-trip trade schema."""

    id: int
    symbol: str
    side: str
    mode: str
    entry_time: datetime
    exit_time: datetime
    entry_price: float
    exit_price: float
    quantity: float
    pnl: float
    pnl_pct: float
    fees: float
    duration_seconds: int
    exit_reason: str


class TradesSummaryResponse(BaseModel):
    """Aggregated trade performance metrics schema."""

    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate_pct: float
    total_net_pnl: float
    total_fees_paid: float
    profit_factor: float
    average_trade_pnl: float
    trades: List[TradeResponse] = []


class RiskEventResponse(BaseModel):
    """Risk audit event schema."""

    id: int
    timestamp: datetime
    event_type: str
    symbol: Optional[str] = None
    reason: str
    details: Dict[str, Any]
    kill_switch_active: bool


class KillSwitchRequest(BaseModel):
    """Request payload for manual circuit breaker trigger/reset."""

    action: str = Field(..., description="'trigger' to halt trading or 'reset' to resume.")
    reason: str = Field(..., min_length=3, description="Audit reason for action.")


class KillSwitchResponse(BaseModel):
    """Response payload for kill switch actions."""

    success: bool
    circuit_breaker_active: bool
    message: str
    timestamp: datetime
