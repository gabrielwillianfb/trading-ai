"""Virtual account and wallet simulator for Paper Trading."""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from app.risk.state import PortfolioState

logger = logging.getLogger(__name__)


@dataclass
class VirtualPosition:
    """Represents an open trading position within the VirtualAccount."""

    symbol: str
    quantity: float
    entry_price: float
    entry_time: datetime
    current_price: float
    stop_loss_price: Optional[float] = None
    take_profit_price: Optional[float] = None
    holding_bars: int = 0

    @property
    def market_value(self) -> float:
        return self.quantity * self.current_price

    @property
    def unrealized_pnl(self) -> float:
        return (self.current_price - self.entry_price) * self.quantity

    @property
    def unrealized_pnl_pct(self) -> float:
        cost = self.entry_price * self.quantity
        return (self.unrealized_pnl / cost) * 100.0 if cost > 0 else 0.0


class VirtualAccount:
    """Tracks virtual cash balances, open positions, PnL, and portfolio equity."""

    def __init__(
        self,
        initial_cash: float = 10000.0,
        commission_rate: float = 0.001,
        slippage_rate: float = 0.0005,
    ):
        self.initial_cash = initial_cash
        self.cash = initial_cash
        self.commission_rate = commission_rate
        self.slippage_rate = slippage_rate

        self.positions: Dict[str, VirtualPosition] = {}
        self.realized_pnl: float = 0.0
        self.daily_pnl: float = 0.0
        self.peak_equity: float = initial_cash
        self.consecutive_losses: int = 0
        self.total_commissions_paid: float = 0.0

    def update_price(self, symbol: str, current_price: float) -> None:
        """Update market price for an open position and update peak equity."""
        if symbol in self.positions:
            self.positions[symbol].current_price = current_price

        current_equity = self.get_total_equity()
        if current_equity > self.peak_equity:
            self.peak_equity = current_equity

    def get_position(self, symbol: str) -> Optional[VirtualPosition]:
        """Get active position for a symbol if open."""
        return self.positions.get(symbol)

    def get_unrealized_pnl(self) -> float:
        """Calculate aggregate unrealized PnL across all open positions."""
        return sum(pos.unrealized_pnl for pos in self.positions.values())

    def get_total_equity(self) -> float:
        """Calculate total portfolio equity: Cash + Market Value of open positions."""
        positions_value = sum(pos.market_value for pos in self.positions.values())
        return self.cash + positions_value

    def get_portfolio_state(self) -> PortfolioState:
        """Extract standardized PortfolioState snapshot for RiskManager validation."""
        equity = self.get_total_equity()
        open_pos_dict = {sym: pos.quantity for sym, pos in self.positions.items()}
        return PortfolioState(
            cash=self.cash,
            equity=equity,
            peak_equity=self.peak_equity,
            daily_pnl=self.daily_pnl,
            consecutive_losses=self.consecutive_losses,
            open_positions=open_pos_dict,
        )

    def execute_buy(
        self,
        symbol: str,
        quantity: float,
        market_price: float,
        stop_loss_price: Optional[float] = None,
        take_profit_price: Optional[float] = None,
        timestamp: Optional[datetime] = None,
    ) -> Dict[str, Any]:
        """Execute a simulated BUY order, debit cash, and open position."""
        ts = timestamp or datetime.now(timezone.utc)
        fill_price = market_price * (1.0 + self.slippage_rate)
        gross_cost = quantity * fill_price
        fee = gross_cost * self.commission_rate
        total_required = gross_cost + fee

        if total_required > self.cash:
            raise ValueError(
                f"Insufficient virtual cash: Required ${total_required:.2f}, Available ${self.cash:.2f}"
            )

        self.cash -= total_required
        self.total_commissions_paid += fee

        pos = VirtualPosition(
            symbol=symbol,
            quantity=quantity,
            entry_price=fill_price,
            entry_time=ts,
            current_price=fill_price,
            stop_loss_price=stop_loss_price,
            take_profit_price=take_profit_price,
            holding_bars=0,
        )
        self.positions[symbol] = pos

        logger.info(
            f"VIRTUAL BUY: {quantity:.6f} {symbol} @ ${fill_price:.2f} (Fee: ${fee:.4f}, Cash Left: ${self.cash:.2f})"
        )

        return {
            "symbol": symbol,
            "side": "BUY",
            "quantity": quantity,
            "price": fill_price,
            "fee": fee,
            "total_cost": total_required,
            "timestamp": ts,
        }

    def execute_sell(
        self,
        symbol: str,
        quantity: float,
        market_price: float,
        exit_reason: str = "signal_exit",
        timestamp: Optional[datetime] = None,
    ) -> Dict[str, Any]:
        """Execute a simulated SELL order, credit cash, calculate realized PnL, and close position."""
        if symbol not in self.positions:
            raise ValueError(f"No open position found to sell for symbol '{symbol}'.")

        pos = self.positions[symbol]
        ts = timestamp or datetime.now(timezone.utc)
        fill_price = market_price * (1.0 - self.slippage_rate)
        gross_proceeds = quantity * fill_price
        fee = gross_proceeds * self.commission_rate
        net_proceeds = gross_proceeds - fee

        self.cash += net_proceeds
        self.total_commissions_paid += fee

        # Calculate Net PnL of the round-trip trade
        entry_cost = quantity * pos.entry_price
        entry_fee = entry_cost * self.commission_rate
        total_fees = entry_fee + fee
        net_pnl = (gross_proceeds - entry_cost) - total_fees
        net_pnl_pct = (net_pnl / entry_cost) * 100.0 if entry_cost > 0 else 0.0

        # Update PnL accounting
        self.realized_pnl += net_pnl
        self.daily_pnl += net_pnl

        # Update consecutive loss counter
        if net_pnl < 0:
            self.consecutive_losses += 1
        elif net_pnl > 0:
            self.consecutive_losses = 0

        holding_bars = pos.holding_bars

        # Close position
        del self.positions[symbol]

        logger.info(
            f"VIRTUAL SELL: {quantity:.6f} {symbol} @ ${fill_price:.2f} | Net PnL: ${net_pnl:+.2f} ({net_pnl_pct:+.2f}%) | "
            f"Reason: {exit_reason} | Total Equity: ${self.get_total_equity():.2f}"
        )

        return {
            "symbol": symbol,
            "side": "SELL",
            "quantity": quantity,
            "entry_price": pos.entry_price,
            "exit_price": fill_price,
            "entry_time": pos.entry_time,
            "exit_time": ts,
            "gross_pnl": gross_proceeds - entry_cost,
            "net_pnl": net_pnl,
            "net_pnl_pct": net_pnl_pct,
            "fees_paid": total_fees,
            "exit_reason": exit_reason,
            "holding_bars": holding_bars,
        }
