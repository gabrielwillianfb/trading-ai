"""Paper Order Manager handling transactional database persistence for simulated orders, fills, and trades."""

import logging
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple
from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.data.storage.models import Fill, Order, Position, Trade
from app.execution.paper.account import VirtualAccount
from app.risk.state import OrderRequest

logger = logging.getLogger(__name__)


class PaperOrderManager:
    """Manages order creation, simulated fills, position lifecycle, and database audit trails."""

    @staticmethod
    async def create_and_fill_buy_order(
        account: VirtualAccount,
        order_req: OrderRequest,
        approved_qty: float,
        session: AsyncSession,
        model_version_id: Optional[int] = None,
    ) -> Tuple[Order, Fill, Position]:
        """Execute and record a BUY order in VirtualAccount and database."""
        client_order_id = f"paper_buy_{uuid.uuid4().hex[:12]}"
        now = order_req.timestamp or datetime.now(timezone.utc)

        # 1. Execute fill in VirtualAccount
        fill_res = account.execute_buy(
            symbol=order_req.symbol,
            quantity=approved_qty,
            market_price=order_req.price,
            stop_loss_price=order_req.stop_loss_price,
            take_profit_price=order_req.take_profit_price,
            timestamp=now,
        )

        # 2. Persist Order record
        order = Order(
            client_order_id=client_order_id,
            exchange_order_id=f"sim_{client_order_id}",
            symbol=order_req.symbol,
            side="BUY",
            order_type="MARKET",
            mode="paper",
            status="FILLED",
            quantity=approved_qty,
            price=order_req.price,
            stop_loss=order_req.stop_loss_price,
            take_profit=order_req.take_profit_price,
            signal_id=order_req.signal_id,
            created_at=now,
            updated_at=now,
        )
        session.add(order)
        await session.flush()

        # 3. Persist Fill record
        fill = Fill(
            order_id=order.id,
            client_order_id=client_order_id,
            symbol=order_req.symbol,
            side="BUY",
            price=fill_res["price"],
            quantity=approved_qty,
            fee=fill_res["fee"],
            fee_asset="USDT",
            slippage_pct=account.slippage_rate,
            timestamp=now,
        )
        session.add(fill)

        # 4. Upsert/Persist Position record
        pos_query = select(Position).where(
            Position.symbol == order_req.symbol,
            Position.is_open == True,  # noqa: E712
        )
        res = await session.execute(pos_query)
        pos_record = res.scalar_one_or_none()

        if pos_record:
            pos_record.quantity += approved_qty
            pos_record.entry_price = fill_res["price"]
            pos_record.stop_loss = order_req.stop_loss_price
            pos_record.take_profit = order_req.take_profit_price
        else:
            pos_record = Position(
                symbol=order_req.symbol,
                side="LONG",
                quantity=approved_qty,
                entry_price=fill_res["price"],
                current_price=fill_res["price"],
                stop_loss=order_req.stop_loss_price,
                take_profit=order_req.take_profit_price,
                unrealized_pnl=0.0,
                realized_pnl=0.0,
                is_open=True,
                mode="paper",
                opened_at=now,
            )
            session.add(pos_record)

        await session.commit()
        await session.refresh(order)
        await session.refresh(fill)
        await session.refresh(pos_record)

        return order, fill, pos_record

    @staticmethod
    async def create_and_fill_sell_order(
        account: VirtualAccount,
        symbol: str,
        quantity: float,
        market_price: float,
        exit_reason: str,
        session: AsyncSession,
        model_version_id: Optional[int] = None,
        signal_id: Optional[int] = None,
        timestamp: Optional[datetime] = None,
    ) -> Tuple[Order, Fill, Trade]:
        """Execute and record a SELL order in VirtualAccount and close position/trade in database."""
        client_order_id = f"paper_sell_{uuid.uuid4().hex[:12]}"
        now = timestamp or datetime.now(timezone.utc)

        # 1. Execute sell in VirtualAccount
        sell_res = account.execute_sell(
            symbol=symbol,
            quantity=quantity,
            market_price=market_price,
            exit_reason=exit_reason,
            timestamp=now,
        )

        # 2. Persist Order record
        order = Order(
            client_order_id=client_order_id,
            exchange_order_id=f"sim_{client_order_id}",
            symbol=symbol,
            side="SELL",
            order_type="MARKET",
            mode="paper",
            status="FILLED",
            quantity=quantity,
            price=market_price,
            signal_id=signal_id,
            created_at=now,
            updated_at=now,
        )
        session.add(order)
        await session.flush()

        # 3. Persist Fill record
        fill = Fill(
            order_id=order.id,
            client_order_id=client_order_id,
            symbol=symbol,
            side="SELL",
            price=sell_res["exit_price"],
            quantity=quantity,
            fee=sell_res["fees_paid"] / 2.0,  # exit portion of fees
            fee_asset="USDT",
            slippage_pct=account.slippage_rate,
            timestamp=now,
        )
        session.add(fill)

        # 4. Mark DB Position as CLOSED
        pos_query = select(Position).where(
            Position.symbol == symbol,
            Position.is_open == True,  # noqa: E712
        )
        res = await session.execute(pos_query)
        pos_record = res.scalar_one_or_none()

        if pos_record:
            pos_record.is_open = False
            pos_record.closed_at = now
            pos_record.realized_pnl = sell_res["net_pnl"]
            pos_record.unrealized_pnl = 0.0

        # 5. Persist Trade record
        duration_sec = int((sell_res["exit_time"] - sell_res["entry_time"]).total_seconds())
        trade = Trade(
            symbol=symbol,
            side="LONG",
            mode="paper",
            entry_time=sell_res["entry_time"],
            exit_time=sell_res["exit_time"],
            entry_price=sell_res["entry_price"],
            exit_price=sell_res["exit_price"],
            quantity=quantity,
            pnl=sell_res["net_pnl"],
            pnl_pct=sell_res["net_pnl_pct"],
            fees=sell_res["fees_paid"],
            duration_seconds=max(0, duration_sec),
            exit_reason=exit_reason,
            exit_order_id=order.id,
        )
        session.add(trade)

        await session.commit()
        await session.refresh(order)
        await session.refresh(fill)
        await session.refresh(trade)

        return order, fill, trade

    @classmethod
    async def check_intrabar_exits(
        cls,
        account: VirtualAccount,
        symbol: str,
        high: float,
        low: float,
        open_price: float,
        timestamp: datetime,
        session: AsyncSession,
        model_version_id: Optional[int] = None,
        max_holding_bars: Optional[int] = 12,
    ) -> Optional[Trade]:
        """Inspect intrabar high and low against open position SL/TP thresholds."""
        pos = account.get_position(symbol)
        if not pos:
            return None

        pos.holding_bars += 1
        exit_price = None
        exit_reason = None

        # Check Stop Loss
        if pos.stop_loss_price is not None and low <= pos.stop_loss_price:
            exit_price = min(open_price, pos.stop_loss_price)
            exit_reason = "stop_loss"

        # Check Take Profit
        elif pos.take_profit_price is not None and high >= pos.take_profit_price:
            exit_price = max(open_price, pos.take_profit_price)
            exit_reason = "take_profit"

        # Check Max Holding Time
        elif max_holding_bars is not None and pos.holding_bars >= max_holding_bars:
            exit_price = open_price
            exit_reason = "time_exit"

        if exit_price is not None and exit_reason is not None:
            _, _, trade = await cls.create_and_fill_sell_order(
                account=account,
                symbol=symbol,
                quantity=pos.quantity,
                market_price=exit_price,
                exit_reason=exit_reason,
                session=session,
                model_version_id=model_version_id,
                timestamp=timestamp,
            )
            return trade

        return None
