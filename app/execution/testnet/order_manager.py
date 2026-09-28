"""Testnet Order Manager for authenticated Binance Spot Sandbox execution."""

import logging
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.data.collectors.binance import BinanceClient
from app.data.storage.models import Fill, Order, Position, Trade
from app.execution.testnet.filters import SymbolFilters
from app.risk.state import OrderRequest, PortfolioState

logger = logging.getLogger(__name__)


class TestnetOrderManager:
    """Manages order submission to Binance Testnet and synchronizes DB state."""

    __test__ = False

    def __init__(self, client: BinanceClient):
        self.client = client
        self._symbol_filters_cache: Dict[str, SymbolFilters] = {}

    async def get_symbol_filters(self, symbol: str) -> SymbolFilters:
        """Fetch and cache trading filters for a symbol from Binance Testnet."""
        if symbol in self._symbol_filters_cache:
            return self._symbol_filters_cache[symbol]

        try:
            info = await self.client.get_exchange_info(symbol)
            symbols = info.get("symbols", [])
            if symbols:
                filters = SymbolFilters.from_exchange_info(symbols[0])
                self._symbol_filters_cache[symbol] = filters
                return filters
        except Exception as exc:
            logger.warning(f"Could not load live filters for {symbol}: {exc}. Using defaults.")

        default_filters = SymbolFilters(symbol=symbol)
        self._symbol_filters_cache[symbol] = default_filters
        return default_filters

    async def fetch_testnet_portfolio_state(
        self,
        symbol: str,
        current_price: float,
        session: Optional[AsyncSession] = None,
    ) -> PortfolioState:
        """Query real Binance Testnet account balance and assemble PortfolioState."""
        account_info = await self.client.get_account_info()
        balances = account_info.get("balances", [])

        usdt_free = 0.0
        usdt_locked = 0.0
        base_asset = symbol.split("/")[0].upper()
        base_free = 0.0

        for b in balances:
            asset = b.get("asset")
            if asset == "USDT":
                usdt_free = float(b.get("free", 0.0))
                usdt_locked = float(b.get("locked", 0.0))
            elif asset == base_asset:
                base_free = float(b.get("free", 0.0))

        cash = usdt_free
        position_val = base_free * current_price
        total_equity = cash + usdt_locked + position_val

        open_positions = {}
        if base_free > 0:
            open_positions[symbol] = base_free

        return PortfolioState(
            cash=cash,
            equity=total_equity,
            peak_equity=total_equity,
            daily_pnl=0.0,
            consecutive_losses=0,
            open_positions=open_positions,
        )

    async def execute_testnet_buy_order(
        self,
        order_req: OrderRequest,
        approved_qty: float,
        session: AsyncSession,
        model_version_id: Optional[int] = None,
    ) -> Tuple[Order, Optional[Fill], Optional[Position]]:
        """Submit a BUY order to Binance Testnet and persist records to database."""
        filters = await self.get_symbol_filters(order_req.symbol)
        formatted_qty = filters.format_quantity(approved_qty)

        if not filters.validate_notional(formatted_qty, order_req.price):
            raise ValueError(
                f"Order notional ${formatted_qty * order_req.price:.2f} is below minNotional ${filters.min_notional:.2f}"
            )

        client_order_id = f"testnet_buy_{uuid.uuid4().hex[:12]}"
        now = datetime.now(timezone.utc)

        # 1. Place order on Binance Testnet
        raw_res = await self.client.create_order(
            symbol=order_req.symbol,
            side="BUY",
            order_type="MARKET",
            quantity=formatted_qty,
            client_order_id=client_order_id,
        )

        exchange_order_id = str(raw_res.get("orderId", ""))
        status = raw_res.get("status", "FILLED")
        executed_qty = float(raw_res.get("executedQty", formatted_qty))
        cummulative_quote = float(raw_res.get("cummulativeQuoteQty", 0.0))
        avg_price = cummulative_quote / executed_qty if executed_qty > 0 else order_req.price

        logger.info(
            f"TESTNET BUY FILLED: {executed_qty} {order_req.symbol} @ ${avg_price:.2f} (Exchange ID: {exchange_order_id})"
        )

        # 2. Persist Order in DB
        order = Order(
            client_order_id=client_order_id,
            exchange_order_id=exchange_order_id,
            symbol=order_req.symbol,
            side="BUY",
            order_type="MARKET",
            mode="testnet",
            status=status,
            quantity=formatted_qty,
            price=order_req.price,
            stop_loss=order_req.stop_loss_price,
            take_profit=order_req.take_profit_price,
            signal_id=order_req.signal_id,
            created_at=now,
            updated_at=now,
        )
        session.add(order)
        await session.flush()

        # 3. Persist Fill in DB
        fill = None
        if executed_qty > 0:
            fill = Fill(
                order_id=order.id,
                client_order_id=client_order_id,
                symbol=order_req.symbol,
                side="BUY",
                price=avg_price,
                quantity=executed_qty,
                fee=cummulative_quote * 0.001,
                fee_asset="USDT",
                slippage_pct=0.0,
                timestamp=now,
            )
            session.add(fill)

            # 4. Upsert Position
            pos_query = select(Position).where(
                Position.symbol == order_req.symbol,
                Position.is_open == True,  # noqa: E712
                Position.mode == "testnet",
            )
            res = await session.execute(pos_query)
            pos_record = res.scalar_one_or_none()

            if pos_record:
                pos_record.quantity += executed_qty
                pos_record.entry_price = avg_price
                pos_record.stop_loss = order_req.stop_loss_price
                pos_record.take_profit = order_req.take_profit_price
            else:
                pos_record = Position(
                    symbol=order_req.symbol,
                    side="LONG",
                    quantity=executed_qty,
                    entry_price=avg_price,
                    current_price=avg_price,
                    stop_loss=order_req.stop_loss_price,
                    take_profit=order_req.take_profit_price,
                    unrealized_pnl=0.0,
                    realized_pnl=0.0,
                    is_open=True,
                    mode="testnet",
                    opened_at=now,
                )
                session.add(pos_record)

        await session.commit()
        await session.refresh(order)
        if fill:
            await session.refresh(fill)
        return order, fill, pos_record

    async def execute_testnet_sell_order(
        self,
        symbol: str,
        quantity: float,
        market_price: float,
        exit_reason: str,
        session: AsyncSession,
        model_version_id: Optional[int] = None,
        signal_id: Optional[int] = None,
    ) -> Tuple[Order, Optional[Fill], Optional[Trade]]:
        """Submit a SELL order to Binance Testnet and close position/trade in DB."""
        filters = await self.get_symbol_filters(symbol)
        formatted_qty = filters.format_quantity(quantity)

        client_order_id = f"testnet_sell_{uuid.uuid4().hex[:12]}"
        now = datetime.now(timezone.utc)

        # 1. Place sell order on Binance Testnet
        raw_res = await self.client.create_order(
            symbol=symbol,
            side="SELL",
            order_type="MARKET",
            quantity=formatted_qty,
            client_order_id=client_order_id,
        )

        exchange_order_id = str(raw_res.get("orderId", ""))
        status = raw_res.get("status", "FILLED")
        executed_qty = float(raw_res.get("executedQty", formatted_qty))
        cummulative_quote = float(raw_res.get("cummulativeQuoteQty", 0.0))
        avg_price = cummulative_quote / executed_qty if executed_qty > 0 else market_price

        logger.info(
            f"TESTNET SELL FILLED: {executed_qty} {symbol} @ ${avg_price:.2f} (Exchange ID: {exchange_order_id})"
        )

        # 2. Persist Order in DB
        order = Order(
            client_order_id=client_order_id,
            exchange_order_id=exchange_order_id,
            symbol=symbol,
            side="SELL",
            order_type="MARKET",
            mode="testnet",
            status=status,
            quantity=formatted_qty,
            price=market_price,
            signal_id=signal_id,
            created_at=now,
            updated_at=now,
        )
        session.add(order)
        await session.flush()

        # 3. Persist Fill in DB
        fill = None
        trade = None
        if executed_qty > 0:
            fill = Fill(
                order_id=order.id,
                client_order_id=client_order_id,
                symbol=symbol,
                side="SELL",
                price=avg_price,
                quantity=executed_qty,
                fee=cummulative_quote * 0.001,
                fee_asset="USDT",
                slippage_pct=0.0,
                timestamp=now,
            )
            session.add(fill)

            # 4. Close Position in DB
            pos_query = select(Position).where(
                Position.symbol == symbol,
                Position.is_open == True,  # noqa: E712
                Position.mode == "testnet",
            )
            res = await session.execute(pos_query)
            pos_record = res.scalar_one_or_none()

            entry_price = avg_price
            opened_at = now
            if pos_record:
                entry_price = pos_record.entry_price
                opened_at = pos_record.opened_at
                pos_record.is_open = False
                pos_record.closed_at = now

            cost = executed_qty * entry_price
            gross_pnl = (avg_price - entry_price) * executed_qty
            fees = cummulative_quote * 0.001 * 2.0
            net_pnl = gross_pnl - fees
            pnl_pct = (net_pnl / cost) * 100.0 if cost > 0 else 0.0
            duration_sec = int((now - opened_at).total_seconds())

            trade = Trade(
                symbol=symbol,
                side="LONG",
                mode="testnet",
                entry_time=opened_at,
                exit_time=now,
                entry_price=entry_price,
                exit_price=avg_price,
                quantity=executed_qty,
                pnl=net_pnl,
                pnl_pct=pnl_pct,
                fees=fees,
                duration_seconds=max(0, duration_sec),
                exit_reason=exit_reason,
                exit_order_id=order.id,
            )
            session.add(trade)

        await session.commit()
        await session.refresh(order)
        if fill:
            await session.refresh(fill)
        if trade:
            await session.refresh(trade)
        return order, fill, trade
