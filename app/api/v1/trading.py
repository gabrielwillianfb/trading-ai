"""Trading activity REST API endpoints for signals, orders, and closed trades."""

from typing import List, Optional
from fastapi import APIRouter, Depends, Query
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.schemas import (
    FillResponse,
    OrderResponse,
    SignalResponse,
    TradeResponse,
    TradesSummaryResponse,
)
from app.data.storage.db import get_session
from app.data.storage.models import Order, Signal, Trade

router = APIRouter(tags=["Trading"])


@router.get("/signals", response_model=List[SignalResponse])
async def list_signals(
    symbol: Optional[str] = Query(None, description="Filter by symbol (e.g. BTC/USDT)"),
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0),
    session: AsyncSession = Depends(get_session),
) -> List[SignalResponse]:
    """Retrieve historical ML decision signals with confidence and probability scores."""
    query = select(Signal).order_by(Signal.timestamp.desc())
    if symbol:
        query = query.where(Signal.symbol == symbol)
    query = query.limit(limit).offset(offset)

    res = await session.execute(query)
    signals = res.scalars().all()

    return [
        SignalResponse(
            id=s.id,
            timestamp=s.timestamp,
            symbol=s.symbol,
            timeframe=s.timeframe,
            signal_type=s.signal_type,
            probability_up=s.probability_up,
            probability_down=s.probability_down,
            confidence=s.confidence,
            threshold_used=s.threshold_used,
            prediction_id=s.prediction_id,
            is_processed=s.is_processed,
        )
        for s in signals
    ]


@router.get("/orders", response_model=List[OrderResponse])
async def list_orders(
    symbol: Optional[str] = Query(None),
    status: Optional[str] = Query(None),
    mode: Optional[str] = Query(None),
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0),
    session: AsyncSession = Depends(get_session),
) -> List[OrderResponse]:
    """Retrieve order history with lifecycle statuses and executed fills."""
    query = select(Order).order_by(Order.created_at.desc())
    if symbol:
        query = query.where(Order.symbol == symbol)
    if status:
        query = query.where(Order.status == status.upper())
    if mode:
        query = query.where(Order.mode == mode.lower())
    query = query.limit(limit).offset(offset)

    res = await session.execute(query)
    orders = res.scalars().all()

    return [
        OrderResponse(
            id=o.id,
            client_order_id=o.client_order_id,
            exchange_order_id=o.exchange_order_id,
            symbol=o.symbol,
            side=o.side,
            order_type=o.order_type,
            quantity=o.quantity,
            price=o.price,
            stop_loss=o.stop_loss,
            take_profit=o.take_profit,
            status=o.status,
            mode=o.mode,
            rejection_reason=o.rejection_reason,
            signal_id=o.signal_id,
            created_at=o.created_at,
            updated_at=o.updated_at,
            fills=[
                FillResponse(
                    id=f.id,
                    price=f.price,
                    quantity=f.quantity,
                    fee=f.fee,
                    fee_asset=f.fee_asset,
                    slippage_pct=f.slippage_pct,
                    timestamp=f.timestamp,
                )
                for f in o.fills
            ],
        )
        for o in orders
    ]


@router.get("/trades", response_model=TradesSummaryResponse)
async def list_trades(
    symbol: Optional[str] = Query(None),
    mode: Optional[str] = Query(None),
    limit: int = Query(100, ge=1, le=1000),
    offset: int = Query(0, ge=0),
    session: AsyncSession = Depends(get_session),
) -> TradesSummaryResponse:
    """Retrieve closed round-trip trades with aggregated quantitative performance metrics."""
    query = select(Trade).order_by(Trade.exit_time.desc())
    if symbol:
        query = query.where(Trade.symbol == symbol)
    if mode:
        query = query.where(Trade.mode == mode.lower())
    query = query.limit(limit).offset(offset)

    res = await session.execute(query)
    trades = res.scalars().all()

    total_trades = len(trades)
    winning_trades = sum(1 for t in trades if t.pnl > 0)
    losing_trades = sum(1 for t in trades if t.pnl < 0)
    win_rate = (winning_trades / total_trades * 100.0) if total_trades > 0 else 0.0

    total_net_pnl = sum(t.pnl for t in trades)
    total_fees = sum(t.fees for t in trades)
    avg_pnl = total_net_pnl / total_trades if total_trades > 0 else 0.0

    gross_profit = sum(t.pnl for t in trades if t.pnl > 0)
    gross_loss = abs(sum(t.pnl for t in trades if t.pnl < 0))
    profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else (1.0 if total_trades > 0 else 0.0)

    trades_data = [
        TradeResponse(
            id=t.id,
            symbol=t.symbol,
            side=t.side,
            mode=t.mode,
            entry_time=t.entry_time,
            exit_time=t.exit_time,
            entry_price=t.entry_price,
            exit_price=t.exit_price,
            quantity=t.quantity,
            pnl=t.pnl,
            pnl_pct=t.pnl_pct,
            fees=t.fees,
            duration_seconds=t.duration_seconds,
            exit_reason=t.exit_reason,
        )
        for t in trades
    ]

    return TradesSummaryResponse(
        total_trades=total_trades,
        winning_trades=winning_trades,
        losing_trades=losing_trades,
        win_rate_pct=win_rate,
        total_net_pnl=total_net_pnl,
        total_fees_paid=total_fees,
        profit_factor=profit_factor,
        average_trade_pnl=avg_pnl,
        trades=trades_data,
    )
