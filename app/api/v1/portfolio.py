"""Portfolio and account financial balance REST API endpoints."""

from datetime import datetime, timezone
from typing import List
from fastapi import APIRouter, Depends
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.schemas import PortfolioResponse, PositionResponse
from app.config.settings import get_settings
from app.data.storage.db import get_session
from app.data.storage.models import PortfolioSnapshot, Position

router = APIRouter(tags=["Portfolio"])


@router.get("/portfolio", response_model=PortfolioResponse)
async def get_portfolio(session: AsyncSession = Depends(get_session)) -> PortfolioResponse:
    """Retrieve the latest portfolio financial snapshot and list of open active positions."""
    settings = get_settings()

    # 1. Query latest snapshot
    snap_query = select(PortfolioSnapshot).order_by(PortfolioSnapshot.timestamp.desc()).limit(1)
    res_snap = await session.execute(snap_query)
    latest_snap = res_snap.scalar_one_or_none()

    # 2. Query open positions
    pos_query = select(Position).where(Position.is_open == True).order_by(Position.opened_at.desc())  # noqa: E712
    res_pos = await session.execute(pos_query)
    open_positions = res_pos.scalars().all()

    positions_data = [
        PositionResponse(
            id=p.id,
            symbol=p.symbol,
            side=p.side,
            quantity=p.quantity,
            entry_price=p.entry_price,
            current_price=p.current_price,
            stop_loss=p.stop_loss,
            take_profit=p.take_profit,
            unrealized_pnl=p.unrealized_pnl,
            realized_pnl=p.realized_pnl,
            is_open=p.is_open,
            mode=p.mode,
            opened_at=p.opened_at,
            closed_at=p.closed_at,
        )
        for p in open_positions
    ]

    if latest_snap:
        return PortfolioResponse(
            timestamp=latest_snap.timestamp,
            mode=latest_snap.mode,
            total_equity=latest_snap.total_equity,
            cash_balance=latest_snap.cash_balance,
            unrealized_pnl=latest_snap.unrealized_pnl,
            realized_pnl=latest_snap.realized_pnl,
            peak_equity=latest_snap.peak_equity,
            current_drawdown_pct=latest_snap.current_drawdown_pct,
            daily_pnl=latest_snap.daily_pnl,
            daily_return_pct=latest_snap.daily_return_pct,
            open_positions_count=len(positions_data),
            open_positions=positions_data,
        )

    # Default fallback when no snapshots exist yet
    return PortfolioResponse(
        timestamp=datetime.now(timezone.utc),
        mode=settings.TRADING_MODE,
        total_equity=10000.0,
        cash_balance=10000.0,
        unrealized_pnl=0.0,
        realized_pnl=0.0,
        peak_equity=10000.0,
        current_drawdown_pct=0.0,
        daily_pnl=0.0,
        daily_return_pct=0.0,
        open_positions_count=len(positions_data),
        open_positions=positions_data,
    )
