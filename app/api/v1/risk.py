"""Risk management REST API endpoints for audit events and emergency kill-switch control."""

from datetime import datetime, timezone
from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.schemas import (
    KillSwitchRequest,
    KillSwitchResponse,
    RiskEventResponse,
)
from app.data.storage.db import get_session
from app.data.storage.models import RiskEvent
from app.risk.manager import RiskManager

router = APIRouter(prefix="/risk", tags=["Risk Management"])


@router.get("/events", response_model=List[RiskEventResponse])
async def list_risk_events(
    event_type: Optional[str] = Query(None),
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0),
    session: AsyncSession = Depends(get_session),
) -> List[RiskEventResponse]:
    """Retrieve immutable audit log of risk limit violations, circuit breaker triggers, and order rejections."""
    query = select(RiskEvent).order_by(RiskEvent.timestamp.desc())
    if event_type:
        query = query.where(RiskEvent.event_type == event_type.upper())
    query = query.limit(limit).offset(offset)

    res = await session.execute(query)
    events = res.scalars().all()

    return [
        RiskEventResponse(
            id=e.id,
            timestamp=e.timestamp,
            event_type=e.event_type,
            symbol=e.symbol,
            reason=e.reason,
            details=e.details,
            kill_switch_active=e.kill_switch_active,
        )
        for e in events
    ]


@router.post("/kill-switch", response_model=KillSwitchResponse)
async def manage_kill_switch(
    payload: KillSwitchRequest,
    session: AsyncSession = Depends(get_session),
) -> KillSwitchResponse:
    """Manually trigger emergency halt or perform supervised reset of the system Circuit Breaker."""
    action = payload.action.strip().lower()
    now = datetime.now(timezone.utc)

    if action == "trigger":
        await RiskManager.record_risk_event(
            event_type="MANUAL_EMERGENCY_STOP",
            reason=f"Manual operator kill switch triggered: {payload.reason}",
            severity="CRITICAL",
            details={"operator_reason": payload.reason, "action": "trigger"},
            kill_switch_active=True,
            session=session,
        )
        return KillSwitchResponse(
            success=True,
            circuit_breaker_active=True,
            message=f"Emergency Circuit Breaker ACTIVATED. All new order authorizations halted. Reason: {payload.reason}",
            timestamp=now,
        )

    elif action == "reset":
        await RiskManager.record_risk_event(
            event_type="CIRCUIT_BREAKER_RESET",
            reason=f"Manual operator reset: {payload.reason}",
            severity="INFO",
            details={"operator_reason": payload.reason, "action": "reset"},
            kill_switch_active=False,
            session=session,
        )
        return KillSwitchResponse(
            success=True,
            circuit_breaker_active=False,
            message=f"Circuit Breaker RESET. Trading system resumed under standard risk rules. Reason: {payload.reason}",
            timestamp=now,
        )

    else:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid action '{payload.action}'. Allowed actions are 'trigger' or 'reset'.",
        )
