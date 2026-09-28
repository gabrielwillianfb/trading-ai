"""Status and Model Registry REST API endpoints."""

from datetime import datetime, timezone
from typing import List
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.schemas import ModelVersionResponse, SystemStatusResponse
from app.config.settings import get_settings
from app.data.storage.db import get_session
from app.data.storage.models import ModelVersion, RiskEvent
from app.models.registry.registry import ModelRegistry

router = APIRouter(tags=["Status & Models"])


@router.get("/status", response_model=SystemStatusResponse)
async def get_system_status(session: AsyncSession = Depends(get_session)) -> SystemStatusResponse:
    """Retrieve overall system health, trading mode, active model, and circuit breaker status."""
    settings = get_settings()

    # 1. Test database connectivity
    db_connected = False
    try:
        await session.execute(text("SELECT 1"))
        db_connected = True
    except Exception:
        db_connected = False

    # 2. Query active model
    active_model = await ModelRegistry.get_active_model(session)
    active_ver = active_model.version if active_model else None

    # 3. Query latest circuit breaker state from risk_events
    cb_active = False
    cb_reason = None
    try:
        latest_risk_event_query = (
            select(RiskEvent)
            .where(RiskEvent.event_type.in_(["CIRCUIT_BREAKER", "MAX_DRAWDOWN_EXCEEDED", "MAX_CONSECUTIVE_LOSSES_EXCEEDED"]))
            .order_by(RiskEvent.timestamp.desc())
            .limit(1)
        )
        res = await session.execute(latest_risk_event_query)
        latest_ev = res.scalar_one_or_none()
        if latest_ev and latest_ev.kill_switch_active:
            cb_active = True
            cb_reason = latest_ev.reason
    except Exception:
        pass

    return SystemStatusResponse(
        app_name=settings.APP_NAME,
        app_env=settings.APP_ENV,
        trading_mode=settings.TRADING_MODE,
        is_live_trading_forbidden=True,
        database_connected=db_connected,
        active_model_version=active_ver,
        circuit_breaker_active=cb_active,
        circuit_breaker_reason=cb_reason,
        server_time_utc=datetime.now(timezone.utc),
    )


@router.get("/models", response_model=List[ModelVersionResponse])
async def list_model_versions(session: AsyncSession = Depends(get_session)) -> List[ModelVersionResponse]:
    """List all registered Machine Learning models, active statuses, and performance metrics."""
    query = select(ModelVersion).order_by(ModelVersion.created_at.desc())
    res = await session.execute(query)
    models = res.scalars().all()

    return [
        ModelVersionResponse(
            id=m.id,
            version=m.version,
            model_type=m.model_type,
            is_active=m.is_active,
            feature_names=m.feature_names,
            hyperparameters=m.hyperparameters,
            validation_metrics=m.validation_metrics,
            test_metrics=m.test_metrics,
            file_path=m.file_path,
            trained_at=m.trained_at,
            created_at=m.created_at,
        )
        for m in models
    ]
