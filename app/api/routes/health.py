"""Health and system status routes."""

from datetime import datetime, timezone
from typing import Any, Dict, List
from fastapi import APIRouter, status
from pydantic import BaseModel, Field

from app.config.settings import get_settings
from app.data.storage.db import check_db_health

router = APIRouter()


class HealthResponse(BaseModel):
    """Schema for health status response."""

    status: str = Field(..., example="healthy")
    app_name: str
    app_env: str
    trading_mode: str = Field(..., description="'paper' or 'testnet'")
    database_connected: bool
    symbols: List[str]
    default_timeframe: str
    timestamp: datetime


@router.get(
    "/health",
    response_model=HealthResponse,
    status_code=status.HTTP_200_OK,
    summary="System Health & Safety Check",
    description="Returns current operational status, database connectivity, and active trading mode.",
)
async def get_health() -> Dict[str, Any]:
    """Check health and security parameters of the system."""
    settings = get_settings()
    db_ok = await check_db_health()

    return {
        "status": "healthy" if db_ok else "degraded",
        "app_name": settings.APP_NAME,
        "app_env": settings.APP_ENV,
        "trading_mode": settings.TRADING_MODE,
        "database_connected": db_ok,
        "symbols": settings.SYMBOLS,
        "default_timeframe": settings.DEFAULT_TIMEFRAME,
        "timestamp": datetime.now(timezone.utc),
    }
