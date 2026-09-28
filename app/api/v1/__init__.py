"""API v1 router combining status, portfolio, trading, and risk endpoints."""

from fastapi import APIRouter

from app.api.v1.portfolio import router as portfolio_router
from app.api.v1.risk import router as risk_router
from app.api.v1.status import router as status_router
from app.api.v1.trading import router as trading_router

api_v1_router = APIRouter(prefix="/api/v1")

api_v1_router.include_router(status_router)
api_v1_router.include_router(portfolio_router)
api_v1_router.include_router(trading_router)
api_v1_router.include_router(risk_router)

__all__ = ["api_v1_router"]
