"""API routes router registry."""

from fastapi import APIRouter

from app.api.routes.health import router as health_router
from app.api.v1 import api_v1_router

api_router = APIRouter()
api_router.include_router(health_router, tags=["Health"])
api_router.include_router(api_v1_router)

__all__ = ["api_router"]
