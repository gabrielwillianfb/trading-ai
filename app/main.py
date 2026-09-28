"""FastAPI application entrypoint for Trading AI."""

import logging
from contextlib import asynccontextmanager
from typing import AsyncGenerator
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse

from app.api.routes import api_router
from app.config.logging import setup_logging
from app.config.settings import get_settings
from app.data.storage.db import get_engine, init_db

logger = logging.getLogger("app.main")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan manager for startup and shutdown procedures."""
    settings = get_settings()
    setup_logging(settings.LOG_LEVEL)

    logger.info("=" * 60)
    logger.info(f"Starting {settings.APP_NAME} in '{settings.TRADING_MODE.upper()}' mode")
    logger.info(f"Environment: {settings.APP_ENV} | Debug: {settings.DEBUG}")
    logger.info("=" * 60)

    # Initialize database schema if possible
    try:
        await init_db()
        logger.info("Database schema initialized and verified.")
    except Exception as exc:
        logger.warning(
            f"Could not automatically initialize DB on startup: {exc}. "
            "Ensure PostgreSQL is running via `docker compose up -d postgres`."
        )

    yield

    logger.info("Shutting down application resources...")
    engine = get_engine()
    await engine.dispose()
    logger.info("Database connections disposed successfully.")


def create_app() -> FastAPI:
    """Factory function to configure and return the FastAPI application instance."""
    settings = get_settings()

    app = FastAPI(
        title=settings.APP_NAME,
        description=(
            "Autonomous Machine Learning Trading System for Research, "
            "Simulation (Paper), and Testnet Experimentation."
        ),
        version="0.1.0",
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
        lifespan=lifespan,
    )

    # Enable CORS for local development and monitoring UI
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Mount API routes
    app.include_router(api_router)

    @app.get("/", include_in_schema=False)
    async def root_redirect() -> RedirectResponse:
        return RedirectResponse(url="/docs")

    return app


app = create_app()
