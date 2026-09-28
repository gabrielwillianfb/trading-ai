"""Database engine, session management, and lifecycle for SQLAlchemy 2.0."""

import logging
from collections.abc import AsyncGenerator
from typing import Optional

from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import DeclarativeBase

from app.config.settings import get_settings

logger = logging.getLogger(__name__)


class Base(DeclarativeBase):
    """Base declarative class for all SQLAlchemy ORM models."""
    pass


# Global engine and session factory references
_engine: Optional[AsyncEngine] = None
_session_factory: Optional[async_sessionmaker[AsyncSession]] = None


def get_engine(db_url: Optional[str] = None) -> AsyncEngine:
    """Get or create the global SQLAlchemy async engine.

    Args:
        db_url: Optional override database URL. Defaults to Settings.DATABASE_URL.

    Returns:
        AsyncEngine instance.
    """
    global _engine
    if _engine is None or db_url is not None:
        url = db_url or get_settings().DATABASE_URL
        # SQLite async requires specific args if used in tests
        connect_args = {}
        if url.startswith("sqlite"):
            connect_args = {"check_same_thread": False}

        _engine = create_async_engine(
            url,
            echo=get_settings().DEBUG,
            future=True,
            connect_args=connect_args,
            pool_pre_ping=True,
        )
    return _engine


def get_session_factory(db_url: Optional[str] = None) -> async_sessionmaker[AsyncSession]:
    """Get or create the async session factory.

    Args:
        db_url: Optional override database URL.

    Returns:
        async_sessionmaker instance.
    """
    global _session_factory
    engine = get_engine(db_url)
    if _session_factory is None or db_url is not None:
        _session_factory = async_sessionmaker(
            bind=engine,
            class_=AsyncSession,
            expire_on_commit=False,
            autoflush=False,
        )
    return _session_factory


async_session_factory = get_session_factory


async def get_db_session() -> AsyncGenerator[AsyncSession, None]:
    """FastAPI dependency for yielding database sessions.

    Yields:
        AsyncSession instance.
    """
    factory = get_session_factory()
    async with factory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


get_session = get_db_session



async def init_db(engine: Optional[AsyncEngine] = None) -> None:
    """Create all database tables defined in Base metadata.

    Args:
        engine: Optional AsyncEngine to bind to.
    """
    target_engine = engine or get_engine()
    # Import models here to register all tables on Base.metadata before creating
    import app.data.storage.models  # noqa: F401

    logger.info("Initializing database schema...")
    async with target_engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    logger.info("Database schema initialized successfully.")


async def check_db_health(engine: Optional[AsyncEngine] = None) -> bool:
    """Ping the database to verify connectivity.

    Returns:
        True if connection succeeded, False otherwise.
    """
    target_engine = engine or get_engine()
    try:
        from sqlalchemy import text
        async with target_engine.connect() as conn:
            await conn.execute(text("SELECT 1"))
        return True
    except Exception as exc:
        logger.warning(f"Database health check failed: {exc}")
        return False
