"""Tests for FastAPI endpoints."""

import pytest
from httpx import AsyncClient


@pytest.mark.asyncio
async def test_health_endpoint(async_client: AsyncClient):
    """Test that GET /health returns status 200 with proper payload."""
    response = await async_client.get("/health")
    assert response.status_code == 200

    data = response.json()
    assert "status" in data
    assert "app_name" in data
    assert "trading_mode" in data
    assert data["trading_mode"] in ("paper", "testnet")
    assert "database_connected" in data
    assert "symbols" in data
    assert isinstance(data["symbols"], list)
    assert "BTC/USDT" in data["symbols"]
    assert "timestamp" in data


@pytest.mark.asyncio
async def test_root_redirect_to_docs(async_client: AsyncClient):
    """Test that GET / redirects to /docs."""
    response = await async_client.get("/", follow_redirects=False)
    assert response.status_code in (302, 307)
    assert response.headers["location"] == "/docs"
