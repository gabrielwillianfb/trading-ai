"""Tests for configuration, security validations, and logging sanitization."""

import logging
import pytest
from pydantic import ValidationError

from app.config.logging import SensitiveDataFilter
from app.config.settings import Settings


def test_default_trading_mode_is_paper():
    """Verify that default TRADING_MODE is always 'paper'."""
    settings = Settings(TRADING_MODE="paper")
    assert settings.TRADING_MODE == "paper"


def test_testnet_mode_is_allowed():
    """Verify that 'testnet' is an allowed TRADING_MODE."""
    settings = Settings(TRADING_MODE="testnet")
    assert settings.TRADING_MODE == "testnet"


@pytest.mark.parametrize("invalid_mode", ["live", "real", "prod", "production", "LIVE", "REAL"])
def test_production_mode_is_strictly_forbidden(invalid_mode: str):
    """Verify that any attempt to set production trading raises a ValidationError."""
    with pytest.raises(ValidationError) as exc_info:
        Settings(TRADING_MODE=invalid_mode)
    assert "CRITICAL SECURITY ERROR" in str(exc_info.value)


def test_symbols_parsing():
    """Verify comma-separated symbols string is parsed to list."""
    settings = Settings(SYMBOLS="BTC/USDT, ETH/USDT, SOL/USDT")
    assert settings.SYMBOLS == ["BTC/USDT", "ETH/USDT", "SOL/USDT"]


def test_timeframes_parsing():
    """Verify comma-separated timeframes string is parsed to list."""
    settings = Settings(TIMEFRAMES="5m, 15m, 1h")
    assert settings.TIMEFRAMES == ["5m", "15m", "1h"]


def test_risk_parameters_default_bounds():
    """Verify default risk parameters are conservative."""
    settings = Settings()
    assert 0 < settings.MAX_RISK_PER_TRADE_PCT <= 0.01  # <= 1%
    assert 0 < settings.MAX_DAILY_LOSS_PCT <= 0.05      # <= 5%
    assert 0 < settings.MAX_DRAWDOWN_PCT <= 0.20        # <= 20%
    assert settings.MAX_OPEN_POSITIONS == 1


def test_sensitive_data_filter_redacts_keys():
    """Verify that SensitiveDataFilter masks API keys and secrets."""
    filter_instance = SensitiveDataFilter()

    test_record = logging.LogRecord(
        name="test_logger",
        level=logging.INFO,
        pathname="",
        lineno=0,
        msg="Connecting with api_key: 'abcdef1234567890' and api_secret='supersecretvalue'",
        args=(),
        exc_info=None,
    )

    filter_instance.filter(test_record)
    assert "abcdef1234567890" not in test_record.msg
    assert "supersecretvalue" not in test_record.msg
    assert "***REDACTED***" in test_record.msg
