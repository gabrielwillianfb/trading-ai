"""Abstract base class and data models for exchange integration.

Ensures that strategies and models never interact with exchange implementations directly.
"""

from abc import ABC, abstractmethod
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field, field_validator


class CandleData(BaseModel):
    """Normalized OHLCV candlestick data structure."""

    symbol: str = Field(..., description="Standardized trading pair (e.g. BTC/USDT)")
    timeframe: str = Field(..., description="Candle interval (e.g. 5m, 15m, 1h, 4h)")
    timestamp: datetime = Field(..., description="Open timestamp in UTC")
    open: float = Field(..., gt=0.0)
    high: float = Field(..., gt=0.0)
    low: float = Field(..., gt=0.0)
    close: float = Field(..., gt=0.0)
    volume: float = Field(..., ge=0.0)
    quote_volume: Optional[float] = Field(default=None, ge=0.0)
    trades_count: Optional[int] = Field(default=None, ge=0)
    is_closed: bool = Field(default=True)

    @field_validator("timestamp", mode="before")
    @classmethod
    def ensure_utc_timestamp(cls, value: Any) -> datetime:
        """Convert millisecond timestamp, second timestamp or datetime to timezone-aware UTC datetime."""
        if isinstance(value, (int, float)):
            # If timestamp is in milliseconds (> 10^11)
            if value > 1e11:
                return datetime.fromtimestamp(value / 1000.0, tz=timezone.utc)
            return datetime.fromtimestamp(value, tz=timezone.utc)
        if isinstance(value, datetime):
            if value.tzinfo is None:
                return value.replace(tzinfo=timezone.utc)
            return value.astimezone(timezone.utc)
        raise ValueError(f"Invalid timestamp format: {value}")

    @field_validator("high")
    @classmethod
    def validate_high(cls, v: float, info: Any) -> float:
        """Validate high price is consistent with low price if low is already parsed."""
        return v

    def validate_integrity(self) -> None:
        """Validate that high is greater than or equal to open, close, and low."""
        if self.high < self.low:
            raise ValueError(f"High ({self.high}) cannot be lower than low ({self.low})")
        if self.high < self.open or self.high < self.close:
            raise ValueError(f"High ({self.high}) must be >= open ({self.open}) and close ({self.close})")
        if self.low > self.open or self.low > self.close:
            raise ValueError(f"Low ({self.low}) must be <= open ({self.open}) and close ({self.close})")


# ------------------------------------------------------------------------------
# Exchange Exception Hierarchy
# ------------------------------------------------------------------------------

class ExchangeError(Exception):
    """Base exception for all exchange client errors."""
    pass


class RateLimitError(ExchangeError):
    """Raised when exchange rate limit is exceeded (HTTP 429/418)."""
    pass


class AuthenticationError(ExchangeError):
    """Raised when API key/secret authentication fails (HTTP 401/403)."""
    pass


class NetworkError(ExchangeError):
    """Raised on connection failures, timeouts, or transient network issues."""
    pass


class OrderExecutionError(ExchangeError):
    """Raised when exchange rejects or fails to place an order."""
    pass


# ------------------------------------------------------------------------------
# Abstract Exchange Client Interface
# ------------------------------------------------------------------------------

class ExchangeClient(ABC):
    """Abstract interface defining required exchange interactions."""

    @abstractmethod
    async def get_ticker(self, symbol: str) -> Dict[str, Any]:
        """Fetch current ticker price and 24h stats for a symbol.

        Args:
            symbol: Standardized symbol (e.g. 'BTC/USDT').

        Returns:
            Dictionary containing 'symbol', 'last_price', 'bid', 'ask', 'volume'.
        """
        pass

    @abstractmethod
    async def get_klines(
        self,
        symbol: str,
        timeframe: str,
        start_time: Optional[int] = None,
        end_time: Optional[int] = None,
        limit: int = 1000,
    ) -> List[CandleData]:
        """Fetch historical candlestick bars from the exchange.

        Args:
            symbol: Standardized symbol (e.g. 'BTC/USDT').
            timeframe: Timeframe interval (e.g. '15m', '1h').
            start_time: Start timestamp in milliseconds (inclusive).
            end_time: End timestamp in milliseconds (inclusive).
            limit: Maximum number of candles to fetch (max 1000 on Binance).

        Returns:
            List of normalized CandleData instances.
        """
        pass

    @abstractmethod
    async def get_balance(self) -> Dict[str, float]:
        """Fetch account asset balances.

        Returns:
            Dictionary mapping asset symbol (e.g. 'USDT', 'BTC') to available balance.
        """
        pass

    @abstractmethod
    async def get_positions(self) -> List[Dict[str, Any]]:
        """Fetch active positions if applicable.

        Returns:
            List of active position dictionaries.
        """
        pass

    @abstractmethod
    async def create_order(
        self,
        symbol: str,
        side: str,
        order_type: str,
        quantity: float,
        price: Optional[float] = None,
        client_order_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Place an order on the exchange testnet.

        Args:
            symbol: Standardized symbol (e.g. 'BTC/USDT').
            side: 'BUY' or 'SELL'.
            order_type: 'MARKET' or 'LIMIT'.
            quantity: Order size in base asset units.
            price: Required for LIMIT orders.
            client_order_id: Optional custom tracking ID.

        Returns:
            Exchange order response payload.
        """
        pass

    @abstractmethod
    async def cancel_order(self, symbol: str, order_id: str) -> Dict[str, Any]:
        """Cancel an open order on the exchange testnet.

        Args:
            symbol: Standardized symbol.
            order_id: Exchange order ID or client order ID.

        Returns:
            Exchange cancellation response.
        """
        pass
