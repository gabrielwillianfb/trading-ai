"""Binance Testnet and REST API client implementation.

Implements the ExchangeClient interface for market data collection and testnet execution.
"""

import asyncio
import hashlib
import hmac
import logging
import time
from typing import Any, Dict, List, Optional
from urllib.parse import urlencode

import httpx

from app.config.settings import get_settings
from app.data.collectors.base import (
    AuthenticationError,
    CandleData,
    ExchangeClient,
    ExchangeError,
    NetworkError,
    OrderExecutionError,
    RateLimitError,
)

logger = logging.getLogger(__name__)


class BinanceClient(ExchangeClient):
    """Client for Binance Spot Testnet & Public REST API."""

    TIMEFRAME_MAP = {
        "1m": "1m",
        "3m": "3m",
        "5m": "5m",
        "15m": "15m",
        "30m": "30m",
        "1h": "1h",
        "2h": "2h",
        "4h": "4h",
        "6h": "6h",
        "8h": "8h",
        "12h": "12h",
        "1d": "1d",
        "3d": "3d",
        "1w": "1w",
        "1M": "1M",
    }

    def __init__(
        self,
        api_key: Optional[str] = None,
        api_secret: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout: float = 10.0,
        max_retries: int = 3,
        backoff_factor: float = 1.5,
    ):
        settings = get_settings()
        self.api_key = api_key if api_key is not None else settings.EXCHANGE_API_KEY
        self.api_secret = api_secret if api_secret is not None else settings.EXCHANGE_API_SECRET
        self.base_url = (base_url or settings.EXCHANGE_TESTNET_BASE_URL).rstrip("/")
        self.timeout = timeout
        self.max_retries = max_retries
        self.backoff_factor = backoff_factor
        self._http_client: Optional[httpx.AsyncClient] = None

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create the underlying HTTPX AsyncClient."""
        if self._http_client is None or self._http_client.is_closed:
            self._http_client = httpx.AsyncClient(
                timeout=httpx.Timeout(self.timeout),
                headers={"User-Agent": "TradingAI-Bot/1.0"},
            )
        return self._http_client

    async def close(self) -> None:
        """Close underlying HTTP client connection pool."""
        if self._http_client is not None and not self._http_client.is_closed:
            await self._http_client.aclose()
            self._http_client = None

    # --------------------------------------------------------------------------
    # Symbol & Timeframe Normalization
    # --------------------------------------------------------------------------
    @staticmethod
    def to_exchange_symbol(symbol: str) -> str:
        """Convert 'BTC/USDT' to 'BTCUSDT'."""
        return symbol.replace("/", "").replace("-", "").replace("_", "").upper()

    @staticmethod
    def from_exchange_symbol(symbol: str) -> str:
        """Convert 'BTCUSDT' to 'BTC/USDT'."""
        clean = symbol.upper()
        for quote in ["USDT", "BUSD", "FDUSD", "USDC", "BTC", "ETH", "EUR"]:
            if clean.endswith(quote):
                base = clean[:-len(quote)]
                return f"{base}/{quote}"
        return clean

    def _normalize_timeframe(self, timeframe: str) -> str:
        """Validate and map timeframe to Binance interval string."""
        tf = timeframe.lower()
        if tf not in self.TIMEFRAME_MAP:
            raise ValueError(
                f"Unsupported timeframe '{timeframe}'. Allowed: {list(self.TIMEFRAME_MAP.keys())}"
            )
        return self.TIMEFRAME_MAP[tf]

    # --------------------------------------------------------------------------
    # Request Signing & Execution
    # --------------------------------------------------------------------------
    def _sign_payload(self, params: Dict[str, Any]) -> str:
        """Sign request parameters with HMAC-SHA256."""
        if not self.api_secret:
            raise AuthenticationError("EXCHANGE_API_SECRET is required to sign authenticated requests.")
        query_string = urlencode(params)
        signature = hmac.new(
            self.api_secret.encode("utf-8"),
            query_string.encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()
        return signature

    async def _request(
        self,
        method: str,
        path: str,
        params: Optional[Dict[str, Any]] = None,
        signed: bool = False,
    ) -> Any:
        """Execute HTTP request with retries, backoff, and signature handling."""
        client = await self._get_client()
        url = f"{self.base_url}/{path.lstrip('/')}"
        params = params.copy() if params else {}

        headers: Dict[str, str] = {}
        if self.api_key:
            headers["X-MBX-APIKEY"] = self.api_key

        if signed:
            if not self.api_key:
                raise AuthenticationError("EXCHANGE_API_KEY is required for authenticated requests.")
            params["timestamp"] = int(time.time() * 1000)
            params["recvWindow"] = 5000
            params["signature"] = self._sign_payload(params)

        for attempt in range(1, self.max_retries + 1):
            try:
                if method.upper() == "GET":
                    response = await client.get(url, params=params, headers=headers)
                elif method.upper() == "POST":
                    response = await client.post(url, params=params, headers=headers)
                elif method.upper() == "DELETE":
                    response = await client.delete(url, params=params, headers=headers)
                else:
                    raise ValueError(f"Unsupported HTTP method: {method}")

                # Check for rate limiting
                if response.status_code in (429, 418):
                    retry_after = int(response.headers.get("Retry-After", 2))
                    logger.warning(
                        f"Binance rate limit hit (HTTP {response.status_code}). Backing off for {retry_after}s..."
                    )
                    if attempt < self.max_retries:
                        await asyncio.sleep(retry_after)
                        continue
                    raise RateLimitError(f"Binance rate limit exceeded: {response.text}")

                # Check for authentication errors
                if response.status_code in (401, 403):
                    raise AuthenticationError(f"Binance authentication failed: {response.text}")

                # Check for other HTTP errors
                if response.status_code >= 500:
                    logger.warning(
                        f"Binance server error (HTTP {response.status_code}) on attempt {attempt}/{self.max_retries}."
                    )
                    if attempt < self.max_retries:
                        await asyncio.sleep(self.backoff_factor ** attempt)
                        continue
                    raise ExchangeError(f"Binance server error ({response.status_code}): {response.text}")

                if response.status_code >= 400:
                    error_json = response.json() if response.content else {}
                    code = error_json.get("code", 0)
                    msg = error_json.get("msg", response.text)
                    raise OrderExecutionError(f"Binance error code {code}: {msg}")

                return response.json()

            except httpx.RequestError as exc:
                logger.warning(f"Network error communicating with Binance: {exc} (Attempt {attempt}/{self.max_retries})")
                if attempt < self.max_retries:
                    await asyncio.sleep(self.backoff_factor ** attempt)
                    continue
                raise NetworkError(f"Failed to connect to Binance after {self.max_retries} attempts: {exc}")

    # --------------------------------------------------------------------------
    # Public Market Data Methods
    # --------------------------------------------------------------------------
    async def get_ticker(self, symbol: str) -> Dict[str, Any]:
        """Fetch current ticker price and 24h stats for a symbol."""
        raw_symbol = self.to_exchange_symbol(symbol)
        data = await self._request("GET", "/v3/ticker/24hr", params={"symbol": raw_symbol})

        return {
            "symbol": symbol,
            "last_price": float(data.get("lastPrice", 0.0)),
            "bid": float(data.get("bidPrice", 0.0)),
            "ask": float(data.get("askPrice", 0.0)),
            "volume": float(data.get("volume", 0.0)),
            "high_24h": float(data.get("highPrice", 0.0)),
            "low_24h": float(data.get("lowPrice", 0.0)),
            "price_change_pct": float(data.get("priceChangePercent", 0.0)),
            "timestamp": int(data.get("closeTime", 0)),
        }

    async def get_klines(
        self,
        symbol: str,
        timeframe: str,
        start_time: Optional[int] = None,
        end_time: Optional[int] = None,
        limit: int = 1000,
    ) -> List[CandleData]:
        """Fetch historical candlestick bars from Binance."""
        raw_symbol = self.to_exchange_symbol(symbol)
        interval = self._normalize_timeframe(timeframe)
        params: Dict[str, Any] = {
            "symbol": raw_symbol,
            "interval": interval,
            "limit": min(limit, 1000),
        }
        if start_time is not None:
            params["startTime"] = int(start_time)
        if end_time is not None:
            params["endTime"] = int(end_time)

        raw_klines = await self._request("GET", "/v3/klines", params=params)

        candles: List[CandleData] = []
        for row in raw_klines:
            candle = CandleData(
                symbol=symbol,
                timeframe=timeframe,
                timestamp=row[0],  # Open time ms
                open=float(row[1]),
                high=float(row[2]),
                low=float(row[3]),
                close=float(row[4]),
                volume=float(row[5]),
                quote_volume=float(row[7]),
                trades_count=int(row[8]),
                is_closed=True,
            )
            candle.validate_integrity()
            candles.append(candle)

        return candles

    # --------------------------------------------------------------------------
    # Authenticated Testnet Execution Methods
    # --------------------------------------------------------------------------
    async def get_account_info(self) -> Dict[str, Any]:
        """Fetch complete raw account information including all asset balances."""
        return await self._request("GET", "/v3/account", signed=True)

    async def get_balance(self) -> Dict[str, float]:
        """Fetch account asset balances (free + locked)."""
        data = await self.get_account_info()
        balances: Dict[str, float] = {}
        for item in data.get("balances", []):
            asset = item.get("asset")
            free = float(item.get("free", 0.0))
            locked = float(item.get("locked", 0.0))
            total = free + locked
            if total > 0:
                balances[asset] = total
        return balances

    async def get_exchange_info(self, symbol: Optional[str] = None) -> Dict[str, Any]:
        """Fetch exchange trading rules, rate limits, and symbol filters."""
        params: Dict[str, Any] = {}
        if symbol:
            params["symbol"] = self.to_exchange_symbol(symbol)
        return await self._request("GET", "/v3/exchangeInfo", params=params)

    async def get_open_orders(self, symbol: Optional[str] = None) -> List[Dict[str, Any]]:
        """Fetch active open orders on testnet."""
        params: Dict[str, Any] = {}
        if symbol:
            params["symbol"] = self.to_exchange_symbol(symbol)
        return await self._request("GET", "/v3/openOrders", params=params, signed=True)

    async def get_positions(self) -> List[Dict[str, Any]]:
        """Fetch active open orders / positions on testnet."""
        return await self.get_open_orders()

    async def get_order(
        self,
        symbol: str,
        order_id: Optional[int] = None,
        orig_client_order_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Query specific order status by orderId or origClientOrderId."""
        raw_symbol = self.to_exchange_symbol(symbol)
        params: Dict[str, Any] = {"symbol": raw_symbol}
        if order_id is not None:
            params["orderId"] = int(order_id)
        elif orig_client_order_id is not None:
            params["origClientOrderId"] = orig_client_order_id
        else:
            raise ValueError("Either order_id or orig_client_order_id must be provided.")

        return await self._request("GET", "/v3/order", params=params, signed=True)

    async def create_order(
        self,
        symbol: str,
        side: str,
        order_type: str = "MARKET",
        quantity: Optional[float] = None,
        price: Optional[float] = None,
        client_order_id: Optional[str] = None,
        time_in_force: str = "GTC",
    ) -> Dict[str, Any]:
        """Place an order on the exchange testnet."""
        raw_symbol = self.to_exchange_symbol(symbol)
        params: Dict[str, Any] = {
            "symbol": raw_symbol,
            "side": side.upper(),
            "type": order_type.upper(),
        }
        if quantity is not None:
            params["quantity"] = f"{quantity:.6f}"

        if order_type.upper() == "LIMIT":
            if price is None:
                raise ValueError("Price is required for LIMIT orders.")
            params["price"] = f"{price:.2f}"
            params["timeInForce"] = time_in_force

        if client_order_id:
            params["newClientOrderId"] = client_order_id

        return await self._request("POST", "/v3/order", params=params, signed=True)

    async def cancel_order(
        self,
        symbol: str,
        order_id: Optional[str | int] = None,
        client_order_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Cancel an open order on the exchange testnet."""
        raw_symbol = self.to_exchange_symbol(symbol)
        params: Dict[str, Any] = {"symbol": raw_symbol}
        if order_id is not None:
            if str(order_id).isdigit():
                params["orderId"] = int(order_id)
            else:
                params["origClientOrderId"] = str(order_id)
        elif client_order_id is not None:
            params["origClientOrderId"] = client_order_id
        else:
            raise ValueError("Either order_id or client_order_id must be provided.")

        return await self._request("DELETE", "/v3/order", params=params, signed=True)
