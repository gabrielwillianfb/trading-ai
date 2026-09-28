"""Symbol filters and decimal precision handlers for Binance Testnet Spot."""

import math
from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass
class SymbolFilters:
    """Trading rules, lot size step sizes, and notional bounds for a trading pair."""

    symbol: str
    min_qty: float = 0.00001
    max_qty: float = 9000.0
    step_size: float = 0.00001
    min_price: float = 0.01
    max_price: float = 1000000.0
    tick_size: float = 0.01
    min_notional: float = 5.0  # Minimum order value in USDT

    @classmethod
    def from_exchange_info(cls, symbol_info: Dict[str, Any]) -> "SymbolFilters":
        """Parse Binance symbol information dictionary into SymbolFilters."""
        symbol = symbol_info.get("symbol", "")
        min_qty = 0.00001
        max_qty = 9000.0
        step_size = 0.00001
        min_price = 0.01
        max_price = 1000000.0
        tick_size = 0.01
        min_notional = 5.0

        for f in symbol_info.get("filters", []):
            f_type = f.get("filterType")
            if f_type == "LOT_SIZE":
                min_qty = float(f.get("minQty", min_qty))
                max_qty = float(f.get("maxQty", max_qty))
                step_size = float(f.get("stepSize", step_size))
            elif f_type == "PRICE_FILTER":
                min_price = float(f.get("minPrice", min_price))
                max_price = float(f.get("maxPrice", max_price))
                tick_size = float(f.get("tickSize", tick_size))
            elif f_type in ("MIN_NOTIONAL", "NOTIONAL"):
                min_notional = float(f.get("minNotional", f.get("notional", min_notional)))

        return cls(
            symbol=symbol,
            min_qty=min_qty,
            max_qty=max_qty,
            step_size=step_size,
            min_price=min_price,
            max_price=max_price,
            tick_size=tick_size,
            min_notional=min_notional,
        )

    def format_quantity(self, quantity: float) -> float:
        """Format quantity strictly adhering to stepSize and min/max bounds."""
        if quantity <= 0:
            return 0.0

        # Truncate down to step_size precision
        precision = self._get_precision(self.step_size)
        factor = 10.0 ** precision
        formatted = math.floor(quantity * factor) / factor

        if formatted < self.min_qty:
            return 0.0
        return min(formatted, self.max_qty)

    def format_price(self, price: float) -> float:
        """Format price adhering to tickSize precision."""
        if price <= 0:
            return 0.0
        precision = self._get_precision(self.tick_size)
        factor = 10.0 ** precision
        formatted = round(price * factor) / factor
        return max(self.min_price, min(formatted, self.max_price))

    def validate_notional(self, quantity: float, price: float) -> bool:
        """Check whether order gross notional meets minimum notional requirement."""
        return (quantity * price) >= self.min_notional

    @staticmethod
    def _get_precision(step: float) -> int:
        """Determine number of decimal places for a step value."""
        s = f"{step:.8f}".rstrip("0")
        if "." in s:
            return len(s.split(".")[1])
        return 0
