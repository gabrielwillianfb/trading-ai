"""Application configuration with strict safety validations.

Ensures that TRADING_MODE only allows non-production modes ('paper' or 'testnet').
"""

from functools import lru_cache
from typing import List, Literal
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

TradingModeType = Literal["paper", "testnet"]


class Settings(BaseSettings):
    """Central configuration with environment variable support and safety guarantees."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # --------------------------------------------------------------------------
    # Safety & Operating Mode
    # --------------------------------------------------------------------------
    TRADING_MODE: TradingModeType = Field(
        default="paper",
        description="Operating mode: 'paper' (internal simulation) or 'testnet' (exchange testnet). Production is strictly disallowed.",
    )

    # --------------------------------------------------------------------------
    # Application Info
    # --------------------------------------------------------------------------
    APP_NAME: str = "TradingAI"
    APP_ENV: str = "development"
    LOG_LEVEL: str = "INFO"
    DEBUG: bool = False

    # --------------------------------------------------------------------------
    # Database Settings (PostgreSQL)
    # --------------------------------------------------------------------------
    POSTGRES_USER: str = "postgres"
    POSTGRES_PASSWORD: str = "postgres"
    POSTGRES_DB: str = "trading_ai"
    POSTGRES_HOST: str = "localhost"
    POSTGRES_PORT: int = 5433

    DATABASE_URL: str = (
        "postgresql+asyncpg://postgres:postgres@localhost:5433/trading_ai"
    )
    DATABASE_URL_SYNC: str = (
        "postgresql://postgres:postgres@localhost:5433/trading_ai"
    )

    # --------------------------------------------------------------------------
    # Exchange Settings (Testnet/Sandbox Only)
    # --------------------------------------------------------------------------
    EXCHANGE_ID: str = "binance_testnet"
    EXCHANGE_API_KEY: str = ""
    EXCHANGE_API_SECRET: str = ""
    EXCHANGE_TESTNET_BASE_URL: str = "https://testnet.binance.vision/api"

    # --------------------------------------------------------------------------
    # Target Assets & Timeframes
    # --------------------------------------------------------------------------
    SYMBOLS: List[str] | str = Field(default_factory=lambda: ["BTC/USDT", "ETH/USDT"])
    DEFAULT_TIMEFRAME: str = "15m"
    TIMEFRAMES: List[str] | str = Field(default_factory=lambda: ["5m", "15m", "1h", "4h"])

    # --------------------------------------------------------------------------
    # Risk Management Parameters
    # --------------------------------------------------------------------------
    MAX_RISK_PER_TRADE_PCT: float = Field(default=0.005, ge=0.0001, le=0.05)   # 0.5% max risk
    MAX_DAILY_LOSS_PCT: float = Field(default=0.02, ge=0.001, le=0.10)        # 2.0% daily loss kill switch
    MAX_DRAWDOWN_PCT: float = Field(default=0.10, ge=0.01, le=0.30)           # 10.0% max drawdown kill switch
    MAX_CONSECUTIVE_LOSSES: int = Field(default=4, ge=1, le=20)               # 4 consecutive losses kill switch
    MAX_OPEN_POSITIONS: int = Field(default=1, ge=1, le=10)                   # Max open positions
    MAX_POSITION_SIZE_PCT: float = Field(default=0.10, ge=0.01, le=1.0)       # 10.0% max position size
    DEFAULT_STOP_LOSS_PCT: float = Field(default=0.01, ge=0.001, le=0.10)     # 1.0% stop loss
    DEFAULT_TAKE_PROFIT_PCT: float = Field(default=0.02, ge=0.001, le=0.20)   # 2.0% take profit

    # --------------------------------------------------------------------------
    # Machine Learning & Feature Parameters
    # --------------------------------------------------------------------------
    TARGET_RETURN_PCT: float = Field(default=0.005, gt=0.0)                   # 0.5% target return for classification
    PREDICTION_HORIZON: int = Field(default=12, ge=1)                         # 12 bars forward
    PROBABILITY_BUY_THRESHOLD: float = Field(default=0.55, ge=0.50, le=1.0)
    PROBABILITY_SELL_THRESHOLD: float = Field(default=0.45, ge=0.0, le=1.0)

    # --------------------------------------------------------------------------
    # Validators
    # --------------------------------------------------------------------------
    @field_validator("TRADING_MODE", mode="before")
    @classmethod
    def validate_trading_mode(cls, value: str) -> str:
        """Validate execution mode and strictly forbid production trading."""
        mode = str(value).strip().lower()
        if mode in ("live", "real", "prod", "production"):
            raise ValueError(
                "CRITICAL SECURITY ERROR: Production trading is strictly forbidden! "
                "Only 'paper' and 'testnet' modes are allowed."
            )
        if mode not in ("paper", "testnet"):
            raise ValueError(
                f"CRITICAL SECURITY ERROR: Invalid TRADING_MODE '{mode}'. "
                "Allowed modes are strictly 'paper' or 'testnet'."
            )
        return mode

    @field_validator("SYMBOLS", mode="before")
    @classmethod
    def parse_symbols(cls, value: object) -> List[str]:
        if isinstance(value, str):
            # Check if json array representation or comma-separated
            val_str = value.strip()
            if val_str.startswith("[") and val_str.endswith("]"):
                import json
                try:
                    return [str(s).strip() for s in json.loads(val_str)]
                except Exception:
                    pass
            return [s.strip() for s in val_str.split(",") if s.strip()]
        if isinstance(value, (list, tuple)):
            return [str(s).strip() for s in value if str(s).strip()]
        return ["BTC/USDT", "ETH/USDT"]

    @field_validator("TIMEFRAMES", mode="before")
    @classmethod
    def parse_timeframes(cls, value: object) -> List[str]:
        if isinstance(value, str):
            val_str = value.strip()
            if val_str.startswith("[") and val_str.endswith("]"):
                import json
                try:
                    return [str(t).strip() for t in json.loads(val_str)]
                except Exception:
                    pass
            return [t.strip() for t in val_str.split(",") if t.strip()]
        if isinstance(value, (list, tuple)):
            return [str(t).strip() for t in value if str(t).strip()]
        return ["5m", "15m", "1h", "4h"]


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return cached application settings singleton."""
    return Settings()
