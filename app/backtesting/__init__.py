"""Backtesting and simulation engine package."""

from app.backtesting.engine import (
    BacktestConfig,
    BacktestEngine,
    BacktestResult,
    TradeRecord,
)
from app.backtesting.metrics import PerformanceMetricsCalculator
from app.backtesting.report import BacktestReportGenerator
from app.backtesting.signals import SignalAction, SignalGenerator, TradingSignal

__all__ = [
    "BacktestEngine",
    "BacktestConfig",
    "BacktestResult",
    "TradeRecord",
    "PerformanceMetricsCalculator",
    "SignalGenerator",
    "TradingSignal",
    "SignalAction",
    "BacktestReportGenerator",
]
