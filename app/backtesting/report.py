"""Report generation and export utilities for backtest results."""

import json
from pathlib import Path
from typing import Any, Dict
import pandas as pd

from app.backtesting.engine import BacktestResult


class BacktestReportGenerator:
    """Exports and formats backtesting simulation metrics and trade histories."""

    @staticmethod
    def export_to_json(result: BacktestResult, file_path: str) -> None:
        """Export metrics and trade logs to structured JSON."""
        data = {
            "symbol": result.symbol,
            "timeframe": result.timeframe,
            "metrics": result.metrics,
            "trades_count": len(result.trades),
            "trades": [t.to_dict() for t in result.trades],
        }
        Path(file_path).parent.mkdir(parents=True, exist_ok=True)
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    @staticmethod
    def export_trades_csv(result: BacktestResult, file_path: str) -> None:
        """Export completed trades history to CSV format."""
        df = result.to_trades_dataframe()
        Path(file_path).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(file_path, index=False)

    @staticmethod
    def export_equity_csv(result: BacktestResult, file_path: str) -> None:
        """Export bar-by-bar equity curve to CSV format."""
        Path(file_path).parent.mkdir(parents=True, exist_ok=True)
        result.equity_curve.to_csv(file_path, index=True)
