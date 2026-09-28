"""Financial performance and risk metrics calculation engine."""

import logging
from typing import Any, Dict, List
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Approximate annual periods by timeframe for Sharpe annualization
ANNUAL_PERIODS_MAP = {
    "1m": 525600,
    "5m": 105120,
    "15m": 35040,
    "30m": 17520,
    "1h": 8760,
    "4h": 2190,
    "1d": 365,
}


class PerformanceMetricsCalculator:
    """Calculates comprehensive quantitative portfolio, risk, and trade performance metrics."""

    @classmethod
    def calculate_metrics(
        cls,
        equity_series: pd.Series,
        trades: List[Any],
        benchmark_series: pd.Series,
        timeframe: str = "15m",
        risk_free_rate: float = 0.0,
    ) -> Dict[str, Any]:
        """Compute complete financial and risk tear sheet metrics.

        Args:
            equity_series: Time series of portfolio equity over time.
            trades: List of completed TradeRecord objects.
            benchmark_series: Time series of underlying asset price.
            timeframe: Candle interval (for annualization factors).
            risk_free_rate: Annualized risk-free rate.

        Returns:
            Dictionary containing all key performance and risk indicators.
        """
        if len(equity_series) < 2:
            return cls._empty_metrics()

        init_capital = float(equity_series.iloc[0])
        final_equity = float(equity_series.iloc[-1])
        total_pnl = final_equity - init_capital
        total_return_pct = (total_pnl / init_capital) * 100.0

        # Benchmark Buy & Hold return
        bm_init = float(benchmark_series.iloc[0])
        bm_final = float(benchmark_series.iloc[-1])
        buy_and_hold_return_pct = ((bm_final - bm_init) / bm_init) * 100.0 if bm_init > 0 else 0.0

        # Bar returns for volatility and Sharpe calculation
        bar_returns = equity_series.pct_change().dropna()
        n_bars = len(bar_returns)

        # Annualization factor
        periods_per_year = ANNUAL_PERIODS_MAP.get(timeframe, 35040)
        years = max(n_bars / periods_per_year, 1.0 / periods_per_year)

        # Compound Annual Growth Rate (CAGR)
        if final_equity > 0 and init_capital > 0:
            cagr = ((final_equity / init_capital) ** (1.0 / years) - 1.0) * 100.0
        else:
            cagr = -100.0

        # Volatility
        mean_bar_ret = float(bar_returns.mean())
        std_bar_ret = float(bar_returns.std())
        annualized_vol = std_bar_ret * np.sqrt(periods_per_year) * 100.0

        # Sharpe Ratio
        rf_per_period = (1.0 + risk_free_rate) ** (1.0 / periods_per_year) - 1.0
        excess_returns = bar_returns - rf_per_period
        if std_bar_ret > 1e-8:
            sharpe_ratio = float((excess_returns.mean() / std_bar_ret) * np.sqrt(periods_per_year))
        else:
            sharpe_ratio = 0.0

        # Sortino Ratio (Downside deviation only)
        downside_returns = bar_returns[bar_returns < 0]
        downside_std = float(downside_returns.std()) if len(downside_returns) > 1 else 0.0
        if downside_std > 1e-8:
            sortino_ratio = float((excess_returns.mean() / downside_std) * np.sqrt(periods_per_year))
        else:
            sortino_ratio = 0.0

        # Drawdown calculation
        rolling_max = equity_series.cummax()
        drawdown = (equity_series - rolling_max) / rolling_max
        max_drawdown_pct = float(abs(drawdown.min())) * 100.0

        # Max Drawdown duration in bars
        is_in_dd = drawdown < 0
        dd_durations = []
        current_dd_len = 0
        for in_dd in is_in_dd:
            if in_dd:
                current_dd_len += 1
            else:
                if current_dd_len > 0:
                    dd_durations.append(current_dd_len)
                    current_dd_len = 0
        if current_dd_len > 0:
            dd_durations.append(current_dd_len)
        max_dd_duration = max(dd_durations) if dd_durations else 0

        # Calmar Ratio
        calmar_ratio = float(cagr / max_drawdown_pct) if max_drawdown_pct > 1e-6 else 0.0

        # Trade analytics
        total_trades = len(trades)
        winning_trades = [t for t in trades if t.net_pnl > 0]
        losing_trades = [t for t in trades if t.net_pnl < 0]
        breakeven_trades = [t for t in trades if t.net_pnl == 0]

        win_count = len(winning_trades)
        loss_count = len(losing_trades)
        win_rate_pct = (win_count / total_trades) * 100.0 if total_trades > 0 else 0.0

        gross_gains = sum(t.net_pnl for t in winning_trades)
        gross_losses = abs(sum(t.net_pnl for t in losing_trades))

        if gross_losses > 1e-6:
            profit_factor = float(gross_gains / gross_losses)
        elif gross_gains > 0:
            profit_factor = 999.0
        else:
            profit_factor = 0.0

        avg_trade_pnl = float(total_pnl / total_trades) if total_trades > 0 else 0.0
        avg_trade_pct = (
            float(np.mean([t.net_pnl_pct for t in trades])) if total_trades > 0 else 0.0
        )

        avg_win_pnl = float(gross_gains / win_count) if win_count > 0 else 0.0
        avg_loss_pnl = float(gross_losses / loss_count) if loss_count > 0 else 0.0
        payoff_ratio = float(avg_win_pnl / avg_loss_pnl) if avg_loss_pnl > 1e-6 else 0.0

        total_fees = sum(t.fees_paid for t in trades)
        avg_holding_bars = (
            float(np.mean([t.holding_bars for t in trades])) if total_trades > 0 else 0.0
        )

        return {
            "initial_capital": init_capital,
            "final_equity": final_equity,
            "total_net_pnl": total_pnl,
            "total_return_pct": total_return_pct,
            "cagr_pct": cagr,
            "buy_and_hold_return_pct": buy_and_hold_return_pct,
            "annualized_volatility_pct": annualized_vol,
            "sharpe_ratio": sharpe_ratio,
            "sortino_ratio": sortino_ratio,
            "calmar_ratio": calmar_ratio,
            "max_drawdown_pct": max_drawdown_pct,
            "max_drawdown_duration_bars": max_dd_duration,
            "total_trades": total_trades,
            "winning_trades": win_count,
            "losing_trades": loss_count,
            "breakeven_trades": len(breakeven_trades),
            "win_rate_pct": win_rate_pct,
            "profit_factor": profit_factor,
            "expectancy_pnl": avg_trade_pnl,
            "avg_trade_return_pct": avg_trade_pct,
            "payoff_ratio": payoff_ratio,
            "total_fees_paid": total_fees,
            "avg_holding_bars": avg_holding_bars,
        }

    @staticmethod
    def _empty_metrics() -> Dict[str, Any]:
        return {
            "initial_capital": 0.0,
            "final_equity": 0.0,
            "total_net_pnl": 0.0,
            "total_return_pct": 0.0,
            "cagr_pct": 0.0,
            "buy_and_hold_return_pct": 0.0,
            "annualized_volatility_pct": 0.0,
            "sharpe_ratio": 0.0,
            "sortino_ratio": 0.0,
            "calmar_ratio": 0.0,
            "max_drawdown_pct": 0.0,
            "max_drawdown_duration_bars": 0,
            "total_trades": 0,
            "winning_trades": 0,
            "losing_trades": 0,
            "breakeven_trades": 0,
            "win_rate_pct": 0.0,
            "profit_factor": 0.0,
            "expectancy_pnl": 0.0,
            "avg_trade_return_pct": 0.0,
            "payoff_ratio": 0.0,
            "total_fees_paid": 0.0,
            "avg_holding_bars": 0.0,
        }
