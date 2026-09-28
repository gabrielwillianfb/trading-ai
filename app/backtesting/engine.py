"""Event-driven chronological backtesting simulation engine with realistic execution modeling."""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd

from app.backtesting.metrics import PerformanceMetricsCalculator
from app.backtesting.signals import SignalAction, TradingSignal

logger = logging.getLogger(__name__)


@dataclass
class BacktestConfig:
    """Configuration parameters for trade execution and risk constraints in backtesting."""

    initial_capital: float = 10000.0
    commission_rate: float = 0.001  # 0.1% maker/taker Binance fee
    slippage_rate: float = 0.0005  # 0.05% slippage on market fills
    position_size_pct: float = 0.95  # 95% of available cash per entry
    stop_loss_pct: Optional[float] = 0.01  # 1.0% stop loss
    take_profit_pct: Optional[float] = 0.02  # 2.0% take profit
    max_holding_bars: Optional[int] = 12  # Maximum holding time limit
    risk_free_rate: float = 0.0  # Annualized risk-free rate for Sharpe


@dataclass
class TradeRecord:
    """Detailed log record of an individual round-trip trade."""

    trade_id: int
    symbol: str
    entry_time: datetime
    exit_time: datetime
    entry_price: float
    exit_price: float
    side: str
    quantity: float
    gross_pnl: float
    net_pnl: float
    net_pnl_pct: float
    fees_paid: float
    exit_reason: str
    holding_bars: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "trade_id": self.trade_id,
            "symbol": self.symbol,
            "entry_time": self.entry_time.isoformat(),
            "exit_time": self.exit_time.isoformat(),
            "entry_price": round(self.entry_price, 4),
            "exit_price": round(self.exit_price, 4),
            "side": self.side,
            "quantity": round(self.quantity, 6),
            "gross_pnl": round(self.gross_pnl, 2),
            "net_pnl": round(self.net_pnl, 2),
            "net_pnl_pct": round(self.net_pnl_pct, 4),
            "fees_paid": round(self.fees_paid, 4),
            "exit_reason": self.exit_reason,
            "holding_bars": self.holding_bars,
        }


@dataclass
class BacktestResult:
    """Comprehensive output container for backtest simulation results."""

    config: BacktestConfig
    symbol: str
    timeframe: str
    metrics: Dict[str, Any]
    trades: List[TradeRecord]
    equity_curve: pd.DataFrame

    def to_trades_dataframe(self) -> pd.DataFrame:
        """Convert closed trades list into structured pandas DataFrame."""
        if not self.trades:
            return pd.DataFrame()
        return pd.DataFrame([t.to_dict() for t in self.trades])

    def summary(self) -> str:
        """Render formatted performance tear sheet."""
        m = self.metrics
        lines = [
            "=" * 70,
            f"BACKTEST PERFORMANCE TEAR SHEET: {self.symbol} [{self.timeframe}]",
            "=" * 70,
            "CAPITAL & RETURNS:",
            f"  - Initial Capital:           ${m['initial_capital']:,.2f}",
            f"  - Final Equity:              ${m['final_equity']:,.2f}",
            f"  - Total Net PnL:             ${m['total_net_pnl']:,.2f} ({m['total_return_pct']:+.2f}%)",
            f"  - CAGR:                      {m['cagr_pct']:+.2f}%",
            f"  - Benchmark (Buy & Hold):    {m['buy_and_hold_return_pct']:+.2f}%",
            "-" * 70,
            "RISK-ADJUSTED METRICS:",
            f"  - Sharpe Ratio:              {m['sharpe_ratio']:.2f}",
            f"  - Sortino Ratio:             {m['sortino_ratio']:.2f}",
            f"  - Calmar Ratio:              {m['calmar_ratio']:.2f}",
            f"  - Max Drawdown (MDD):        {m['max_drawdown_pct']:.2f}% ({m['max_drawdown_duration_bars']} bars duration)",
            f"  - Annualized Volatility:     {m['annualized_volatility_pct']:.2f}%",
            "-" * 70,
            "TRADE ANALYTICS:",
            f"  - Total Trades:              {m['total_trades']} (Wins: {m['winning_trades']}, Losses: {m['losing_trades']})",
            f"  - Win Rate:                  {m['win_rate_pct']:.1f}%",
            f"  - Profit Factor:             {m['profit_factor']:.2f}",
            f"  - Payoff Ratio (Win/Loss):   {m['payoff_ratio']:.2f}",
            f"  - Expectancy (Avg Net PnL):  ${m['expectancy_pnl']:+.2f} ({m['avg_trade_return_pct']:+.2f}%)",
            f"  - Avg Holding Time:          {m['avg_holding_bars']:.1f} bars",
            f"  - Total Commissions Paid:    ${m['total_fees_paid']:,.2f}",
            "=" * 70,
        ]
        return "\n".join(lines)


class BacktestEngine:
    """Chronological event-driven simulation engine for algorithmic strategy backtesting."""

    def __init__(self, config: Optional[BacktestConfig] = None):
        self.config = config or BacktestConfig()

    def run(
        self,
        candles_df: pd.DataFrame,
        signals: List[TradingSignal],
        symbol: str = "BTC/USDT",
        timeframe: str = "15m",
    ) -> BacktestResult:
        """Execute chronological bar-by-bar backtest simulation.

        Zero Look-Ahead Guarantee:
            - Signal emitted at close of candle t is executed strictly at candle t+1 Open.

        Args:
            candles_df: OHLCV DataFrame with DatetimeIndex in UTC.
            signals: List of TradingSignals generated by strategy/model.
            symbol: Trading asset symbol.
            timeframe: Candle interval.

        Returns:
            BacktestResult instance.
        """
        if candles_df.empty:
            raise ValueError("Cannot run backtest on empty candles DataFrame.")

        # Map signals by timestamp for fast lookup
        signal_map = {}
        for s in signals:
            ts = s.timestamp
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=timezone.utc)
            signal_map[ts] = s

        cash = self.config.initial_capital
        position_qty = 0.0
        position_entry_price = 0.0
        position_entry_time = None
        position_holding_bars = 0
        trade_counter = 0

        trades: List[TradeRecord] = []
        pending_signal: Optional[TradingSignal] = None

        timestamps = []
        cash_history = []
        pos_value_history = []
        equity_history = []

        cfg = self.config

        for i, (bar_ts, row) in enumerate(candles_df.iterrows()):
            current_time = bar_ts.to_pydatetime() if hasattr(bar_ts, "to_pydatetime") else bar_ts
            if current_time.tzinfo is None:
                current_time = current_time.replace(tzinfo=timezone.utc)

            bar_open = float(row["open"])
            bar_high = float(row["high"])
            bar_low = float(row["low"])
            bar_close = float(row["close"])

            # -------------------------------------------------------------
            # 1. Check Intrabar SL / TP / Time Exit on open position
            # -------------------------------------------------------------
            if position_qty > 0:
                position_holding_bars += 1
                exit_price = None
                exit_reason = None

                # Check Stop Loss
                if cfg.stop_loss_pct is not None:
                    sl_trigger = position_entry_price * (1.0 - cfg.stop_loss_pct)
                    if bar_low <= sl_trigger:
                        # Filled at stop price or open (if gap down)
                        exit_price = min(bar_open, sl_trigger) * (1.0 - cfg.slippage_rate)
                        exit_reason = "stop_loss"

                # Check Take Profit
                if exit_price is None and cfg.take_profit_pct is not None:
                    tp_trigger = position_entry_price * (1.0 + cfg.take_profit_pct)
                    if bar_high >= tp_trigger:
                        # Filled at take profit price or open (if gap up)
                        exit_price = max(bar_open, tp_trigger) * (1.0 - cfg.slippage_rate)
                        exit_reason = "take_profit"

                # Check Time-based Exit limit
                if exit_price is None and cfg.max_holding_bars is not None:
                    if position_holding_bars >= cfg.max_holding_bars:
                        exit_price = bar_open * (1.0 - cfg.slippage_rate)
                        exit_reason = "time_exit"

                # Execute Exit if triggered
                if exit_price is not None:
                    gross_proceeds = position_qty * exit_price
                    exit_fee = gross_proceeds * cfg.commission_rate
                    net_proceeds = gross_proceeds - exit_fee
                    cash += net_proceeds

                    entry_cost = position_qty * position_entry_price
                    entry_fee = entry_cost * cfg.commission_rate
                    total_fees = entry_fee + exit_fee
                    net_pnl = (gross_proceeds - entry_cost) - total_fees
                    net_pnl_pct = (net_pnl / entry_cost) * 100.0 if entry_cost > 0 else 0.0

                    trade_counter += 1
                    trades.append(
                        TradeRecord(
                            trade_id=trade_counter,
                            symbol=symbol,
                            entry_time=position_entry_time,
                            exit_time=current_time,
                            entry_price=position_entry_price,
                            exit_price=exit_price,
                            side="BUY",
                            quantity=position_qty,
                            gross_pnl=gross_proceeds - entry_cost,
                            net_pnl=net_pnl,
                            net_pnl_pct=net_pnl_pct,
                            fees_paid=total_fees,
                            exit_reason=exit_reason,
                            holding_bars=position_holding_bars,
                        )
                    )

                    # Reset position
                    position_qty = 0.0
                    position_entry_price = 0.0
                    position_entry_time = None
                    position_holding_bars = 0

            # -------------------------------------------------------------
            # 2. Process Pending Signal from previous candle close (t-1)
            # -------------------------------------------------------------
            if pending_signal is not None:
                if pending_signal.action == SignalAction.BUY and position_qty == 0:
                    fill_price = bar_open * (1.0 + cfg.slippage_rate)
                    alloc_capital = cash * cfg.position_size_pct
                    qty = alloc_capital / (fill_price * (1.0 + cfg.commission_rate))

                    if qty > 0:
                        gross_cost = qty * fill_price
                        fee = gross_cost * cfg.commission_rate
                        total_cost = gross_cost + fee

                        if cash >= total_cost:
                            cash -= total_cost
                            position_qty = qty
                            position_entry_price = fill_price
                            position_entry_time = current_time
                            position_holding_bars = 0

                elif pending_signal.action == SignalAction.SELL and position_qty > 0:
                    fill_price = bar_open * (1.0 - cfg.slippage_rate)
                    gross_proceeds = position_qty * fill_price
                    exit_fee = gross_proceeds * cfg.commission_rate
                    net_proceeds = gross_proceeds - exit_fee
                    cash += net_proceeds

                    entry_cost = position_qty * position_entry_price
                    entry_fee = entry_cost * cfg.commission_rate
                    total_fees = entry_fee + exit_fee
                    net_pnl = (gross_proceeds - entry_cost) - total_fees
                    net_pnl_pct = (net_pnl / entry_cost) * 100.0 if entry_cost > 0 else 0.0

                    trade_counter += 1
                    trades.append(
                        TradeRecord(
                            trade_id=trade_counter,
                            symbol=symbol,
                            entry_time=position_entry_time,
                            exit_time=current_time,
                            entry_price=position_entry_price,
                            exit_price=fill_price,
                            side="BUY",
                            quantity=position_qty,
                            gross_pnl=gross_proceeds - entry_cost,
                            net_pnl=net_pnl,
                            net_pnl_pct=net_pnl_pct,
                            fees_paid=total_fees,
                            exit_reason="signal_exit",
                            holding_bars=position_holding_bars,
                        )
                    )

                    position_qty = 0.0
                    position_entry_price = 0.0
                    position_entry_time = None
                    position_holding_bars = 0

                pending_signal = None

            # -------------------------------------------------------------
            # 3. Mark-to-Market Equity at Bar Close
            # -------------------------------------------------------------
            pos_val = position_qty * bar_close
            total_equity = cash + pos_val

            timestamps.append(current_time)
            cash_history.append(cash)
            pos_value_history.append(pos_val)
            equity_history.append(total_equity)

            # -------------------------------------------------------------
            # 4. Queue Signal emitted at current candle Close for t+1 Open
            # -------------------------------------------------------------
            if current_time in signal_map:
                pending_signal = signal_map[current_time]

        # -----------------------------------------------------------------
        # Close open position at end of simulation if any
        # -----------------------------------------------------------------
        if position_qty > 0:
            final_close = float(candles_df["close"].iloc[-1])
            exit_price = final_close * (1.0 - cfg.slippage_rate)
            gross_proceeds = position_qty * exit_price
            exit_fee = gross_proceeds * cfg.commission_rate
            net_proceeds = gross_proceeds - exit_fee
            cash += net_proceeds

            entry_cost = position_qty * position_entry_price
            entry_fee = entry_cost * cfg.commission_rate
            total_fees = entry_fee + exit_fee
            net_pnl = (gross_proceeds - entry_cost) - total_fees
            net_pnl_pct = (net_pnl / entry_cost) * 100.0 if entry_cost > 0 else 0.0

            trade_counter += 1
            trades.append(
                TradeRecord(
                    trade_id=trade_counter,
                    symbol=symbol,
                    entry_time=position_entry_time,
                    exit_time=timestamps[-1],
                    entry_price=position_entry_price,
                    exit_price=exit_price,
                    side="BUY",
                    quantity=position_qty,
                    gross_pnl=gross_proceeds - entry_cost,
                    net_pnl=net_pnl,
                    net_pnl_pct=net_pnl_pct,
                    fees_paid=total_fees,
                    exit_reason="end_of_data",
                    holding_bars=position_holding_bars,
                )
            )
            # Update final equity
            equity_history[-1] = cash
            pos_value_history[-1] = 0.0

        # Build equity DataFrame
        equity_series = pd.Series(equity_history, index=pd.to_datetime(timestamps, utc=True))
        benchmark_series = candles_df["close"].astype(float)
        benchmark_normalized = (benchmark_series / benchmark_series.iloc[0]) * cfg.initial_capital

        cummax = equity_series.cummax()
        drawdown_pct = ((cummax - equity_series) / cummax) * 100.0

        equity_df = pd.DataFrame(
            {
                "cash": cash_history,
                "position_value": pos_value_history,
                "equity": equity_history,
                "drawdown_pct": drawdown_pct.values,
                "benchmark_equity": benchmark_normalized.values,
            },
            index=equity_series.index,
        )

        metrics = PerformanceMetricsCalculator.calculate_metrics(
            equity_series=equity_series,
            trades=trades,
            benchmark_series=benchmark_series,
            timeframe=timeframe,
            risk_free_rate=cfg.risk_free_rate,
        )

        return BacktestResult(
            config=cfg,
            symbol=symbol,
            timeframe=timeframe,
            metrics=metrics,
            trades=trades,
            equity_curve=equity_df,
        )
