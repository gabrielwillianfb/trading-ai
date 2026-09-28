"""Tests for BacktestEngine, PerformanceMetricsCalculator, SignalGenerator, and execution modeling."""

from datetime import datetime, timedelta, timezone
import numpy as np
import pandas as pd
import pytest

from app.backtesting.engine import BacktestConfig, BacktestEngine, TradeRecord
from app.backtesting.metrics import PerformanceMetricsCalculator
from app.backtesting.report import BacktestReportGenerator
from app.backtesting.signals import SignalAction, SignalGenerator, TradingSignal
from app.models.inference.predictor import PredictionResult


def create_deterministic_candles(prices: list[float], start_date="2026-01-01") -> pd.DataFrame:
    """Helper to create deterministic OHLCV candles from a list of close prices."""
    dates = pd.date_range(start_date, periods=len(prices), freq="15min", tz="UTC")
    opens = [prices[0]] + prices[:-1]
    highs = [max(o, c) * 1.001 for o, c in zip(opens, prices)]
    lows = [min(o, c) * 0.999 for o, c in zip(opens, prices)]
    volumes = [100.0] * len(prices)

    return pd.DataFrame(
        {
            "open": opens,
            "high": highs,
            "low": lows,
            "close": prices,
            "volume": volumes,
        },
        index=dates,
    )


# ------------------------------------------------------------------------------
# 1. SignalGenerator Tests
# ------------------------------------------------------------------------------

def test_signal_generator_thresholds():
    """Verify probability threshold mapping to BUY, SELL, and HOLD actions."""
    pred_buy = PredictionResult(
        timestamp=datetime.now(timezone.utc),
        symbol="BTC/USDT",
        timeframe="15m",
        probability_up=0.75,
        probability_down=0.25,
        predicted_class=1,
        model_version="v1",
    )
    sig_buy = SignalGenerator.generate_signal_from_prediction(pred_buy, buy_threshold=0.60, sell_threshold=0.40)
    assert sig_buy.action == SignalAction.BUY

    pred_sell = PredictionResult(
        timestamp=datetime.now(timezone.utc),
        symbol="BTC/USDT",
        timeframe="15m",
        probability_up=0.30,
        probability_down=0.70,
        predicted_class=0,
        model_version="v1",
    )
    sig_sell = SignalGenerator.generate_signal_from_prediction(pred_sell, buy_threshold=0.60, sell_threshold=0.40)
    assert sig_sell.action == SignalAction.SELL

    pred_hold = PredictionResult(
        timestamp=datetime.now(timezone.utc),
        symbol="BTC/USDT",
        timeframe="15m",
        probability_up=0.50,
        probability_down=0.50,
        predicted_class=0,
        model_version="v1",
    )
    sig_hold = SignalGenerator.generate_signal_from_prediction(pred_hold, buy_threshold=0.60, sell_threshold=0.40)
    assert sig_hold.action == SignalAction.HOLD


# ------------------------------------------------------------------------------
# 2. PerformanceMetricsCalculator Tests
# ------------------------------------------------------------------------------

def test_metrics_calculator_math():
    """Verify performance metrics calculation on known equity curve."""
    dates = pd.date_range("2026-01-01", periods=5, freq="15min", tz="UTC")
    equity_series = pd.Series([10000.0, 10200.0, 10100.0, 10500.0, 10400.0], index=dates)
    benchmark_series = pd.Series([50000.0, 50500.0, 50000.0, 51000.0, 52000.0], index=dates)

    trades = [
        TradeRecord(
            trade_id=1,
            symbol="BTC/USDT",
            entry_time=dates[0].to_pydatetime(),
            exit_time=dates[2].to_pydatetime(),
            entry_price=100.0,
            exit_price=105.0,
            side="BUY",
            quantity=10.0,
            gross_pnl=50.0,
            net_pnl=48.0,
            net_pnl_pct=4.8,
            fees_paid=2.0,
            exit_reason="take_profit",
            holding_bars=2,
        ),
        TradeRecord(
            trade_id=2,
            symbol="BTC/USDT",
            entry_time=dates[2].to_pydatetime(),
            exit_time=dates[4].to_pydatetime(),
            entry_price=105.0,
            exit_price=103.0,
            side="BUY",
            quantity=10.0,
            gross_pnl=-20.0,
            net_pnl=-22.0,
            net_pnl_pct=-2.09,
            fees_paid=2.0,
            exit_reason="stop_loss",
            holding_bars=2,
        ),
    ]

    metrics = PerformanceMetricsCalculator.calculate_metrics(
        equity_series=equity_series,
        trades=trades,
        benchmark_series=benchmark_series,
        timeframe="15m",
    )

    assert metrics["total_return_pct"] == 4.0  # (10400 - 10000) / 10000 * 100
    assert metrics["total_trades"] == 2
    assert metrics["winning_trades"] == 1
    assert metrics["losing_trades"] == 1
    assert metrics["win_rate_pct"] == 50.0
    assert np.isclose(metrics["profit_factor"], 48.0 / 22.0)
    assert metrics["max_drawdown_pct"] > 0.0


# ------------------------------------------------------------------------------
# 3. BacktestEngine Execution & Zero Look-Ahead Tests
# ------------------------------------------------------------------------------

def test_backtest_execution_at_next_bar_open():
    """CRITICAL TEST: Signal emitted at candle t close is filled at candle t+1 OPEN."""
    prices = [100.0, 102.0, 105.0, 108.0]
    df = create_deterministic_candles(prices)

    # Signal BUY at t=0 Close
    sig_buy = TradingSignal(
        timestamp=df.index[0].to_pydatetime(),
        symbol="BTC/USDT",
        action=SignalAction.BUY,
        probability=0.80,
        reason="Test buy",
    )
    # Signal SELL at t=2 Close
    sig_sell = TradingSignal(
        timestamp=df.index[2].to_pydatetime(),
        symbol="BTC/USDT",
        action=SignalAction.SELL,
        probability=0.20,
        reason="Test sell",
    )

    config = BacktestConfig(
        initial_capital=10000.0,
        commission_rate=0.0,
        slippage_rate=0.0,
        stop_loss_pct=None,
        take_profit_pct=None,
        max_holding_bars=None,
    )
    engine = BacktestEngine(config=config)
    result = engine.run(df, signals=[sig_buy, sig_sell])

    assert len(result.trades) == 1
    trade = result.trades[0]

    # Entry occurred at t=1 Open (which is equal to prices[0] = 100.0)
    assert trade.entry_time == df.index[1].to_pydatetime()
    assert trade.entry_price == float(df["open"].iloc[1])

    # Exit occurred at t=3 Open (which is equal to prices[2] = 105.0)
    assert trade.exit_time == df.index[3].to_pydatetime()
    assert trade.exit_price == float(df["open"].iloc[3])
    assert trade.exit_reason == "signal_exit"


def test_backtest_take_profit_trigger():
    """Verify Take Profit is triggered intrabar when price reaches TP threshold."""
    # Prices: 100 -> 100 -> 103 (+3% > 2% TP)
    df = create_deterministic_candles([100.0, 100.0, 103.0])
    df.iloc[2, df.columns.get_loc("high")] = 103.5  # High touches TP target

    sig_buy = TradingSignal(
        timestamp=df.index[0].to_pydatetime(),
        symbol="BTC/USDT",
        action=SignalAction.BUY,
        probability=0.90,
        reason="Buy for TP test",
    )

    config = BacktestConfig(
        initial_capital=10000.0,
        commission_rate=0.0,
        slippage_rate=0.0,
        take_profit_pct=0.02,  # +2.0% Take Profit
        stop_loss_pct=None,
        max_holding_bars=None,
    )
    engine = BacktestEngine(config=config)
    result = engine.run(df, signals=[sig_buy])

    assert len(result.trades) == 1
    trade = result.trades[0]
    assert trade.exit_reason == "take_profit"
    assert np.isclose(trade.entry_price, 100.0)
    assert np.isclose(trade.exit_price, 102.0)  # 100 * (1 + 0.02)


def test_backtest_stop_loss_trigger():
    """Verify Stop Loss is triggered intrabar when price breaches SL threshold."""
    # Prices: 100 -> 100 -> 98 (-2% < -1% SL)
    df = create_deterministic_candles([100.0, 100.0, 98.0])
    df.iloc[2, df.columns.get_loc("low")] = 97.5  # Low breaches SL target

    sig_buy = TradingSignal(
        timestamp=df.index[0].to_pydatetime(),
        symbol="BTC/USDT",
        action=SignalAction.BUY,
        probability=0.90,
        reason="Buy for SL test",
    )

    config = BacktestConfig(
        initial_capital=10000.0,
        commission_rate=0.0,
        slippage_rate=0.0,
        take_profit_pct=None,
        stop_loss_pct=0.01,  # -1.0% Stop Loss
        max_holding_bars=None,
    )
    engine = BacktestEngine(config=config)
    result = engine.run(df, signals=[sig_buy])

    assert len(result.trades) == 1
    trade = result.trades[0]
    assert trade.exit_reason == "stop_loss"
    assert np.isclose(trade.entry_price, 100.0)
    assert np.isclose(trade.exit_price, 99.0)  # 100 * (1 - 0.01)


def test_backtest_report_export(tmp_path):
    """Verify export utilities for JSON and CSV."""
    df = create_deterministic_candles([100.0, 101.0, 102.0])
    sig_buy = TradingSignal(
        timestamp=df.index[0].to_pydatetime(),
        symbol="BTC/USDT",
        action=SignalAction.BUY,
        probability=0.85,
        reason="Test export",
    )
    engine = BacktestEngine(BacktestConfig())
    result = engine.run(df, [sig_buy])

    json_path = str(tmp_path / "report.json")
    csv_trades = str(tmp_path / "trades.csv")
    csv_equity = str(tmp_path / "equity.csv")

    BacktestReportGenerator.export_to_json(result, json_path)
    BacktestReportGenerator.export_trades_csv(result, csv_trades)
    BacktestReportGenerator.export_equity_csv(result, csv_equity)

    assert (tmp_path / "report.json").exists()
    assert (tmp_path / "trades.csv").exists()
    assert (tmp_path / "equity.csv").exists()
