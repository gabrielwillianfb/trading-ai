"""CLI script to run event-driven chronological backtest using Model Registry models."""

import argparse
import asyncio
import logging
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.backtesting.engine import BacktestConfig, BacktestEngine
from app.backtesting.report import BacktestReportGenerator
from app.backtesting.signals import SignalGenerator
from app.config.logging import setup_logging
from app.config.settings import get_settings
from app.data.storage.db import get_engine, get_session_factory
from app.data.storage.repository import CandleRepository
from app.features.pipeline import FeaturePipeline
from app.models.inference.predictor import ModelPredictor
from app.models.registry.registry import ModelRegistry


async def main() -> None:
    parser = argparse.ArgumentParser(description="Run event-driven backtest simulation.")
    parser.add_argument("--symbol", type=str, default="BTC/USDT", help="Trading symbol.")
    parser.add_argument("--timeframe", type=str, default="15m", help="Candle timeframe.")
    parser.add_argument(
        "--model-version", type=str, default=None, help="Model version tag. Uses active model if None."
    )
    parser.add_argument("--initial-capital", type=float, default=10000.0, help="Starting capital ($).")
    parser.add_argument("--buy-threshold", type=float, default=0.55, help="Probability threshold for BUY.")
    parser.add_argument("--sell-threshold", type=float, default=0.45, help="Probability threshold for SELL.")
    parser.add_argument("--stop-loss", type=float, default=0.01, help="Stop loss fraction (e.g. 0.01 for 1%).")
    parser.add_argument("--take-profit", type=float, default=0.02, help="Take profit fraction (e.g. 0.02 for 2%).")
    parser.add_argument("--max-holding-bars", type=int, default=12, help="Max holding period in bars.")
    parser.add_argument("--export-trades", type=str, default=None, help="Optional CSV file to export trades.")
    parser.add_argument("--export-equity", type=str, default=None, help="Optional CSV file to export equity.")
    parser.add_argument("--db-url", type=str, default=None, help="Database connection URL.")
    args = parser.parse_args()

    settings = get_settings()
    setup_logging(settings.LOG_LEVEL)
    logger = logging.getLogger("scripts.run_backtest")

    db_url = args.db_url or settings.DATABASE_URL
    engine = get_engine(db_url)
    session_factory = get_session_factory(db_url)

    logger.info("=" * 70)
    logger.info("EVENT-DRIVEN STRATEGY BACKTESTING ENGINE")
    logger.info(f"Target:       {args.symbol} [{args.timeframe}]")
    logger.info(f"Capital:      ${args.initial_capital:,.2f} | Buy Thr: {args.buy_threshold} | Sell Thr: {args.sell_threshold}")
    logger.info(f"Risk Rules:   Stop Loss: {args.stop_loss*100:.1f}% | Take Profit: {args.take_profit*100:.1f}% | Max Hold: {args.max_holding_bars} bars")
    logger.info("=" * 70)

    try:
        async with session_factory() as session:
            # 1. Fetch model from Model Registry
            if args.model_version:
                model_record = await ModelRegistry.get_model_by_version(args.model_version, session)
            else:
                model_record = await ModelRegistry.get_active_model(session)

            if not model_record:
                logger.error("No model found in Model Registry. Run train_model.py first.")
                return

            logger.info(f"Loaded Model: {model_record.version} (ID: {model_record.id}, Active: {model_record.is_active})")
            xgb_model = ModelRegistry.load_model_artifact(model_record)

            # 2. Fetch candles from database
            df_candles = await CandleRepository.get_as_dataframe(args.symbol, args.timeframe, session=session)
            if df_candles.empty:
                logger.error(f"No candles found for {args.symbol} [{args.timeframe}].")
                return
            logger.info(f"Loaded {len(df_candles)} candles from DB.")

            # 3. Compute features
            features_df = FeaturePipeline.compute_features(df_candles, drop_na=True)
            logger.info(f"Computed {len(features_df)} feature vectors.")

            # Align candles with features index
            df_candles_aligned = df_candles.loc[features_df.index]

            # 4. Generate batch predictions
            predictor = ModelPredictor(
                model=xgb_model,
                feature_names=model_record.feature_names,
                model_version=model_record.version,
                model_version_id=model_record.id,
            )
            predictions = predictor.predict_batch(features_df, symbol=args.symbol, timeframe=args.timeframe)
            logger.info(f"Generated {len(predictions)} model predictions.")

            # 5. Generate discrete signals
            signals = SignalGenerator.generate_signals(
                predictions=predictions,
                buy_threshold=args.buy_threshold,
                sell_threshold=args.sell_threshold,
            )
            buy_signals_count = sum(1 for s in signals if s.action == "BUY")
            sell_signals_count = sum(1 for s in signals if s.action == "SELL")
            logger.info(f"Signals generated: BUY={buy_signals_count}, SELL={sell_signals_count}, HOLD={len(signals)-buy_signals_count-sell_signals_count}")

            # 6. Run backtest simulation
            config = BacktestConfig(
                initial_capital=args.initial_capital,
                commission_rate=0.001,
                slippage_rate=0.0005,
                position_size_pct=0.95,
                stop_loss_pct=args.stop_loss,
                take_profit_pct=args.take_profit,
                max_holding_bars=args.max_holding_bars,
            )
            bt_engine = BacktestEngine(config=config)
            result = bt_engine.run(
                candles_df=df_candles_aligned,
                signals=signals,
                symbol=args.symbol,
                timeframe=args.timeframe,
            )

            # 7. Print Performance Tear Sheet
            logger.info("\n" + result.summary() + "\n")

            # 8. Export if requested
            if args.export_trades:
                BacktestReportGenerator.export_trades_csv(result, args.export_trades)
                logger.info(f"Exported trades to: {args.export_trades}")

            if args.export_equity:
                BacktestReportGenerator.export_equity_csv(result, args.export_equity)
                logger.info(f"Exported equity curve to: {args.export_equity}")

    finally:
        await engine.dispose()


if __name__ == "__main__":
    asyncio.run(main())
