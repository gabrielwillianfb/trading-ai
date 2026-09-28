"""Institutional Quantitative Optimization and Model Training Script."""

import argparse
import asyncio
import logging
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.backtesting.engine import BacktestConfig, BacktestEngine
from app.backtesting.signals import SignalGenerator
from app.config.logging import setup_logging
from app.config.settings import get_settings
from app.data.collectors.binance import BinanceClient
from app.data.collectors.historical import HistoricalDataCollector
from app.data.preprocessing.dataset import DatasetBuilder
from app.data.preprocessing.validator import DataValidator
from app.data.storage.db import get_engine, get_session_factory, init_db
from app.data.storage.repository import CandleRepository
from app.features.pipeline import FeaturePipeline
from app.features.repository import FeatureRepository
from app.models.inference.predictor import ModelPredictor
from app.models.registry.registry import ModelRegistry
from app.models.training.optimizer import StrategyOptimizer
from app.models.training.trainer import ModelTrainer


async def run_optimization(
    symbol: str = "BTC/USDT",
    timeframe: str = "15m",
    days: int = 60,
    horizon: int = 12,
    initial_cash: float = 10000.0,
    db_url: str = None,
) -> None:
    settings = get_settings()
    setup_logging(settings.LOG_LEVEL)
    logger = logging.getLogger("scripts.train_optimized_model")

    db_connection_url = db_url or settings.DATABASE_URL
    engine = get_engine(db_connection_url)
    await init_db(engine)
    session_factory = get_session_factory(db_connection_url)

    logger.info("=" * 75)
    logger.info("INSTITUTIONAL QUANTITATIVE OPTIMIZATION & MODEL TRAINING")
    logger.info(f"Symbol:         {symbol} [{timeframe}]")
    logger.info(f"History Window: {days} days")
    logger.info(f"Prediction:     {horizon} bars forward ({(horizon * 15) // 60} hours)")
    logger.info(f"Initial Cash:   ${initial_cash:,.2f} USDT")
    logger.info("=" * 75)

    client = BinanceClient(base_url="https://api.binance.com/api")
    collector = HistoricalDataCollector(exchange_client=client)

    try:
        # 1. Coleta e atualização de dados
        end_time = datetime.now(timezone.utc)
        start_time = end_time - timedelta(days=days)
        candles = await collector.collect_range(
            symbol=symbol,
            timeframe=timeframe,
            start_time=start_time,
            end_time=end_time,
        )
        logger.info(f"Candles coletados: {len(candles)}")

        async with session_factory() as session:
            await CandleRepository.save_candles(candles, session)
            df_candles = await CandleRepository.get_as_dataframe(
                symbol=symbol, timeframe=timeframe, session=session
            )

            # 2. Features com detecção de regime
            logger.info("Calculando 32 features incluindo regime de mercado e tendência macro...")
            features_df = FeaturePipeline.compute_features(df_candles, drop_na=True)
            await FeatureRepository.save_features(features_df, symbol, timeframe, session)

            # 3. Dataset
            dataset = DatasetBuilder.build(
                candles_df=df_candles,
                symbol=symbol,
                timeframe=timeframe,
                horizon=horizon,
                target_return=0.006,  # 0.6% retorno alvo para cobrir taxas de 0.1% e slippage com folga
                drop_na=True,
            )
            splits = DatasetBuilder.split_dataset(dataset, train_pct=0.70, val_pct=0.15, test_pct=0.15)
            logger.info(f"Dataset construído: {dataset.total_samples} amostras | Features: {len(dataset.feature_names)}")

            # 4. Otimização de Hiperparâmetros e Threshold
            opt_result = StrategyOptimizer.optimize(
                dataset=dataset,
                splits=splits,
                candles_df=df_candles,
                symbol=symbol,
                timeframe=timeframe,
                horizon=horizon,
                initial_capital=initial_cash,
            )

            # 5. Treinamento do Modelo Otimizado Final
            logger.info(f"\nTreinando modelo final com hiperparâmetros institucionais: {opt_result.best_hyperparameters}")
            trainer = ModelTrainer()
            train_res = trainer.train(splits, hyperparameters=opt_result.best_hyperparameters)

            # 6. Registro no Model Registry
            model_ver = await ModelRegistry.register_model(
                training_result=train_res,
                symbol=symbol,
                timeframe=timeframe,
                session=session,
                set_active=True,
                model_dir="models",
            )
            logger.info(f"Modelo Otimizado Registrado e Ativado: {model_ver.version}")

            # 7. Backtest Final com Threshold Otimizado de Alta Convicção
            xgb_model = ModelRegistry.load_model_artifact(model_ver)
            predictor = ModelPredictor(
                model=xgb_model,
                feature_names=model_ver.feature_names,
                model_version=model_ver.version,
                model_version_id=model_ver.id,
            )
            predictions = predictor.predict_batch(splits.X_test, symbol=symbol, timeframe=timeframe)
            signals = [
                SignalGenerator.generate_signal_from_prediction(
                    p,
                    buy_threshold=opt_result.best_buy_threshold,
                    sell_threshold=opt_result.best_sell_threshold,
                )
                for p in predictions
            ]

            test_candles_df = df_candles.loc[splits.X_test.index]
            bt_engine = BacktestEngine(
                config=BacktestConfig(
                    initial_capital=initial_cash,
                    commission_rate=0.001,
                    slippage_rate=0.0005,
                    stop_loss_pct=0.01,
                    take_profit_pct=0.02,
                    max_holding_bars=horizon,
                )
            )
            bt_res = bt_engine.run(
                candles_df=test_candles_df,
                signals=signals,
                symbol=symbol,
                timeframe=timeframe,
            )
            m = bt_res.metrics

            logger.info("\n" + "=" * 75)
            logger.info("RESULTADOS DO MODELO INSTITUCIONAL OTIMIZADO:")
            logger.info(f"  - Modelo Ativo:             {model_ver.version}")
            logger.info(f"  - Limiar de Alta Convicção: P(UP) >= {opt_result.best_buy_threshold:.2f}")
            logger.info(f"  - Retorno Líquido Total:    {m['total_return_pct']:+.2f}%")
            logger.info(f"  - Retorno Buy & Hold:       {m['buy_and_hold_return_pct']:+.2f}%")
            logger.info(f"  - Sharpe Ratio:             {m['sharpe_ratio']:.2f}")
            logger.info(f"  - Sortino Ratio:            {m['sortino_ratio']:.2f}")
            logger.info(f"  - Rebaixamento Máximo:      {m['max_drawdown_pct']:.2f}%")
            logger.info(f"  - Total de Trades:          {m['total_trades']} (Win Rate: {m['win_rate_pct']:.1f}%)")
            logger.info(f"  - Profit Factor:            {m['profit_factor']:.2f}")
            logger.info(f"  - Payoff Ratio:             {m['payoff_ratio']:.2f}")
            logger.info(f"  - Expectância por Trade:    ${m['expectancy_pnl']:+,.2f}")
            logger.info(f"  - Corretagem Total Paga:    ${m['total_fees_paid']:,.2f}")
            logger.info("=" * 75)

    finally:
        await client.close()
        await engine.dispose()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Institutional Quantitative Optimization.")
    parser.add_argument("--symbol", type=str, default="BTC/USDT")
    parser.add_argument("--timeframe", type=str, default="15m")
    parser.add_argument("--days", type=int, default=60)
    parser.add_argument("--horizon", type=int, default=12)
    parser.add_argument("--initial-cash", type=float, default=10000.0)
    parser.add_argument("--db-url", type=str, default=None)
    args = parser.parse_args()

    asyncio.run(
        run_optimization(
            symbol=args.symbol,
            timeframe=args.timeframe,
            days=args.days,
            horizon=args.horizon,
            initial_cash=args.initial_cash,
            db_url=args.db_url,
        )
    )
