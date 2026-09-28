"""High-Velocity 30-Day Scalping Simulation for $400 USDT Starting Capital."""

import asyncio
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd

from app.backtesting.engine import BacktestConfig, BacktestEngine, TradeRecord
from app.backtesting.signals import SignalAction, TradingSignal
from app.config.logging import setup_logging
from app.data.preprocessing.dataset import DatasetBuilder
from app.config.settings import get_settings
from app.data.storage.db import get_engine, get_session_factory
from app.data.storage.repository import CandleRepository
from app.features.pipeline import FeaturePipeline
from app.models.training.multi_model_trainer import MultiModelTrainer


async def main() -> None:
    settings = get_settings()
    setup_logging(settings.LOG_LEVEL)
    engine = get_engine(settings.DATABASE_URL)
    session_factory = get_session_factory(settings.DATABASE_URL)

    STARTING_CAPITAL = 400.0
    DAYS_WINDOW = 30

    print("=" * 80)
    print(f"HIGH-VELOCITY ACCELERATED TRADING SIMULATION ({DAYS_WINDOW} DAYS | ${STARTING_CAPITAL:.2f} USDT)")
    print("=" * 80)

    # High-beta assets evaluated on 1h (sniper setups with 2:1 and 3:1 reward-to-risk)
    assets_to_evaluate = [
        {"symbol": "SOL/USDT", "timeframe": "1h", "horizon": 16, "sl": 0.020, "tp": 0.050, "q": 0.85},
        {"symbol": "BTC/USDT", "timeframe": "1h", "horizon": 16, "sl": 0.015, "tp": 0.035, "q": 0.85},
        {"symbol": "ETH/USDT", "timeframe": "1h", "horizon": 16, "sl": 0.018, "tp": 0.040, "q": 0.85},
    ]

    async with session_factory() as session:
        for cfg in assets_to_evaluate:
            sym = cfg["symbol"]
            tf = cfg["timeframe"]

            df = await CandleRepository.get_as_dataframe(sym, tf, session=session)
            if df.empty:
                print(f"No data for {sym} [{tf}]")
                continue

            # Filter to the most recent 30 days
            cutoff_date = df.index.max() - timedelta(days=DAYS_WINDOW)
            df_recent = df[df.index >= cutoff_date]

            print(f"\n--- Strategy: {sym} [{tf}] (Last {DAYS_WINDOW} Days: {len(df_recent)} candles) ---")

            # Compute features & build dataset
            feats = FeaturePipeline.compute_features(df_recent, drop_na=False)
            ds = DatasetBuilder.build(
                candles_df=df_recent,
                features_df=feats,
                symbol=sym,
                timeframe=tf,
                horizon=cfg["horizon"],
                target_return=0.010,
                label_method="peak",
                drop_na=True,
            )
            if len(ds.X) < 100:
                print(f"Insufficient samples: {len(ds.X)}")
                continue

            splits = DatasetBuilder.split_dataset(ds, train_pct=0.60, val_pct=0.20, test_pct=0.20)

            tr = MultiModelTrainer.train_model(
                "extra_trees",
                splits,
                hyperparameters={"n_estimators": 100, "max_depth": 5, "min_samples_split": 6, "min_samples_leaf": 3, "class_weight": "balanced"}
            )

            # Generate signals on the 30-day window
            probs = tr.model.predict_proba(ds.X[tr.feature_names])[:, 1]
            s_probs = pd.Series(probs, index=ds.X.index)
            thresh = s_probs.quantile(cfg["q"])

            signals = [
                TradingSignal(
                    timestamp=ts,
                    symbol=sym,
                    action=SignalAction.BUY if p >= thresh else SignalAction.HOLD,
                    probability=p,
                    reason="high_velocity_sniper",
                )
                for ts, p in s_probs.items()
            ]

            # Compare: 1) Standard Sizing (10% per trade) vs 2) Full Compounding Sizing (90% per trade)
            for alloc_pct in [0.25, 0.50, 0.90]:
                bt_engine = BacktestEngine(
                    config=BacktestConfig(
                        initial_capital=STARTING_CAPITAL,
                        commission_rate=0.001,
                        slippage_rate=0.0005,
                        stop_loss_pct=cfg["sl"],
                        take_profit_pct=cfg["tp"],
                        max_holding_bars=cfg["horizon"],
                    )
                )

                aligned_candles = df_recent.loc[ds.X.index]
                res = bt_engine.run(aligned_candles, signals, sym, tf)
                m = res.metrics

                # Calculate compounding multiplier if full capital is redeployed
                gross_ret = m["total_return_pct"]
                # Scale return proportionally with allocation leverage
                scaled_return_pct = gross_ret * (alloc_pct / 0.10)
                final_cash = STARTING_CAPITAL * (1.0 + scaled_return_pct / 100.0)

                print(
                    f"  Alloc: {alloc_pct*100:2.0f}% Capital -> "
                    f"Trades: {m['total_trades']:3d} | "
                    f"Win Rate: {m['win_rate_pct']:4.1f}% | "
                    f"Profit Factor: {m['profit_factor']:4.2f} | "
                    f"Return: {scaled_return_pct:+6.2f}% | "
                    f"Final Balance: ${final_cash:,.2f} USDT"
                )

    await engine.dispose()


if __name__ == "__main__":
    asyncio.run(main())
