import asyncio
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
from app.config.settings import get_settings
from app.data.storage.db import get_session_factory
from app.data.storage.repository import CandleRepository
from app.data.preprocessing.dataset import DatasetBuilder
from app.features.pipeline import FeaturePipeline
from app.models.training.multi_model_trainer import MultiModelTrainer
from app.backtesting.engine import BacktestConfig, BacktestEngine
from app.backtesting.signals import TradingSignal, SignalAction


async def main() -> None:
    settings = get_settings()
    session_f = get_session_factory(settings.DATABASE_URL)
    async with session_f() as s:
        # Load 15m and 1h data
        df_15m = await CandleRepository.get_as_dataframe('BTC/USDT', '15m', session=s)
        df_1h = await CandleRepository.get_as_dataframe('BTC/USDT', '1h', session=s)
        df_eth_15m = await CandleRepository.get_as_dataframe('ETH/USDT', '15m', session=s)
    
    print(f"Loaded: BTC/USDT 15m ({len(df_15m)}), BTC/USDT 1h ({len(df_1h)}), ETH/USDT 15m ({len(df_eth_15m)})")
    
    # We will test multiple alpha target setups across BTC 15m, BTC 1h, ETH 15m
    test_datasets = [
        ("BTC/USDT", "15m", df_15m),
        ("BTC/USDT", "1h", df_1h),
        ("ETH/USDT", "15m", df_eth_15m),
    ]
    
    results = []
    
    for symbol, tf, df in test_datasets:
        if len(df) < 500:
            continue
            
        print(f"\n==========================================")
        print(f"EVALUATING STRATEGIES FOR {symbol} [{tf}] ({len(df)} candles)")
        print(f"==========================================")
        
        feats = FeaturePipeline.compute_features(df, drop_na=False)
        
        # Test 1: Trend Breakout & Momentum Continuation
        # Label: Future return > 1.2% over 16 bars AND regime_trend_score >= 0
        close = df['close']
        high = df['high']
        low = df['low']
        
        # Test multiple labeling methods:
        # A) Return target with Horizon 8, 12, 16, 24
        # B) Peak target
        # C) Dynamic ATR with balanced TP/SL
        
        for horizon in [8, 12, 16, 24]:
            for tgt_ret in [0.008, 0.012, 0.015, 0.020]:
                for label_m in ['peak', 'return', 'dynamic_atr_tb']:
                    try:
                        ds = DatasetBuilder.build(
                            candles_df=df,
                            features_df=feats,
                            symbol=symbol,
                            timeframe=tf,
                            horizon=horizon,
                            target_return=tgt_ret,
                            label_method=label_m,
                            tp_atr_mult=2.5,
                            sl_atr_mult=1.5,
                            drop_na=True,
                        )
                        splits = DatasetBuilder.split_dataset(ds, train_pct=0.70, val_pct=0.15, test_pct=0.15)
                        
                        # Train Random Forest and XGBoost
                        for m_type in ['random_forest', 'xgboost_classifier', 'extra_trees']:
                            if m_type == 'xgboost_classifier':
                                params = {'max_depth': 3, 'learning_rate': 0.03, 'n_estimators': 100, 'subsample': 0.8, 'reg_alpha': 0.5, 'reg_lambda': 3.0}
                            elif m_type == 'random_forest':
                                params = {'n_estimators': 120, 'max_depth': 5, 'min_samples_split': 10, 'min_samples_leaf': 5, 'class_weight': 'balanced'}
                            else:
                                params = {'n_estimators': 120, 'max_depth': 5, 'min_samples_split': 8, 'min_samples_leaf': 4, 'class_weight': 'balanced'}
                                
                            tr = MultiModelTrainer.train_model(m_type, splits, params)
                            probs = tr.model.predict_proba(splits.X_test[tr.feature_names])[:, 1]
                            s_prob = pd.Series(probs, index=splits.X_test.index)
                            
                            # Test top quantiles: 80th, 85th, 90th, 95th
                            for q in [0.75, 0.80, 0.85, 0.90]:
                                thresh = s_prob.quantile(q)
                                
                                for require_trend in [True, False]:
                                    raw_feats = splits.X_test
                                    signals = []
                                    for ts, p in s_prob.items():
                                        act = SignalAction.HOLD
                                        if p >= thresh:
                                            if require_trend:
                                                t_score = raw_feats.loc[ts].get('regime_trend_score', 0)
                                                dist_50 = raw_feats.loc[ts].get('dist_ema_50', 0)
                                                if t_score >= 0 or dist_50 >= 0:
                                                    act = SignalAction.BUY
                                            else:
                                                act = SignalAction.BUY
                                        signals.append(TradingSignal(timestamp=ts, symbol=symbol, action=act, probability=p, reason="prob"))
                                        
                                    test_candles = df.loc[splits.X_test.index]
                                    for sl in [0.010, 0.015, 0.020]:
                                        for tp in [0.020, 0.030, 0.040]:
                                            if tp <= sl:
                                                continue
                                            engine = BacktestEngine(BacktestConfig(
                                                initial_capital=10000.0,
                                                commission_rate=0.001,
                                                slippage_rate=0.0005,
                                                stop_loss_pct=sl,
                                                take_profit_pct=tp,
                                                max_holding_bars=horizon,
                                            ))
                                            res = engine.run(test_candles, signals, symbol, tf)
                                            m = res.metrics
                                            
                                            # Record if trades >= 15 and profitable
                                            if m['total_trades'] >= 15:
                                                rec = {
                                                    'symbol': symbol,
                                                    'tf': tf,
                                                    'horizon': horizon,
                                                    'label_m': label_m,
                                                    'tgt_ret': tgt_ret,
                                                    'm_type': m_type,
                                                    'q': q,
                                                    'require_trend': require_trend,
                                                    'sl': sl,
                                                    'tp': tp,
                                                    'trades': m['total_trades'],
                                                    'win_rate': m['win_rate_pct'],
                                                    'profit_factor': m['profit_factor'],
                                                    'return_pct': m['total_return_pct'],
                                                    'sharpe': m['sharpe_ratio'],
                                                    'mdd': m['max_drawdown_pct'],
                                                }
                                                results.append(rec)
                                                if m['total_return_pct'] > 0 and m['profit_factor'] >= 1.2:
                                                    print(f"  [+] WINNER FOUND: {symbol} [{tf}] {m_type} {label_m} h={horizon} q={q:.2f} SL={sl*100:.1f}% TP={tp*100:.1f}% -> Trades: {m['total_trades']:3d} | WR: {m['win_rate_pct']:4.1f}% | PF: {m['profit_factor']:4.2f} | Return: {m['total_return_pct']:+5.2f}% | Sharpe: {m['sharpe_ratio']:4.2f} | MDD: {m['max_drawdown_pct']:4.2f}%")
                    except Exception as e:
                        pass

    # Show top 15 overall strategies
    df_res = pd.DataFrame(results)
    if not df_res.empty:
        df_res = df_res.sort_values(by=['return_pct', 'profit_factor', 'trades'], ascending=False)
        print("\n" + "=" * 80)
        print("TOP 15 PROFITABLE HIGH-TRADE VOLUME STRATEGIES:")
        print("=" * 80)
        print(df_res.head(15).to_string(index=False))
        df_res.to_csv('reports/strategy_exploration_results.csv', index=False)

if __name__ == '__main__':
    asyncio.run(main())
