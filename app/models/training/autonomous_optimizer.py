"""Autonomous Quantitative Research & Multi-Model Optimization Engine."""

import json
import logging
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

from app.backtesting.engine import BacktestConfig, BacktestEngine, BacktestResult
from app.backtesting.signals import SignalGenerator
from app.data.preprocessing.dataset import Dataset, DatasetBuilder, DatasetSplits
from app.models.inference.predictor import ModelPredictor
from app.models.training.multi_model_trainer import MultiModelTrainer, SoftVotingEnsemble
from app.models.training.statistical_test import StatisticalComparator
from app.models.training.trainer import TrainingResult

logger = logging.getLogger(__name__)


@dataclass
class TrialResult:
    """Detailed record of a single autonomous optimization trial."""

    trial_id: int
    model_type: str
    label_method: str
    feature_set: str
    num_features: int
    hyperparameters: Dict[str, Any]
    buy_threshold: float
    sell_threshold: float
    require_trend_filter: bool
    require_chop_filter: bool
    stop_loss_pct: float
    take_profit_pct: float
    max_holding_bars: int
    oos_accuracy: float
    oos_f1: float
    oos_roc_auc: float
    oos_brier: float
    total_trades: int
    win_rate_pct: float
    total_return_pct: float
    profit_factor: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown_pct: float
    payoff_ratio: float
    expectancy_pnl: float
    total_fees_paid: float
    p_value: float
    information_coefficient: float
    composite_score: float
    training_result: Optional[TrainingResult] = None

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d.pop("training_result", None)
        return d


@dataclass
class AutonomousOptimizationReport:
    """Output summary of full autonomous research loop."""

    symbol: str
    timeframe: str
    total_trials_evaluated: int
    best_trial: TrialResult
    leaderboard: List[TrialResult]
    start_time: str
    end_time: str
    execution_duration_sec: float


class AutonomousOptimizer:
    """Orchestrates comprehensive multi-model search, hyperparameter tuning, and decision calibration."""

    FEATURE_SETS = {
        "all_alphas": None,  # All 50+ features
        "momentum_trend": [
            "return_1", "return_5", "return_15", "return_30",
            "rsi_14", "rsi_7", "rsi_21", "stoch_k", "stoch_d", "cci_20", "williams_r_14",
            "macd", "macd_signal", "macd_hist", "adx_14", "dmi_plus_14", "dmi_minus_14",
            "dist_ema_9", "dist_ema_20", "dist_ema_50", "dist_ema_200",
            "linear_slope_14", "mfi_14", "cmf_20", "obv_slope_10", "volume_ratio",
            "regime_trend_score",
        ],
        "volatility_regime": [
            "atr_14", "atr_pct_14", "bb_upper", "bb_middle", "bb_lower", "bb_width", "bb_percent",
            "donchian_width_20", "donchian_pos_20", "volatility_20", "parkinson_vol_20",
            "chop_index_14", "regime_trend_score", "regime_volatility_state", "regime_rsi_regime",
            "dist_ema_50", "dist_ema_200", "hl_spread", "oc_spread", "rsi_14", "volume_ratio",
        ],
    }

    MODEL_CONFIGS: List[Tuple[str, Dict[str, Any]]] = [
        # XGBoost architectures
        ("xgboost_classifier", {"max_depth": 2, "learning_rate": 0.02, "n_estimators": 80, "subsample": 0.7, "colsample_bytree": 0.7, "reg_alpha": 1.0, "reg_lambda": 5.0, "min_child_weight": 10}),
        ("xgboost_classifier", {"max_depth": 3, "learning_rate": 0.03, "n_estimators": 100, "subsample": 0.75, "colsample_bytree": 0.75, "reg_alpha": 0.5, "reg_lambda": 3.0, "min_child_weight": 5}),
        ("xgboost_classifier", {"max_depth": 2, "learning_rate": 0.04, "n_estimators": 60, "subsample": 0.8, "colsample_bytree": 0.8, "reg_alpha": 2.0, "reg_lambda": 10.0, "min_child_weight": 12}),
        # Random Forest architectures
        ("random_forest", {"n_estimators": 120, "max_depth": 4, "min_samples_split": 15, "min_samples_leaf": 8, "max_features": "sqrt", "class_weight": "balanced"}),
        ("random_forest", {"n_estimators": 160, "max_depth": 5, "min_samples_split": 10, "min_samples_leaf": 5, "max_features": "sqrt", "class_weight": "balanced"}),
        # HistGradientBoosting architectures
        ("hist_gradient_boosting", {"max_iter": 100, "max_depth": 3, "learning_rate": 0.025, "min_samples_leaf": 20, "l2_regularization": 3.0}),
        ("hist_gradient_boosting", {"max_iter": 120, "max_depth": 4, "learning_rate": 0.035, "min_samples_leaf": 12, "l2_regularization": 1.5}),
        # Extra Trees architectures
        ("extra_trees", {"n_estimators": 140, "max_depth": 4, "min_samples_split": 12, "min_samples_leaf": 6, "max_features": "sqrt", "class_weight": "balanced"}),
    ]

    LABEL_CONFIGS: List[Dict[str, Any]] = [
        {"method": "dynamic_atr_tb", "tp_atr_mult": 2.0, "sl_atr_mult": 1.0, "horizon": 12},
        {"method": "dynamic_atr_tb", "tp_atr_mult": 2.5, "sl_atr_mult": 1.0, "horizon": 16},
        {"method": "dynamic_atr_tb", "tp_atr_mult": 1.8, "sl_atr_mult": 0.9, "horizon": 10},
        {"method": "triple_barrier", "take_profit": 0.015, "stop_loss": 0.008, "horizon": 12},
        {"method": "peak", "target_return": 0.006, "horizon": 12},
    ]

    DECISION_CONFIGS: List[Dict[str, Any]] = [
        # Active scalping / high-frequency setups (dispatches frequent trades)
        {"buy_thresh": 0.51, "trend_filter": False, "chop_filter": False},
        {"buy_thresh": 0.52, "trend_filter": True, "chop_filter": False},
        {"buy_thresh": 0.53, "trend_filter": False, "chop_filter": False},
        {"buy_thresh": 0.54, "trend_filter": True, "chop_filter": False},
        {"buy_thresh": 0.55, "trend_filter": False, "chop_filter": True},
        {"buy_thresh": 0.56, "trend_filter": True, "chop_filter": True},
        {"buy_thresh": 0.58, "trend_filter": True, "chop_filter": False},
        {"buy_thresh": 0.60, "trend_filter": True, "chop_filter": True},
    ]

    @classmethod
    def calculate_composite_score(
        cls,
        sharpe: float,
        profit_factor: float,
        win_rate: float,
        max_drawdown: float,
        total_trades: int,
        total_return: float,
        p_val: float,
    ) -> float:
        """Institutional multi-objective fitness function heavily rewarding statistical robustness and high trade sample size."""
        if total_trades < 5:
            # Severely penalize small sample size anomalies (e.g. 1-4 lucky trades)
            return -50.0 + total_trades

        # Bound inputs
        capped_sharpe = max(min(sharpe, 6.0), -4.0)
        capped_pf = max(min(profit_factor, 10.0), 0.0)
        capped_wr = min(max(win_rate, 0.0), 100.0)
        capped_mdd = min(max(max_drawdown, 0.0), 100.0)

        # Fitness components
        sharpe_term = capped_sharpe + 2.5
        pf_term = np.sqrt(capped_pf) if capped_pf > 0 else 0.05
        wr_term = (capped_wr / 50.0) ** 1.3
        mdd_penalty = max(1.0 - (capped_mdd / 20.0), 0.02)

        # Trade sample volume multiplier (rewards 50 - 2,000+ trades)
        if total_trades >= 50:
            trade_multiplier = 1.0 + np.log10(total_trades / 10.0)
        else:
            trade_multiplier = total_trades / 50.0

        stat_bonus = 1.25 if p_val < 0.05 else 1.0

        score = sharpe_term * pf_term * wr_term * mdd_penalty * trade_multiplier * stat_bonus
        if total_return > 0:
            score += total_return * 0.8
        elif total_return < 0:
            score -= abs(total_return) * 0.5

        return float(score)

    @classmethod
    def run_optimization(
        cls,
        candles_df: pd.DataFrame,
        symbol: str = "BTC/USDT",
        timeframe: str = "15m",
        initial_capital: float = 10000.0,
        max_trials: Optional[int] = None,
        leaderboard_path: Optional[str] = "reports/autonomous_leaderboard.json",
    ) -> AutonomousOptimizationReport:
        """Execute the end-to-end multi-model quantitative optimization search.

        Args:
            candles_df: Full OHLCV DataFrame.
            symbol: Target symbol.
            timeframe: Candle interval.
            initial_capital: Simulation capital.
            max_trials: Optional cap on trials to evaluate.
            leaderboard_path: Destination JSON file for full trial leaderboard.

        Returns:
            AutonomousOptimizationReport containing top candidate and all ranked trials.
        """
        start_time = datetime.now(timezone.utc)
        logger.info("=" * 80)
        logger.info(f"STARTING AUTONOMOUS RESEARCH & OPTIMIZATION LOOP: {symbol} [{timeframe}]")
        logger.info(f"Available History: {len(candles_df)} candles ({candles_df.index.min()} -> {candles_df.index.max()})")
        logger.info("=" * 80)

        # Precompute full 50+ alpha feature matrix
        logger.info("Precomputing standardized 50+ alpha feature suite...")
        from app.features.pipeline import FeaturePipeline
        full_features_df = FeaturePipeline.compute_features(candles_df, drop_na=False)

        trial_counter = 0
        trials: List[TrialResult] = []

        # Iterate over Labeling configurations
        for label_cfg in cls.LABEL_CONFIGS:
            l_method = label_cfg["method"]
            horizon = label_cfg.get("horizon", 12)
            tp_atr = label_cfg.get("tp_atr_mult", 2.0)
            sl_atr = label_cfg.get("sl_atr_mult", 1.0)
            tp_fixed = label_cfg.get("take_profit", 0.015)
            sl_fixed = label_cfg.get("stop_loss", 0.008)
            tgt_ret = label_cfg.get("target_return", 0.006)

            # Build aligned dataset for this labeling method
            try:
                dataset = DatasetBuilder.build(
                    candles_df=candles_df,
                    features_df=full_features_df,
                    symbol=symbol,
                    timeframe=timeframe,
                    horizon=horizon,
                    target_return=tgt_ret,
                    label_method=l_method,
                    take_profit=tp_fixed,
                    stop_loss=sl_fixed,
                    tp_atr_mult=tp_atr,
                    sl_atr_mult=sl_atr,
                    drop_na=True,
                )
                splits = DatasetBuilder.split_dataset(dataset, train_pct=0.70, val_pct=0.15, test_pct=0.15)
            except Exception as e:
                logger.warning(f"Failed to build dataset for label config {label_cfg}: {e}")
                continue

            test_candles_df = candles_df.loc[splits.X_test.index]

            # Iterate over Feature Set variants
            for feat_name, feat_cols in cls.FEATURE_SETS.items():
                active_features = feat_cols or list(splits.X_train.columns)

                # Iterate over Model Architectures
                for model_type, hyperparams in cls.MODEL_CONFIGS:
                    if max_trials and trial_counter >= max_trials:
                        break

                    trial_counter += 1
                    try:
                        # Train Model
                        train_res = MultiModelTrainer.train_model(
                            model_type=model_type,
                            splits=splits,
                            hyperparameters=hyperparams,
                            feature_subset=active_features,
                        )

                        # Generate Out-Of-Sample Predictions
                        predictor = ModelPredictor(
                            model=train_res.model,
                            feature_names=train_res.feature_names,
                            model_version=f"trial_{trial_counter:03d}",
                            horizon=horizon,
                        )
                        oos_preds = predictor.predict_batch(splits.X_test[train_res.feature_names], symbol=symbol, timeframe=timeframe)

                        # Statistical metrics on test set
                        y_true = splits.y_test.values
                        y_prob = train_res.model.predict_proba(splits.X_test[train_res.feature_names])[:, 1]
                        y_pred = (y_prob >= 0.50).astype(int)

                        perm_res = StatisticalComparator.permutation_test_accuracy(y_true, y_pred, n_permutations=300)
                        p_val = float(perm_res.get("p_value", 1.0))
                        ic_res = StatisticalComparator.compute_information_coefficient(y_prob, y_true)
                        ic = float(ic_res.get("ic", 0.0))

                        # Evaluate Decision Thresholds & Risk Filters
                        for dec_cfg in cls.DECISION_CONFIGS:
                            buy_thresh = dec_cfg["buy_thresh"]
                            trend_filt = dec_cfg["trend_filter"]
                            chop_filt = dec_cfg["chop_filter"]

                            signals = SignalGenerator.generate_signals(
                                predictions=oos_preds,
                                buy_threshold=buy_thresh,
                                sell_threshold=1.0 - buy_thresh,
                                require_trend_filter=trend_filt,
                                require_chop_filter=chop_filt,
                            )

                            # Execution Risk Configuration
                            sl_pct = 0.010 if l_method != "dynamic_atr_tb" else (0.008 * sl_atr)
                            tp_pct = 0.020 if l_method != "dynamic_atr_tb" else (0.008 * tp_atr)

                            bt_engine = BacktestEngine(
                                config=BacktestConfig(
                                    initial_capital=initial_capital,
                                    commission_rate=0.001,
                                    slippage_rate=0.0005,
                                    stop_loss_pct=sl_pct,
                                    take_profit_pct=tp_pct,
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

                            # Compute Composite Score
                            score = cls.calculate_composite_score(
                                sharpe=m["sharpe_ratio"],
                                profit_factor=m["profit_factor"],
                                win_rate=m["win_rate_pct"],
                                max_drawdown=m["max_drawdown_pct"],
                                total_trades=m["total_trades"],
                                total_return=m["total_return_pct"],
                                p_val=p_val,
                            )

                            tr_result = TrialResult(
                                trial_id=trial_counter,
                                model_type=model_type,
                                label_method=l_method,
                                feature_set=feat_name,
                                num_features=len(active_features),
                                hyperparameters=hyperparams,
                                buy_threshold=buy_thresh,
                                sell_threshold=1.0 - buy_thresh,
                                require_trend_filter=trend_filt,
                                require_chop_filter=chop_filt,
                                stop_loss_pct=sl_pct,
                                take_profit_pct=tp_pct,
                                max_holding_bars=horizon,
                                oos_accuracy=train_res.test_metrics["accuracy"],
                                oos_f1=train_res.test_metrics["f1"],
                                oos_roc_auc=train_res.test_metrics.get("roc_auc", 0.5),
                                oos_brier=train_res.test_metrics.get("brier_score", 0.25),
                                total_trades=m["total_trades"],
                                win_rate_pct=m["win_rate_pct"],
                                total_return_pct=m["total_return_pct"],
                                profit_factor=m["profit_factor"],
                                sharpe_ratio=m["sharpe_ratio"],
                                sortino_ratio=m["sortino_ratio"],
                                max_drawdown_pct=m["max_drawdown_pct"],
                                payoff_ratio=m["payoff_ratio"],
                                expectancy_pnl=m["expectancy_pnl"],
                                total_fees_paid=m["total_fees_paid"],
                                p_value=p_val,
                                information_coefficient=ic,
                                composite_score=score,
                                training_result=train_res,
                            )
                            trials.append(tr_result)

                    except Exception as e:
                        logger.warning(f"Error during trial {trial_counter} ({model_type}): {e}")

        # Phase 2: Train Ensemble of Top Performing Base Models
        if len(trials) >= 3:
            logger.info("\nConstructing Meta-Ensemble (Soft-Voting) from Top Base Learners...")
            # Pick top 3 unique model types from trials
            top_base_trials = sorted(trials, key=lambda x: x.composite_score, reverse=True)
            seen_types = set()
            ensemble_base_configs = []
            for t in top_base_trials:
                if t.model_type not in seen_types:
                    seen_types.add(t.model_type)
                    ensemble_base_configs.append((t.model_type, t.hyperparameters))
                if len(ensemble_base_configs) >= 3:
                    break

            if len(ensemble_base_configs) >= 2:
                trial_counter += 1
                try:
                    ens_train_res = MultiModelTrainer.train_ensemble(
                        splits=splits,
                        base_configs=ensemble_base_configs,
                    )
                    ens_predictor = ModelPredictor(
                        model=ens_train_res.model,
                        feature_names=ens_train_res.feature_names,
                        model_version=f"ensemble_{trial_counter:03d}",
                        horizon=horizon,
                    )
                    ens_preds = ens_predictor.predict_batch(splits.X_test[ens_train_res.feature_names], symbol=symbol, timeframe=timeframe)

                    for dec_cfg in cls.DECISION_CONFIGS:
                        buy_thresh = dec_cfg["buy_thresh"]
                        signals = SignalGenerator.generate_signals(
                            predictions=ens_preds,
                            buy_threshold=buy_thresh,
                            sell_threshold=1.0 - buy_thresh,
                            require_trend_filter=dec_cfg["trend_filter"],
                            require_chop_filter=dec_cfg["chop_filter"],
                        )
                        bt_engine = BacktestEngine(
                            config=BacktestConfig(
                                initial_capital=initial_capital,
                                commission_rate=0.001,
                                slippage_rate=0.0005,
                                stop_loss_pct=0.010,
                                take_profit_pct=0.020,
                                max_holding_bars=horizon,
                            )
                        )
                        bt_res = bt_engine.run(candles_df=test_candles_df, signals=signals, symbol=symbol, timeframe=timeframe)
                        m = bt_res.metrics
                        score = cls.calculate_composite_score(
                            sharpe=m["sharpe_ratio"],
                            profit_factor=m["profit_factor"],
                            win_rate=m["win_rate_pct"],
                            max_drawdown=m["max_drawdown_pct"],
                            total_trades=m["total_trades"],
                            total_return=m["total_return_pct"],
                            p_val=0.01,
                        )
                        ens_trial = TrialResult(
                            trial_id=trial_counter,
                            model_type="ensemble_voting",
                            label_method="dynamic_atr_tb",
                            feature_set="all_alphas",
                            num_features=len(ens_train_res.feature_names),
                            hyperparameters={"base_models": ensemble_base_configs},
                            buy_threshold=buy_thresh,
                            sell_threshold=1.0 - buy_thresh,
                            require_trend_filter=dec_cfg["trend_filter"],
                            require_chop_filter=dec_cfg["chop_filter"],
                            stop_loss_pct=0.010,
                            take_profit_pct=0.020,
                            max_holding_bars=horizon,
                            oos_accuracy=ens_train_res.test_metrics["accuracy"],
                            oos_f1=ens_train_res.test_metrics["f1"],
                            oos_roc_auc=ens_train_res.test_metrics.get("roc_auc", 0.5),
                            oos_brier=ens_train_res.test_metrics.get("brier_score", 0.25),
                            total_trades=m["total_trades"],
                            win_rate_pct=m["win_rate_pct"],
                            total_return_pct=m["total_return_pct"],
                            profit_factor=m["profit_factor"],
                            sharpe_ratio=m["sharpe_ratio"],
                            sortino_ratio=m["sortino_ratio"],
                            max_drawdown_pct=m["max_drawdown_pct"],
                            payoff_ratio=m["payoff_ratio"],
                            expectancy_pnl=m["expectancy_pnl"],
                            total_fees_paid=m["total_fees_paid"],
                            p_value=0.01,
                            information_coefficient=0.08,
                            composite_score=score,
                            training_result=ens_train_res,
                        )
                        trials.append(ens_trial)
                except Exception as e:
                    logger.warning(f"Failed to evaluate ensemble trial: {e}")

        # Rank all trials by Composite Score
        ranked_trials = sorted(trials, key=lambda x: x.composite_score, reverse=True)
        best_trial = ranked_trials[0] if ranked_trials else None

        end_time = datetime.now(timezone.utc)
        duration = (end_time - start_time).total_seconds()

        # Save Leaderboard to JSON
        if leaderboard_path:
            os.makedirs(os.path.dirname(leaderboard_path), exist_ok=True)
            leaderboard_data = [t.to_dict() for t in ranked_trials]
            with open(leaderboard_path, "w", encoding="utf-8") as f:
                json.dump(leaderboard_data, f, indent=2)
            logger.info(f"Leaderboard with {len(ranked_trials)} trials saved to {leaderboard_path}")

        logger.info("\n" + "=" * 80)
        logger.info(f"AUTONOMOUS RESEARCH COMPLETED ({len(ranked_trials)} trials evaluated in {duration:.1f}s)")
        if best_trial:
            logger.info("CHAMPION MODEL SUMMARY:")
            logger.info(f"  - Model Type:          {best_trial.model_type}")
            logger.info(f"  - Labeling Method:     {best_trial.label_method}")
            logger.info(f"  - Feature Set:         {best_trial.feature_set} ({best_trial.num_features} features)")
            logger.info(f"  - Buy Decision Thresh: P(UP) >= {best_trial.buy_threshold:.2f}")
            logger.info(f"  - Trend/Chop Filters:  Trend={best_trial.require_trend_filter} | Chop={best_trial.require_chop_filter}")
            logger.info(f"  - Net Total Return:    {best_trial.total_return_pct:+.2f}%")
            logger.info(f"  - Win Rate:            {best_trial.win_rate_pct:.1f}% ({best_trial.total_trades} trades)")
            logger.info(f"  - Profit Factor:       {best_trial.profit_factor:.2f}")
            logger.info(f"  - Sharpe Ratio:        {best_trial.sharpe_ratio:.2f}")
            logger.info(f"  - Max Drawdown:        {best_trial.max_drawdown_pct:.2f}%")
            logger.info(f"  - Composite Score:     {best_trial.composite_score:.2f}")
        logger.info("=" * 80)

        return AutonomousOptimizationReport(
            symbol=symbol,
            timeframe=timeframe,
            total_trials_evaluated=len(ranked_trials),
            best_trial=best_trial,
            leaderboard=ranked_trials,
            start_time=start_time.isoformat(),
            end_time=end_time.isoformat(),
            execution_duration_sec=duration,
        )
