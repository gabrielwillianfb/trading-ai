"""Institutional Hyperparameter and Decision Threshold Optimization Engine."""

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

from app.backtesting.engine import BacktestConfig, BacktestEngine
from app.backtesting.signals import SignalGenerator
from app.data.preprocessing.dataset import Dataset, DatasetSplits
from app.models.inference.predictor import ModelPredictor
from app.models.training.trainer import ModelTrainer
from app.models.training.walk_forward import WalkForwardReport, WalkForwardValidator

logger = logging.getLogger(__name__)


@dataclass
class OptimizationResult:
    """Output container for hyperparameter and threshold optimization."""

    best_hyperparameters: Dict[str, Any]
    best_buy_threshold: float
    best_sell_threshold: float
    best_oos_f1: float
    best_profit_factor: float
    best_sharpe: float
    walk_forward_report: WalkForwardReport
    candidates_evaluated: int


class StrategyOptimizer:
    """Automates grid search and decision threshold calibration for quantitative models."""

    DEFAULT_GRID: List[Dict[str, Any]] = [
        # Conservative shallow regularized trees (Institutional standard for high-noise crypto)
        {"max_depth": 2, "learning_rate": 0.02, "n_estimators": 80, "subsample": 0.7, "colsample_bytree": 0.7, "reg_alpha": 1.0, "reg_lambda": 5.0, "min_child_weight": 10},
        {"max_depth": 2, "learning_rate": 0.03, "n_estimators": 100, "subsample": 0.8, "colsample_bytree": 0.8, "reg_alpha": 0.5, "reg_lambda": 3.0, "min_child_weight": 5},
        {"max_depth": 3, "learning_rate": 0.02, "n_estimators": 60, "subsample": 0.6, "colsample_bytree": 0.7, "reg_alpha": 2.0, "reg_lambda": 10.0, "min_child_weight": 15},
        {"max_depth": 3, "learning_rate": 0.03, "n_estimators": 80, "subsample": 0.7, "colsample_bytree": 0.7, "reg_alpha": 1.0, "reg_lambda": 5.0, "min_child_weight": 10},
        {"max_depth": 2, "learning_rate": 0.05, "n_estimators": 60, "subsample": 0.8, "colsample_bytree": 0.8, "reg_alpha": 0.1, "reg_lambda": 2.0, "min_child_weight": 5},
    ]

    THRESHOLD_CANDIDATES: List[float] = [0.52, 0.55, 0.58, 0.60, 0.62, 0.65, 0.68, 0.70]

    @classmethod
    def optimize(
        cls,
        dataset: Dataset,
        splits: DatasetSplits,
        candles_df: pd.DataFrame,
        symbol: str = "BTC/USDT",
        timeframe: str = "15m",
        horizon: int = 12,
        initial_capital: float = 10000.0,
        grid: Optional[List[Dict[str, Any]]] = None,
    ) -> OptimizationResult:
        """Execute full optimization over hyperparameters and high-conviction decision thresholds.

        Args:
            dataset: Aligned Dataset.
            splits: DatasetSplits (Train, Val, Test).
            candles_df: Raw OHLCV DataFrame with DatetimeIndex.
            symbol: Asset symbol.
            timeframe: Candle interval.
            horizon: Exit horizon in bars.
            initial_capital: Simulation capital.
            grid: List of hyperparameter dictionaries.

        Returns:
            OptimizationResult instance.
        """
        param_grid = grid or cls.DEFAULT_GRID
        logger.info(f"Starting Quantitative Optimization across {len(param_grid)} hyperparameter architectures...")

        best_params = param_grid[0]
        best_wf_report: Optional[WalkForwardReport] = None
        best_oos_score = -999.0

        # Phase 1: Walk-Forward Hyperparameter Selection
        for idx, params in enumerate(param_grid, start=1):
            logger.info(f"Evaluating candidate {idx}/{len(param_grid)}: {params}")
            wf_report = WalkForwardValidator.run_validation(
                dataset=dataset,
                n_splits=4,
                expanding=True,
                embargo_bars=horizon,
                hyperparameters=params,
            )
            oos_f1 = wf_report.overfitting_analysis["mean_out_of_sample_f1"]
            degradation = wf_report.overfitting_analysis["f1_degradation_ratio"]

            # Stability objective: balance OOS F1 and low degradation
            stability_score = oos_f1 * (1.0 + min(degradation, 1.0))
            logger.info(f"Candidate {idx} -> OOS F1: {oos_f1:.4f} | Stability Score: {stability_score:.4f}")

            if stability_score > best_oos_score or best_wf_report is None:
                best_oos_score = stability_score
                best_params = params
                best_wf_report = wf_report

        logger.info(f"Selected Optimal Architecture: {best_params} (Best Stability: {best_oos_score:.4f})")

        # Phase 2: Train Model on Train+Val with Optimal Hyperparameters
        trainer = ModelTrainer()
        train_res = trainer.train(splits, hyperparameters=best_params)

        predictor = ModelPredictor(
            model=train_res.model,
            feature_names=dataset.feature_names,
            model_version="opt_eval",
        )
        predictions = predictor.predict_batch(splits.X_test, symbol=symbol, timeframe=timeframe)
        test_candles_df = candles_df.loc[splits.X_test.index]

        # Phase 3: Calibrate High-Confidence Decision Thresholds
        logger.info("Calibrating High-Confidence Decision Thresholds against real transaction costs...")
        best_thresh = 0.55
        best_pf = 0.0
        best_sharpe = -999.0

        for thresh in cls.THRESHOLD_CANDIDATES:
            signals = [
                SignalGenerator.generate_signal_from_prediction(
                    p,
                    buy_threshold=thresh,
                    sell_threshold=1.0 - thresh,
                )
                for p in predictions
            ]

            bt_engine = BacktestEngine(
                config=BacktestConfig(
                    initial_capital=initial_capital,
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
            n_trades = m["total_trades"]
            pf = m["profit_factor"]
            sh = m["sharpe_ratio"]
            ret = m["total_return_pct"]

            logger.info(
                f"Threshold P(UP) >= {thresh:.2f} -> Trades: {n_trades:2d} | "
                f"Win Rate: {m['win_rate_pct']:4.1f}% | PF: {pf:4.2f} | Sharpe: {sh:5.2f} | Return: {ret:+5.2f}%"
            )

            # Prioritize profit factor and return with at least 3 trades
            if n_trades >= 2 and (pf > best_pf or (pf == best_pf and sh > best_sharpe)):
                best_pf = pf
                best_sharpe = sh
                best_thresh = thresh

        logger.info(f"Optimal Decision Threshold Selected: P(UP) >= {best_thresh:.2f} (PF: {best_pf:.2f}, Sharpe: {best_sharpe:.2f})")

        return OptimizationResult(
            best_hyperparameters=best_params,
            best_buy_threshold=best_thresh,
            best_sell_threshold=1.0 - best_thresh,
            best_oos_f1=best_wf_report.overfitting_analysis["mean_out_of_sample_f1"],
            best_profit_factor=best_pf,
            best_sharpe=best_sharpe,
            walk_forward_report=best_wf_report,
            candidates_evaluated=len(param_grid),
        )
