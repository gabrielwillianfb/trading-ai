"""Walk-Forward validation engine for chronological multi-period cross-validation and overfitting diagnosis."""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd
import xgboost as xgb

from app.data.preprocessing.dataset import Dataset
from app.data.preprocessing.splitters import WalkForwardSplitter
from app.models.training.statistical_test import StatisticalComparator
from app.models.training.trainer import ModelTrainer

logger = logging.getLogger(__name__)


@dataclass
class WalkForwardFoldResult:
    """Evaluation output for a single chronological walk-forward fold."""

    fold_index: int
    train_start: datetime
    train_end: datetime
    test_start: datetime
    test_end: datetime
    in_sample_metrics: Dict[str, Any]
    out_of_sample_metrics: Dict[str, Any]
    best_iteration: int
    train_samples: int
    test_samples: int


@dataclass
class WalkForwardReport:
    """Aggregated walk-forward validation and statistical advantage report."""

    symbol: str
    timeframe: str
    n_folds: int
    total_oos_samples: int
    folds: List[WalkForwardFoldResult]
    aggregate_metrics: Dict[str, float]
    overfitting_analysis: Dict[str, Any]
    statistical_test: Dict[str, Any]

    def summary(self) -> str:
        """Render comprehensive human-readable walk-forward diagnostic report."""
        lines = [
            "=" * 70,
            f"WALK-FORWARD TEMPORAL VALIDATION REPORT: {self.symbol} [{self.timeframe}]",
            f"Total Folds: {self.n_folds} | Total Out-of-Sample (OOS) Samples: {self.total_oos_samples}",
            "=" * 70,
            "FOLD-BY-FOLD BREAKDOWN:",
            "-" * 70,
            f"{'Fold':<6} | {'Train Range':<32} | {'Test Range':<32} | {'OOS Acc':<8} | {'OOS F1':<8} | {'OOS AUC':<8}",
            "-" * 70,
        ]

        for f in self.folds:
            tr_str = f"{f.train_start.strftime('%m-%d %H:%M')}->{f.train_end.strftime('%m-%d %H:%M')}"
            te_str = f"{f.test_start.strftime('%m-%d %H:%M')}->{f.test_end.strftime('%m-%d %H:%M')}"
            m = f.out_of_sample_metrics
            lines.append(
                f"#{f.fold_index:<5} | {tr_str:<32} | {te_str:<32} | "
                f"{m['accuracy']:.4f}   | {m['f1']:.4f}   | {m.get('roc_auc', 0.0):.4f}"
            )

        lines.extend(
            [
                "-" * 70,
                "AGGREGATED OUT-OF-SAMPLE (OOS) METRICS (Mean +/- Std):",
                f"  - Accuracy:  {self.aggregate_metrics['mean_accuracy']:.4f} +/- {self.aggregate_metrics['std_accuracy']:.4f}",
                f"  - Precision: {self.aggregate_metrics['mean_precision']:.4f} +/- {self.aggregate_metrics['std_precision']:.4f}",
                f"  - Recall:    {self.aggregate_metrics['mean_recall']:.4f} +/- {self.aggregate_metrics['std_recall']:.4f}",
                f"  - F1-Score:  {self.aggregate_metrics['mean_f1']:.4f} +/- {self.aggregate_metrics['std_f1']:.4f}",
                f"  - ROC-AUC:   {self.aggregate_metrics['mean_roc_auc']:.4f} +/- {self.aggregate_metrics['std_roc_auc']:.4f}",
                "-" * 70,
                "OVERFITTING & STABILITY DIAGNOSTICS:",
                f"  - Mean In-Sample F1:       {self.overfitting_analysis['mean_in_sample_f1']:.4f}",
                f"  - Mean Out-of-Sample F1:   {self.overfitting_analysis['mean_out_of_sample_f1']:.4f}",
                f"  - F1 Degradation Ratio:    {self.overfitting_analysis['f1_degradation_ratio']:.2f} "
                f"({self.overfitting_analysis['f1_degradation_ratio']*100:.1f}% of train performance retained)",
                f"  - Overfitting Risk:        {self.overfitting_analysis['overfitting_risk_level']}",
                "-" * 70,
                "STATISTICAL ADVANTAGE TESTS (vs Random Baseline):",
                f"  - Monte Carlo Permutation p-value: {self.statistical_test['permutation_test']['p_value']:.5f}",
                f"  - Brier Skill Score (BSS):         {self.statistical_test['brier_skill_score']:.4f}",
                f"  - Information Coefficient (IC):    {self.statistical_test['information_coefficient']:.4f} "
                f"(p-val: {self.statistical_test['ic_pvalue']:.4f})",
                f"  - Statistical Advantage Confirmed: {self.statistical_test['has_statistical_advantage']}",
                "=" * 70,
            ]
        )
        return "\n".join(lines)


class WalkForwardValidator:
    """Executes multi-fold walk-forward validation and computes statistical advantage."""

    @classmethod
    def run_validation(
        cls,
        dataset: Dataset,
        n_splits: int = 4,
        expanding: bool = True,
        embargo_bars: int = 12,
        hyperparameters: Optional[Dict[str, Any]] = None,
        early_stopping_rounds: int = 25,
    ) -> WalkForwardReport:
        """Run complete chronological walk-forward cross-validation on a dataset.

        Args:
            dataset: Aligned Dataset instance.
            n_splits: Number of walk-forward folds.
            expanding: Whether to use expanding or rolling window.
            embargo_bars: Embargo gap between train and test.
            hyperparameters: Optional dictionary of XGBoost parameters.
            early_stopping_rounds: Early stopping patience.

        Returns:
            WalkForwardReport instance.
        """
        logger.info(
            f"Initiating Walk-Forward Validation ({n_splits} splits, embargo={embargo_bars} bars, expanding={expanding})..."
        )

        folds_def = WalkForwardSplitter.generate_folds(
            df=dataset.X,
            n_splits=n_splits,
            expanding=expanding,
            embargo_bars=embargo_bars,
        )

        fold_results: List[WalkForwardFoldResult] = []
        all_oos_y_true: List[int] = []
        all_oos_y_pred: List[int] = []
        all_oos_y_proba: List[float] = []

        params = ModelTrainer.DEFAULT_HYPERPARAMETERS.copy()
        if hyperparameters:
            params.update(hyperparameters)
        params["early_stopping_rounds"] = early_stopping_rounds

        for fold in folds_def:
            X_train_full = dataset.X.iloc[fold.train_indices]
            y_train_full = dataset.y.iloc[fold.train_indices]

            X_test = dataset.X.iloc[fold.test_indices]
            y_test = dataset.y.iloc[fold.test_indices]

            # Internal train/validation split (80% / 20%) for early stopping
            n_train = len(X_train_full)
            split_point = int(n_train * 0.80)
            X_tr, y_tr = X_train_full.iloc[:split_point], y_train_full.iloc[:split_point]
            X_va, y_va = X_train_full.iloc[split_point:], y_train_full.iloc[split_point:]

            # Handle class weighting
            pos_c = (y_tr == 1).sum()
            neg_c = (y_tr == 0).sum()
            fold_params = params.copy()
            if "scale_pos_weight" not in fold_params and pos_c > 0:
                fold_params["scale_pos_weight"] = float(neg_c / pos_c)

            # Fit model
            model = xgb.XGBClassifier(**fold_params)
            model.fit(
                X_tr,
                y_tr,
                eval_set=[(X_tr, y_tr), (X_va, y_va)],
                verbose=False,
            )
            best_iter = getattr(model, "best_iteration", fold_params["n_estimators"] - 1)

            # In-Sample Metrics (evaluated on full fold train set)
            is_proba = model.predict_proba(X_train_full)[:, 1]
            is_pred = (is_proba >= 0.50).astype(int)
            is_metrics = ModelTrainer.evaluate_predictions(y_train_full, is_pred, is_proba)

            # Out-of-Sample Metrics (evaluated on unseen future fold test set)
            oos_proba = model.predict_proba(X_test)[:, 1]
            oos_pred = (oos_proba >= 0.50).astype(int)
            oos_metrics = ModelTrainer.evaluate_predictions(y_test, oos_pred, oos_proba)

            # Collect global OOS predictions
            all_oos_y_true.extend(y_test.values.tolist())
            all_oos_y_pred.extend(oos_pred.tolist())
            all_oos_y_proba.extend(oos_proba.tolist())

            fold_res = WalkForwardFoldResult(
                fold_index=fold.fold_index,
                train_start=fold.train_start,
                train_end=fold.train_end,
                test_start=fold.test_start,
                test_end=fold.test_end,
                in_sample_metrics=is_metrics,
                out_of_sample_metrics=oos_metrics,
                best_iteration=int(best_iter),
                train_samples=len(X_train_full),
                test_samples=len(X_test),
            )
            fold_results.append(fold_res)

        # Compute aggregate metrics across folds
        oos_accs = [f.out_of_sample_metrics["accuracy"] for f in fold_results]
        oos_precs = [f.out_of_sample_metrics["precision"] for f in fold_results]
        oos_recs = [f.out_of_sample_metrics["recall"] for f in fold_results]
        oos_f1s = [f.out_of_sample_metrics["f1"] for f in fold_results]
        oos_aucs = [f.out_of_sample_metrics.get("roc_auc", 0.5) for f in fold_results]

        is_f1s = [f.in_sample_metrics["f1"] for f in fold_results]

        mean_is_f1 = float(np.mean(is_f1s)) if is_f1s else 0.0
        mean_oos_f1 = float(np.mean(oos_f1s)) if oos_f1s else 0.0

        f1_deg_ratio = mean_oos_f1 / max(mean_is_f1, 1e-6)
        if f1_deg_ratio >= 0.70:
            overfit_risk = "LOW"
        elif f1_deg_ratio >= 0.40:
            overfit_risk = "MODERATE"
        else:
            overfit_risk = "HIGH (Severe In-Sample Overfitting)"

        agg_metrics = {
            "mean_accuracy": float(np.mean(oos_accs)),
            "std_accuracy": float(np.std(oos_accs)),
            "mean_precision": float(np.mean(oos_precs)),
            "std_precision": float(np.std(oos_precs)),
            "mean_recall": float(np.mean(oos_recs)),
            "std_recall": float(np.std(oos_recs)),
            "mean_f1": float(np.mean(oos_f1s)),
            "std_f1": float(np.std(oos_f1s)),
            "mean_roc_auc": float(np.mean(oos_aucs)),
            "std_roc_auc": float(np.std(oos_aucs)),
        }

        overfitting_diag = {
            "mean_in_sample_f1": mean_is_f1,
            "mean_out_of_sample_f1": mean_oos_f1,
            "f1_degradation_ratio": float(f1_deg_ratio),
            "overfitting_risk_level": overfit_risk,
        }

        # Statistical significance test across all concatenated OOS predictions
        stat_test = StatisticalComparator.evaluate_statistical_advantage(
            y_true=np.array(all_oos_y_true),
            y_pred=np.array(all_oos_y_pred),
            y_proba=np.array(all_oos_y_proba),
        )

        return WalkForwardReport(
            symbol=dataset.symbol,
            timeframe=dataset.timeframe,
            n_folds=len(fold_results),
            total_oos_samples=len(all_oos_y_true),
            folds=fold_results,
            aggregate_metrics=agg_metrics,
            overfitting_analysis=overfitting_diag,
            statistical_test=stat_test,
        )
