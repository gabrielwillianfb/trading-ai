"""Supervised model training pipeline using XGBoost with early stopping and comprehensive metrics."""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)

from app.data.preprocessing.dataset import DatasetSplits

logger = logging.getLogger(__name__)


@dataclass
class TrainingResult:
    """Comprehensive training output containing fitted model, metrics, and metadata."""

    model: xgb.XGBClassifier
    train_metrics: Dict[str, Any]
    validation_metrics: Dict[str, Any]
    test_metrics: Dict[str, Any]
    feature_importances: Dict[str, float]
    hyperparameters: Dict[str, Any]
    best_iteration: int
    dataset_hash: str
    feature_names: List[str]
    start_date: datetime
    end_date: datetime
    model_type: str = "xgboost_classifier"

    def summary(self) -> str:
        """Format human-readable training report."""
        lines = [
            "=" * 60,
            f"TRAINING REPORT | Type: {self.model_type} | Best Iteration: {self.best_iteration}",
            f"Date Range: {self.start_date} -> {self.end_date}",
            f"Dataset Hash: {self.dataset_hash[:16]}...",
            "-" * 60,
            f"TRAIN METRICS:      Accuracy={self.train_metrics['accuracy']:.4f}, "
            f"Precision={self.train_metrics['precision']:.4f}, Recall={self.train_metrics['recall']:.4f}, "
            f"F1={self.train_metrics['f1']:.4f}, ROC-AUC={self.train_metrics.get('roc_auc', 0.0):.4f}",
            f"VALIDATION METRICS: Accuracy={self.validation_metrics['accuracy']:.4f}, "
            f"Precision={self.validation_metrics['precision']:.4f}, Recall={self.validation_metrics['recall']:.4f}, "
            f"F1={self.validation_metrics['f1']:.4f}, ROC-AUC={self.validation_metrics.get('roc_auc', 0.0):.4f}",
            f"TEST METRICS (OOS): Accuracy={self.test_metrics['accuracy']:.4f}, "
            f"Precision={self.test_metrics['precision']:.4f}, Recall={self.test_metrics['recall']:.4f}, "
            f"F1={self.test_metrics['f1']:.4f}, ROC-AUC={self.test_metrics.get('roc_auc', 0.0):.4f}",
            "-" * 60,
            "TOP 5 FEATURE IMPORTANCES (Gain):",
        ]
        sorted_fi = sorted(self.feature_importances.items(), key=lambda x: x[1], reverse=True)[:5]
        for name, score in sorted_fi:
            lines.append(f"  - {name:<18}: {score:.5f}")
        lines.append("=" * 60)
        return "\n".join(lines)


class ModelTrainer:
    """Trains and evaluates XGBoost classification models with anti-overfitting controls."""

    DEFAULT_HYPERPARAMETERS: Dict[str, Any] = {
        "n_estimators": 300,
        "max_depth": 4,
        "learning_rate": 0.03,
        "subsample": 0.80,
        "colsample_bytree": 0.80,
        "min_child_weight": 3,
        "gamma": 0.10,
        "eval_metric": "logloss",
        "random_state": 42,
        "n_jobs": -1,
    }

    @staticmethod
    def evaluate_predictions(
        y_true: pd.Series,
        y_pred: np.ndarray,
        y_proba: np.ndarray,
    ) -> Dict[str, Any]:
        """Compute statistical classification performance metrics."""
        y_true_arr = y_true.values if hasattr(y_true, "values") else np.array(y_true)

        # Unique classes present in y_true
        unique_classes = np.unique(y_true_arr)

        acc = float(accuracy_score(y_true_arr, y_pred))
        prec = float(precision_score(y_true_arr, y_pred, zero_division=0))
        rec = float(recall_score(y_true_arr, y_pred, zero_division=0))
        f1 = float(f1_score(y_true_arr, y_pred, zero_division=0))

        # ROC-AUC requires at least 2 classes in y_true
        if len(unique_classes) >= 2:
            roc_auc = float(roc_auc_score(y_true_arr, y_proba))
            loss = float(log_loss(y_true_arr, y_proba, labels=[0, 1]))
        else:
            roc_auc = 0.5
            loss = 0.0

        brier = float(brier_score_loss(y_true_arr, y_proba))
        cm = confusion_matrix(y_true_arr, y_pred, labels=[0, 1]).tolist()

        return {
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "roc_auc": roc_auc,
            "log_loss": loss,
            "brier_score": brier,
            "confusion_matrix": cm,
            "samples_count": len(y_true_arr),
            "positive_count": int(np.sum(y_true_arr == 1)),
        }

    @classmethod
    def train(
        cls,
        splits: DatasetSplits,
        hyperparameters: Optional[Dict[str, Any]] = None,
        early_stopping_rounds: int = 30,
    ) -> TrainingResult:
        """Train XGBoost classifier on train split, tune on validation split, and score on test split.

        Args:
            splits: DatasetSplits containing X_train, y_train, X_val, y_val, X_test, y_test.
            hyperparameters: Optional dictionary overriding default hyperparameters.
            early_stopping_rounds: Early stopping patience on validation loss.

        Returns:
            TrainingResult instance with trained model and metrics.
        """
        params = cls.DEFAULT_HYPERPARAMETERS.copy()
        if hyperparameters:
            params.update(hyperparameters)

        # Handle class imbalance weighting if not explicitly set
        pos_count = (splits.y_train == 1).sum()
        neg_count = (splits.y_train == 0).sum()
        if "scale_pos_weight" not in params and pos_count > 0:
            scale_pos = neg_count / pos_count
            params["scale_pos_weight"] = float(scale_pos)

        # Early stopping configuration in modern XGBoost
        params["early_stopping_rounds"] = early_stopping_rounds

        logger.info(
            f"Training XGBoost on {len(splits.X_train)} samples with {splits.X_train.shape[1]} features..."
        )

        model = xgb.XGBClassifier(**params)
        model.fit(
            splits.X_train,
            splits.y_train,
            eval_set=[(splits.X_train, splits.y_train), (splits.X_val, splits.y_val)],
            verbose=False,
        )

        best_iteration = getattr(model, "best_iteration", params["n_estimators"] - 1)
        logger.info(f"Model converged at best iteration: {best_iteration}")

        # Predictions on Train
        train_proba = model.predict_proba(splits.X_train)[:, 1]
        train_pred = (train_proba >= 0.50).astype(int)
        train_metrics = cls.evaluate_predictions(splits.y_train, train_pred, train_proba)

        # Predictions on Validation
        val_proba = model.predict_proba(splits.X_val)[:, 1]
        val_pred = (val_proba >= 0.50).astype(int)
        val_metrics = cls.evaluate_predictions(splits.y_val, val_pred, val_proba)

        # Predictions on Out-of-Sample Test
        test_proba = model.predict_proba(splits.X_test)[:, 1]
        test_pred = (test_proba >= 0.50).astype(int)
        test_metrics = cls.evaluate_predictions(splits.y_test, test_pred, test_proba)

        # Feature importances (gain-based)
        booster = model.get_booster()
        score_gain = booster.get_score(importance_type="gain")
        feature_names = list(splits.X_train.columns)

        feature_importances: Dict[str, float] = {}
        for feat in feature_names:
            feature_importances[feat] = float(score_gain.get(feat, 0.0))

        # Normalize importances
        total_gain = sum(feature_importances.values())
        if total_gain > 0:
            feature_importances = {k: v / total_gain for k, v in feature_importances.items()}

        return TrainingResult(
            model=model,
            train_metrics=train_metrics,
            validation_metrics=val_metrics,
            test_metrics=test_metrics,
            feature_importances=feature_importances,
            hyperparameters=params,
            best_iteration=int(best_iteration),
            dataset_hash=splits.dataset.dataset_hash,
            feature_names=feature_names,
            start_date=splits.dataset.start_date,
            end_date=splits.dataset.end_date,
        )
