"""Multi-Model Training Engine supporting XGBoost, Random Forest, HistGradientBoosting, ExtraTrees, and Ensembles."""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Union
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
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
import xgboost as xgb

from app.data.preprocessing.dataset import DatasetSplits
from app.models.training.trainer import TrainingResult

logger = logging.getLogger(__name__)


class SoftVotingEnsemble:
    """Combines predictions from multiple fitted probabilistic base classifiers with normalized weights."""

    def __init__(self, models: List[Any], weights: Optional[List[float]] = None, model_names: Optional[List[str]] = None):
        self.models = models
        self.model_names = model_names or [getattr(m, "__class__", type(m)).__name__ for m in models]
        if weights is None or len(weights) != len(models):
            self.weights = [1.0 / len(models)] * len(models)
        else:
            w_sum = sum(weights) if sum(weights) > 0 else 1.0
            self.weights = [w / w_sum for w in weights]

    def predict_proba(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """Calculate weighted average predicted probability across all ensemble members."""
        X_arr = X.values if hasattr(X, "values") else np.array(X)
        total_proba = np.zeros((len(X_arr), 2), dtype=np.float64)

        for model, weight in zip(self.models, self.weights):
            if hasattr(model, "predict_proba"):
                p = model.predict_proba(X_arr)
            else:
                pred = model.predict(X_arr)
                p = np.column_stack([1.0 - pred, pred])
            total_proba += weight * p

        return total_proba

    def predict(self, X: Union[np.ndarray, pd.DataFrame], threshold: float = 0.50) -> np.ndarray:
        """Predict binary class labels based on ensemble probability and threshold."""
        probas = self.predict_proba(X)
        return (probas[:, 1] >= threshold).astype(int)


class MultiModelTrainer:
    """Unified training and evaluation pipeline for diverse Machine Learning model families."""

    SUPPORTED_MODELS = [
        "xgboost_classifier",
        "random_forest",
        "hist_gradient_boosting",
        "extra_trees",
        "ensemble_voting",
    ]

    @staticmethod
    def evaluate_predictions(
        y_true: pd.Series,
        y_pred: np.ndarray,
        y_proba: np.ndarray,
    ) -> Dict[str, Any]:
        """Compute comprehensive statistical classification metrics."""
        y_true_arr = y_true.values if hasattr(y_true, "values") else np.array(y_true)
        unique_classes = np.unique(y_true_arr)

        acc = float(accuracy_score(y_true_arr, y_pred))
        prec = float(precision_score(y_true_arr, y_pred, zero_division=0))
        rec = float(recall_score(y_true_arr, y_pred, zero_division=0))
        f1 = float(f1_score(y_true_arr, y_pred, zero_division=0))

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
    def train_model(
        cls,
        model_type: str,
        splits: DatasetSplits,
        hyperparameters: Optional[Dict[str, Any]] = None,
        feature_subset: Optional[List[str]] = None,
    ) -> TrainingResult:
        """Train and evaluate any supported model family on chronological splits.

        Args:
            model_type: Model identifier ('xgboost_classifier', 'random_forest', 'hist_gradient_boosting', 'extra_trees').
            splits: Chronological DatasetSplits.
            hyperparameters: Optional dictionary of hyperparameters.
            feature_subset: Optional subset of features to train on.

        Returns:
            TrainingResult container.
        """
        params = hyperparameters.copy() if hyperparameters else {}
        feature_cols = feature_subset or list(splits.X_train.columns)

        X_tr = splits.X_train[feature_cols]
        y_tr = splits.y_train
        X_va = splits.X_val[feature_cols]
        y_va = splits.y_val
        X_te = splits.X_test[feature_cols]
        y_te = splits.y_test

        pos_count = int((y_tr == 1).sum())
        neg_count = int((y_tr == 0).sum())
        scale_pos = (neg_count / max(pos_count, 1)) if pos_count > 0 else 1.0

        best_iteration = 0
        feature_importances: Dict[str, float] = {}

        if model_type == "xgboost_classifier":
            default_xgb = {
                "n_estimators": 120,
                "max_depth": 3,
                "learning_rate": 0.03,
                "subsample": 0.75,
                "colsample_bytree": 0.75,
                "min_child_weight": 5,
                "reg_alpha": 1.0,
                "reg_lambda": 5.0,
                "random_state": 42,
                "n_jobs": -1,
                "eval_metric": "logloss",
                "early_stopping_rounds": 25,
            }
            default_xgb.update(params)
            if "scale_pos_weight" not in default_xgb:
                default_xgb["scale_pos_weight"] = float(scale_pos)

            model = xgb.XGBClassifier(**default_xgb)
            model.fit(
                X_tr,
                y_tr,
                eval_set=[(X_tr, y_tr), (X_va, y_va)],
                verbose=False,
            )
            best_iteration = int(getattr(model, "best_iteration", default_xgb["n_estimators"] - 1))

            booster = model.get_booster()
            score_gain = booster.get_score(importance_type="gain")
            total_gain = sum(score_gain.values()) if score_gain else 1.0
            for feat in feature_cols:
                feature_importances[feat] = float(score_gain.get(feat, 0.0)) / max(total_gain, 1e-6)

        elif model_type == "random_forest":
            default_rf = {
                "n_estimators": 150,
                "max_depth": 5,
                "min_samples_split": 10,
                "min_samples_leaf": 5,
                "max_features": "sqrt",
                "class_weight": "balanced",
                "random_state": 42,
                "n_jobs": -1,
            }
            default_rf.update(params)
            model = RandomForestClassifier(**default_rf)
            model.fit(X_tr, y_tr)
            best_iteration = default_rf["n_estimators"]

            importances = model.feature_importances_
            for feat, imp in zip(feature_cols, importances):
                feature_importances[feat] = float(imp)

        elif model_type == "hist_gradient_boosting":
            default_hgb = {
                "max_iter": 120,
                "max_depth": 4,
                "learning_rate": 0.03,
                "min_samples_leaf": 15,
                "l2_regularization": 2.0,
                "random_state": 42,
                "early_stopping": True,
                "n_iter_no_change": 15,
            }
            default_hgb.update(params)
            model = HistGradientBoostingClassifier(**default_hgb)
            model.fit(X_tr, y_tr)
            best_iteration = int(getattr(model, "n_iter_", default_hgb["max_iter"]))
            feature_importances = {f: 1.0 / len(feature_cols) for f in feature_cols}

        elif model_type == "extra_trees":
            default_et = {
                "n_estimators": 150,
                "max_depth": 5,
                "min_samples_split": 8,
                "min_samples_leaf": 4,
                "max_features": "sqrt",
                "class_weight": "balanced",
                "random_state": 42,
                "n_jobs": -1,
            }
            default_et.update(params)
            model = ExtraTreesClassifier(**default_et)
            model.fit(X_tr, y_tr)
            best_iteration = default_et["n_estimators"]

            importances = model.feature_importances_
            for feat, imp in zip(feature_cols, importances):
                feature_importances[feat] = float(imp)

        else:
            raise ValueError(f"Unsupported model_type: '{model_type}'. Choose from {cls.SUPPORTED_MODELS}")

        # Evaluate on all 3 sets
        tr_prob = model.predict_proba(X_tr)[:, 1]
        tr_pred = (tr_prob >= 0.50).astype(int)
        tr_metrics = cls.evaluate_predictions(y_tr, tr_pred, tr_prob)

        va_prob = model.predict_proba(X_va)[:, 1]
        va_pred = (va_prob >= 0.50).astype(int)
        va_metrics = cls.evaluate_predictions(y_va, va_pred, va_prob)

        te_prob = model.predict_proba(X_te)[:, 1]
        te_pred = (te_prob >= 0.50).astype(int)
        te_metrics = cls.evaluate_predictions(y_te, te_pred, te_prob)

        return TrainingResult(
            model=model,
            train_metrics=tr_metrics,
            validation_metrics=va_metrics,
            test_metrics=te_metrics,
            feature_importances=feature_importances,
            hyperparameters=params,
            best_iteration=best_iteration,
            dataset_hash=splits.dataset.dataset_hash,
            feature_names=feature_cols,
            start_date=splits.dataset.start_date,
            end_date=splits.dataset.end_date,
            model_type=model_type,
        )

    @classmethod
    def train_ensemble(
        cls,
        splits: DatasetSplits,
        base_configs: List[Tuple[str, Dict[str, Any]]],
        weights: Optional[List[float]] = None,
        feature_subset: Optional[List[str]] = None,
    ) -> TrainingResult:
        """Train multiple distinct base learners and combine into a SoftVotingEnsemble.

        Args:
            splits: Chronological DatasetSplits.
            base_configs: List of (model_type, hyperparameters) tuples.
            weights: Optional explicit weights for each base model.
            feature_subset: Optional feature columns to use.

        Returns:
            TrainingResult containing SoftVotingEnsemble and aggregated metrics.
        """
        fitted_models = []
        model_names = []
        val_f1_scores = []
        feature_cols = feature_subset or list(splits.X_train.columns)

        for m_type, m_params in base_configs:
            res = cls.train_model(m_type, splits, hyperparameters=m_params, feature_subset=feature_cols)
            fitted_models.append(res.model)
            model_names.append(m_type)
            val_f1_scores.append(res.validation_metrics["f1"])

        # Auto-compute weights based on Validation F1 if not supplied
        if weights is None:
            weights = [max(f1, 0.05) for f1 in val_f1_scores]

        ensemble = SoftVotingEnsemble(models=fitted_models, weights=weights, model_names=model_names)

        X_tr = splits.X_train[feature_cols]
        y_tr = splits.y_train
        X_va = splits.X_val[feature_cols]
        y_va = splits.y_val
        X_te = splits.X_test[feature_cols]
        y_te = splits.y_test

        tr_prob = ensemble.predict_proba(X_tr)[:, 1]
        tr_pred = (tr_prob >= 0.50).astype(int)
        tr_metrics = cls.evaluate_predictions(y_tr, tr_pred, tr_prob)

        va_prob = ensemble.predict_proba(X_va)[:, 1]
        va_pred = (va_prob >= 0.50).astype(int)
        va_metrics = cls.evaluate_predictions(y_va, va_pred, va_prob)

        te_prob = ensemble.predict_proba(X_te)[:, 1]
        te_pred = (te_prob >= 0.50).astype(int)
        te_metrics = cls.evaluate_predictions(y_te, te_pred, te_prob)

        # Combine feature importances across all base models
        avg_importances: Dict[str, float] = {f: 0.0 for f in feature_cols}
        for m, w in zip(fitted_models, ensemble.weights):
            if hasattr(m, "feature_importances_"):
                for f, imp in zip(feature_cols, m.feature_importances_):
                    avg_importances[f] += w * float(imp)
            elif hasattr(m, "get_booster"):
                scores = m.get_booster().get_score(importance_type="gain")
                total = sum(scores.values()) if scores else 1.0
                for f in feature_cols:
                    avg_importances[f] += w * (float(scores.get(f, 0.0)) / max(total, 1e-6))
            else:
                for f in feature_cols:
                    avg_importances[f] += w * (1.0 / len(feature_cols))

        return TrainingResult(
            model=ensemble,
            train_metrics=tr_metrics,
            validation_metrics=va_metrics,
            test_metrics=te_metrics,
            feature_importances=avg_importances,
            hyperparameters={"base_models": base_configs, "weights": ensemble.weights},
            best_iteration=len(fitted_models),
            dataset_hash=splits.dataset.dataset_hash,
            feature_names=feature_cols,
            start_date=splits.dataset.start_date,
            end_date=splits.dataset.end_date,
            model_type="ensemble_voting",
        )
