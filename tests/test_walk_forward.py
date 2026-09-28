"""Tests for WalkForwardValidator, StatisticalComparator, and multi-period temporal stability."""

import numpy as np
import pandas as pd
import pytest

from app.data.preprocessing.dataset import DatasetBuilder
from app.models.training.statistical_test import StatisticalComparator
from app.models.training.walk_forward import WalkForwardReport, WalkForwardValidator


def generate_synthetic_candles_for_walk_forward(n_bars: int = 240, seed: int = 77) -> pd.DataFrame:
    """Generate synthetic price series with alternating upward and consolidation regimes."""
    np.random.seed(seed)
    returns = np.random.normal(loc=0.0004, scale=0.008, size=n_bars)
    price_path = 60000.0 * np.cumprod(1 + returns)

    dates = pd.date_range("2026-01-01", periods=n_bars, freq="15min", tz="UTC")

    opens = price_path
    closes = price_path * (1 + np.random.normal(0, 0.001, size=n_bars))
    highs = np.maximum(opens, closes) * 1.002
    lows = np.minimum(opens, closes) * 0.998
    volumes = np.random.uniform(20.0, 80.0, size=n_bars)

    return pd.DataFrame(
        {
            "open": opens,
            "high": highs,
            "low": lows,
            "close": closes,
            "volume": volumes,
        },
        index=dates,
    )


# ------------------------------------------------------------------------------
# 1. StatisticalComparator Tests
# ------------------------------------------------------------------------------

def test_statistical_comparator_information_coefficient():
    """Verify Information Coefficient calculates Spearman correlation."""
    probas = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
    returns = np.array([0.001, 0.003, 0.005, 0.008, 0.012])  # perfectly monotonic

    ic_dict = StatisticalComparator.compute_information_coefficient(probas, returns)
    assert np.isclose(ic_dict["ic"], 1.0)
    assert ic_dict["ic_pvalue"] < 0.05


def test_statistical_comparator_brier_skill_score():
    """Verify Brier Skill Score calculation."""
    y_true = np.array([0, 0, 1, 1])
    # Perfect predictions -> BSS = 1.0
    perfect_proba = np.array([0.0, 0.0, 1.0, 1.0])
    bss_perfect = StatisticalComparator.compute_brier_skill_score(y_true, perfect_proba)
    assert np.isclose(bss_perfect, 1.0)

    # Base rate predictions -> BSS = 0.0
    base_proba = np.array([0.5, 0.5, 0.5, 0.5])
    bss_base = StatisticalComparator.compute_brier_skill_score(y_true, base_proba)
    assert np.isclose(bss_base, 0.0, atol=1e-3)


def test_statistical_comparator_permutation_test():
    """Verify Monte Carlo permutation test for accuracy."""
    y_true = np.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0])
    y_pred = np.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0])  # 100% accuracy

    perm_res = StatisticalComparator.permutation_test_accuracy(
        y_true=y_true, y_pred=y_pred, n_permutations=200, seed=42
    )

    assert perm_res["observed_accuracy"] == 1.0
    assert perm_res["p_value"] < 0.05  # Statistically significant


# ------------------------------------------------------------------------------
# 2. WalkForwardValidator Tests
# ------------------------------------------------------------------------------

def test_walk_forward_validator_execution():
    """Verify WalkForwardValidator runs multi-fold validation and produces full report."""
    df = generate_synthetic_candles_for_walk_forward(200)
    dataset = DatasetBuilder.build(
        candles_df=df,
        symbol="BTC/USDT",
        timeframe="15m",
        horizon=6,
        target_return=0.003,
        label_method="peak",
        drop_na=True,
    )

    hyperparams = {
        "n_estimators": 25,
        "max_depth": 3,
        "learning_rate": 0.05,
    }

    report = WalkForwardValidator.run_validation(
        dataset=dataset,
        n_splits=3,
        expanding=True,
        embargo_bars=6,
        hyperparameters=hyperparams,
        early_stopping_rounds=5,
    )

    assert isinstance(report, WalkForwardReport)
    assert report.n_folds == 3
    assert len(report.folds) == 3
    assert report.total_oos_samples > 0

    # Verify fold metrics
    for f in report.folds:
        assert "accuracy" in f.out_of_sample_metrics
        assert "f1" in f.out_of_sample_metrics
        assert "roc_auc" in f.out_of_sample_metrics
        assert f.train_samples > 0
        assert f.test_samples > 0

    # Verify aggregated metrics
    assert "mean_accuracy" in report.aggregate_metrics
    assert "mean_f1" in report.aggregate_metrics
    assert "std_f1" in report.aggregate_metrics

    # Verify overfitting diagnostics
    assert "f1_degradation_ratio" in report.overfitting_analysis
    assert "overfitting_risk_level" in report.overfitting_analysis

    # Verify summary text generation
    summary_text = report.summary()
    assert "WALK-FORWARD TEMPORAL VALIDATION REPORT" in summary_text
    assert "FOLD-BY-FOLD BREAKDOWN" in summary_text
    assert "STATISTICAL ADVANTAGE TESTS" in summary_text
