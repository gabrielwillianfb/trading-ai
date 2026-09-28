"""Tests for MultiModelTrainer, Dynamic ATR Labeling, SoftVotingEnsemble, and AutonomousOptimizer."""

from datetime import datetime, timezone
import numpy as np
import pandas as pd
import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from app.backtesting.signals import SignalAction, SignalGenerator
from app.data.preprocessing.dataset import DatasetBuilder
from app.data.preprocessing.labeling import LabelGenerator
from app.features.pipeline import FeaturePipeline
from app.models.inference.predictor import ModelPredictor, PredictionResult
from app.models.registry.registry import ModelRegistry
from app.models.training.autonomous_optimizer import AutonomousOptimizer
from app.models.training.multi_model_trainer import MultiModelTrainer, SoftVotingEnsemble


@pytest.fixture
def sample_ohlcv_dataframe() -> pd.DataFrame:
    """Generate 150 synthetic bars with realistic trend and volatility."""
    dates = pd.date_range("2026-01-01", periods=150, freq="15min", tz="UTC")
    np.random.seed(42)
    returns = np.random.normal(0.0005, 0.005, 150)
    prices = 50000.0 * np.cumprod(1.0 + returns)

    high = prices * (1.0 + np.abs(np.random.normal(0.002, 0.001, 150)))
    low = prices * (1.0 - np.abs(np.random.normal(0.002, 0.001, 150)))
    open_p = (prices + low) / 2.0
    volume = np.random.uniform(10.0, 50.0, 150)

    return pd.DataFrame(
        {
            "open": open_p,
            "high": high,
            "low": low,
            "close": prices,
            "volume": volume,
        },
        index=dates,
    )


def test_dynamic_atr_triple_barrier_labeling(sample_ohlcv_dataframe):
    """Verify dynamic ATR triple-barrier labeling generates valid binary outputs with zero look-ahead at the tail."""
    df = sample_ohlcv_dataframe
    from app.features.indicators import compute_atr

    atr = compute_atr(df["high"], df["low"], df["close"], period=14)
    labels = LabelGenerator.compute_dynamic_atr_triple_barrier_labels(
        high=df["high"],
        low=df["low"],
        close=df["close"],
        atr=atr,
        horizon=10,
        tp_atr_mult=2.0,
        sl_atr_mult=1.0,
    )
    assert len(labels) == len(df)
    assert labels.iloc[-10:].isna().all()
    valid_labels = labels.dropna()
    assert set(valid_labels.unique()).issubset({0.0, 1.0})
    assert len(valid_labels) > 50


def test_multi_model_trainer_all_architectures(sample_ohlcv_dataframe):
    """Verify training of XGBoost, Random Forest, HistGradientBoosting, ExtraTrees, and SoftVotingEnsemble."""
    df = sample_ohlcv_dataframe
    dataset = DatasetBuilder.build(
        candles_df=df,
        symbol="BTC/USDT",
        timeframe="15m",
        horizon=8,
        label_method="dynamic_atr_tb",
        tp_atr_mult=2.0,
        sl_atr_mult=1.0,
        drop_na=True,
    )
    splits = DatasetBuilder.split_dataset(dataset, train_pct=0.60, val_pct=0.20, test_pct=0.20)

    # 1. Test XGBoost
    xgb_res = MultiModelTrainer.train_model(
        "xgboost_classifier",
        splits,
        hyperparameters={"n_estimators": 20, "max_depth": 2, "early_stopping_rounds": 5},
    )
    assert xgb_res.model_type == "xgboost_classifier"
    assert "accuracy" in xgb_res.test_metrics

    # 2. Test Random Forest
    rf_res = MultiModelTrainer.train_model(
        "random_forest",
        splits,
        hyperparameters={"n_estimators": 20, "max_depth": 3},
    )
    assert rf_res.model_type == "random_forest"
    assert "accuracy" in rf_res.test_metrics

    # 3. Test HistGradientBoosting
    hgb_res = MultiModelTrainer.train_model(
        "hist_gradient_boosting",
        splits,
        hyperparameters={"max_iter": 20, "max_depth": 3},
    )
    assert hgb_res.model_type == "hist_gradient_boosting"

    # 4. Test Extra Trees
    et_res = MultiModelTrainer.train_model(
        "extra_trees",
        splits,
        hyperparameters={"n_estimators": 20, "max_depth": 3},
    )
    assert et_res.model_type == "extra_trees"

    # 5. Test Ensemble
    ens_res = MultiModelTrainer.train_ensemble(
        splits,
        base_configs=[
            ("xgboost_classifier", {"n_estimators": 15, "max_depth": 2, "early_stopping_rounds": 5}),
            ("random_forest", {"n_estimators": 15, "max_depth": 3}),
        ],
    )
    assert ens_res.model_type == "ensemble_voting"
    assert isinstance(ens_res.model, SoftVotingEnsemble)
    assert len(ens_res.model.models) == 2


@pytest.mark.asyncio
async def test_model_registry_universal_serialization(db_session: AsyncSession, sample_ohlcv_dataframe, tmp_path):
    """Verify ModelRegistry correctly registers, serializes, and loads Random Forest via joblib."""
    df = sample_ohlcv_dataframe
    dataset = DatasetBuilder.build(candles_df=df, horizon=8, drop_na=True)
    splits = DatasetBuilder.split_dataset(dataset)

    rf_res = MultiModelTrainer.train_model("random_forest", splits, hyperparameters={"n_estimators": 15, "max_depth": 3})
    model_dir = str(tmp_path / "models")

    model_record = await ModelRegistry.register_model(
        training_result=rf_res,
        session=db_session,
        set_active=True,
        model_dir=model_dir,
    )
    assert model_record.model_type == "random_forest"
    assert model_record.file_path.endswith(".joblib")
    assert model_record.is_active is True

    # Load artifact back
    loaded_model = ModelRegistry.load_model_artifact(model_record)
    assert hasattr(loaded_model, "predict_proba")
    test_X = splits.X_test.iloc[:5]
    probas = loaded_model.predict_proba(test_X)
    assert probas.shape == (5, 2)


def test_signal_generator_with_trend_and_chop_filters():
    """Verify SignalGenerator enforces macro trend and choppiness filters."""
    ts = datetime.now(timezone.utc)
    pred_up_clean = PredictionResult(
        timestamp=ts,
        symbol="BTC/USDT",
        timeframe="15m",
        probability_up=0.68,
        probability_down=0.32,
        predicted_class=1,
        model_version="test_v1",
        raw_features={"regime_trend_score": 1.0, "dist_ema_50": 0.01, "chop_index_14": 40.0, "rsi_14": 55.0},
    )

    # Case 1: All filters pass -> BUY
    sig1 = SignalGenerator.generate_signal_from_prediction(
        pred_up_clean,
        buy_threshold=0.60,
        require_trend_filter=True,
        require_chop_filter=True,
        max_entry_rsi=70.0,
    )
    assert sig1.action == SignalAction.BUY

    # Case 2: Market is choppy -> HOLD
    pred_up_choppy = PredictionResult(
        timestamp=ts,
        symbol="BTC/USDT",
        timeframe="15m",
        probability_up=0.68,
        probability_down=0.32,
        predicted_class=1,
        model_version="test_v1",
        raw_features={"regime_trend_score": 1.0, "dist_ema_50": 0.01, "chop_index_14": 72.0, "rsi_14": 55.0},
    )
    sig2 = SignalGenerator.generate_signal_from_prediction(
        pred_up_choppy,
        buy_threshold=0.60,
        require_chop_filter=True,
    )
    assert sig2.action == SignalAction.HOLD
    assert "Chop filter" in sig2.reason


def test_autonomous_optimizer_mini_loop(sample_ohlcv_dataframe, tmp_path):
    """Verify AutonomousOptimizer runs search across configurations and produces ranked Leaderboard."""
    df = sample_ohlcv_dataframe
    lb_path = str(tmp_path / "leaderboard.json")

    report = AutonomousOptimizer.run_optimization(
        candles_df=df,
        symbol="BTC/USDT",
        timeframe="15m",
        max_trials=8,
        leaderboard_path=lb_path,
    )
    assert report.total_trials_evaluated > 0
    assert report.best_trial is not None
    assert len(report.leaderboard) > 0
    assert report.best_trial.composite_score >= report.leaderboard[-1].composite_score
