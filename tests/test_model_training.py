"""Tests for ModelTrainer, ModelRegistry, ModelPredictor, and ML evaluation metrics."""

import os
from datetime import datetime, timezone
import numpy as np
import pandas as pd
import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from app.data.preprocessing.dataset import DatasetBuilder
from app.models.inference.predictor import ModelPredictor, PredictionResult
from app.models.registry.registry import ModelRegistry
from app.models.training.trainer import ModelTrainer, TrainingResult


def generate_synthetic_candles_for_training(n_bars: int = 200, seed: int = 42) -> pd.DataFrame:
    """Generate deterministic synthetic market candles for model testing."""
    np.random.seed(seed)
    returns = np.random.normal(loc=0.0005, scale=0.01, size=n_bars)
    price_path = 50000.0 * np.cumprod(1 + returns)

    dates = pd.date_range("2026-01-01", periods=n_bars, freq="15min", tz="UTC")

    opens = price_path
    closes = price_path * (1 + np.random.normal(0, 0.002, size=n_bars))
    highs = np.maximum(opens, closes) * 1.003
    lows = np.minimum(opens, closes) * 0.997
    volumes = np.random.uniform(10.0, 100.0, size=n_bars)

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
# 1. ModelTrainer Tests
# ------------------------------------------------------------------------------

def test_model_trainer_pipeline():
    """Verify ModelTrainer trains XGBoost and generates comprehensive evaluation metrics."""
    df = generate_synthetic_candles_for_training(180)
    dataset = DatasetBuilder.build(
        candles_df=df,
        symbol="BTC/USDT",
        timeframe="15m",
        horizon=6,
        target_return=0.003,
        label_method="peak",
        drop_na=True,
    )
    splits = DatasetBuilder.split_dataset(dataset, train_pct=0.70, val_pct=0.15, test_pct=0.15)

    hyperparams = {
        "n_estimators": 50,
        "max_depth": 3,
        "learning_rate": 0.05,
    }
    result = ModelTrainer.train(splits, hyperparameters=hyperparams, early_stopping_rounds=10)

    assert isinstance(result, TrainingResult)
    assert result.model is not None
    assert result.best_iteration >= 0

    # Verify metrics structure
    for m in [result.train_metrics, result.validation_metrics, result.test_metrics]:
        assert "accuracy" in m
        assert "precision" in m
        assert "recall" in m
        assert "f1" in m
        assert "roc_auc" in m
        assert "log_loss" in m
        assert "brier_score" in m
        assert "confusion_matrix" in m
        assert 0.0 <= m["accuracy"] <= 1.0

    # Verify feature importances
    assert len(result.feature_importances) == len(splits.X_train.columns)
    assert np.isclose(sum(result.feature_importances.values()), 1.0, atol=1e-3)


# ------------------------------------------------------------------------------
# 2. ModelRegistry Tests
# ------------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_model_registry_save_load_and_activation(
    db_session: AsyncSession, tmp_path
):
    """Verify ModelRegistry serialization, versioning, database record, and activation."""
    df = generate_synthetic_candles_for_training(160)
    dataset = DatasetBuilder.build(
        candles_df=df,
        symbol="ETH/USDT",
        timeframe="15m",
        horizon=6,
        target_return=0.003,
        drop_na=True,
    )
    splits = DatasetBuilder.split_dataset(dataset)
    training_result = ModelTrainer.train(
        splits,
        hyperparameters={"n_estimators": 20, "max_depth": 3},
        early_stopping_rounds=5,
    )

    model_dir = str(tmp_path / "models")

    # 1. Register model_v001
    model_record_1 = await ModelRegistry.register_model(
        training_result=training_result,
        symbol="ETH/USDT",
        timeframe="15m",
        session=db_session,
        set_active=True,
        model_dir=model_dir,
    )

    assert model_record_1.version == "model_v001"
    assert model_record_1.is_active is True
    assert os.path.exists(model_record_1.file_path)

    # 2. Register model_v002 (not active)
    model_record_2 = await ModelRegistry.register_model(
        training_result=training_result,
        symbol="ETH/USDT",
        timeframe="15m",
        session=db_session,
        set_active=False,
        model_dir=model_dir,
    )
    assert model_record_2.version == "model_v002"
    assert model_record_2.is_active is False

    # 3. Active model check
    active = await ModelRegistry.get_active_model(db_session)
    assert active is not None
    assert active.version == "model_v001"

    # 4. Promote model_v002 to active
    await ModelRegistry.set_active_model("model_v002", db_session)
    active_updated = await ModelRegistry.get_active_model(db_session)
    assert active_updated is not None
    assert active_updated.version == "model_v002"

    # 5. Load model artifact back into memory
    loaded_xgb = ModelRegistry.load_model_artifact(active_updated)
    assert loaded_xgb is not None
    # Test that loaded model can predict
    sample_X = splits.X_test.iloc[:5]
    probas = loaded_xgb.predict_proba(sample_X)
    assert probas.shape == (5, 2)


# ------------------------------------------------------------------------------
# 3. ModelPredictor Tests
# ------------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_model_predictor_probabilities_and_persistence(db_session: AsyncSession, tmp_path):
    """Verify ModelPredictor output probabilities and database logging."""
    df = generate_synthetic_candles_for_training(160)
    dataset = DatasetBuilder.build(
        candles_df=df,
        symbol="BTC/USDT",
        timeframe="15m",
        horizon=6,
        target_return=0.003,
        drop_na=True,
    )
    splits = DatasetBuilder.split_dataset(dataset)
    training_result = ModelTrainer.train(
        splits,
        hyperparameters={"n_estimators": 20, "max_depth": 3},
        early_stopping_rounds=5,
    )

    model_dir = str(tmp_path / "models")
    model_record = await ModelRegistry.register_model(
        training_result=training_result,
        symbol="BTC/USDT",
        timeframe="15m",
        session=db_session,
        set_active=True,
        model_dir=model_dir,
    )

    predictor = ModelPredictor(
        model=training_result.model,
        feature_names=training_result.feature_names,
        model_version=model_record.version,
        model_version_id=model_record.id,
        horizon=6,
    )

    # 1. Single prediction test
    single_feat = splits.X_test.iloc[0].to_dict()
    pred_single = predictor.predict_single(
        features=single_feat,
        timestamp=splits.X_test.index[0].to_pydatetime(),
        symbol="BTC/USDT",
        timeframe="15m",
    )
    assert isinstance(pred_single, PredictionResult)
    assert np.isclose(pred_single.probability_up + pred_single.probability_down, 1.0)
    assert pred_single.predicted_class in (0, 1)

    json_dict = pred_single.to_dict()
    assert "probability_up" in json_dict
    assert "probability_down" in json_dict
    assert "predicted_class" in json_dict

    # 2. Batch predictions test
    batch_preds = predictor.predict_batch(splits.X_test, symbol="BTC/USDT", timeframe="15m")
    assert len(batch_preds) == len(splits.X_test)
    for p in batch_preds:
        assert np.isclose(p.probability_up + p.probability_down, 1.0)

    # 3. Missing features error check
    bad_features = {"return_1": 0.01}  # Missing most features
    with pytest.raises(ValueError) as exc:
        predictor.predict_single(bad_features)
    assert "missing required model columns" in str(exc.value)

    # 4. Save predictions to database
    saved_count = await ModelPredictor.save_predictions(batch_preds, db_session)
    assert saved_count == len(batch_preds)
