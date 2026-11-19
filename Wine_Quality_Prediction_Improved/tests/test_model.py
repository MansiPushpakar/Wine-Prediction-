import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier

from model import (
    FEATURE_COLUMNS,
    REQUIRED_COLUMNS,
    build_features_and_target,
    predict_quality,
    train_model,
    validate_dataset,
)


@pytest.fixture
def sample_data():
    # Small deterministic test fixture. The production app loads the full UCI CSV.
    rows = []
    for i in range(40):
        row = {
            "fixed acidity": 6.0 + (i % 5) * 0.2,
            "volatile acidity": 0.3 + (i % 4) * 0.05,
            "citric acid": 0.1 + (i % 3) * 0.05,
            "residual sugar": 1.5 + (i % 6) * 0.2,
            "chlorides": 0.05 + (i % 5) * 0.005,
            "free sulfur dioxide": 10 + i % 10,
            "total sulfur dioxide": 30 + i % 15,
            "density": 0.994 + (i % 4) * 0.001,
            "pH": 3.1 + (i % 5) * 0.05,
            "sulphates": 0.5 + (i % 4) * 0.05,
            "alcohol": 9.0 + (i % 6) * 0.3,
            "quality": 7 if i % 3 == 0 else 5,
        }
        rows.append(row)
    return pd.DataFrame(rows, columns=REQUIRED_COLUMNS)


def test_dataset_schema_and_shape(sample_data):
    clean = validate_dataset(sample_data)
    assert len(clean) == 40
    assert list(clean.columns) == REQUIRED_COLUMNS
    assert len(FEATURE_COLUMNS) == 11


def test_original_binary_target_rule(sample_data):
    X, y = build_features_and_target(sample_data)
    assert X.shape == (40, 11)
    assert set(y.unique()) == {0, 1}
    assert (y == 1).sum() == 14
    assert (y == 0).sum() == 26


def test_missing_column_is_rejected(sample_data):
    with pytest.raises(ValueError, match="missing required columns"):
        validate_dataset(sample_data.drop(columns=["alcohol"]))


def test_missing_values_are_rejected(sample_data):
    broken = sample_data.copy()
    broken.loc[0, "pH"] = np.nan
    with pytest.raises(ValueError, match="missing or non-numeric"):
        validate_dataset(broken)


def test_invalid_quality_is_rejected(sample_data):
    broken = sample_data.copy()
    broken.loc[0, "quality"] = 12
    with pytest.raises(ValueError, match="between 0 and 10"):
        validate_dataset(broken)


def test_model_trains_and_predicts(sample_data):
    model, X_test, y_test = train_model(sample_data)
    assert isinstance(model, RandomForestClassifier)
    assert len(X_test) == 8
    assert len(y_test) == 8
    values = {feature: float(sample_data.iloc[0][feature]) for feature in FEATURE_COLUMNS}
    result = predict_quality(model, values)
    assert result["label"] in (0, 1)
    assert 0.0 <= result["good_probability"] <= 1.0
    assert 0.0 <= result["not_good_probability"] <= 1.0
    assert result["good_probability"] + result["not_good_probability"] == pytest.approx(1.0)


def test_missing_prediction_input_is_rejected(sample_data):
    model, _, _ = train_model(sample_data)
    values = {feature: float(sample_data.iloc[0][feature]) for feature in FEATURE_COLUMNS}
    values.pop("alcohol")
    with pytest.raises(ValueError, match="Missing measurements"):
        predict_quality(model, values)


def test_non_finite_prediction_input_is_rejected(sample_data):
    model, _, _ = train_model(sample_data)
    values = {feature: float(sample_data.iloc[0][feature]) for feature in FEATURE_COLUMNS}
    values["alcohol"] = float("nan")
    with pytest.raises(ValueError, match="finite numbers"):
        predict_quality(model, values)
