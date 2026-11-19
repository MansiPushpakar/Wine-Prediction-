"""Data loading, validation, training and prediction for Wine Quality."""

from io import StringIO
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd
import requests
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split

DATA_URL = (
    "https://huggingface.co/osanseviero/wine-quality/resolve/main/"
    "winequality-red.csv"
)
LOCAL_DATA_PATH = Path(__file__).resolve().parent / "winequality-red.csv"

FEATURE_COLUMNS = [
    "fixed acidity",
    "volatile acidity",
    "citric acid",
    "residual sugar",
    "chlorides",
    "free sulfur dioxide",
    "total sulfur dioxide",
    "density",
    "pH",
    "sulphates",
    "alcohol",
]
TARGET_COLUMN = "quality"
REQUIRED_COLUMNS = FEATURE_COLUMNS + [TARGET_COLUMN]


def validate_dataset(data: pd.DataFrame) -> pd.DataFrame:
    """Validate and return a clean copy of the expected red-wine table."""
    if not isinstance(data, pd.DataFrame):
        raise ValueError("Dataset must be a pandas DataFrame.")

    missing = [column for column in REQUIRED_COLUMNS if column not in data.columns]
    if missing:
        raise ValueError(f"Dataset is missing required columns: {', '.join(missing)}")

    clean = data[REQUIRED_COLUMNS].copy()
    for column in REQUIRED_COLUMNS:
        clean[column] = pd.to_numeric(clean[column], errors="coerce")

    if clean.isna().any().any():
        raise ValueError("Dataset contains missing or non-numeric values.")
    if not np.isfinite(clean.to_numpy(dtype=float)).all():
        raise ValueError("Dataset contains infinite values.")
    if len(clean) < 20:
        raise ValueError("Dataset must contain at least 20 rows.")
    if not clean[TARGET_COLUMN].between(0, 10).all():
        raise ValueError("Wine quality scores must be between 0 and 10.")

    return clean


def load_wine_data() -> pd.DataFrame:
    """Load a local CSV if present; otherwise download the public red-wine CSV."""
    if LOCAL_DATA_PATH.exists():
        raw = pd.read_csv(LOCAL_DATA_PATH, sep=";")
    else:
        response = requests.get(DATA_URL, timeout=(5, 20))
        response.raise_for_status()
        raw = pd.read_csv(StringIO(response.text), sep=";")
    return validate_dataset(raw)


def build_features_and_target(data: pd.DataFrame):
    """Apply the original notebook's binary target rule."""
    clean = validate_dataset(data)
    X = clean[FEATURE_COLUMNS]
    y = clean[TARGET_COLUMN].apply(lambda score: 1 if score >= 7 else 0)
    return X, y


def train_model(data: pd.DataFrame):
    """Train Random Forest using the original 80:20 split and split seed."""
    X, y = build_features_and_target(data)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=3,
    )
    # The original notebook used RandomForestClassifier() with default settings.
    # A fixed seed makes this deployment reproducible while keeping the same model family.
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    return model, X_test, y_test


def evaluate_model(model, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    """Return held-out classification metrics."""
    predictions = model.predict(X_test)
    return {
        "accuracy": float(accuracy_score(y_test, predictions)),
        "precision": float(precision_score(y_test, predictions, zero_division=0)),
        "recall": float(recall_score(y_test, predictions, zero_division=0)),
        "f1": float(f1_score(y_test, predictions, zero_division=0)),
    }


def predict_quality(model, measurements: Mapping[str, float]) -> dict:
    """Predict the binary quality class after validating all 11 inputs."""
    if not isinstance(measurements, Mapping):
        raise ValueError("Measurements must be supplied as a feature-to-value mapping.")

    missing = [name for name in FEATURE_COLUMNS if name not in measurements]
    if missing:
        raise ValueError(f"Missing measurements: {', '.join(missing)}")

    try:
        values = [float(measurements[name]) for name in FEATURE_COLUMNS]
    except (TypeError, ValueError) as exc:
        raise ValueError("All measurements must be numeric.") from exc

    if not np.isfinite(values).all():
        raise ValueError("Measurements must be finite numbers.")

    sample = pd.DataFrame([values], columns=FEATURE_COLUMNS)
    probabilities = model.predict_proba(sample)[0]
    label = int(model.predict(sample)[0])

    class_probabilities = {
        int(class_label): float(probability)
        for class_label, probability in zip(model.classes_, probabilities)
    }
    return {
        "label": label,
        "good_probability": class_probabilities.get(1, 0.0),
        "not_good_probability": class_probabilities.get(0, 0.0),
    }
