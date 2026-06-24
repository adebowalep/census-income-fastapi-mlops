"""Unit tests for the machine learning data and model functions."""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier

from starter.ml.data import clean_data, process_data
from starter.ml.model import (
    compute_model_metrics,
    compute_slice_metrics,
    inference,
    train_model,
)

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "census.csv"
CAT_FEATURES = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]


@pytest.fixture(scope="module")
def cleaned_data() -> pd.DataFrame:
    """Load and clean the census dataset once for all tests in this module."""
    return clean_data(DATA_PATH)


@pytest.fixture(scope="module")
def processed(cleaned_data: pd.DataFrame):
    """Process a small sample of the cleaned data for fast, isolated tests."""
    sample = cleaned_data.sample(n=200, random_state=42)
    X, y, encoder, lb = process_data(
        sample, categorical_features=CAT_FEATURES, label="salary", training=True
    )
    return X, y, encoder, lb, sample


def test_clean_data_strips_whitespace(cleaned_data: pd.DataFrame) -> None:
    """clean_data should remove leading/trailing whitespace from values."""
    assert cleaned_data.columns.tolist()[0] == "age"
    sample_values = cleaned_data["workclass"].unique()
    assert all(value == value.strip() for value in sample_values)


def test_process_data_returns_expected_types(processed) -> None:
    """process_data should return numpy arrays for X and y in training mode."""
    X, y = processed[0], processed[1]
    assert isinstance(X, np.ndarray)
    assert isinstance(y, np.ndarray)
    assert X.shape[0] == y.shape[0]


def test_process_data_inference_mode_uses_passed_encoder(processed) -> None:
    """In inference mode, process_data should reuse the supplied encoder/lb."""
    _, _, encoder, lb, sample = processed
    X_infer, y_infer, returned_encoder, returned_lb = process_data(
        sample,
        categorical_features=CAT_FEATURES,
        label="salary",
        training=False,
        encoder=encoder,
        lb=lb,
    )
    assert returned_encoder is encoder
    assert returned_lb is lb
    assert isinstance(X_infer, np.ndarray)


def test_train_model_returns_fitted_random_forest(processed) -> None:
    """train_model should return a fitted RandomForestClassifier instance."""
    X, y = processed[0], processed[1]
    model = train_model(X, y)
    assert isinstance(model, RandomForestClassifier)
    # A fitted estimator exposes learned attributes ending in an underscore.
    assert hasattr(model, "feature_importances_")


def test_inference_returns_array_of_expected_length(processed) -> None:
    """inference should return one prediction per input row."""
    X, y = processed[0], processed[1]
    model = train_model(X, y)
    preds = inference(model, X)
    assert isinstance(preds, np.ndarray)
    assert preds.shape[0] == X.shape[0]
    assert set(np.unique(preds)).issubset({0, 1})


def test_compute_model_metrics_returns_three_floats() -> None:
    """compute_model_metrics should return precision, recall, fbeta as floats."""
    y = np.array([1, 0, 1, 1, 0, 1])
    preds = np.array([1, 0, 0, 1, 0, 1])
    precision, recall, fbeta = compute_model_metrics(y, preds)
    assert isinstance(precision, float)
    assert isinstance(recall, float)
    assert isinstance(fbeta, float)
    assert 0.0 <= precision <= 1.0
    assert 0.0 <= recall <= 1.0
    assert 0.0 <= fbeta <= 1.0


def test_compute_slice_metrics_writes_output_file(processed, tmp_path) -> None:
    """compute_slice_metrics should write a non-empty slice_output file."""
    X, y, encoder, lb, sample = processed
    model = train_model(X, y)
    output_path = tmp_path / "slice_output.txt"
    compute_slice_metrics(
        model,
        sample,
        categorical_features=CAT_FEATURES,
        encoder=encoder,
        lb=lb,
        label="salary",
        output_path=str(output_path),
    )
    assert output_path.exists()
    content = output_path.read_text()
    assert "sex=" in content
    assert "precision=" in content
