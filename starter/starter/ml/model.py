"""Model training, inference, and evaluation utilities.

This module implements the core machine learning functions required by the
Census Income Classification project: training a classifier, running
inference, computing aggregate classification metrics, and computing
classification metrics on slices of the data held fixed on a single
categorical feature value.
"""
from typing import List

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import fbeta_score, precision_score, recall_score

from starter.ml.data import process_data


def train_model(X_train: np.ndarray, y_train: np.ndarray) -> RandomForestClassifier:
    """
    Trains a machine learning model and returns it.

    Inputs
    ------
    X_train : np.ndarray
        Training data.
    y_train : np.ndarray
        Labels.
    Returns
    -------
    model : RandomForestClassifier
        Trained machine learning model.
    """
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=12,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    return model


def compute_model_metrics(y: np.ndarray, preds: np.ndarray):
    """
    Validates the trained machine learning model using precision, recall, and F1.

    Inputs
    ------
    y : np.ndarray
        Known labels, binarized.
    preds : np.ndarray
        Predicted labels, binarized.
    Returns
    -------
    precision : float
    recall : float
    fbeta : float
    """
    fbeta = fbeta_score(y, preds, beta=1, zero_division=1)
    precision = precision_score(y, preds, zero_division=1)
    recall = recall_score(y, preds, zero_division=1)
    return precision, recall, fbeta


def inference(model: RandomForestClassifier, X: np.ndarray) -> np.ndarray:
    """ Run model inferences and return the predictions.

    Inputs
    ------
    model : RandomForestClassifier
        Trained machine learning model.
    X : np.ndarray
        Data used for prediction.
    Returns
    -------
    preds : np.ndarray
        Predictions from the model.
    """
    return model.predict(X)


def compute_slice_metrics(
    model: RandomForestClassifier,
    data: pd.DataFrame,
    categorical_features: List[str],
    encoder,
    lb,
    label: str = "salary",
    output_path: str = "slice_output.txt",
) -> None:
    """Compute classification metrics on slices of the data.

    For every categorical feature in ``categorical_features``, this loops
    through every unique value that feature takes on, holds the feature
    fixed at that value, and computes precision/recall/fbeta on that subset
    of the data only. Results are written to ``output_path``.

    Inputs
    ------
    model : RandomForestClassifier
        Trained machine learning model.
    data : pd.DataFrame
        Full (unencoded) dataframe containing features and the label column.
    categorical_features : List[str]
        List of categorical column names to slice on.
    encoder : sklearn.preprocessing.OneHotEncoder
        Trained encoder used during training.
    lb : sklearn.preprocessing.LabelBinarizer
        Trained label binarizer used during training.
    label : str
        Name of the label column in ``data``.
    output_path : str
        Path of the file the slice metrics will be written to.

    Returns
    -------
    None
    """
    lines = []
    for feature in categorical_features:
        lines.append(f"# Feature: {feature}")
        for value in sorted(data[feature].unique(), key=str):
            slice_df = data[data[feature] == value]
            X_slice, y_slice, _, _ = process_data(
                slice_df,
                categorical_features=categorical_features,
                label=label,
                training=False,
                encoder=encoder,
                lb=lb,
            )
            preds = inference(model, X_slice)
            precision, recall, fbeta = compute_model_metrics(y_slice, preds)
            lines.append(f"{feature}={value}")
            lines.append(f"  n_samples={len(slice_df)}")
            lines.append(f"  precision={precision:.4f}")
            lines.append(f"  recall={recall:.4f}")
            lines.append(f"  fbeta={fbeta:.4f}")
        lines.append("")

    with open(output_path, "w") as f:
        f.write("\n".join(lines))
