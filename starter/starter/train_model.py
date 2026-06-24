"""Script to train, evaluate, and persist the census income classifier.

Run from the ``starter`` directory:

    python -m starter.train_model
"""
from pathlib import Path

import joblib
from sklearn.model_selection import train_test_split

from starter.ml.data import clean_data, process_data
from starter.ml.model import (
    compute_model_metrics,
    compute_slice_metrics,
    inference,
    train_model,
)

# Paths are resolved relative to this file so the script works regardless
# of the caller's current working directory.
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "census.csv"
MODEL_DIR = BASE_DIR / "model"
SLICE_OUTPUT_PATH = BASE_DIR / "slice_output.txt"

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


def main() -> None:
    """Load data, clean it, train the model, evaluate it, and save artifacts."""
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    data = clean_data(DATA_PATH)

    train, test = train_test_split(
        data, test_size=0.20, random_state=42, stratify=data["salary"]
    )

    X_train, y_train, encoder, lb = process_data(
        train, categorical_features=CAT_FEATURES, label="salary", training=True
    )

    X_test, y_test, _, _ = process_data(
        test,
        categorical_features=CAT_FEATURES,
        label="salary",
        training=False,
        encoder=encoder,
        lb=lb,
    )

    model = train_model(X_train, y_train)

    preds = inference(model, X_test)
    precision, recall, fbeta = compute_model_metrics(y_test, preds)
    print(f"Overall test metrics -> precision: {precision:.4f}, "
          f"recall: {recall:.4f}, fbeta: {fbeta:.4f}")

    joblib.dump(model, MODEL_DIR / "model.pkl")
    joblib.dump(encoder, MODEL_DIR / "encoder.pkl")
    joblib.dump(lb, MODEL_DIR / "lb.pkl")
    print(f"Saved model, encoder, and label binarizer to {MODEL_DIR}")

    compute_slice_metrics(
        model,
        test,
        categorical_features=CAT_FEATURES,
        encoder=encoder,
        lb=lb,
        label="salary",
        output_path=str(SLICE_OUTPUT_PATH),
    )
    print(f"Wrote slice metrics to {SLICE_OUTPUT_PATH}")


if __name__ == "__main__":
    main()
