"""API tests for the FastAPI application defined in main.py."""
from fastapi.testclient import TestClient

from main import app

client = TestClient(app)

# Verified against the trained model: this record reliably predicts <=50K.
LOW_INCOME_PAYLOAD = {
    "age": 18,
    "workclass": "Private",
    "fnlgt": 226956,
    "education": "HS-grad",
    "education-num": 9,
    "marital-status": "Never-married",
    "occupation": "Other-service",
    "relationship": "Own-child",
    "race": "White",
    "sex": "Female",
    "capital-gain": 0,
    "capital-loss": 0,
    "hours-per-week": 30,
    "native-country": "?",
}

# Verified against the trained model: this record reliably predicts >50K.
HIGH_INCOME_PAYLOAD = {
    "age": 46,
    "workclass": "Private",
    "fnlgt": 188386,
    "education": "Doctorate",
    "education-num": 16,
    "marital-status": "Married-civ-spouse",
    "occupation": "Exec-managerial",
    "relationship": "Husband",
    "race": "White",
    "sex": "Male",
    "capital-gain": 15024,
    "capital-loss": 0,
    "hours-per-week": 60,
    "native-country": "United-States",
}


def test_get_root() -> None:
    """GET / should return a 200 status and the expected welcome message."""
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Welcome to Census Income Prediction API"}


def test_post_prediction_low_income() -> None:
    """POST /predict should return a <=50K prediction for a low-income profile."""
    response = client.post("/predict", json=LOW_INCOME_PAYLOAD)
    assert response.status_code == 200
    body = response.json()
    assert "prediction" in body
    assert body["prediction"] == "<=50K"


def test_post_prediction_high_income() -> None:
    """POST /predict should return a >50K prediction for a high-income profile."""
    response = client.post("/predict", json=HIGH_INCOME_PAYLOAD)
    assert response.status_code == 200
    body = response.json()
    assert "prediction" in body
    assert body["prediction"] == ">50K"


def test_post_prediction_missing_field_returns_422() -> None:
    """POST /predict with a missing required field should be rejected."""
    bad_payload = dict(LOW_INCOME_PAYLOAD)
    del bad_payload["age"]
    response = client.post("/predict", json=bad_payload)
    assert response.status_code == 422
