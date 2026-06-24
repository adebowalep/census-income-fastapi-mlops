"""FastAPI application serving the Census Income Classification model.

Exposes:
    GET  /         -> welcome message
    POST /predict  -> runs model inference on a single census record
"""
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, ConfigDict, Field

from starter.ml.data import process_data
from starter.ml.model import inference

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "model"

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

_model = None
_encoder = None
_lb = None


def _load_artifacts() -> None:
    """Load the trained model, encoder, and label binarizer from disk."""
    global _model, _encoder, _lb
    if _model is None:
        _model = joblib.load(MODEL_DIR / "model.pkl")
        _encoder = joblib.load(MODEL_DIR / "encoder.pkl")
        _lb = joblib.load(MODEL_DIR / "lb.pkl")


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    """Load model artifacts once when the application starts."""
    _load_artifacts()
    yield


app = FastAPI(
    title="Census Income Prediction API",
    description="Predicts whether an individual's income is <=50K or >50K "
    "based on 1994 US Census data.",
    version="1.0.0",
    lifespan=lifespan,
)


class CensusData(BaseModel):
    """Schema for a single census record submitted for inference.

    Field aliases are used for columns whose names contain hyphens, since
    those are not valid Python identifiers. The original column names from
    ``census.csv`` are preserved exactly via the aliases so the data does
    not need to be renamed.
    """

    model_config = ConfigDict(
        populate_by_name=True,
        json_schema_extra={
            "examples": [
                {
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
                },
                {
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
                },
            ]
        },
    )

    age: int = Field(..., json_schema_extra={"example": 39})
    workclass: str = Field(..., json_schema_extra={"example": "State-gov"})
    fnlgt: int = Field(..., json_schema_extra={"example": 77516})
    education: str = Field(..., json_schema_extra={"example": "Bachelors"})
    education_num: int = Field(
        ..., alias="education-num", json_schema_extra={"example": 13}
    )
    marital_status: str = Field(
        ..., alias="marital-status", json_schema_extra={"example": "Never-married"}
    )
    occupation: str = Field(..., json_schema_extra={"example": "Adm-clerical"})
    relationship: str = Field(..., json_schema_extra={"example": "Not-in-family"})
    race: str = Field(..., json_schema_extra={"example": "White"})
    sex: str = Field(..., json_schema_extra={"example": "Male"})
    capital_gain: int = Field(
        ..., alias="capital-gain", json_schema_extra={"example": 2174}
    )
    capital_loss: int = Field(
        ..., alias="capital-loss", json_schema_extra={"example": 0}
    )
    hours_per_week: int = Field(
        ..., alias="hours-per-week", json_schema_extra={"example": 40}
    )
    native_country: str = Field(
        ...,
        alias="native-country",
        json_schema_extra={"example": "United-States"},
    )


class PredictionResponse(BaseModel):
    """Schema for the response returned by the ``/predict`` endpoint."""

    prediction: str = Field(..., json_schema_extra={"example": "<=50K"})


@app.get("/")
def read_root() -> dict:
    """Return a welcome message.

    Returns
    -------
    dict
        A JSON-serializable greeting.
    """
    return {"message": "Welcome to Census Income Prediction API"}


@app.post("/predict", response_model=PredictionResponse)
def predict(data: CensusData) -> PredictionResponse:
    """Run model inference on a single census record.

    Inputs
    ------
    data : CensusData
        A single census record using the original (hyphenated) column
        names via field aliases.

    Returns
    -------
    PredictionResponse
        The predicted salary category: ``"<=50K"`` or ``">50K"``.
    """
    _load_artifacts()

    record = pd.DataFrame([data.model_dump(by_alias=True)])

    X, _, _, _ = process_data(
        record,
        categorical_features=CAT_FEATURES,
        training=False,
        encoder=_encoder,
        lb=_lb,
    )
    pred = inference(_model, X)
    label = _lb.inverse_transform(pred)[0]
    return PredictionResponse(prediction=label)
