"""Send a single POST request to the deployed Census Income Prediction API.

Usage:
    python live_post.py [API_URL]

If API_URL is not provided, it defaults to the value of the
CENSUS_API_URL environment variable, or to a placeholder Render URL that
must be replaced with your actual deployed app's URL.
"""
import os
import sys

import requests

DEFAULT_URL = "https://census-income-fastapi-mlops.onrender.com/predict"

# A sample census record. This particular record reflects a high-education,
# high-hours, married executive profile and is expected to predict ">50K".
SAMPLE_RECORD = {
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


def main() -> None:
    """POST a sample record to the live API and print the result."""
    url = sys.argv[1] if len(sys.argv) > 1 else os.environ.get(
        "CENSUS_API_URL", DEFAULT_URL
    )

    response = requests.post(url, json=SAMPLE_RECORD, timeout=30)

    print(f"Status code: {response.status_code}")
    print(f"Result: {response.json()}")


if __name__ == "__main__":
    main()
