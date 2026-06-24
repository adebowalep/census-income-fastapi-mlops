# Census Income Classification — FastAPI on a Cloud Application Platform

## Project Overview

This project trains a machine learning model to predict whether a person's annual income exceeds $50,000 based on 1994 U.S. Census demographic and employment data, then deploys that model behind a FastAPI REST API. The repository implements the full MLOps loop required by the rubric: data cleaning, model training and persistence, unit testing, model evaluation on data slices, a model card, a FastAPI inference service, API tests, GitHub Actions CI, and continuous deployment to Render.

## Dataset

The data is the UCI "Census Income" (Adult) dataset (`starter/data/census.csv`), 32,561 rows with 14 features (age, workclass, fnlgt, education, education-num, marital-status, occupation, relationship, race, sex, capital-gain, capital-loss, hours-per-week, native-country) and a binary `salary` label (`<=50K` or `>50K`). The raw CSV has stray leading whitespace after every delimiter; `starter/starter/ml/data.py:clean_data()` removes it before training.

## Installation

```bash
# Clone the repository
git clone https://github.com/adebowalep/census-income-fastapi-mlops.git
cd census-income-fastapi-mlops/starter

# Create and activate a virtual environment (Python 3.13)
python3.13 -m venv .venv
source .venv/bin/activate   # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Training

Train the model, persist the artifacts, and generate the slice metrics report:

```bash
cd starter
python -m starter.train_model
```

This loads and cleans `data/census.csv`, performs an 80/20 stratified train/test split, trains a `RandomForestClassifier`, prints overall precision/recall/F1 on the test set, saves `model/model.pkl`, `model/encoder.pkl`, and `model/lb.pkl` with `joblib`, and writes per-category slice metrics to `slice_output.txt`.

## Running FastAPI

```bash
cd starter
uvicorn main:app --reload
```

The interactive API docs (with the request example) are available at `http://127.0.0.1:8000/docs`.

* `GET /` returns a welcome message.
* `POST /predict` accepts a single census record (original hyphenated column names, e.g. `marital-status`, via Pydantic field aliases) and returns the predicted salary category.

## Running Tests

```bash
cd starter
pytest -q
flake8 .
```

There are 11 tests total: 7 unit tests for the data/model functions in `tests/test_ml.py` and 4 API tests in `tests/test_main.py` (GET, a `<=50K` prediction, a `>50K` prediction, and an invalid-payload case). You can also run the rubric's heuristic sanity checker:

```bash
python sanitycheck.py
# when prompted, enter: tests/test_main.py
```

## GitHub Actions

`.github/workflows/python-app.yml` runs on every push and pull request to `main`/`master`. It installs dependencies, runs `flake8`, trains the model so the artifacts exist for the API tests, and runs `pytest`, all under Python 3.13 (matching local development). Both flake8 and pytest must pass without error.

## Deployment

The app is deployed to [Render](https://render.com) as a Web Service, configured via `render.yaml` at the repo root:

1. Push this repository to GitHub.
2. In Render, choose **New > Web Service**, connect the GitHub repo, and let Render detect `render.yaml` (or configure manually: root directory `starter`, build command `pip install -r requirements.txt`, start command `uvicorn main:app --host 0.0.0.0 --port $PORT`).
3. Enable **Auto-Deploy** so Render redeploys automatically on every push to the default branch, after GitHub Actions CI has passed.

## API Usage

Example request to the live API with `requests`:

```bash
curl -X POST "https://<your-render-app>.onrender.com/predict" \
  -H "Content-Type: application/json" \
  -d '{
        "age": 46, "workclass": "Private", "fnlgt": 188386,
        "education": "Doctorate", "education-num": 16,
        "marital-status": "Married-civ-spouse", "occupation": "Exec-managerial",
        "relationship": "Husband", "race": "White", "sex": "Male",
        "capital-gain": 15024, "capital-loss": 0, "hours-per-week": 60,
        "native-country": "United-States"
      }'
```

`starter/live_post.py` does the same thing via the `requests` library and prints the status code and prediction:

```bash
python starter/live_post.py https://<your-render-app>.onrender.com/predict
```

## Model Card

See [`starter/model_card.md`](starter/model_card.md) for model details, intended use, training/evaluation data, metrics (precision 0.8083 / recall 0.5727 / F1 0.6704 on the held-out test set), ethical considerations, and caveats/recommendations.

## Screenshots Required

The following screenshots must be captured after pushing to GitHub and deploying to Render, and committed to `starter/screenshots/`:

| Screenshot | What it shows |
|---|---|
| `continuous_integration.png` | The GitHub Actions workflow run passing (flake8 + pytest green). |
| `example.png` | The `/docs` Swagger UI showing the `/predict` request body example. |
| `continuous_deployment.png` | The Render dashboard showing Auto-Deploy enabled. |
| `live_get.png` | A browser showing the JSON response of `GET /` on the live Render URL. |
| `live_post.png` | The terminal output of `python starter/live_post.py <live-url>/predict`. |

## Repository Link

https://github.com/adebowalep/census-income-fastapi-mlops
