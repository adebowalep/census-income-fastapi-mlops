# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details

This model is a `RandomForestClassifier` from scikit-learn (`n_estimators=100`, `max_depth=12`, `random_state=42`) trained to predict whether a person's annual income exceeds $50,000 based on demographic and employment attributes from 1994 U.S. Census data. Categorical features are encoded with a one-hot encoder, and the binary target label is encoded with a label binarizer. The model, the fitted encoder, and the fitted label binarizer are persisted to disk with `joblib` and are loaded by the FastAPI inference service. The model was trained on October 2025 era scikit-learn (1.7.2) under Python 3.13.

## Intended Use

The model is intended for educational and demonstration purposes within an MLOps course project. It illustrates an end-to-end workflow for training a classifier, packaging it behind a REST API, and deploying that API to a cloud application platform with continuous integration and continuous delivery. It is not intended to be used for real-world decisions about income, creditworthiness, employment, or any other consequential outcome about real individuals.

## Training Data

The data is the UCI "Census Income" (Adult) dataset, sourced from the 1994 U.S. Census database. It contains 32,561 rows and 14 features (age, workclass, fnlgt, education, education-num, marital-status, occupation, relationship, race, sex, capital-gain, capital-loss, hours-per-week, and native-country) plus the binary target column, salary (`<=50K` or `>50K`). Before training, the raw CSV is cleaned by stripping the leading whitespace present after every delimiter in both the header and the data rows. The cleaned data was split 80/20 into training and test sets using a stratified split on the salary label, with a fixed random seed of 42 for reproducibility.

## Evaluation Data

The evaluation (test) set is the 20% holdout split described above, totaling 6,513 rows, processed with the same one-hot encoder and label binarizer that were fit on the training data (encoder and binarizer are not refit on test data).

## Metrics

The model was evaluated with precision, recall, and F1 (fbeta with beta=1), the standard classification metrics for this rubric. On the held-out test set, the model achieved a precision of 0.8083, a recall of 0.5727, and an F1 score of 0.6704. These numbers indicate that when the model predicts an individual earns more than $50,000, it is correct roughly 81% of the time, but it misses a substantial share of the actual high-income individuals, identifying only about 57% of them.

Performance also varies meaningfully across categorical slices of the data; the full breakdown is available in `slice_output.txt`, which reports precision, recall, and F1 for every unique value of every categorical feature (workclass, education, marital-status, occupation, relationship, race, sex, and native-country) held fixed. For example, performance for `native-country=United-States` (n=5,835) closely tracks the overall metrics (precision 0.8198, recall 0.5680, F1 0.6710), while smaller slices with few samples (e.g., several individual countries with fewer than 10 rows) show much higher variance, including some slices with perfect scores and others with zero recall, which reflects sample-size noise rather than genuine subgroup performance differences.

## Ethical Considerations

This dataset and model encode demographic attributes, including race, sex, and native country, that are legally protected characteristics in many jurisdictions. The slice-level evaluation shows that model performance is not uniform across these subgroups, which means using this model for any real decision-making could perpetuate or amplify existing societal biases present in the 1994 census data (for example, historical income disparities correlated with sex or national origin). The dataset is also three decades old and reflects the labor market, demographics, and income distribution of 1994, which is not representative of the present day. This model should not be used to make or inform decisions about real people.

## Caveats and Recommendations

The model's recall (0.57) is meaningfully lower than its precision (0.81), meaning it systematically under-predicts the higher-income class; downstream consumers of this model's predictions should not treat a `<=50K` prediction as a confident negative. Several categorical slices have very small sample counts (single digits in some `native-country` categories), so metrics for those slices should be interpreted cautiously and not treated as reliable estimates of subgroup performance. Recommended next steps before any further use include: rebalancing or reweighting the training data to address class imbalance, performing hyperparameter tuning or trying alternative model families (e.g., gradient boosting), and collecting more representative, contemporary data if this model were ever to be used for purposes beyond this educational exercise.
