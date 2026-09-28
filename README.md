# Diabetes Prediction

Machine learning models that predict diabetes from patient health indicators, with a Flask web app and JSON API serving the final model.

> **Educational project only.** This is not a medical device and must not be used for diagnosis or treatment decisions.

## Overview

Team capstone project exploring classical ML and neural networks on two public diabetes datasets. The deployed model is a Random Forest trained on the Pima-style dataset and served through a Flask app with input validation, a JSON endpoint and a health check.

## Data

1. **`Diabetes/kaggle_diabetes.csv`** (Kaggle, Pima-style): pregnancies, glucose, blood pressure, skin thickness, insulin, BMI, diabetes pedigree function, age. The file has 2,000 rows, but **only 744 are unique**; the rest are exact duplicates and are removed before training.
2. **`Diabetes/diabetes_prediction_dataset.csv`** (Kaggle, 100,000 records): demographics, hypertension, heart disease, smoking history, HbA1c, blood glucose. Used for the exploratory neural network.

`Database/` holds PostgreSQL schemas for both datasets.

## Approach

`src/train.py` builds the production model:

1. Drop duplicate rows, so no record appears in both the training and test sets.
2. Stratified 80/20 train/test split (595 / 149 records).
3. Pipeline: zero readings for glucose, blood pressure, skin thickness, insulin and BMI are treated as missing, then imputed (mean for glucose and blood pressure, median for the skewed columns). Imputation is fitted on training data only.
4. Random Forest tuned with 5-fold `GridSearchCV` on ROC AUC.
5. Final evaluation once on the held-out test set.

Logistic regression, a decision tree and an SVC were also compared by cross-validation. All scored within noise of the Random Forest (ROC AUC 0.74 to 0.83).

## Results

Held-out test set: 149 patients, 51 with diabetes. Produced by `src/train.py`; full output in `models/metadata.json`.

| Metric | Value |
|---|---|
| Accuracy | 0.779 |
| ROC AUC | 0.819 (cross-validation: 0.827) |
| Precision (diabetes) | 0.750 |
| Recall (diabetes) | 0.529 |
| F1 (diabetes) | 0.621 |

Confusion matrix: 27 diabetic patients correctly flagged, **24 missed**, 9 false alarms, 89 correctly cleared.

The model misses about half of diabetic patients at the default 0.5 threshold. For a screening tool, recall matters more than accuracy, so this is the main limitation.

**On earlier figures:** the original notebook reported 98.75% test accuracy. That result came from duplicate rows appearing in both the training and test sets; on deduplicated data, realistic accuracy is about 78%.

### Neural network (exploratory)

`Diabetes/nn_model/NN_code_diabetes.ipynb` trains a TensorFlow/Keras network on the 100,000-record dataset. It is kept as exploratory work and its metrics are not reported, because the output layer uses `relu` instead of `sigmoid`.

## Web App and API

| Route | Method | Description |
|---|---|---|
| `/` | GET | HTML form |
| `/predict` | POST | Form submission (HTML) or JSON request (JSON response) |
| `/health` | GET | Health check |

```bash
curl -X POST http://127.0.0.1:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"pregnancies": 2, "glucose": 148, "bloodpressure": 72, "skinthickness": 35,
       "insulin": 0, "bmi": 33.6, "dpf": 0.627, "age": 50}'
# {"prediction": 1, "probability": 0.579}
```

Invalid, missing or out-of-range fields return HTTP 400 with a per-field error message.

## Project Structure

```
├── app.py                      # Flask app
├── src/
│   ├── diabetes_model.py       # data loading and model pipeline
│   └── train.py                # training and evaluation
├── models/                     # trained model and metadata.json
├── templates/, static/         # web front end
├── tests/                      # pytest suite
├── Diabetes/                   # datasets and exploratory notebooks
├── Database/                   # PostgreSQL schemas
├── JR Code/                    # Dash dashboard prototype
└── Dockerfile
```

## How to Run

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt
pytest -q
python src/train.py        # optional: retrain the model
python app.py              # http://127.0.0.1:5000
```

With Docker:

```bash
docker build -t diabetes-app .
docker run -p 5000:5000 diabetes-app
```

The notebooks and Dash prototype need the extra packages in `requirements-notebooks.txt`.

## Tech Stack

Python · scikit-learn · pandas · Flask · Gunicorn · Docker · pytest · GitHub Actions · TensorFlow/Keras (exploratory) · PostgreSQL

## Team

Team capstone project. Contributors: [@Faraz-Tab](https://github.com/Faraz-Tab), [@jeffreymrobertson](https://github.com/jeffreymrobertson), [@kkevin1999](https://github.com/kkevin1999), and commit author `cindyliwho`.
