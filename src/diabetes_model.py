"""Data preparation and model pipeline for the diabetes classifier."""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "Diabetes" / "kaggle_diabetes.csv"
TARGET = "Outcome"
FEATURES = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI",
            "DiabetesPedigreeFunction", "Age"]
# A value of 0 is physiologically impossible for these measurements and marks a missing reading
ZERO_AS_MISSING = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
MEAN_IMPUTED = ["Glucose", "BloodPressure"]
MEDIAN_IMPUTED = ["SkinThickness", "Insulin", "BMI"]
RANDOM_STATE = 42


def load_data(path=DATA_PATH):
    """Load the dataset and drop exact duplicate rows, which otherwise leak across the train/test split."""
    df = pd.read_csv(path).drop_duplicates().reset_index(drop=True)
    return df[FEATURES], df[TARGET]


def _zeros_to_nan(X):
    X = X.copy()
    X[ZERO_AS_MISSING] = X[ZERO_AS_MISSING].replace(0, np.nan)
    return X


def build_preprocessor():
    impute = ColumnTransformer(
        [("mean", SimpleImputer(strategy="mean"), MEAN_IMPUTED),
         ("median", SimpleImputer(strategy="median"), MEDIAN_IMPUTED)],
        remainder="passthrough",
        verbose_feature_names_out=False,
    )
    return Pipeline([("zeros", FunctionTransformer(_zeros_to_nan, feature_names_out="one-to-one")),
                     ("impute", impute)])


def build_model(**rf_params):
    return Pipeline([("prep", build_preprocessor()),
                     ("clf", RandomForestClassifier(random_state=RANDOM_STATE, **rf_params))])
