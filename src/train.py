"""Train the Random Forest, evaluate it on a held-out test set, and save the model with its metrics."""
import json
import platform

import joblib
import sklearn
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score, precision_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split

from diabetes_model import FEATURES, RANDOM_STATE, ROOT, build_model, load_data

MODEL_PATH = ROOT / "models" / "diabetes_rf.joblib"
METADATA_PATH = ROOT / "models" / "metadata.json"

PARAM_GRID = {
    "clf__n_estimators": [200, 400],
    "clf__max_depth": [None, 6, 10],
    "clf__min_samples_leaf": [1, 3, 5],
    "clf__class_weight": [None, "balanced"],
}


def main():
    X, y = load_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE)

    search = GridSearchCV(build_model(), PARAM_GRID, scoring="roc_auc", n_jobs=-1,
                          cv=StratifiedKFold(5, shuffle=True, random_state=RANDOM_STATE))
    search.fit(X_train, y_train)
    model = search.best_estimator_

    pred = model.predict(X_test)
    proba = model.predict_proba(X_test)[:, 1]
    metrics = {
        "accuracy": accuracy_score(y_test, pred),
        "precision": precision_score(y_test, pred),
        "recall": recall_score(y_test, pred),
        "f1": f1_score(y_test, pred),
        "roc_auc": roc_auc_score(y_test, proba),
        "cv_roc_auc": search.best_score_,
    }
    metadata = {
        "features": FEATURES,
        "best_params": {k.removeprefix("clf__"): v for k, v in search.best_params_.items()},
        "metrics": {k: round(v, 3) for k, v in metrics.items()},
        "confusion_matrix": confusion_matrix(y_test, pred).tolist(),
        "rows": {"total_unique": len(X), "train": len(X_train), "test": len(X_test)},
        "sklearn_version": sklearn.__version__,
        "python_version": platform.python_version(),
    }

    MODEL_PATH.parent.mkdir(exist_ok=True)
    joblib.dump(model, MODEL_PATH, compress=3)
    METADATA_PATH.write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
