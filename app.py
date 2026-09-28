"""Flask app serving the diabetes Random Forest model through an HTML form and a JSON API."""
import json
import logging
import os
import sys
from pathlib import Path

import joblib
import pandas as pd
import sklearn
from flask import Flask, jsonify, render_template, request
from werkzeug.exceptions import HTTPException

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))  # the saved pipeline references functions in diabetes_model

MODEL_PATH = Path(os.environ.get("MODEL_PATH", ROOT / "models" / "diabetes_rf.joblib"))
METADATA_PATH = MODEL_PATH.with_name("metadata.json")

# Form field name -> (model feature, type, min, max). Ranges reject impossible values, not unusual ones.
FIELDS = {
    "pregnancies": ("Pregnancies", int, 0, 20),
    "glucose": ("Glucose", float, 0, 300),
    "bloodpressure": ("BloodPressure", float, 0, 200),
    "skinthickness": ("SkinThickness", float, 0, 100),
    "insulin": ("Insulin", float, 0, 900),
    "bmi": ("BMI", float, 0, 80),
    "dpf": ("DiabetesPedigreeFunction", float, 0, 3),
    "age": ("Age", int, 1, 120),
}

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("diabetes-app")


class ValidationError(ValueError):
    pass


def load_model(path=MODEL_PATH, metadata_path=METADATA_PATH):
    if not path.exists():
        raise FileNotFoundError(f"Model not found at {path}. Run `python src/train.py` first.")
    metadata = json.loads(metadata_path.read_text())
    if metadata["sklearn_version"] != sklearn.__version__:
        log.warning("Model trained with scikit-learn %s, running %s",
                    metadata["sklearn_version"], sklearn.__version__)
    return joblib.load(path), metadata


def parse_input(data):
    errors, row = {}, {}
    for field, (feature, cast, low, high) in FIELDS.items():
        raw = data.get(field)
        if raw is None or str(raw).strip() == "":
            errors[field] = "required"
            continue
        try:
            value = cast(str(raw).strip()) if cast is int else cast(raw)
        except (TypeError, ValueError):
            errors[field] = f"must be a {'whole number' if cast is int else 'number'}"
            continue
        if not low <= value <= high:
            errors[field] = f"must be between {low} and {high}"
            continue
        row[feature] = value
    if errors:
        raise ValidationError(errors)
    return pd.DataFrame([row])


def create_app(model=None, metadata=None):
    app = Flask(__name__)
    if model is None:
        model, metadata = load_model()

    def predict(data):
        X = parse_input(data)
        probability = float(model.predict_proba(X)[0, 1])
        return {"prediction": int(probability >= 0.5), "probability": round(probability, 3)}

    @app.get("/")
    def home():
        return render_template("index.html", values={})

    @app.post("/predict")
    def predict_endpoint():
        if request.is_json:
            try:
                return jsonify(predict(request.get_json(silent=True) or {}))
            except ValidationError as exc:
                return jsonify({"errors": exc.args[0]}), 400
        values = request.form.to_dict()
        try:
            return render_template("index.html", result=predict(values), values=values)
        except ValidationError as exc:
            return render_template("index.html", errors=exc.args[0], values=values), 400

    @app.get("/health")
    def health():
        return jsonify({"status": "ok", "model_sklearn_version": metadata["sklearn_version"]})

    @app.errorhandler(Exception)
    def unhandled(exc):
        if isinstance(exc, HTTPException):
            return exc
        log.exception("Unhandled error")
        return jsonify({"error": "internal server error"}), 500

    return app


if __name__ == "__main__":
    create_app().run(host="127.0.0.1", port=int(os.environ.get("PORT", 5000)))
