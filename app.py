import os
import numpy as np
import pandas as pd
import joblib
from flask import Flask, render_template, request, jsonify

APP_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(APP_DIR, "carrier_status_model.pkl")

app = Flask(__name__)

# ---------------------------------------------------------------------------
# Load the model bundle once at startup
# ---------------------------------------------------------------------------
bundle = joblib.load(MODEL_PATH)
pipeline = bundle["pipeline"]
MODEL_NAME = bundle["model_name"]
FEATURE_NUM = bundle["feature_num"]
FEATURE_CAT = bundle["feature_cat"]
CAT_OPTIONS = bundle["cat_options"]
NUM_RANGES = {
    col: {k: round(v, 1) for k, v in ranges.items()}
    for col, ranges in bundle["num_ranges"].items()
}
TARGET_NAMES = bundle["target_names"]
METRICS = bundle["metrics"]

ALL_FEATURES = FEATURE_NUM + FEATURE_CAT

# Friendly labels for the form
FIELD_LABELS = {
    "FLEETSIZE": "Fleet size",
    "TRUCK_UNITS": "Truck units",
    "POWER_UNITS": "Power units",
    "BUS_UNITS": "Bus units",
    "TOTAL_DRIVERS": "Total drivers",
    "TOTAL_CDL": "Total CDL drivers",
    "MCS150_MILEAGE": "Annual mileage (MCS-150)",
    "DRIVER_INTER_TOTAL": "Interstate drivers",
    "num_cargo_types": "Number of cargo types hauled",
    "carrier_age_years": "Carrier age (years)",
    "is_hazmat": "Hauls hazardous materials",
    "prior_revoke": "Prior registration revoked",
    "has_safety_review": "Has completed a safety review",
    "CARRIER_OPERATION": "Carrier operation type",
    "PHY_STATE": "Physical state",
    "CLASSDEF": "Operating classification",
    "BUSINESS_ORG_DESC": "Business organization type",
}

CARRIER_OPERATION_LABELS = {
    "A": "Interstate",
    "B": "Intrastate (Hazmat)",
    "C": "Intrastate",
}

BINARY_FIELDS = {"is_hazmat", "prior_revoke", "has_safety_review"}


@app.route("/")
def index():
    """Render the prediction form, pre-populated with training-data ranges/options."""
    return render_template(
        "index.html",
        model_name=MODEL_NAME,
        metrics=METRICS,
        feature_num=FEATURE_NUM,
        feature_cat=FEATURE_CAT,
        binary_fields=BINARY_FIELDS,
        cat_options=CAT_OPTIONS,
        num_ranges=NUM_RANGES,
        field_labels=FIELD_LABELS,
        op_labels=CARRIER_OPERATION_LABELS,
    )


@app.route("/api/model-info")
def model_info():
    """Small JSON endpoint describing the deployed model (used by the page on load)."""
    return jsonify(
        {
            "model_name": MODEL_NAME,
            "metrics": METRICS,
            "features": ALL_FEATURES,
        }
    )


def _to_float_or_nan(value):
    if value is None or value == "":
        return np.nan
    try:
        return float(value)
    except (TypeError, ValueError):
        return np.nan


def _to_str_or_nan(value):
    if value is None or str(value).strip() == "":
        return np.nan
    return str(value).strip()


@app.route("/predict", methods=["POST"])
def predict():
    """Accept a JSON payload of carrier attributes and return a live prediction."""
    payload = request.get_json(force=True, silent=True) or {}

    row = {}
    for col in FEATURE_NUM:
        row[col] = _to_float_or_nan(payload.get(col))
    for col in FEATURE_CAT:
        row[col] = _to_str_or_nan(payload.get(col))

    X = pd.DataFrame([row], columns=ALL_FEATURES)

    try:
        pred = int(pipeline.predict(X)[0])
        proba_active = float(pipeline.predict_proba(X)[0, 1])
    except Exception as exc:  # pragma: no cover
        return jsonify({"error": str(exc)}), 400

    return jsonify(
        {
            "prediction": TARGET_NAMES[pred],
            "prediction_code": pred,
            "probability_active": round(proba_active, 4),
            "probability_inactive": round(1 - proba_active, 4),
            "model_name": MODEL_NAME,
        }
    )


@app.route("/health")
def health():
    return jsonify({"status": "ok", "model_loaded": pipeline is not None})


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
