import numpy as np
import pandas as pd
import joblib
from flask import Flask, request, render_template

app = Flask(__name__)
MODEL_PATH = "pipeline.pkl"
pipeline = joblib.load(MODEL_PATH)

FEATURE_MAP = {
    "age":        "age",
    "sex":        "sex",
    "cp":         "chest_pain_type",
    "trestbps":   "resting_blood_pressure",
    "chol":       "cholesterol",
    "fbs":        "fasting_blood_sugar",
    "restecg":    "resting_ecg",
    "thalach":    "max_heart_rate",
    "exang":      "exercise_induced_angina",
    "oldpeak":    "st_depression",
    "slope":      "slope",
    "ca":         "num_major_vessels",
    "thal":       "thalassemia",
}

FEATURE_NAMES = list(FEATURE_MAP.values())


def _top_drivers(coefs: np.ndarray, n: int = 3) -> list[tuple[str, float]]:
    """Return the n features with the largest absolute coefficient magnitude."""
    pairs = list(zip(FEATURE_NAMES, coefs))
    pairs.sort(key=lambda x: abs(x[1]), reverse=True)
    return pairs[:n]


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    # Parse and validate all 13 form values as floats.
    try:
        values = [float(request.form[field]) for field in FEATURE_MAP]
    except (KeyError, ValueError) as exc:
        return f"Invalid input: {exc}", 400

    # Build a single-row DataFrame with the correct column names.
    X = pd.DataFrame([values], columns=FEATURE_NAMES)

    # --- Inference ---
    proba = pipeline.predict_proba(X)[0]   # [P(no disease), P(disease)]
    prediction = int(pipeline.predict(X)[0])
    risk_pct = round(float(proba[1]) * 100, 1)

    # --- Explainability: extract LR coefficients ---
    coefs = pipeline.named_steps["lr"].coef_[0]
    top_drivers = _top_drivers(coefs, n=3)

    return render_template(
        "result.html",
        prediction=prediction,        # 0 = no disease, 1 = disease
        risk_pct=risk_pct,            # e.g. 73.4  (percentage)
        top_drivers=top_drivers,      # [(feature_name, coef), ...]
    )


if __name__ == "__main__":
    # Bind only to localhost — patient data must never leave this machine.
    app.run(host="127.0.0.1", port=5000, debug=False)
