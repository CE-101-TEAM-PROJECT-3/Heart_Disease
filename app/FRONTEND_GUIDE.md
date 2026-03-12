# Frontend Engineer Guide — Clinical Triage Dashboard

This document explains the Flask backend (`app.py`) so you know exactly what the server expects from your HTML and what it will hand back to your templates.

---

## How the App Starts

```python
pipeline = joblib.load("pipeline.pkl")
```

The trained ML model is loaded **once** when the server boots — not on every request. You don't need to do anything here, but it means the server must have `pipeline.pkl` in the same directory before it starts.

---

## Routes

### `GET /`

Renders `templates/index.html`. This is your input form page.

No data is passed to this template — it is a blank form.

---

### `POST /predict`

This is the route your form must submit to.

**Your form must:**
- Use `method="POST"` and `action="/predict"`
- Have exactly **13 named inputs**, all numeric

| HTML `name` attribute | What it represents              | Expected values                        |
|-----------------------|---------------------------------|----------------------------------------|
| `age`                 | Patient age in years            | Integer (e.g. `63`)                    |
| `sex`                 | Biological sex                  | `0` = female, `1` = male               |
| `cp`                  | Chest pain type                 | `0` – `3`                              |
| `trestbps`            | Resting blood pressure (mmHg)   | Integer (e.g. `145`)                   |
| `chol`                | Serum cholesterol (mg/dl)       | Integer (e.g. `233`)                   |
| `fbs`                 | Fasting blood sugar > 120 mg/dl | `0` = false, `1` = true                |
| `restecg`             | Resting ECG results             | `0` – `2`                              |
| `thalach`             | Max heart rate achieved         | Integer (e.g. `150`)                   |
| `exang`               | Exercise-induced angina         | `0` = no, `1` = yes                    |
| `oldpeak`             | ST depression induced           | Float (e.g. `2.3`)                     |
| `slope`               | Slope of peak exercise ST       | `0` – `2`                              |
| `ca`                  | Number of major vessels (0–3)   | `0` – `3`                              |
| `thal`                | Thalassemia                     | `1` = normal, `2` = fixed, `3` = reversible defect |

**If any field is missing or non-numeric**, the server returns a plain-text `400 Bad Request`. You should add client-side `required` and `type="number"` attributes to all inputs to prevent this.

---

## What the Backend Does on `/predict`

1. Reads all 13 form fields and casts them to `float`.
2. Builds a single-row table with the correct column names the model expects.
3. Runs two model calls:
   - `predict_proba()` — probability of heart disease (0.0 – 1.0)
   - `predict()` — binary classification (0 or 1)
4. Pulls the model's internal coefficients to identify the **top 3 features** that drove the prediction.
5. Renders `templates/result.html` with the variables below.

---

## Template Variables Passed to `result.html`

| Variable      | Type                          | Description                                               |
|---------------|-------------------------------|-----------------------------------------------------------|
| `prediction`  | `int` — `0` or `1`            | `0` = no heart disease predicted, `1` = disease predicted |
| `risk_pct`    | `float` — e.g. `73.4`         | Probability of heart disease as a percentage              |
| `top_drivers` | `list` of `(str, float)` tuples | Top 3 model features by impact — `(feature_name, coefficient)` |

### Using these in Jinja2

```html
<!-- Binary result -->
{% if prediction == 1 %}
  <p>High Risk — Heart Disease Detected</p>
{% else %}
  <p>Low Risk — No Heart Disease Detected</p>
{% endif %}

<!-- Probability score -->
<p>Risk Score: {{ risk_pct }}%</p>

<!-- Top driving factors -->
<ul>
  {% for feature, coef in top_drivers %}
    <li>{{ feature }} (coefficient: {{ "%.3f" | format(coef) }})</li>
  {% endfor %}
</ul>
```

### Understanding `top_drivers`

Each tuple is `(feature_name, coefficient)`:

- A **positive coefficient** means the feature pushes the prediction toward heart disease.
- A **negative coefficient** means it pushes away from it.
- They are sorted by **absolute magnitude** — the feature with the biggest impact comes first.

Example value:
```python
[
    ("st_depression",         1.842),
    ("num_major_vessels",     1.603),
    ("exercise_induced_angina", 0.971)
]
```

---

## File Structure Expected

```
Heart_Disease/app/
├── app.py
├── pipeline.pkl
├── FRONTEND_GUIDE.md
├── static/
│   └── css/
└── templates/
    ├── index.html
    └── result.html
```
## My Recommendation

Adding that "This model provides probabilistic risk assessments based on historical data. It is designed to assist, not replace, professional medical judgment. All triage flags must be independently verified by a licensed clinician prior to patient diagnosis." would allow us to support our safety first approach.

---

## Running the Server

```bash
pip install flask pandas numpy scikit-learn joblib
python app.py
```

The server binds to `127.0.0.1:5000` only. It will **not** be accessible from other machines — this is intentional for patient data privacy.

Visit `http://127.0.0.1:5000` in your browser.
