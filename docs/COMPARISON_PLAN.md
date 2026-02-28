# Plan: ML Model Comparison Script (All Models × All Datasets)

## Task
Write a single Python script (`model_comparison.py`) that trains every ML model on every
cleaned dataset, collects evaluation metrics, and produces comparison graphs.

---

## Datasets (4 total)

| Label in chart | File | Notes |
|---|---|---|
| Raw | `heart.xlsx` | 303 rows, abbreviated col names. Drop 1 duplicate row at load time. |
| Team Cleaned | `cleaned_heart_data (1).csv` | 303 rows. Drop extra `Unnamed: 0` index column. |
| Enes Cleaned | `enes_final_cleaned_data.csv` | 302 rows, fully snake_case. Best quality. |
| Script Cleaned | generated in-script from `heart.xlsx` | Script's own cleaning: drop duplicate + rename cols. |

`final_cleaned_data.csv` (root folder) is identical to `enes_final_cleaned_data.csv` — skip it.

### Script's own cleaning steps (heart.xlsx → "Script Cleaned")
1. Read `heart.xlsx`
2. `drop_duplicates()`
3. Rename columns:
   - `cp` → `chest_pain_type`
   - `trestbps` → `resting_blood_pressure`
   - `chol` → `cholesterol`
   - `fbs` → `fasting_blood_sugar`
   - `restecg` → `resting_ecg`
   - `thalach` → `max_heart_rate`
   - `exang` → `exercise_induced_angina`
   - `oldpeak` → `st_depression`
   - `ca` → `num_major_vessels`
   - `thal` → `thalassemia`

---

## Models (5 total)

| Name | Algorithm | Key config |
|---|---|---|
| K-NN (tuned k) | KNeighborsClassifier | Best k via 5-fold CV on k=1..20, StandardScaler |
| K-NN (GridSearchCV) | KNeighborsClassifier | GridSearchCV over n_neighbors 1-20, weights, metric; n_jobs=-1 |
| Logistic Regression | LogisticRegression | max_iter=1000, StandardScaler |
| SVM (RBF) | SVC | kernel=rbf, C=1.0, gamma=scale, StandardScaler |
| Random Forest | RandomForestClassifier | n_estimators=100 |

---

## Constants (no hardcoding)
```python
RANDOM_STATE = 42
TEST_SIZE    = 0.2
CV_FOLDS     = 5
```

---

## Evaluation — 20 combinations (5 models × 4 datasets)
Per combination collect:
- Test Accuracy
- Precision (macro)
- Recall (macro)
- F1-Score (macro)
- CV Accuracy (5-fold on full dataset)

---

## Visualisation

### Figure 1 — Accuracy by Model, grouped by Dataset
- Grouped bar chart
- X-axis: 5 model names
- Bar groups (colors): 4 datasets
- Y-axis: Test Accuracy (%)
- Value labels above each bar
- Saved as `comparison_by_model.png`

### Figure 2 — Heatmap (Model × Dataset)
- Rows = models, Columns = datasets, values = Test Accuracy
- Annotated cells, color scale low=red → high=green
- Saved as `comparison_heatmap.png`

Both figures displayed with `plt.show()`. Full results table printed to console.

---

## Internal structure
- A `load_dataset(label, path)` helper that reads CSV or XLSX, drops `Unnamed: 0` if present, returns `(X, y)`
- A `build_models(X_train_scaled, y_train)` or dict of pipelines built per dataset run
- K-NN tuning (CV loop) and GridSearchCV run once per dataset, then reused
- Outer loop: datasets → inner loop: models → store results in a dict keyed by `(model, dataset)`

---

## Output file
`Heart_Disease/model_comparison.py` — single self-contained script, ~180-200 lines

---

## How to run
```bash
cd /path/to/Heart_Disease
python model_comparison.py
```

Produces:
- `comparison_by_model.png`
- `comparison_heatmap.png`
- Console results table
