# Data Leak Analysis: Alice's Logistic Regression Model

## Summary

The high accuracy scores (97-98%) in `Alice_CE101Lab (2).ipynb` are caused by a **data leak** -- the `Unnamed: 0` column (CSV row index) was accidentally left in the feature set.

---

## The Problem

When loading the dataset:

```python
df = pd.read_csv("cleaned_heart_data.csv")
X = df.drop('target', axis=1)
```

`X` contains **14 columns** instead of 13, because `Unnamed: 0` (the row index from the CSV file) was not dropped. This column contains values 0, 1, 2, 3, ..., 302 -- a unique identifier for each row.

You can confirm this in cell 20 of the notebook:

```
Train shape: (151, 14)   <-- should be 13
Test shape: (76, 14)
```

---

## Why This Inflates Accuracy

The dataset is partially ordered by target class -- rows with heart disease are clustered together. This means the row index strongly correlates with the target variable. The model learns a rule like "if row number > threshold, predict disease" instead of learning from actual medical features.

---

## Evidence

| Metric | Reported (with leak) | Expected (without leak) |
|--------|---------------------|------------------------|
| Validation Accuracy | 100.0% | ~82-83% |
| Test Accuracy (basic LR) | 97.37% | ~82-83% |
| Test Accuracy (GridSearchCV) | 98.68% | ~83-84% |
| Cross-validation Accuracy | 96.69% | ~82-83% |

- 100% validation accuracy is a strong indicator of data leakage in any real-world medical dataset.
- After removing the `Unnamed: 0` column, Logistic Regression achieves ~82-83% across repeated stratified 5-fold cross-validation, which is consistent with all other models on this dataset.

---

## The Fix

Add one line after loading the CSV:

```python
df = pd.read_csv("cleaned_heart_data.csv")
df = df.drop(columns=["Unnamed: 0"])  # <-- add this line
X = df.drop('target', axis=1)
```

Alternatively, load with `index_col=0`:

```python
df = pd.read_csv("cleaned_heart_data.csv", index_col=0)
```

---

## Verification

The `model_comparison.py` script already handles this correctly:

```python
team = pd.read_csv("cleaned_heart_data (1).csv")
if "Unnamed: 0" in team.columns:
    team = team.drop(columns=["Unnamed: 0"])
```

With the fix applied, Logistic Regression scores **82.7% +/- 3.3%** (mean +/- std over 15 folds), which is realistic for the Cleveland Heart Disease dataset.
