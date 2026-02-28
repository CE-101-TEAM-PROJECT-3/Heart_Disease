# Model Issues Analysis: All Team Members

## Overview

Every script and notebook in the project was reviewed for data leaks, methodology issues, and bugs. Alice's `Unnamed: 0` data leak is covered separately in `data_leak_analysis.md`. This document covers issues found in all other files.

---

## Summary Table

| File | Author | Issue | Severity |
|------|--------|-------|----------|
| `svc_model_evalution2.ipynb` / `(1).ipynb` | (SVC eval) | Scaler fit before GridSearchCV without Pipeline | Low-Medium |
| `akagra_svc.py` | Akagra | `cross_val_score` run on full dataset instead of training set | Low-Medium |
| `Jim's Logistic Regression Model.py` | Jim | No scaling, no duplicate handling, incomplete evaluation | Low |
| `Sayok_Logisctic_&_K_NN_Model (1).ipynb` | Sayok | Bug: wrong model object reused, reports incorrect 56.5% accuracy | Bug |
| `enes_knn_model.py` | Enes | Scaler fit before CV loop (minor) | Low |
| `ce101_cleaned_data_code.py` | (Team) | Saves CSV without `index=False` -- root cause of Alice's leak | Root cause |

### Clean Files (no issues found)

- `enes_grid_search.py` -- Pipeline used correctly inside GridSearchCV
- `enes_knn_and_grid_search.ipynb` -- Pipeline used correctly, best practice
- `Toby-final-model.ipynb` -- Pipelines throughout, duplicates handled, feature engineering excluded properly
- `Mayukh Support Vector.py` -- Correct scaling after split
- `Mayukhheart.ipynb` -- Duplicates handled, clean scaling, saved output with `index=False`
- `nicholas_updated_lr_testing.py` -- Pipeline used correctly inside cross-validation
- `updated_lr_testing.py` -- Same as above (identical logic)

---

## Detailed Findings

---

### 1. `svc_model_evalution2.ipynb` / `svc_model_evalution2 (1).ipynb`

**Issue: Scaler fit before GridSearchCV without a Pipeline**

The code fits `StandardScaler` on `X_train` first, transforms it into a numpy array, then passes the already-scaled array into `GridSearchCV(SVC(), ...)`:

```python
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# GridSearchCV receives pre-scaled data
grid = GridSearchCV(SVC(), param_grid, cv=3)
grid.fit(X_train, y_train)
```

**Why this is a problem:** During GridSearchCV's 3-fold cross-validation, the scaler's mean and standard deviation were computed using all of `X_train`, including the rows that become the validation fold in each CV split. The validation fold was "seen" by the scaler during fitting. This makes the CV score mildly over-optimistic.

**The correct approach:**

```python
from sklearn.pipeline import Pipeline

pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("svc", SVC())
])
grid = GridSearchCV(pipeline, param_grid, cv=3)
grid.fit(X_train, y_train)
```

This way the scaler is re-fit on only the training portion of each CV fold.

**Impact:** The held-out test set evaluation is not affected (scaler was fit on `X_train` only). Only the CV accuracy during hyperparameter tuning is mildly inflated. In practice, the difference is small for this dataset.

---

### 2. `akagra_svc.py`

**Issue: Cross-validation run on the full dataset instead of the training set**

```python
X_train, X_test, y_train, y_test = train_test_split(X, y, ...)

# ... model is trained on X_train ...

# But CV is run on the FULL X (includes test rows)
cv_scores = cross_val_score(pipeline, X, y, cv=5, scoring="accuracy")
```

**Why this is a problem:** The purpose of holding out a test set is to get an unbiased estimate of model performance on unseen data. Running `cross_val_score` on the full `X` means some CV folds will train on test set rows and validate on other test set rows. The CV score is no longer an independent estimate -- it overlaps with the test set.

**The correct approach:**

```python
cv_scores = cross_val_score(pipeline, X_train, y_train, cv=5, scoring="accuracy")
```

**Impact:** The reported CV accuracy is unreliable as a generalisation estimate. The separate test set evaluation (`pipeline.score(X_test, y_test)`) is still valid since the pipeline was fit on `X_train` only.

---

### 3. `Jim's Logistic Regression Model.py`

**Issue 1: No feature scaling**

Logistic Regression is trained directly on raw feature values without `StandardScaler`. Features like `cholesterol` (100-500 range) and `fasting_blood_sugar` (0 or 1) are on very different scales, which affects convergence and coefficient magnitudes.

**Issue 2: No duplicate handling**

The raw `heart.csv` has 303 rows including 1 known duplicate. The duplicate is not removed before training.

**Issue 3: No stratified split**

```python
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
```

Missing `stratify=y`. With a small dataset (303 rows), this can cause class imbalance in the splits.

**Issue 4: Incomplete evaluation**

The test set (`X_test`, `y_test`) is created but never used for prediction. Only validation accuracy is reported.

---

### 4. `Sayok_Logisctic_&_K_NN_Model (1).ipynb`

**Issue: Bug in cell 28 -- wrong model object reused**

Cell 28 reports a test accuracy of **56.5%** with this warning:

```
UserWarning: X has feature names, but LogisticRegression was fitted
without feature names
```

This happens because `lr_model` (from an earlier cell, fit on numpy arrays from the first `train_test_split`) is accidentally called again on `X_test` (a pandas DataFrame from a different split context). The model and test set don't match.

This is not a data leak -- it's a code bug that produces a meaningless accuracy number. The correctly evaluated results in other cells (LR: 76%, KNN GridSearch: 80.43%) are fine.

**Note:** The rest of the notebook is well-structured. Duplicates are handled (`df.drop_duplicates()`), Pipelines are used for GridSearch, and the `age_group` EDA column is properly excluded from features.

---

### 5. `enes_knn_model.py`

**Issue: Scaler fit before cross-validation loop (minor)**

```python
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# CV runs on pre-scaled X_train_scaled
for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, X_train_scaled, y_train, cv=5)
```

The scaler is fit on `X_train` before the CV loop, so CV validation folds were "seen" by the scaler. This is the same class of issue as the SVC notebooks above, but less severe because:

- The test set is not contaminated (scaler fit on `X_train` only)
- The effect on KNN is minimal compared to parametric models

**Note:** `enes_grid_search.py` and `enes_knn_and_grid_search.ipynb` fix this by using a Pipeline inside GridSearchCV.

---

### 6. `ce101_cleaned_data_code.py`

**Issue: Root cause of the `Unnamed: 0` leak**

This data-cleaning script saves the output CSV without `index=False`:

```python
data.to_csv(filename)  # writes pandas row index as first column
```

The saved CSV gets an `Unnamed: 0` column containing values 0, 1, 2, ..., 302. Every file that reads this CSV without dropping that column inherits the leak (Alice's notebooks).

Additionally, `drop_duplicates()` is never called, so the known duplicate row from the raw data is preserved in the output.

**The fix:**

```python
data = data.drop_duplicates()
data.to_csv(filename, index=False)
```
