# Final Model Analysis: Why K-NN (GridSearchCV)

## Overview

Nine machine learning models were trained and evaluated on the Cleveland Heart Disease dataset
(302 samples, 13 features, binary target) using **Repeated Stratified 5-Fold Cross-Validation
(3 repeats = 15 folds)** with sklearn Pipelines to prevent data leakage. All models include
StandardScaler inside the pipeline.

This document analyses every model and explains why **K-NN with GridSearchCV** was selected as
the final production model.

---

## Results Summary

| Rank | Model                  | Accuracy (%) | ROC-AUC | Notes                        |
|------|------------------------|-------------|---------|------------------------------|
| 1    | **K-NN (GridSearchCV)**| **83.2**    | **0.911** | Best accuracy + best AUC   |
| 2    | LR (Robust+SMOTE)     | 83.1        | ~0.896  | Uses synthetic data (SMOTE)  |
| 3    | SVM (RBF)             | 82.9        | ~0.905  | Black-box kernel             |
| 4    | Soft Voting Ensemble   | 82.8        | ~0.907  | 4 models combined            |
| 5    | Logistic Regression    | 82.7        | ~0.905  | Linear baseline              |
| 6    | LR (feature select)   | 82.5        | ~0.904  | Drops fbs + resting_ecg      |
| 7    | K-NN (tuned k)        | 82.4        | ~0.901  | Manual k search only         |
| 8    | XGBoost (Tuned)       | 82.2        | ~0.891  | Gradient boosting            |
| 9    | Random Forest          | 81.7        | ~0.907  | Bagging ensemble             |

*Accuracy values from the cleaned dataset (Cleaned Data 2). ROC-AUC values from the ROC curve
analysis. All evaluated via Repeated Stratified 5-Fold CV (15 folds).*

---

## Individual Model Analysis

### 1. K-NN (GridSearchCV) -- SELECTED

- **Accuracy:** 83.2%
- **ROC-AUC:** 0.911 (highest of all models)
- **Best parameters:** k=16, manhattan distance, uniform weights
- **Pipeline:** StandardScaler -> KNeighborsClassifier

GridSearchCV explored 180 parameter combinations (k=1-20, 2 weight schemes, 3 distance
metrics) across 5 folds, totalling 900 fits. The resulting model is rigorously tuned with
no manual guesswork.

**Strengths:**
- Highest accuracy (83.2%) and highest ROC-AUC (0.911) across all models
- No assumptions about data distribution -- purely distance-based
- Systematically optimised via exhaustive grid search
- Clean pipeline with no extra dependencies
- Consistent across all 4 dataset variants (83.2% on 3 of 4 datasets)

**Weaknesses:**
- Stores entire training set in memory (negligible for 302 samples)
- Slower inference than parametric models (negligible for 302 samples)

---

### 2. LR (Robust+SMOTE) -- Runner-up

- **Accuracy:** 83.1%
- **Pipeline:** RobustScaler -> SMOTE -> LogisticRegression

Only 0.1% behind K-NN, but **rejected** because:
- **SMOTE generates synthetic training samples** by interpolating between real patients. On a
  302-row medical dataset, this means the model is partially trained on fabricated data
- The class imbalance is mild (165 vs 137, ratio 1.2:1) -- SMOTE is not justified
- Requires the `imbalanced-learn` library as an extra dependency
- Harder to justify in a medical context: "we invented fake patient records to train on"

---

### 3. SVM (RBF)

- **Accuracy:** 82.9%
- **ROC-AUC:** ~0.905
- **Pipeline:** StandardScaler -> SVC(kernel=rbf, C=1.0, gamma=scale)

**Why not selected:**
- 0.3% lower accuracy than K-NN (GridSearchCV)
- Lower ROC-AUC (0.905 vs 0.911) -- worse at ranking patients by risk
- Black-box model: the RBF kernel maps data to infinite-dimensional space, making it
  impossible to explain which features drive a specific prediction
- Probability estimates use Platt scaling (a post-hoc sigmoid fit), which is unreliable on
  small datasets

---

### 4. Soft Voting Ensemble

- **Accuracy:** 82.8%
- **ROC-AUC:** ~0.907
- **Pipeline:** StandardScaler -> VotingClassifier(LR + SVC + RF + GradientBoosting)

**Why not selected:**
- Lower accuracy than K-NN despite combining 4 models
- Over-engineered for a 302-sample, 13-feature dataset
- Each sub-model adds complexity, training time, and potential failure points
- Harder to explain: "our model is actually 4 models voting" is not a clean story
- The ensemble failed to beat simpler models, suggesting the dataset is too small to benefit
  from model diversity

---

### 5. Logistic Regression

- **Accuracy:** 82.7%
- **ROC-AUC:** ~0.905
- **Pipeline:** StandardScaler -> LogisticRegression(max_iter=1000)

**Why not selected:**
- 0.5% lower accuracy
- Lower AUC (0.905 vs 0.911)
- While interpretable, it assumes a linear decision boundary. Heart disease prediction may
  involve non-linear feature interactions that LR cannot capture
- Good baseline, but K-NN simply performs better on this dataset

---

### 6. LR (Feature Selection)

- **Accuracy:** 82.5%
- **Pipeline:** DropColumns(fbs, resting_ecg) -> StandardScaler -> LogisticRegression

**Why not selected:**
- Dropping features did not help -- accuracy decreased from 82.7% to 82.5%
- Removing features discards potentially useful information
- The hypothesis that fasting_blood_sugar and resting_ecg are noise was not supported by
  the results

---

### 7. K-NN (Tuned k)

- **Accuracy:** 82.4%
- **Pipeline:** StandardScaler -> KNeighborsClassifier(k=8)

**Why not selected:**
- Only tunes k (number of neighbours), leaving weights and distance metric at defaults
  (uniform, minkowski/euclidean)
- GridSearchCV version gains 0.8% by also optimising weights and distance metric
- This model demonstrates that the full grid search was worth the effort

---

### 8. XGBoost (Tuned)

- **Accuracy:** 82.2%
- **ROC-AUC:** ~0.891
- **Pipeline:** StandardScaler -> XGBClassifier(n_estimators=100, max_depth=3, lr=0.05,
  reg_lambda=10)

**Why not selected:**
- 1.0% lower accuracy than K-NN
- Lowest ROC-AUC among all models (~0.891)
- Heavy regularisation (lambda=10) was needed to prevent overfitting, confirming the dataset
  is too small for gradient boosting to shine
- Requires the `xgboost` library as an extra dependency
- Over-engineered for the problem size

---

### 9. Random Forest

- **Accuracy:** 81.7%
- **ROC-AUC:** ~0.907
- **Pipeline:** StandardScaler -> RandomForestClassifier(n_estimators=100)

**Why not selected:**
- Lowest accuracy of all models (81.7%)
- 100 decision trees is excessive for 302 samples and 13 features
- Prone to overfitting on small datasets despite bagging
- High variance between folds compared to simpler models

---

## Why K-NN (GridSearchCV) Is the Best Choice

### 1. It has the best numbers

K-NN (GridSearchCV) achieves the **highest accuracy (83.2%)** and the **highest ROC-AUC
(0.911)** of all 9 models. In a medical screening context, ROC-AUC is particularly important
because it measures how well the model ranks patients from low-risk to high-risk across all
possible thresholds.

### 2. It is consistent

K-NN (GridSearchCV) scored 83.2% on 3 out of 4 dataset variants and 82.3% on the fourth.
This consistency across differently cleaned versions of the data shows the model is robust
and not overfitting to a specific preprocessing choice.

### 3. It makes no assumptions about the data

Unlike Logistic Regression (assumes linear boundaries) or SVM (maps to kernel space), K-NN
is a non-parametric, instance-based learner. It makes predictions based on the actual
similarity between patients. This is intuitive in a medical context: "patients with similar
measurements tend to have similar outcomes."

### 4. It is rigorously optimised

GridSearchCV tested 180 hyperparameter combinations across 5 folds (900 total fits) to find
the optimal configuration: **k=16 neighbours, manhattan distance, uniform weights**. This
exhaustive search ensures the model is not under-tuned or relying on lucky defaults.

### 5. It is simple and dependency-free

The final model is a standard sklearn Pipeline with just two steps (StandardScaler +
KNeighborsClassifier). No extra libraries (no xgboost, no imbalanced-learn), no synthetic
data, no ensemble complexity. It can be deployed with scikit-learn alone.

### 6. It does not fabricate data

Unlike the SMOTE-based model (the only close competitor at 83.1%), K-NN trains exclusively on
real patient data. In a medical application, training on synthetic data is a liability that is
difficult to justify to stakeholders.

---

## Final Model Configuration

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier

final_model = Pipeline([
    ("scaler", StandardScaler()),
    ("knn", KNeighborsClassifier(
        n_neighbors=16,
        weights="uniform",
        metric="manhattan",
    )),
])
```

---

## Evaluation Methodology

- **Cross-validation:** Repeated Stratified 5-Fold CV with 3 repeats (15 total folds)
- **Stratification:** Preserves class distribution (165 disease / 137 no disease) in every fold
- **Pipeline:** Scaler is fit inside each fold to prevent data leakage
- **Metrics:** Accuracy, Precision (macro), Recall (macro), F1 (macro), ROC-AUC
- **Datasets tested:** 4 variants (Raw, Cleaned Data 1, Cleaned Data 2, Cleaned Data 3)
- **Models compared:** 9 (K-NN x2, LR x3, SVM, Random Forest, XGBoost, Voting Ensemble)

---

## Conclusion

All 9 models cluster tightly between 81.7% and 83.2% accuracy, which is expected for
classical ML on the Cleveland Heart Disease dataset (a well-studied benchmark with a known
accuracy ceiling around 83-85%). Within this narrow range, **K-NN (GridSearchCV) leads on
both accuracy and ROC-AUC**, is consistent across dataset variants, uses no synthetic data,
requires no extra dependencies, and produces an intuitive similarity-based prediction that
aligns well with medical reasoning.
