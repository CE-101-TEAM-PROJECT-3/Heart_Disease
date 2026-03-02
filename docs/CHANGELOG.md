# Changelog -- Since Last GitHub Update

**Base commit:** `b06a092` -- Repo reorganization, `model_comparison.py` with 6 baseline models

---

## Commit `812a784` -- "testing new model"

### Added
- `advanced_ensembles.py` -- new file exporting two advanced pipelines:
  - **XGBoost (Tuned):** `StandardScaler` + `XGBClassifier` with heavy regularization (`max_depth=3`, `learning_rate=0.05`, `reg_lambda=10`) to prevent overfitting on 302 rows.
  - **Soft Voting Ensemble:** `StandardScaler` + `VotingClassifier(voting='soft')` combining Logistic Regression, SVM, Random Forest (`max_depth=3`), and Gradient Boosting (`max_depth=3`).

### Changed
- `model_comparison.py` -- imported and injected both new pipelines into the evaluation roster. Model count went from 6 to 8. No changes to CV logic, datasets, or plotting functions.
- `results/` -- all 4 charts regenerated with 8 models (bar chart, heatmap, ROC curves, confusion matrices).

### Results
- XGBoost (Tuned): ~82.2% accuracy, 0.900 AUC
- Soft Voting Ensemble: ~82.8% accuracy, 0.912 AUC (highest AUC of all models)
- Neither broke the ~83% accuracy ceiling

---

## Uncommitted changes (current working state)

### Added
- **LR (Robust+SMOTE)** pipeline in `advanced_ensembles.py`:
  - Uses `imblearn.pipeline.Pipeline` (not sklearn) so SMOTE runs safely inside each CV fold without data leakage.
  - Steps: `RobustScaler` -> `SMOTE(random_state=42)` -> `LogisticRegression(max_iter=1000)`.
  - Installed `imbalanced-learn` dependency.

### Changed
- `model_comparison.py` -- imported and injected `ultimate_pipeline` as "LR (Robust+SMOTE)". Model count went from 8 to 9.
- `docs/chart_explanations.md` -- updated all sections to reflect 9 models, added LR (Robust+SMOTE) observations and new conclusion about advanced preprocessing.
- `results/` -- all 4 charts regenerated with 9 models.

### Results
- LR (Robust+SMOTE): ~83.1% accuracy (Raw/Cleaned 2/Cleaned 3), ~83.0% (Cleaned Data 1)
- Tightest standard deviation of any model (~0.034 on Cleaned Data 1) -- most stable predictor
- AUC of ~0.901 -- comparable to standard Logistic Regression, confirming SMOTE does not improve ranking on this near-balanced dataset
- Still does not break the ~83% ceiling

---

## Current Model Roster (9 models)

| Model | Accuracy | AUC | Notes |
|-------|----------|-----|-------|
| SVM (RBF) | **83.3%** | 0.896 | Best accuracy (Cleaned Data 1) |
| K-NN (GridSearchCV) | 83.2% | 0.911 | Best accuracy (302-row datasets) |
| LR (Robust+SMOTE) | 83.1% | 0.901 | Most stable (lowest std) |
| Logistic Regression | 82.7% | 0.901 | Solid baseline |
| SVM (RBF) | 82.9% | 0.896 | On 302-row datasets |
| Soft Voting Ensemble | 82.8% | **0.912** | Best AUC (risk ranking) |
| LR (feature select) | 82.5% | 0.906 | Dropping fbs + restecg |
| K-NN (tuned k) | 82.4% | 0.899 | Manual k selection |
| XGBoost (Tuned) | 82.2% | 0.900 | Heavily regularized |
| Random Forest | 81.7% | 0.909 | Unrestricted depth |

---

## Why We Built These Models (Team Context)

We hit a hard technical wall at ~83% accuracy with our baseline SVM and K-NN models. We needed to determine whether that ceiling was a limitation of our implementation or the mathematical limit of this 303-row, 13-feature dataset.

So we stress-tested the data with advanced ensemble methods and preprocessing to see what would happen. Here is what we found:

**The advanced models did not outperform the baselines clinically.** XGBoost actually performed worse where it matters most -- it missed ~24 patients with heart disease (False Negatives) compared to K-NN (GridSearchCV) which only missed ~16. In a hospital setting, missing a sick patient is the most dangerous error a model can make.

**The Soft Voting Ensemble gave us the best overall risk-ranking score** (0.912 AUC), but it still missed ~19 patients with heart disease. Better at ranking risk, worse at catching every case.

**LR (Robust+SMOTE) matched the accuracy ceiling** at ~83.1% with the most stable predictions across folds, but SMOTE had minimal effect because the dataset is already near balanced (54%/46% target split). The RobustScaler helped slightly with outlier handling but did not fundamentally shift the results.

**The ~83% ceiling is real.** All 9 models -- from a basic K-NN to a 4-model voting ensemble -- land between 81% and 83.3%. This is the known performance ceiling for classical ML on the Cleveland Heart Disease dataset with 13 features, consistent with published research.

### What this means for the presentation

We are not presenting these complex models as our final product. We are using them as evidence. When asked why we are recommending a "basic" K-NN or SVM, we can show these charts and explain:

1. We engineered pipelines (XGBoost, ensemble voting, SMOTE oversampling) and ran them through the same rigorous evaluation as our baselines.
2. We deliberately chose not to use the more complex models because they put more patients at risk through higher False Negative rates.
3. We proved that ~83% is the dataset's information limit, not a failure of our approach.
4. We prioritised clinical safety over chasing a higher number.

This demonstrates that we understand the real world medical implications of model selection, not just the technical implementation.
