# Heart Disease Prediction

CE101 Team Project Challenge -- University of Essex, LABa01 Team 3 (2025-26)

Supervised by Dr Haider Raza

## About

A machine learning pipeline that predicts whether a patient has heart disease based on 13
clinical measurements. Built on the Cleveland Heart Disease dataset (302 samples) from the
UCI Machine Learning Repository.

Nine models were trained and compared. **K-Nearest Neighbours (GridSearchCV)** was selected
as the final model with **83.2% accuracy** and **0.911 ROC-AUC** -- the highest on both
metrics.

## Dataset

- **Source:** Cleveland Heart Disease Database (UCI ML Repository)
- **Samples:** 302 (1 duplicate removed from original 303)
- **Features:** 13 (age, sex, chest pain type, resting blood pressure, cholesterol, fasting
  blood sugar, resting ECG, max heart rate, exercise-induced angina, ST depression, slope,
  number of major vessels, thalassemia)
- **Target:** Binary (1 = heart disease, 0 = no heart disease)

## Final Model

```python
Pipeline([
    ("scaler", StandardScaler()),
    ("knn", KNeighborsClassifier(n_neighbors=16, weights="uniform", metric="manhattan")),
])
```

Selected over 8 other models (Logistic Regression, SVM, Random Forest, XGBoost, Voting
Ensemble, and others). Full analysis in [docs/final_model_analysis.md](docs/final_model_analysis.md).

## Results

| Model                  | Accuracy | ROC-AUC |
|------------------------|----------|---------|
| **K-NN (GridSearchCV)**| **83.2%**| **0.911** |
| LR (Robust+SMOTE)     | 83.1%    | ~0.896  |
| SVM (RBF)             | 82.9%    | ~0.905  |
| Soft Voting Ensemble   | 82.8%    | ~0.907  |
| Logistic Regression    | 82.7%    | ~0.905  |
| LR (feature select)   | 82.5%    | ~0.904  |
| K-NN (tuned k)        | 82.4%    | ~0.901  |
| XGBoost (Tuned)       | 82.2%    | ~0.891  |
| Random Forest          | 81.7%    | ~0.907  |

Evaluated using Repeated Stratified 5-Fold Cross-Validation (3 repeats, 15 folds total)
with sklearn Pipelines to prevent data leakage.

## Project Structure

```
Heart_Disease/
├── data/                   # Datasets (heart.xlsx, cleaned CSVs)
├── docs/                   # Analysis documents
│   ├── final_model_analysis.md
│   ├── data_leak_analysis.md
│   ├── model_issues_analysis.md
│   └── chart_explanations.md
├── results/                # Generated comparison charts
│   ├── comparison_by_model.png
│   ├── comparison_heatmap.png
│   ├── comparison_roc.png
│   └── comparison_confusion.png
├── model_comparison.py     # Unified comparison script (9 models x 4 datasets)
├── akagra/                 # Akagra -- SVC (RBF kernel)
├── alice/                  # Alice -- Logistic Regression, SVM
├── enes/                   # Enes -- K-NN (manual + GridSearchCV)
├── jim/                    # Jim -- Logistic Regression
├── mayukh/                 # Mayukh -- SVM (SVC)
├── nicholas/               # Nicholas -- Logistic Regression, Random Forest
├── sayok/                  # Sayok -- Logistic Regression, K-NN
└── toby/                   # Toby -- Final model notebook
```

## Team

| Member   | Model(s)                              |
|----------|---------------------------------------|
| Enes     | K-NN (manual tuning + GridSearchCV)   |
| Alice    | Logistic Regression, SVM              |
| Akagra   | SVM (SVC with RBF kernel)             |
| Jim      | Logistic Regression                   |
| Mayukh   | SVM (SVC)                             |
| Nicholas | Logistic Regression, Random Forest    |
| Sayok    | Logistic Regression, K-NN             |
| Toby     | SVM (RBF)                             |

## Requirements

- Python 3.10+
- scikit-learn
- pandas
- numpy
- matplotlib
- seaborn
- xgboost
- imbalanced-learn
