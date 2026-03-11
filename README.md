# Heart Disease Prediction

CE101 Team Project Challenge -- University of Essex, LABa01 Team 3 (2025-26)

Supervised by Dr Haider Raza

## About

A machine learning pipeline that predicts whether a patient has heart disease based on 13
clinical measurements. Built on the Cleveland Heart Disease dataset (302 samples) from the
UCI Machine Learning Repository.

Nine models were trained and compared. **Logistic Regression** was selected as the final
model -- it detects **90.9% of heart disease cases** with the lowest overall failure rate
(16.2%) among the safest models. Full analysis in
[docs/final_model_analysis.md](docs/final_model_analysis.md).

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
    ("lr", LogisticRegression(max_iter=1000, random_state=42)),
])
```

Selected over 8 other models because it provides the best balance of safety (fewest missed
disease cases) and accuracy, while being fully explainable and dependency-free.

## Results

| Model                        | Accuracy | Disease Missed (FN) | Detection Rate |
|------------------------------|----------|--------------------:|---------------:|
| K-NN (GridSearchCV)          | 83.2%    | 18/164 (11.0%)      | 89.0%          |
| LR (Robust+SMOTE)           | 83.1%    | 15/164 (9.1%)       | 90.9%          |
| SVM (RBF)                   | 82.9%    | 19/164 (11.6%)      | 88.4%          |
| Soft Voting Ensemble         | 82.8%    | 19/164 (11.6%)      | 88.4%          |
| **Logistic Regression**     | **82.7%**| **15/164 (9.1%)**   | **90.9%**      |
| LR (feature select)         | 82.5%    | 16/164 (9.8%)       | 90.2%          |
| K-NN (tuned k)              | 82.4%    | 14/164 (8.5%)       | 91.5%          |
| XGBoost (Tuned)             | 82.2%    | 24/164 (14.6%)      | 85.4%          |
| Random Forest                | 81.7%    | 23/164 (14.0%)      | 86.0%          |

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
