# Chart Explanations

This document explains what each chart produced by `model_comparison.py` shows, how to read it, and what conclusions can be drawn.

---

## Figure 1: Grouped Bar Chart (`comparison_by_model.png`)

**What it shows:** Test accuracy (%) for each of the 9 models, with separate bars for each of the 4 datasets. Error bars represent the standard deviation across 15 folds (5-fold CV repeated 3 times).

**How to read it:**
- X-axis = model names. The best overall model is marked with `**` and bold text.
- Each colour represents a different dataset.
- The height of the bar = mean accuracy. The error bar = how much the accuracy varies across folds.
- Gold stars appear above the bars of the best-performing model.

**Key takeaway:** All models cluster tightly between 81–83%. The error bars overlap significantly, meaning the differences between models are **not statistically significant** on this dataset. SVM (RBF) has the highest mean accuracy overall but the advantage is marginal (~0.4% over the next best). The ensemble models (XGBoost Tuned at ~82.2% and Soft Voting Ensemble at ~82.8%) perform competitively but do not surpass the simpler models. LR (Robust+SMOTE) at ~83.1% is the closest challenger to SVM, and notably has one of the tightest standard deviations (~0.034–0.043), meaning it generalizes the most consistently across folds.

---

## Figure 2: Heatmap (`comparison_heatmap.png`)

**What it shows:** A matrix view of mean accuracy -- rows are models, columns are datasets. Each cell contains the accuracy value. Gold borders highlight the best model per dataset.

**How to read it:**
- Colour scale runs from red (60%) to green (100%). Darker green = higher accuracy.
- Bold text with a gold border = the best accuracy in that column (dataset).
- Reading across a row tells you how consistent a model is across different datasets.
- Reading down a column tells you which model performs best on a specific dataset.

**Key takeaway:** The heatmap reveals that Cleaned Data 1 (the team-cleaned CSV with 303 rows) produces slightly different results than the other three datasets. On Cleaned Data 1, Logistic Regression and SVM (RBF) tie at 83.3%. On the other three datasets, K-NN (GridSearchCV) wins at 83.2%. This difference exists because Cleaned Data 1 has 303 rows (keeps the duplicate), while the others have 302. XGBoost (Tuned) is the weakest on Cleaned Data 1 at 81.7%. Soft Voting Ensemble is consistently mid-range at 82.8–82.9%. LR (Robust+SMOTE) sits at 83.1% on Raw/Cleaned Data 2/Cleaned Data 3 and 83.0% on Cleaned Data 1, making it one of the most consistent performers across all datasets.

---

## Figure 3: ROC Curves (`comparison_roc.png`)

**What it shows:** Receiver Operating Characteristic curves for all 9 models, with one subplot per dataset. The ROC curve plots the True Positive Rate against the False Positive Rate at every classification threshold.

**How to read it:**
- The diagonal dashed line represents a random classifier (AUC = 0.5).
- The further a curve bows toward the top-left corner, the better the model.
- AUC (Area Under the Curve) summarises performance in a single number: 1.0 = perfect, 0.5 = random.
- The best model per dataset has a thicker line and `*BEST*` in the legend.

**Key takeaway:** ROC-AUC tells a slightly different story than accuracy. Soft Voting Ensemble has the highest AUC (~0.912 on Raw, Cleaned Data 2, and Cleaned Data 3; ~0.907 on Cleaned Data 1), making it the best model at **ranking** patients by risk. K-NN (GridSearchCV) is close behind at ~0.911. Random Forest achieves ~0.907–0.909. XGBoost (Tuned) and LR (Robust+SMOTE) land at ~0.900 and ~0.901 respectively. SVM (RBF) has the lowest AUC (~0.896) despite having the highest accuracy, meaning its probability calibration is weaker than other models. LR (Robust+SMOTE) does not improve AUC over standard Logistic Regression (~0.901 vs ~0.901), confirming that SMOTE helps with class balance at the decision boundary but does not fundamentally change ranking ability. AUC is a more robust metric than accuracy for medical datasets because it is not affected by the choice of classification threshold.

**Why AUC and accuracy can disagree:**
- Accuracy measures performance at a single threshold (default 0.5).
- AUC measures performance across all possible thresholds.
- A model can have high AUC but lower accuracy if the default threshold is not ideal for the class distribution.

---

## Figure 4: Confusion Matrices (`comparison_confusion.png`)

**What it shows:** A grid of 36 confusion matrices (4 datasets x 9 models). Each 2x2 matrix shows how many patients were correctly and incorrectly classified.

**How to read it:**
- Rows = datasets, Columns = models.
- Inside each matrix:
  - **Top-left (True Negative):** Correctly predicted "No Disease"
  - **Top-right (False Positive):** Predicted "Disease" but actually healthy
  - **Bottom-left (False Negative):** Predicted "No Disease" but actually has disease
  - **Bottom-right (True Positive):** Correctly predicted "Disease"
- The best model per dataset uses a warm colour scheme (yellow/red) with a gold border, while others use blue.

**Key takeaway:** The confusion matrices reveal that most models make similar types of errors. The main errors are:

- **False Positives (~33-41):** Healthy patients incorrectly flagged as having heart disease. This leads to unnecessary follow-up tests but is the "safer" error in a medical context.
- **False Negatives (~14-24):** Patients with heart disease missed by the model. This is the more dangerous error in healthcare.

K-NN (GridSearchCV) tends to have fewer false negatives (~16) compared to Random Forest (~23) and XGBoost (~24), making it a better choice when the priority is not missing any disease cases. However, it achieves this by having more false positives (~36 vs ~33). Soft Voting Ensemble balances both error types well with ~35 false positives and ~19 false negatives. Logistic Regression and LR (feature select) also show a good balance at ~34-35 FP and ~15-16 FN. LR (Robust+SMOTE) produces a similar error profile to standard Logistic Regression, showing that SMOTE's synthetic oversampling does not dramatically shift the confusion matrix on this near-balanced dataset (target split is roughly 54%/46%).

---

## Which Chart Should You Use When?

| Question | Best Chart |
|----------|-----------|
| Which model has the highest accuracy? | Heatmap |
| Are the differences between models significant? | Bar chart (check if error bars overlap) |
| Which model is best at ranking patients by risk? | ROC curves (compare AUC) |
| What types of mistakes does each model make? | Confusion matrices |
| Does the dataset choice matter? | Heatmap (compare across columns) |

---

## Overall Conclusions

1. **Best model by accuracy:** SVM (RBF) -- 83.3% on Cleaned Data 1, ~82.9% on other datasets. LR (Robust+SMOTE) is the closest challenger at 83.1% on Raw/Cleaned Data 2/Cleaned Data 3 and 83.0% on Cleaned Data 1.
2. **Best model by AUC:** Soft Voting Ensemble -- 0.912 AUC, meaning it ranks patients most effectively by combining predictions from Logistic Regression, SVM, Random Forest, and Gradient Boosting. K-NN (GridSearchCV) is a close second at 0.911.
3. **All models are very close** (81–83% accuracy, 0.896–0.912 AUC). The Cleveland Heart Disease dataset is well-studied and ~83% is the expected ceiling for these classical ML algorithms on 13 features.
4. **Ensemble methods do not dramatically outperform simpler models.** XGBoost (Tuned) at ~82.2% and Soft Voting Ensemble at ~82.8% are competitive but do not beat SVM (RBF) on accuracy. However, Soft Voting Ensemble achieves the best AUC, showing its strength in probability estimation and risk ranking.
5. **Advanced preprocessing (RobustScaler + SMOTE) provides marginal gains.** LR (Robust+SMOTE) at ~83.1% nearly matches SVM (RBF) and has the tightest standard deviation of any model (~0.034 on Cleaned Data 1), making it the most stable predictor. However, it does not break the ~83% ceiling, confirming the dataset's information limit.
6. **Dataset choice has minimal impact.** Raw, Cleaned Data 2, and Cleaned Data 3 produce identical results (same data after deduplication, just different column names). Cleaned Data 1 differs slightly because it keeps the duplicate row.
7. **Feature selection (Nicholas's approach)** of dropping `fasting_blood_sugar` and `resting_ecg` does not improve accuracy, confirming those two features carry little predictive signal but also do not hurt.
