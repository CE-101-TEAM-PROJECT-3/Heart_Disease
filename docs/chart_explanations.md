# Chart Explanations

This document explains what each chart produced by `model_comparison.py` shows, how to read it, and what conclusions can be drawn.

---

## Figure 1: Grouped Bar Chart (`comparison_by_model.png`)

**What it shows:** Test accuracy (%) for each of the 6 models, with separate bars for each of the 4 datasets. Error bars represent the standard deviation across 15 folds (5-fold CV repeated 3 times).

**How to read it:**
- X-axis = model names. The best overall model is marked with `**` and bold text.
- Each colour represents a different dataset.
- The height of the bar = mean accuracy. The error bar = how much the accuracy varies across folds.
- Gold stars appear above the bars of the best-performing model.

**Key takeaway:** All models cluster tightly between 81-83%. The error bars overlap significantly, meaning the differences between models are **not statistically significant** on this dataset. SVM (RBF) has the highest mean accuracy overall but the advantage is marginal (~0.5% over the next best).

---

## Figure 2: Heatmap (`comparison_heatmap.png`)

**What it shows:** A matrix view of mean accuracy -- rows are models, columns are datasets. Each cell contains the accuracy value. Gold borders highlight the best model per dataset.

**How to read it:**
- Colour scale runs from red (60%) to green (100%). Darker green = higher accuracy.
- Bold text with a gold border = the best accuracy in that column (dataset).
- Reading across a row tells you how consistent a model is across different datasets.
- Reading down a column tells you which model performs best on a specific dataset.

**Key takeaway:** The heatmap reveals that Cleaned Data 1 (the team-cleaned CSV with 303 rows) produces slightly different results than the other three datasets. On Cleaned Data 1, Logistic Regression and SVM tie at 83.3%. On the other three datasets, K-NN (GridSearchCV) wins at 83.2%. This difference exists because Cleaned Data 1 has 303 rows (keeps the duplicate), while the others have 302.

---

## Figure 3: ROC Curves (`comparison_roc.png`)

**What it shows:** Receiver Operating Characteristic curves for all 6 models, with one subplot per dataset. The ROC curve plots the True Positive Rate against the False Positive Rate at every classification threshold.

**How to read it:**
- The diagonal dashed line represents a random classifier (AUC = 0.5).
- The further a curve bows toward the top-left corner, the better the model.
- AUC (Area Under the Curve) summarises performance in a single number: 1.0 = perfect, 0.5 = random.
- The best model per dataset has a thicker line and `*BEST*` in the legend.

**Key takeaway:** ROC-AUC tells a slightly different story than accuracy. K-NN (GridSearchCV) and Random Forest have the highest AUC (~0.91), even though Random Forest has the lowest accuracy. This means Random Forest is better at **ranking** patients by risk (separating high-risk from low-risk) even if its binary yes/no threshold is not optimal. AUC is a more robust metric than accuracy for medical datasets because it is not affected by the choice of classification threshold.

**Why AUC and accuracy can disagree:**
- Accuracy measures performance at a single threshold (default 0.5).
- AUC measures performance across all possible thresholds.
- A model can have high AUC but lower accuracy if the default threshold is not ideal for the class distribution.

---

## Figure 4: Confusion Matrices (`comparison_confusion.png`)

**What it shows:** A grid of 24 confusion matrices (4 datasets x 6 models). Each 2x2 matrix shows how many patients were correctly and incorrectly classified.

**How to read it:**
- Rows = datasets, Columns = models.
- Inside each matrix:
  - **Top-left (True Negative):** Correctly predicted "No Disease"
  - **Top-right (False Positive):** Predicted "Disease" but actually healthy
  - **Bottom-left (False Negative):** Predicted "No Disease" but actually has disease
  - **Bottom-right (True Positive):** Correctly predicted "Disease"
- The best model per dataset uses a warm colour scheme (yellow/red) with a gold border, while others use blue.

**Key takeaway:** The confusion matrices reveal that most models make similar types of errors. The main errors are:

- **False Positives (~34-41):** Healthy patients incorrectly flagged as having heart disease. This leads to unnecessary follow-up tests but is the "safer" error in a medical context.
- **False Negatives (~14-23):** Patients with heart disease missed by the model. This is the more dangerous error in healthcare.

K-NN (GridSearchCV) tends to have fewer false negatives (~16) compared to Random Forest (~23), making it a better choice when the priority is not missing any disease cases. However, it achieves this by having more false positives (~36 vs ~33).

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

1. **Best model by accuracy:** SVM (RBF) -- 83.3% on Cleaned Data 1, ~82.9% on other datasets.
2. **Best model by AUC:** K-NN (GridSearchCV) -- 0.911 AUC, meaning it ranks patients most effectively.
3. **All models are very close** (81-83% accuracy, 0.90-0.91 AUC). The Cleveland Heart Disease dataset is well-studied and ~83% is the expected ceiling for these classical ML algorithms on 13 features.
4. **Dataset choice has minimal impact.** Raw, Cleaned Data 2, and Cleaned Data 3 produce identical results (same data after deduplication, just different column names). Cleaned Data 1 differs slightly because it keeps the duplicate row.
5. **Feature selection (Nicholas's approach)** of dropping `fasting_blood_sugar` and `resting_ecg` does not improve accuracy, confirming those two features carry little predictive signal but also do not hurt.
