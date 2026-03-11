# Note of Commitment: Why We Compromise on Accuracy

**Project:** CE101 Team Project -- Heart Disease Prediction
**Team:** LABa01 Team 3, University of Essex (2025-26)
**Date:** March 2026

---

## The Decision

We selected **Logistic Regression** as our final model with **82.7% accuracy**, knowingly
passing over K-NN (GridSearchCV) which achieves **83.2% accuracy** -- the highest of all
nine models tested.

We made this decision deliberately and stand by it.

---

## Why We Accept Lower Accuracy

Our model is a **medical screening tool for heart disease**. In this context, not all errors
are equal.

There are two ways our model can be wrong:

1. **A healthy patient is flagged as having heart disease.**
   They undergo further tests, are cleared, and go home. Inconvenient, but safe.

2. **A patient with heart disease is told they are healthy.**
   They go home undiagnosed. This could cost them their life.

These two failure types are not comparable. One is a false alarm. The other is a missed
diagnosis.

---

## The Numbers Behind the Decision

| Model               | Accuracy | Patients Missed (FN) | Detection Rate |
|---------------------|----------|---------------------|----------------|
| K-NN (GridSearchCV) | 83.2%    | 18 out of 164       | 89.0%          |
| **Logistic Regression** | **82.7%** | **15 out of 164** | **90.9%**   |

Choosing K-NN (GridSearchCV) for its higher accuracy would mean **3 additional heart disease
patients sent home undiagnosed** for every 302 patients screened. That is not a trade-off we
are willing to make for a 0.5% improvement in accuracy.

Logistic Regression detects **90.9% of all heart disease cases** -- catching 3 more patients
than the "more accurate" model.

---

## Why the Accuracy Gap Does Not Matter

The 0.5% accuracy gap between Logistic Regression and K-NN (GridSearchCV) translates to
roughly 1-2 patients on our 302-sample dataset. This difference:

- Falls within the normal statistical noise of cross-validation on small data
- Would likely disappear with a different random seed or fold split
- Is smaller than the standard deviation of the accuracy scores across folds

Selecting a model based on a 0.5% accuracy advantage on 302 samples is not scientifically
sound. The safety difference of 3 fewer missed diagnoses is far more meaningful and consistent.

---

## Commitment

We commit to Logistic Regression as our production model because:

- It minimises the most dangerous failure mode in medical screening
- It is fully explainable -- every prediction can be traced back to specific patient features
- It trains only on real patient data with no synthetic augmentation
- It is robust, dependency-free, and simple to maintain
- The accuracy compromise is statistically insignificant but the safety gain is real

**In a tool designed to protect patients, we choose the model that protects the most patients
-- not the model with the highest number on a leaderboard.**

---

*Signed off by LABa01 Team 3*
