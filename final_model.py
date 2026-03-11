import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix

RANDOM_STATE = 42
DATA_PATH = "data/enes_final_cleaned_data.csv"

data = pd.read_csv(DATA_PATH)
X = data.drop(columns=["target"])
y = data["target"]

model = Pipeline([
    ("scaler", StandardScaler()),
    ("lr", LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)),
])

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
y_pred = cross_val_predict(model, X, y, cv=cv)

cm = confusion_matrix(y, y_pred)
tn, fp, fn, tp = cm.ravel()

print("=" * 50)
print("FINAL MODEL: Logistic Regression")
print("=" * 50)
print()
print("Confusion Matrix:")
print(f"                    Predicted Healthy    Predicted Disease")
print(f"Actually Healthy:        {tn}                  {fp}")
print(f"Actually Disease:         {fn}                 {tp}")
print()
print(f"Total patients: {len(y)}")
print(f"Correctly classified: {tn + tp}/{len(y)} ({(tn + tp) / len(y) * 100:.1f}%)")
print(f"Misclassified:        {fp + fn}/{len(y)} ({(fp + fn) / len(y) * 100:.1f}%)")
print()
print(f"Disease missed (FN):     {fn}/{fn + tp} ({fn / (fn + tp) * 100:.1f}%)")
print(f"False alarms (FP):       {fp}/{tn + fp} ({fp / (tn + fp) * 100:.1f}%)")
print(f"Disease detection rate:  {tp / (fn + tp) * 100:.1f}%")
print()
print(classification_report(y, y_pred, target_names=["No Disease", "Disease"]))

model.fit(X, y)
print("Model trained on full dataset and ready for predictions.")
