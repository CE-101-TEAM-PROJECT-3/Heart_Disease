#Support Vector Machine (SVC)
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score , classification report , confusion_matrix
# Load Dataset
data = pd.read_csv("heart.csv")
X = data.drop("target" , axis=1)
y = data["target"]

# Split the data set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_State=42,stratify=y)
# Scale Features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train and predict
svc = SVC(kernel='rbf')
svc.fit(X_train_scaled, y_train)
y_pred_svc = svc.predict(X_test_scaled)

print("SVC - Answer")
print("Validation:",accuracy_score(y_test, y_pred))
print("Classification report:\n", classification_report(y_test, y_pred_svc))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_svc))
