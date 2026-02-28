# Created by Alice

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from sklearn.svm import SVC


df = pd.read_csv("/content/drive/MyDrive/CE101 Team Project/cleaned_heart_data.csv")

X = df.drop('target', axis=1)
y = df['target']

# Step 1: Train
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y,
    test_size=0.4,
    random_state=42,
    stratify=y
)

# Step 2: Validation
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp,
    test_size=0.5,
    random_state=42,
    stratify=y_temp
)

print("Train:", X_train.shape)
print("Validation:", X_val.shape)
print("Test:", X_test.shape)

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

lr = LogisticRegression(max_iter=1000, random_state=42)
lr.fit(X_train_scaled, y_train)

y_val_pred_lr = lr.predict(X_val_scaled)

print("Logistic Regression – Validation Accuracy:",
      accuracy_score(y_val, y_val_pred_lr))
print(classification_report(y_val, y_val_pred_lr))

svc = SVC(kernel='rbf', random_state=42)
svc.fit(X_train_scaled, y_train)

y_val_pred_svc = svc.predict(X_val_scaled)

print("SVC – Validation Accuracy:",
      accuracy_score(y_val, y_val_pred_svc))
print(classification_report(y_val, y_val_pred_svc))
