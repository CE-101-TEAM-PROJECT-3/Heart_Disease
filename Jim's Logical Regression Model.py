import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

data = pd.read_csv("heart.csv")

X = data.drop("target", axis=1)
y = data["target"]
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)
y_val_pred = model.predict(X_val)
accuracy = accuracy_score(y_val, y_val_pred)

print("Validation Accuracy:", accuracy)
