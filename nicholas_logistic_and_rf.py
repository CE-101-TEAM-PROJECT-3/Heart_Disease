import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report


df = pd.read_csv('health_data.csv')

df = df.rename(columns = {
    'cp': 'chest_pain_type',
    'trestbps': 'resting_blood_pressure_mm_hg',
    'chol': 'serum_cholestoral_mg_per_dl',
    'fbs': 'fasting_blood_sugar',
    'restecg': 'resting_ecg',
    'thalach': 'max_heart_rate',
    'exang': 'exercise_induced_angina',
    'ca': 'num_major_fluorosopic_vessels'
})

x = df.drop(columns = ['target'])
y = df['target']

# Split the data for training and testing, usually preserving 20-30% (default 0.25) of the data for testing.
# The random_state argument ensures that subsequent calls to this method produce constant output.
# x refers to the features (e.g., age, sex, etc.) and y refers to the target.
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 5)

model = LogisticRegression(max_iter = 1000)
model.fit(x_train, y_train)

y_pred = model.predict(x_test)

print(accuracy_score(y_test, y_pred))
# https://www.nb-data.com/p/breaking-down-the-classification
print(classification_report(y_test, y_pred))

model = SVC()
model.fit(x_train, y_train)

y_pred = model.predict(x_test)

print(accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))
