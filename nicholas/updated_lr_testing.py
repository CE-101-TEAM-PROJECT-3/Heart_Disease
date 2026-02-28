from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, cross_val_score, KFold

import numpy as np
import pandas as pd


df = pd.read_csv('drive/MyDrive/health_data.csv')

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

x = df.drop(columns = ['target', 'fasting_blood_sugar', 'resting_ecg'])
y = df['target']

x_dev, x_test, y_dev, y_test = train_test_split(x, y, test_size = 0.2, random_state = 5)

pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('log_reg', LogisticRegression())
])

kf = KFold(n_splits = 10, shuffle = True, random_state = 5)

fold_accuracies = cross_val_score(pipeline, x_dev, y_dev, cv = kf)

print(f"mean cv accuracy: {np.mean(fold_accuracies):.4f}")

pipeline.fit(x_dev, y_dev)

final_score = pipeline.score(x_test, y_test)

print(f"final accuracy: {final_score}")
