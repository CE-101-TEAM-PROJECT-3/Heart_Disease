import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

print("Loading dataset...")
data = pd.read_csv('enes_final_cleaned_data.csv')

X = data.drop(columns=['target'])
y = data['target']

print("Splitting data...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print("Setting up pipeline and parameters...")
# Pipeline: scaling + KNN
pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("knn", KNeighborsClassifier())
])

# Define the parameter grid for KNN
param_grid = {
    'knn__n_neighbors': list(range(1, 31)),
    'knn__weights': ['uniform', 'distance'],
    'knn__metric': ['euclidean', 'manhattan', 'minkowski']
}

print("Starting Grid Search...")
# Initialize GridSearchCV
grid_search = GridSearchCV(
    estimator=pipeline,
    param_grid=param_grid,
    cv=5,               # 5 fold cross validation
    scoring='accuracy', # Optimize for accuracy
    verbose=1,
    n_jobs=-1           # Use all available CPU cores
)

# Train using Grid Search
grid_search.fit(X_train, y_train)

# Output best parameters
print("\n" + "="*50)
print("Best Parameters Found:")
print(grid_search.best_params_)
print("Best Cross-Validation Accuracy: {:.4f}".format(grid_search.best_score_))
print("="*50 + "\n")

# Predict using the best model found
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)

print("EVALUATION ON TEST SET")
print(f"Test Set Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
