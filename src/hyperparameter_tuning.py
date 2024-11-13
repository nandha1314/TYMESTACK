import joblib
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import mean_squared_error
import os

# Constants
DATA_PATH = r'C:\Users\harikarank\Desktop\tymestack\data\boston_house_prices (1).csv'
MODEL_SAVE_PATH = 'models/new_model.joblib'
PARAM_DIST = {
    'n_estimators': [50, 100, 150],
    'max_depth': [10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# Load dataset
try:
    data = pd.read_csv(DATA_PATH)
except FileNotFoundError:
    print(f"Data file not found at {DATA_PATH}. Please check the path.")
    raise

# Prepare features and target variable
try:
    X = data.drop('MEDV', axis=1)
    y = data['MEDV']
except KeyError:
    print("Target column 'MEDV' not found in dataset.")
    raise

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the model and hyperparameter search
model = RandomForestRegressor(random_state=42)
random_search = RandomizedSearchCV(
    estimator=model,
    param_distributions=PARAM_DIST,
    n_iter=10,
    scoring='neg_mean_squared_error',
    cv=3,
    verbose=2,
    random_state=42,
    n_jobs=-1
)

# Train model with hyperparameter tuning
print("Starting hyperparameter tuning...")
random_search.fit(X_train, y_train)
print("Hyperparameter tuning completed.")

# Save the best model
os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
joblib.dump(random_search.best_estimator_, MODEL_SAVE_PATH)
print(f"Best model saved at {MODEL_SAVE_PATH}.")

# Display best parameters and evaluate on test data
print("Best Parameters:", random_search.best_params_)
y_pred = random_search.best_estimator_.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error on Test Data:", mse)
