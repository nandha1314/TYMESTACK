import joblib
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

# Load data
data = pd.read_csv(r'C:\Users\harikarank\Desktop\tymestack\data\boston_house_prices (1).csv')
print("Columns in dataset:", data.columns)  # Print column names to verify

# Print the first few rows to ensure data is loaded correctly
print("First few rows of dataset:\n", data.head())

# Set the correct target column name (e.g., 'MEDV' for Boston housing data)
target_column = 'MEDV'.strip()  # Adjust target column name as necessary

if target_column not in data.columns:
    print(f"Error: Column '{target_column}' not found in dataset. Please check column names.")
else:
    # Prepare features and target
    X = data.drop(target_column, axis=1)  # Drop target column
    y = data[target_column]  # Target column

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train model
    model = RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42)
    model.fit(X_train, y_train)

    # Save model
    joblib.dump(model, 'models/new_model.joblib')
    print("Model training and saving successful.")



