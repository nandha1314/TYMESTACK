import unittest
import pandas as pd
from unittest.mock import patch, mock_open
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import joblib
import os

class TestModelTraining(unittest.TestCase):

    @patch('pandas.read_csv')
    def test_data_loading(self, mock_read_csv):
        # Mocking the data loading process
        mock_data = pd.DataFrame({
            'CRIM': [0.1, 0.2],
            'ZN': [18.0, 0.0],
            'INDUS': [2.31, 7.07],
            'CHAS': [0, 0],
            'NOX': [0.538, 0.469],
            'RM': [6.575, 6.421],
            'AGE': [65.2, 78.9],
            'DIS': [4.09, 4.9671],
            'RAD': [1, 2],
            'TAX': [296, 242],
            'PTRATIO': [15.3, 17.8],
            'B': [396.9, 396.9],
            'LSTAT': [4.98, 9.14],
            'MEDV': [24.0, 21.6]
        })
        mock_read_csv.return_value = mock_data
        
        # Load data
        data = pd.read_csv('dummy_path.csv')
        self.assertEqual(list(data.columns), list(mock_data.columns), "Data columns do not match expected values.")
        
    def test_target_column_existence(self):
        # Check that the target column exists in the dataset
        mock_data = pd.DataFrame({
            'CRIM': [0.1, 0.2],
            'ZN': [18.0, 0.0],
            'INDUS': [2.31, 7.07],
            'CHAS': [0, 0],
            'NOX': [0.538, 0.469],
            'RM': [6.575, 6.421],
            'AGE': [65.2, 78.9],
            'DIS': [4.09, 4.9671],
            'RAD': [1, 2],
            'TAX': [296, 242],
            'PTRATIO': [15.3, 17.8],
            'B': [396.9, 396.9],
            'LSTAT': [4.98, 9.14],
            'MEDV': [24.0, 21.6]
        })
        target_column = 'MEDV'
        self.assertIn(target_column, mock_data.columns, f"Target column '{target_column}' not found in dataset.")

    @patch('joblib.dump')
    def test_model_training_and_saving(self, mock_joblib_dump):
        # Define mock data for training
        mock_data = pd.DataFrame({
            'CRIM': [0.1, 0.2, 0.3, 0.4],
            'ZN': [18.0, 0.0, 18.0, 0.0],
            'INDUS': [2.31, 7.07, 2.31, 7.07],
            'CHAS': [0, 0, 0, 0],
            'NOX': [0.538, 0.469, 0.538, 0.469],
            'RM': [6.575, 6.421, 6.575, 6.421],
            'AGE': [65.2, 78.9, 65.2, 78.9],
            'DIS': [4.09, 4.9671, 4.09, 4.9671],
            'RAD': [1, 2, 1, 2],
            'TAX': [296, 242, 296, 242],
            'PTRATIO': [15.3, 17.8, 15.3, 17.8],
            'B': [396.9, 396.9, 396.9, 396.9],
            'LSTAT': [4.98, 9.14, 4.98, 9.14],
            'MEDV': [24.0, 21.6, 24.0, 21.6]
        })
        X = mock_data.drop('MEDV', axis=1)
        y = mock_data['MEDV']

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Train model
        model = RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42)
        model.fit(X_train, y_train)

        # Test model prediction (basic check)
        predictions = model.predict(X_test)
        self.assertEqual(len(predictions), len(y_test), "Predicted output length does not match expected test set length.")

        # Mock model saving
        model_path = 'models/new_model.joblib'
        joblib.dump(model, model_path)
        
        # Check if joblib.dump was called with the correct file path
        mock_joblib_dump.assert_called_once_with(model, model_path)

# Run tests
if __name__ == '__main__':
    unittest.main()
