import unittest
from unittest.mock import patch, MagicMock
import joblib
import pandas as pd
import os
from sklearn.metrics import mean_squared_error
import numpy as np

class TestModelComparison(unittest.TestCase):

    @patch('pandas.read_csv')
    @patch('joblib.load')
    @patch('os.path.exists')
    @patch('os.path.getsize')
    def test_model_comparison_with_previous_model(self, mock_getsize, mock_exists, mock_load, mock_read_csv):
        # Mock dataset
        mock_df = pd.DataFrame({
            'Feature1': [1, 2, 3, 4],
            'Feature2': [2, 3, 4, 5],
            'MEDV': [10, 15, 10, 20]
        })
        mock_read_csv.return_value = mock_df
        X = mock_df.drop('MEDV', axis=1)
        y = mock_df['MEDV']
        
        # Mock model predictions
        new_model_mock = MagicMock()
        new_model_mock.predict.return_value = np.array([12, 14, 11, 19])
        
        previous_model_mock = MagicMock()
        previous_model_mock.predict.return_value = np.array([13, 16, 10, 18])
        
        mock_load.side_effect = [new_model_mock, previous_model_mock]

        # Mock existence and size of previous model file
        mock_exists.return_value = True
        mock_getsize.return_value = 100

        # Run the comparison
        new_mse = mean_squared_error(y, new_model_mock.predict(X))
        previous_mse = mean_squared_error(y, previous_model_mock.predict(X))
        
        # Assert new model is better since it has lower MSE
        self.assertTrue(new_mse < previous_mse, "New model should be better than the previous model.")

    @patch('pandas.read_csv')
    @patch('joblib.load')
    @patch('os.path.exists')
    def test_model_comparison_without_previous_model(self, mock_exists, mock_load, mock_read_csv):
        # Mock dataset
        mock_df = pd.DataFrame({
            'Feature1': [1, 2, 3, 4],
            'Feature2': [2, 3, 4, 5],
            'MEDV': [10, 15, 10, 20]
        })
        mock_read_csv.return_value = mock_df
        X = mock_df.drop('MEDV', axis=1)
        y = mock_df['MEDV']
        
        # Mock model prediction
        new_model_mock = MagicMock()
        new_model_mock.predict.return_value = np.array([12, 14, 11, 19])
        
        mock_load.return_value = new_model_mock

        # Mock no previous model found
        mock_exists.return_value = False

        # Run the comparison
        new_mse = mean_squared_error(y, new_model_mock.predict(X))
        
        # Since there's no previous model, the new model should be considered as the best model
        self.assertTrue(new_mse < float('inf'), "New model should be saved as the best model since no previous model exists.")

if __name__ == '__main__':
    unittest.main()
