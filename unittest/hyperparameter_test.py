import unittest
import os
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import mean_squared_error

# Constants
DATA_PATH = r'C:\Users\harikarank\Desktop\tymestack\data\boston_house_prices (1).csv'
MODEL_SAVE_PATH = 'models/new_model.joblib'
PARAM_DIST = {
    'n_estimators': [50, 100, 150],
    'max_depth': [10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

class TestModelTraining(unittest.TestCase):
    
    def setUp(self):
        """Set up test environment."""
        try:
            self.data = pd.read_csv(DATA_PATH)
            self.X = self.data.drop('MEDV', axis=1)
            self.y = self.data['MEDV']
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=42)
        except FileNotFoundError:
            self.skipTest(f"Data file not found at {DATA_PATH}.")
        except KeyError:
            self.skipTest("Target column 'MEDV' not found in dataset.")
    
    def test_model_training(self):
        """Test if the model can be trained successfully."""
        model = RandomForestRegressor(random_state=42)
        random_search = RandomizedSearchCV(
            estimator=model,
            param_distributions=PARAM_DIST,
            n_iter=10,
            scoring='neg_mean_squared_error',
            cv=3,
            verbose=0,
            random_state=42,
            n_jobs=-1
        )
        random_search.fit(self.X_train, self.y_train)
        self.assertIsNotNone(random_search.best_estimator_)
    
    def test_save_model(self):
        """Test if the model is saved correctly."""
        model = RandomForestRegressor(random_state=42)
        random_search = RandomizedSearchCV(
            estimator=model,
            param_distributions=PARAM_DIST,
            n_iter=10,
            scoring='neg_mean_squared_error',
            cv=3,
            verbose=0,
            random_state=42,
            n_jobs=-1
        )
        random_search.fit(self.X_train, self.y_train)
        joblib.dump(random_search.best_estimator_, MODEL_SAVE_PATH)
        self.assertTrue(os.path.exists(MODEL_SAVE_PATH))

    def tearDown(self):
        """Clean up after tests."""
        if os.path.exists(MODEL_SAVE_PATH):
            os.remove(MODEL_SAVE_PATH)

if __name__ == '__main__':
    unittest.main()