import joblib
import pandas as pd
from sklearn.metrics import mean_squared_error

# Load dataset
data = pd.read_csv(r'C:\Users\harikarank\Desktop\tymestack\data\boston_house_prices (1).csv')
X = data.drop('MEDV', axis=1)
y = data['MEDV']

# Load new and previous models
new_model = joblib.load('models/new_model.joblib')
try:
    previous_model = joblib.load('models/previous_model.joblib')
    previous_mse = mean_squared_error(y, previous_model.predict(X))
except FileNotFoundError:
    print("No previous model found; setting previous_mse to infinity")
    previous_mse = float('inf')

new_mse = mean_squared_error(y, new_model.predict(X))
print(f"New Model MSE: {new_mse}, Previous Model MSE: {previous_mse}")

# Determine if new model is better
if new_mse < previous_mse:
    print("New model is better.")
else:
    print("Previous model is betTter.")
