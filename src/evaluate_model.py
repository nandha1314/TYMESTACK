# import joblib
# import pandas as pd
# from sklearn.metrics import mean_squared_error
# import os

# # Load data
# data = pd.read_csv(r'C:\Users\harikarank\Desktop\tymestack\data\boston_house_prices (1).csv')
# X = data.drop('MEDV', axis=1)
# y = data['MEDV']

# # Load new model
# new_model = joblib.load('models/new_model.joblib')

# # Check if previous model exists and is loadable
# previous_model_path = 'models/previous_model.joblib'
# previous_model = None  # Initialize with None

# if os.path.exists(previous_model_path) and os.path.getsize(previous_model_path) > 0:
#     try:
#         previous_model = joblib.load(previous_model_path)
#     except EOFError:
#         print("Error: The previous model file is corrupted. Replacing with the new model.")
# else:
#     print("Previous model not found or is empty. Assuming the new model is better.")

# # Evaluate new model
# new_model_mse = mean_squared_error(y, new_model.predict(X))
# print(f"New Model MSE: {new_model_mse}")

# # Evaluate and compare with the previous model if it exists and is loadable
# if previous_model:
#     previous_model_mse = mean_squared_error(y, previous_model.predict(X))
#     print(f"Previous Model MSE: {previous_model_mse}")

#     # Improvement threshold
#     improvement_threshold = 0.01
#     if new_model_mse < previous_model_mse - improvement_threshold:
#         print("New model is better and will replace the previous model.")
#         joblib.dump(new_model, previous_model_path)  # Save the new model as the best model
#     else:
#         print("New model is not significantly better.")
# else:
#     # If no previous model, consider the new model as the best and save it
#     print("Saving new model as the best model.")
#     joblib.dump(new_model, previous_model_path)






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
    print("Previous model is better.")
