import os
import shutil

new_model_path = 'models/new_model.joblib'
previous_model_path = 'models/previous_model.joblib'

# Deploy new model if it exists
if os.path.exists(new_model_path):
    if os.path.exists(previous_model_path):
        os.remove(previous_model_path)
    shutil.move(new_model_path, previous_model_path)
    print("Model deployed successfully.")
else:
    print("New model not found.")
