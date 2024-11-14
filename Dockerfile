# Use an official Python runtime as a parent image
FROM python:3.8-slim

# Set the working directory
WORKDIR /home/tis/Downloads/tymestack

# Copy the current directory contents into the container
COPY . .

# Install any needed packages
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install joblib
RUN pip install scikit-learn
RUN pip install pandas
# RUN mkdir models

# Define the command to run your model training, tuning, evaluation, and deployment scripts in sequence
CMD ["sh", "-c", "python src/train_model.py && python src/hyperparameter_tuning.py && python src/evaluate_model.py && python src/deploy_model.py && tail -f /dev/null"]
