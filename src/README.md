# House Price Prediction Model Retraining Pipeline

This project uses a machine learning pipeline with hyperparameter tuning, evaluation, and deployment for house price prediction. The pipeline is automated with GitHub Actions.

## Project Structure

```plaintext
my_ml_project/
├── .github/
│   └── workflows/
│       └── retrain_pipeline.yml
├── data/
│   └── dataset.csv
├── models/
│   ├── previous_model.joblib
│   └── new_model.joblib
├── src/
│   ├── hyperparameter_tuning.py
│   ├── evaluate_model.py
│   └── deploy_model.py
├── requirements.txt
└── README.md
