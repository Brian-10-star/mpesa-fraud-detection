# test_model.py
# Tests that the trained ML model loads correctly from MLflow and produces valid predictions.
# We don't test model accuracy here as that's done during training.
# We test that the model interface works correctly:
# - Loads without errors
# - Accepts the correct input shape
# - Returns probabilities between 0 and 1
# - Returns binary predictions (0 or 1 only)

import pytest
import sys
import os
import pandas as pd
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))


@pytest.fixture(scope='module')
def loaded_model():
    """
    Loads the model from MLflow once for all tests in this file.
    scope='module' means this fixture runs once, not before every test.
    """
    import mlflow
    import mlflow.sklearn
    from dotenv import load_dotenv
    load_dotenv()

    mlflow.set_tracking_uri(
        os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000"))
    model = mlflow.sklearn.load_model("models:/mpesa-fraud-detector/latest")
    return model


@pytest.fixture(scope='module')
def sample_features():
    """
    Creates a sample feature vector for testing predictions.
    Must have exactly the same columns as the training data.
    """
    return pd.DataFrame([{
        'amount': 5000.0,
        'hour_of_day': 14,
        'day_of_week': 0,
        'is_night': 0,
        'is_weekend': 0,
        'is_month_start': 0,
        'is_month_end': 0,
        'txn_count_last_10min': 1,
        'txn_count_last_1hr': 3,
        'txn_sum_last_10min': 5000.0,
        'txn_sum_last_1hr': 15000.0,
        'amount_zscore': 0.5,
        'amount_vs_sender_mean': 1.2,
        'is_large_amount': 0,
        'is_new_device': 1,
        'is_new_location': 0,
        'unique_receivers_last_1hr': 1,
        'type_frequency': 0.3
    }])


def test_model_loads(loaded_model):
    """Model must load without errors and not be None."""
    assert loaded_model is not None


def test_model_has_predict_method(loaded_model):
    """Model must have a predict() method — basic scikit-learn interface."""
    assert hasattr(loaded_model, 'predict')


def test_model_has_predict_proba(loaded_model):
    """Model must support predict_proba() for fraud probability scoring."""
    assert hasattr(loaded_model, 'predict_proba')


def test_prediction_returns_valid_probability(loaded_model, sample_features):
    """Fraud probability must be between 0.0 and 1.0."""
    proba = loaded_model.predict_proba(sample_features)[:, 1][0]
    assert 0.0 <= proba <= 1.0, f"Invalid probability: {proba}"


def test_prediction_returns_binary(loaded_model, sample_features):
    """predict() must return only 0 or 1."""
    pred = loaded_model.predict(sample_features)[0]
    assert pred in [0, 1], f"Invalid prediction: {pred}"


def test_fraud_transaction_scored(loaded_model):
    """
    A clearly fraudulent transaction (3am, large amount, new device,
    high velocity) should receive a high fraud probability.
    This tests that the model learned meaningful patterns.
    """
    fraud_features = pd.DataFrame([{
        'amount': 75000.0,
        'hour_of_day': 3,
        'day_of_week': 6,
        'is_night': 1,
        'is_weekend': 1,
        'is_month_start': 0,
        'is_month_end': 0,
        'txn_count_last_10min': 8,
        'txn_count_last_1hr': 20,
        'txn_sum_last_10min': 200000.0,
        'txn_sum_last_1hr': 500000.0,
        'amount_zscore': 4.5,
        'amount_vs_sender_mean': 5.0,
        'is_large_amount': 1,
        'is_new_device': 1,
        'is_new_location': 1,
        'unique_receivers_last_1hr': 8,
        'type_frequency': 0.0
    }])
    proba = loaded_model.predict_proba(fraud_features)[:, 1][0]
    assert proba >= 0.5, f"Fraud transaction scored too low: {proba}"