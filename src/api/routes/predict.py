# routes/predict.py
# POST /predict is the core endpoint.
# Accepts a transaction, engineers its features, runs the model, logs the prediction, and returns the fraud probability.
# This is what a payment system calls in real time before approving a transaction.

from fastapi import APIRouter, HTTPException, Depends
from src.api.auth import verify_api_key
from src.api.schemas import TransactionRequest, PredictionResponse
from src.api.model_loader import get_model, get_model_version, is_model_loaded
from src.api.prediction_logger import log_prediction
from src.features.temporal_features import extract_temporal_features
from src.features.velocity_features import extract_velocity_features
from src.features.amount_features import extract_amount_features
from src.features.behavioral_features import extract_behavioral_features
from sqlalchemy import create_engine
from dotenv import load_dotenv
import pandas as pd
import os

load_dotenv()

router = APIRouter()

# Feature columns must match exactly what the model was trained on
FEATURE_COLUMNS = [
    'amount', 'hour_of_day', 'day_of_week', 'is_night', 'is_weekend',
    'is_month_start', 'is_month_end', 'txn_count_last_10min',
    'txn_count_last_1hr', 'txn_sum_last_10min', 'txn_sum_last_1hr',
    'amount_zscore', 'amount_vs_sender_mean', 'is_large_amount',
    'is_new_device', 'is_new_location', 'unique_receivers_last_1hr',
    'type_frequency'
]


def get_db_engine():
    url = (
        f"postgresql+psycopg2://{os.getenv('DB_USER')}:{os.getenv('DB_PASSWORD')}"
        f"@{os.getenv('DB_HOST')}:{os.getenv('DB_PORT')}/{os.getenv('DB_NAME')}"
    )
    return create_engine(url)


@router.post("/predict", response_model=PredictionResponse)
def predict(transaction: TransactionRequest, _: str = Depends(verify_api_key)):
    """
    Main prediction endpoint.

    Steps:
    1. Check model is loaded
    2. Convert request to dict
    3. Engineer all features (same modules as training pipeline)
    4. Build feature vector as pandas DataFrame
    5. Run model.predict_proba() to get fraud probability
    6. Log prediction to PostgreSQL
    7. Return result
    """
    if not is_model_loaded():
        raise HTTPException(status_code=503,
                           detail="Model not loaded yet. Try again shortly.")

    # Convert Pydantic model to plain dict for feature modules
    txn = transaction.model_dump()

    engine = get_db_engine()

    # Engineer features — same pipeline as training
    temporal = extract_temporal_features(txn)
    velocity = extract_velocity_features(txn, engine)
    amount = extract_amount_features(txn, engine)
    behavioral = extract_behavioral_features(txn, engine)

    # Merge all features
    features = {
        'amount': txn['amount'],
        **temporal,
        **velocity,
        **amount,
        **behavioral
    }

    # Convert booleans to int — model expects numeric input
    bool_cols = ['is_night', 'is_weekend', 'is_month_start', 'is_month_end',
                 'is_large_amount', 'is_new_device', 'is_new_location']
    for col in bool_cols:
        features[col] = int(features[col])

    # Build DataFrame with exact column order the model expects
    X = pd.DataFrame([features])[FEATURE_COLUMNS]

    # predict_proba returns [[prob_legit, prob_fraud]]
    # [:,1] takes the fraud probability column
    model = get_model()
    fraud_probability = float(model.predict_proba(X)[:, 1][0])
    is_fraud = fraud_probability >= 0.5

    # Log to database
    log_prediction(
        transaction_id=txn['transaction_id'],
        transaction_type=txn['transaction_type'],
        amount=txn['amount'],
        fraud_probability=fraud_probability,
        is_fraud=is_fraud,
        model_version=get_model_version()
    )

    message = "FRAUD DETECTED" if is_fraud else "Transaction appears legitimate"

    return PredictionResponse(
        transaction_id=txn['transaction_id'],
        fraud_probability=round(fraud_probability, 4),
        is_fraud=is_fraud,
        model_version=get_model_version(),
        message=message
    )