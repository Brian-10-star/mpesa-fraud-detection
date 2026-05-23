# batch_scorer.py
# Scores all unscored transactions in bulk by loading the model directly
# and running predictions without going through the HTTP API.
# Used to populate prediction_log with enough data for drift monitoring.
# In production, this would run nightly on the previous day's transactions.

import os
import sys
import mlflow
import mlflow.sklearn
import pandas as pd
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
load_dotenv()

from src.features.temporal_features import extract_temporal_features
from src.features.velocity_features import extract_velocity_features
from src.features.amount_features import extract_amount_features
from src.features.behavioral_features import extract_behavioral_features
from src.api.prediction_logger import log_prediction

FEATURE_COLUMNS = [
    'amount', 'hour_of_day', 'day_of_week', 'is_night', 'is_weekend',
    'is_month_start', 'is_month_end', 'txn_count_last_10min',
    'txn_count_last_1hr', 'txn_sum_last_10min', 'txn_sum_last_1hr',
    'amount_zscore', 'amount_vs_sender_mean', 'is_large_amount',
    'is_new_device', 'is_new_location', 'unique_receivers_last_1hr',
    'type_frequency'
]

BOOL_COLS = ['is_night', 'is_weekend', 'is_month_start', 'is_month_end',
             'is_large_amount', 'is_new_device', 'is_new_location']


def get_db_engine():
    url = (
        f"postgresql+psycopg2://{os.getenv('DB_USER')}:{os.getenv('DB_PASSWORD')}"
        f"@{os.getenv('DB_HOST')}:{os.getenv('DB_PORT')}/{os.getenv('DB_NAME')}"
    )
    return create_engine(url)


def get_unscored_transactions(engine):
    """
    Fetches transactions that exist in raw_transactions but have
    no corresponding row in prediction_log yet.
    """
    sql = text("""
        SELECT r.* FROM raw_transactions r
        LEFT JOIN prediction_log p ON r.transaction_id = p.transaction_id
        WHERE p.transaction_id IS NULL
        ORDER BY r.created_at ASC
    """)
    with engine.connect() as conn:
        result = conn.execute(sql)
        return [dict(row._mapping) for row in result]


def main():
    print("Starting batch scorer...")

    # Load model directly from MLflow
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000"))
    model = mlflow.sklearn.load_model(f"models:/mpesa-fraud-detector/latest")
    print("Model loaded.")

    engine = get_db_engine()
    transactions = get_unscored_transactions(engine)
    print(f"Found {len(transactions)} unscored transactions.")

    if not transactions:
        print("Nothing to score.")
        return

    scored = 0
    fraud_count = 0

    for txn in transactions:
        try:
            # Convert timestamp to string if needed
            if not isinstance(txn['timestamp'], str):
                txn['timestamp'] = txn['timestamp'].isoformat()

            temporal = extract_temporal_features(txn)
            velocity = extract_velocity_features(txn, engine)
            amount = extract_amount_features(txn, engine)
            behavioral = extract_behavioral_features(txn, engine)

            features = {
                'amount': float(txn['amount']),
                **temporal, **velocity, **amount, **behavioral
            }

            for col in BOOL_COLS:
                features[col] = int(features[col])

            X = pd.DataFrame([features])[FEATURE_COLUMNS]
            fraud_probability = float(model.predict_proba(X)[:, 1][0])
            is_fraud = fraud_probability >= 0.5

            log_prediction(
                transaction_id=txn['transaction_id'],
                transaction_type=txn['transaction_type'],
                amount=float(txn['amount']),
                fraud_probability=fraud_probability,
                is_fraud=is_fraud,
                model_version='latest'
            )

            scored += 1
            if is_fraud:
                fraud_count += 1
                print(f"[FRAUD]  {txn['transaction_id']} | "
                      f"{txn['transaction_type']} | KES {float(txn['amount']):,.2f} | "
                      f"prob={fraud_probability:.4f}")
            else:
                print(f"[LEGIT]  {txn['transaction_id']} | "
                      f"{txn['transaction_type']} | KES {float(txn['amount']):,.2f}")

        except Exception as e:
            print(f"[ERROR] {txn.get('transaction_id')} | {e}")

    print(f"\nBatch scoring complete.")
    print(f"Scored: {scored} | Fraud: {fraud_count} | "
          f"Legitimate: {scored - fraud_count}")


if __name__ == "__main__":
    main()