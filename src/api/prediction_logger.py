# prediction_logger.py
# Writes every prediction the API makes to the prediction_log table.
# This is critical for two reasons:
# 1. Monitoring — we can track how many frauds are being detected over time
# 2. Drift detection — Evidently AI reads this table in Phase 8 to check
#    if the model's predictions are drifting from the training distribution

from sqlalchemy import create_engine, text
from dotenv import load_dotenv
import os

load_dotenv()


def get_db_engine():
    url = (
        f"postgresql+psycopg2://{os.getenv('DB_USER')}:{os.getenv('DB_PASSWORD')}"
        f"@{os.getenv('DB_HOST')}:{os.getenv('DB_PORT')}/{os.getenv('DB_NAME')}"
    )
    return create_engine(url)


# Create engine once at module load — reused across all requests
engine = get_db_engine()


def log_prediction(transaction_id: str, transaction_type: str,
                   amount: float, fraud_probability: float,
                   is_fraud: bool, model_version: str):
    """
    Inserts one prediction record into prediction_log.
    Called by the /predict route after every successful prediction.
    """
    sql = text("""
        INSERT INTO prediction_log (
            transaction_id, transaction_type, amount,
            fraud_probability, is_fraud, model_version
        ) VALUES (
            :transaction_id, :transaction_type, :amount,
            :fraud_probability, :is_fraud, :model_version
        )
    """)
    with engine.connect() as conn:
        conn.execute(sql, {
            'transaction_id': transaction_id,
            'transaction_type': transaction_type,
            'amount': amount,
            'fraud_probability': round(fraud_probability, 4),
            'is_fraud': is_fraud,
            'model_version': model_version
        })
        conn.commit()


def get_prediction_metrics() -> dict:
    """
    Returns aggregate stats from prediction_log.
    Called by the GET /metrics endpoint.
    """
    sql = text("""
        SELECT
            COUNT(*) as total,
            COALESCE(SUM(CASE WHEN is_fraud THEN 1 ELSE 0 END), 0) as fraud_count
        FROM prediction_log
    """)
    with engine.connect() as conn:
        row = conn.execute(sql).fetchone()

    total = int(row.total)
    fraud = int(row.fraud_count)
    legit = total - fraud
    rate = round((fraud / total * 100), 2) if total > 0 else 0.0

    return {
        'total_predictions': total,
        'fraud_detected': fraud,
        'legitimate': legit,
        'fraud_rate': rate
    }