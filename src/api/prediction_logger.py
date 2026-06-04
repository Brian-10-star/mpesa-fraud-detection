# prediction_logger.py
# Writes every prediction the API makes to the prediction_log table.
# Now includes latency_ms so the metrics endpoint can compute real percentile latency (p50, p95, p99) from actual request data.

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

engine = get_db_engine()


def log_prediction(transaction_id: str, transaction_type: str,
                   amount: float, fraud_probability: float,
                   is_fraud: bool, model_version: str,
                   latency_ms: float = None):
    """
    Inserts one prediction record into prediction_log.
    Called by the /predict route after every successful prediction.
    latency_ms is optional so that batch_scorer.py, which does not
    measure latency, can still call this function without changes.
    """
    sql = text("""
        INSERT INTO prediction_log (
            transaction_id, transaction_type, amount,
            fraud_probability, is_fraud, model_version, latency_ms
        ) VALUES (
            :transaction_id, :transaction_type, :amount,
            :fraud_probability, :is_fraud, :model_version, :latency_ms
        )
    """)
    with engine.connect() as conn:
        conn.execute(sql, {
            'transaction_id': transaction_id,
            'transaction_type': transaction_type,
            'amount': amount,
            'fraud_probability': round(fraud_probability, 4),
            'is_fraud': is_fraud,
            'model_version': model_version,
            'latency_ms': latency_ms
        })
        conn.commit()


def get_prediction_metrics() -> dict:
    """
    Returns comprehensive metrics from prediction_log.

    Percentile latency is computed using PostgreSQL's PERCENTILE_CONT function, which is a standard SQL window function for ordered set aggregates. 
    
    PERCENTILE_CONT(0.95) means: find the value at the 95th percentile of the ordered latency_ms column.
    WITHIN GROUP (ORDER BY latency_ms) tells PostgreSQL how to order the values before computing the percentile.

    We compute two fraud rates: all-time and last 24 hours. The 24-hour rate is more useful operationally because it reflects what the model is doing right now and not the entire history.

    Throughput is computed as predictions per minute over the last hour. We count rows in the last 60 minutes and divide by 60.
    """
    sql = text("""
        SELECT
            COUNT(*) AS total,
            COALESCE(SUM(CASE WHEN is_fraud THEN 1 ELSE 0 END), 0)
                AS fraud_count,

            ROUND(PERCENTILE_CONT(0.50) WITHIN GROUP
                (ORDER BY latency_ms)::NUMERIC, 2)
                AS p50_latency_ms,

            ROUND(PERCENTILE_CONT(0.95) WITHIN GROUP
                (ORDER BY latency_ms)::NUMERIC, 2)
                AS p95_latency_ms,

            ROUND(PERCENTILE_CONT(0.99) WITHIN GROUP
                (ORDER BY latency_ms)::NUMERIC, 2)
                AS p99_latency_ms,

            COUNT(CASE WHEN predicted_at >= NOW() - INTERVAL '24 hours'
                THEN 1 END)
                AS total_last_24h,

            COALESCE(SUM(CASE WHEN is_fraud AND predicted_at >= NOW() - INTERVAL '24 hours'
                THEN 1 ELSE 0 END), 0)
                AS fraud_last_24h,

            COUNT(CASE WHEN predicted_at >= NOW() - INTERVAL '1 hour'
                THEN 1 END)
                AS total_last_1h

        FROM prediction_log
        WHERE latency_ms IS NOT NULL
    """)

    sql_all = text("""
        SELECT
            COUNT(*) AS total,
            COALESCE(SUM(CASE WHEN is_fraud THEN 1 ELSE 0 END), 0)
                AS fraud_count
        FROM prediction_log
    """)
    with engine.connect() as conn:
        row = conn.execute(sql).fetchone()
        row_all = conn.execute(sql_all).fetchone()

    total = int(row_all.total)
    fraud = int(row_all.fraud_count)
    legit = total - fraud
    fraud_rate = round((fraud / total * 100), 2) if total > 0 else 0.0

    total_24h = int(row.total_last_24h) if row.total_last_24h else 0
    fraud_24h = int(row.fraud_last_24h) if row.fraud_last_24h else 0
    fraud_rate_24h = round((fraud_24h / total_24h * 100), 2) if total_24h > 0 else 0.0

    total_1h = int(row.total_last_1h) if row.total_last_1h else 0
    # Throughput: predictions per minute over the last hour
    throughput_per_minute = round(total_1h / 60, 2)

    return {
        'total_predictions': total,
        'fraud_detected': fraud,
        'legitimate': legit,
        'fraud_rate': fraud_rate,
        'fraud_rate_24h': fraud_rate_24h,
        'total_last_24h': total_24h,
        'fraud_last_24h': fraud_24h,
        'p50_latency_ms': float(row.p50_latency_ms) if row.p50_latency_ms else None,
        'p95_latency_ms': float(row.p95_latency_ms) if row.p95_latency_ms else None,
        'p99_latency_ms': float(row.p99_latency_ms) if row.p99_latency_ms else None,
        'throughput_per_minute': throughput_per_minute,
    }