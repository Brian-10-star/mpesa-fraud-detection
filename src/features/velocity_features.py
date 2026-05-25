# velocity_features.py
# Velocity refers to how fast transactions are happening from one sender like if someone sends 10 transactions in 10 minutes, that's a red flag.
# These features look back in time and count/sum recent activity.

from sqlalchemy import text
from datetime import datetime


def extract_velocity_features(txn: dict, engine) -> dict:
    """
    Queries raw_transactions to count and sum recent transactions
    from the same sender phone number.
    
    We look back two windows:
    - Last 10 minutes: catches rapid-fire fraud bursts
    - Last 1 hour: catches sustained fraud campaigns
    
    engine is the SQLAlchemy connection to PostgreSQL — we need it
    to query historical transactions for this sender.
    """
    sender_phone = txn['sender_phone']
    ts = txn['timestamp']
    timestamp = ts if isinstance(ts, datetime) else datetime.fromisoformat(str(ts))

    with engine.connect() as conn:

        # Count and sum transactions in the last 10 minutes
        result_10min = conn.execute(text("""
            SELECT COUNT(*) as count, COALESCE(SUM(amount), 0) as total
            FROM raw_transactions
            WHERE sender_phone = :phone
              AND timestamp >= :timestamp - INTERVAL '10 minutes'
              AND timestamp <= :timestamp
        """), {'phone': sender_phone, 'timestamp': timestamp}).fetchone()

        # Count and sum transactions in the last 1 hour
        result_1hr = conn.execute(text("""
            SELECT COUNT(*) as count, COALESCE(SUM(amount), 0) as total
            FROM raw_transactions
            WHERE sender_phone = :phone
              AND timestamp >= :timestamp - INTERVAL '1 hour'
              AND timestamp <= :timestamp
        """), {'phone': sender_phone, 'timestamp': timestamp}).fetchone()

    return {
        'txn_count_last_10min': int(result_10min.count),
        'txn_sum_last_10min': float(result_10min.total),
        'txn_count_last_1hr': int(result_1hr.count),
        'txn_sum_last_1hr': float(result_1hr.total),
    }