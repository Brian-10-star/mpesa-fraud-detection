# amount_features.py
# Compares a transaction's amount to the sender's historical behaviour.
# These features should give the ML model context about whether the amount is unusual for a specific sender

from sqlalchemy import text
from datetime import datetime


def extract_amount_features(txn: dict, engine) -> dict:
    """
    Calculates how unusual this transaction's amount is relative to the sender's historical average and standard deviation.

    Z-score = (amount - mean) / std_dev
    A z-score of 0 means perfectly average.
    A z-score of 3+ means this amount is 3 standard deviations above the sender's average which reflects very unusual and a high fraud risk.

    COALESCE(x, fallback) returns fallback if x is NULL.
    We use this because new senders have no history and therefore we default their stats to the transaction amount itself (z-score = 0).
    """
    sender_phone = txn['sender_phone']
    amount = float(txn['amount'])
    ts = txn['timestamp']
    timestamp = ts if isinstance(ts, datetime) else datetime.fromisoformat(str(ts))

    with engine.connect() as conn:
        result = conn.execute(text("""
            SELECT
                COALESCE(AVG(amount), :amount)   AS mean_amount,
                COALESCE(STDDEV(amount), 0)      AS std_amount,
                COUNT(*)                          AS txn_count
            FROM raw_transactions
            WHERE sender_phone = :phone
              AND timestamp < :timestamp
        """), {'phone': sender_phone, 'amount': amount,
               'timestamp': timestamp}).fetchone()

    mean_amount = float(result.mean_amount)
    std_amount = float(result.std_amount)

    # Avoid division by zero
    if std_amount > 0:
        amount_zscore = (amount - mean_amount) / std_amount
    else:
        amount_zscore = 0.0


    if mean_amount > 0:
        amount_vs_sender_mean = amount / mean_amount
    else:
        amount_vs_sender_mean = 1.0

    # Flag transactions above KES 50,000 as large
    is_large_amount = amount > 50000

    return {
        'amount_zscore': round(amount_zscore, 4),
        'amount_vs_sender_mean': round(amount_vs_sender_mean, 4),
        'is_large_amount': is_large_amount,
    }