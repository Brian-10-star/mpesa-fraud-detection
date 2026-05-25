# temporal_features.py
# Extracts time-based features from a transaction's timestamp as fraud often happens at unusual hours.
# These features will give the ML model a sense of WHEN the transaction happened.

from datetime import datetime


def extract_temporal_features(txn: dict) -> dict:
    """
    Takes a transaction dict and returns time-based features.
    
    txn['timestamp'] is an ISO format string like "2026-05-15T03:42:11.123456"
    We parse it into a datetime object so we can extract hour, day, etc.
    """
    ts = txn['timestamp']
    timestamp = ts if isinstance(ts, datetime) else datetime.fromisoformat(str(ts))

    hour = timestamp.hour          # 0-23
    day_of_week = timestamp.weekday()  # 0=Monday, 6=Sunday

    # is_night: transactions between midnight and 5am are high-risk
    is_night = hour < 5 or hour >= 23

    # is_weekend: fraud rates differ on weekends
    is_weekend = day_of_week >= 5  # Saturday=5, Sunday=6

    # is_month_start/end: salary days and bill payment days see unusual spikes
    is_month_start = timestamp.day <= 3
    is_month_end = timestamp.day >= 28

    return {
        'hour_of_day': hour,
        'day_of_week': day_of_week,
        'is_night': is_night,
        'is_weekend': is_weekend,
        'is_month_start': is_month_start,
        'is_month_end': is_month_end,
    }