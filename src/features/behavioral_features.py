# behavioral_features.py
# Looks at patterns in WHO the sender transacts with and HOW.
# Account takeover fraud often shows: new device, new location, sending to many different people rapidly.

from sqlalchemy import text
from datetime import datetime


def extract_behavioral_features(txn: dict, engine) -> dict:
    """
    Detects behavioural anomalies by comparing this transaction
    to the sender's history stored in raw_transactions.

    Key signals:
    - is_new_device: sender using a device fingerprint never seen before
    - is_new_location: transaction from a location this sender hasn't used
    - unique_receivers_last_1hr: how many different people they sent to
      in the last hour (scammers often spray money to many accounts)
    - type_frequency: how often this sender uses this transaction type
      (0.0 = never used it before, 1.0 = always uses this type)
    """
    sender_phone = txn['sender_phone']
    ts = txn['timestamp']
    timestamp = ts if isinstance(ts, datetime) else datetime.fromisoformat(str(ts))
    device = txn['device_fingerprint']
    location = txn['location']
    txn_type = txn['transaction_type']

    with engine.connect() as conn:

        # Has this sender used this device before?
        device_count = conn.execute(text("""
            SELECT COUNT(*) as cnt FROM raw_transactions
            WHERE sender_phone = :phone
              AND device_fingerprint = :device
              AND timestamp < :timestamp
        """), {'phone': sender_phone, 'device': device,
               'timestamp': timestamp}).fetchone().cnt

        # Has this sender transacted from this location before?
        location_count = conn.execute(text("""
            SELECT COUNT(*) as cnt FROM raw_transactions
            WHERE sender_phone = :phone
              AND location = :location
              AND timestamp < :timestamp
        """), {'phone': sender_phone, 'location': location,
               'timestamp': timestamp}).fetchone().cnt

        # How many unique receivers in the last 1 hour?
        unique_receivers = conn.execute(text("""
            SELECT COUNT(DISTINCT receiver_phone) as cnt
            FROM raw_transactions
            WHERE sender_phone = :phone
              AND timestamp >= :timestamp - INTERVAL '1 hour'
              AND timestamp <= :timestamp
        """), {'phone': sender_phone, 'timestamp': timestamp}).fetchone().cnt

        # What fraction of this sender's transactions use this type?
        type_stats = conn.execute(text("""
            SELECT
                COUNT(*) FILTER (WHERE transaction_type = :txn_type) AS type_count,
                COUNT(*) AS total_count
            FROM raw_transactions
            WHERE sender_phone = :phone
              AND timestamp < :timestamp
        """), {'phone': sender_phone, 'txn_type': txn_type,
               'timestamp': timestamp}).fetchone()

    total = type_stats.total_count
    type_frequency = (float(type_stats.type_count) / total) if total > 0 else 0.0

    return {
        'is_new_device': device_count == 0,
        'is_new_location': location_count == 0,
        'unique_receivers_last_1hr': int(unique_receivers),
        'type_frequency': round(type_frequency, 4),
    }