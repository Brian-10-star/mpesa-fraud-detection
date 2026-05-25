# feature_store.py
# Saves a completed feature row to the features table in frauddb. This is called once per transaction after all feature modules have run.
# The features table is what the ML model reads during training.

from sqlalchemy import text


def save_features(engine, features: dict):
    """
    Inserts one feature row into the features table.
    ON CONFLICT DO NOTHING handles duplicates safely by skipping the insertion if a row with the same transaction_id already exists.
    """
    sql = text("""
        INSERT INTO features (
            transaction_id, amount, transaction_type,
            hour_of_day, day_of_week, is_night, is_weekend,
            is_month_start, is_month_end,
            txn_count_last_10min, txn_count_last_1hr,
            txn_sum_last_10min, txn_sum_last_1hr,
            amount_zscore, amount_vs_sender_mean, is_large_amount,
            is_new_device, is_new_location,
            unique_receivers_last_1hr, type_frequency
        ) VALUES (
            :transaction_id, :amount, :transaction_type,
            :hour_of_day, :day_of_week, :is_night, :is_weekend,
            :is_month_start, :is_month_end,
            :txn_count_last_10min, :txn_count_last_1hr,
            :txn_sum_last_10min, :txn_sum_last_1hr,
            :amount_zscore, :amount_vs_sender_mean, :is_large_amount,
            :is_new_device, :is_new_location,
            :unique_receivers_last_1hr, :type_frequency
        )
        ON CONFLICT (transaction_id) DO NOTHING
    """)

    with engine.connect() as conn:
        conn.execute(sql, features)
        conn.commit()