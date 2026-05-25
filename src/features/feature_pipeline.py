# feature_pipeline.py
# Reads raw transactions from PostgreSQL, runs all five feature modules on each one, and saves results to the features table.

import sys
import os
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from src.features.temporal_features import extract_temporal_features
from src.features.velocity_features import extract_velocity_features
from src.features.amount_features import extract_amount_features
from src.features.behavioral_features import extract_behavioral_features
from src.features.feature_store import save_features

load_dotenv()


def get_db_engine():
    url = (
        f"postgresql+psycopg2://{os.getenv('DB_USER')}:{os.getenv('DB_PASSWORD')}"
        f"@{os.getenv('DB_HOST')}:{os.getenv('DB_PORT')}/{os.getenv('DB_NAME')}"
    )
    return create_engine(url)


def get_unprocessed_transactions(engine):
    """
    Fetches all transactions from raw_transactions that don't yet have a corresponding row in the features table.
    LEFT JOIN + WHERE f.transaction_id IS NULL is the standard SQL pattern for finding rows in one table missing from another.
    """
    sql = text("""
        SELECT r.*
        FROM raw_transactions r
        LEFT JOIN features f ON r.transaction_id = f.transaction_id
        WHERE f.transaction_id IS NULL
        ORDER BY r.created_at ASC
    """)
    with engine.connect() as conn:
        result = conn.execute(sql)
        # Convert each row to a plain dict so our feature modules can read it
        return [dict(row._mapping) for row in result]


def process_transaction(txn: dict, engine) -> dict:
    """
    Runs all four feature extraction modules on one transaction and merges the results into a single flat dictionary.
    {**a, **b} merges two dicts.
    """
    temporal = extract_temporal_features(txn)
    velocity = extract_velocity_features(txn, engine)
    amount = extract_amount_features(txn, engine)
    behavioral = extract_behavioral_features(txn, engine)

    # Merge all features into one dict, starting with the identifiers
    return {
        'transaction_id': txn['transaction_id'],
        'amount': float(txn['amount']),
        'transaction_type': txn['transaction_type'],
        **temporal,
        **velocity,
        **amount,
        **behavioral,
    }


def main():
    print("Starting feature engineering pipeline...")
    engine = get_db_engine()

    transactions = get_unprocessed_transactions(engine)
    print(f"Found {len(transactions)} unprocessed transactions.")

    if not transactions:
        print("Nothing to process. Run the producer first to generate transactions.")
        return

    success = 0
    failed = 0

    for txn in transactions:
        try:
            features = process_transaction(txn, engine)
            save_features(engine, features)
            success += 1
            print(f"[{success}/{len(transactions)}] Processed: "
                  f"{txn['transaction_id']} | "
                  f"{txn['transaction_type']} | "
                  f"KES {float(txn['amount']):,.2f} | "
                  f"z-score: {features['amount_zscore']}")
        except Exception as e:
            failed += 1
            print(f"[FAILED] {txn['transaction_id']} | Error: {e}")

    print(f"\nFeature pipeline complete.")
    print(f"Processed: {success} | Failed: {failed}")


if __name__ == "__main__":
    main()