# label_generator.py
# Assigns fraud/not-fraud labels to transactions in the features table using rule-based heuristics. This is weak supervision.
# How it works:
# 1. Read every feature row that hasn't been labeled yet
# 2. Apply each fraud rule and each rule adds 1 to the fraud_score
# 3. If fraud_score >= 2, mark the transaction as fraud
# 4. Save the label back to the features table

import os
import sys
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
load_dotenv()


def get_db_engine():
    url = (
        f"postgresql+psycopg2://{os.getenv('DB_USER')}:{os.getenv('DB_PASSWORD')}"
        f"@{os.getenv('DB_HOST')}:{os.getenv('DB_PORT')}/{os.getenv('DB_NAME')}"
    )
    return create_engine(url)


def apply_fraud_rules(row: dict) -> tuple[int, list[str]]:
    """
    Applies all fraud detection rules to one feature row.
    Returns a tuple of (fraud_score, list_of_reasons).

    Each rule is independent as we check all of them and accumulate the score. This way, multiple weak signals combine into a strong fraud detector.

    tuple[int, list[str]] means this function returns two things:
    an integer score and a list of string reasons.
    """
    score = 0
    reasons = []

    # Rule 1: Night transaction + amount more than 3x sender's mean
    if row['is_night'] and row['amount_vs_sender_mean'] > 3.0:
        score += 1
        reasons.append("night_high_amount")

    # Rule 2: High velocity: 5+ transactions in 10 minutes
    if row['txn_count_last_10min'] >= 5:
        score += 1
        reasons.append("high_velocity_10min")

    # Rule 3: New device and new location simultaneously
    # Using an unknown device from an unknown place = account takeover signal
    if row['is_new_device'] and row['is_new_location']:
        score += 1
        reasons.append("new_device_and_location")

    # Rule 4: Statistically extreme amount (z-score > 3)
    # This amount is 3 standard deviations above this sender's average
    if row['amount_zscore'] > 3.0:
        score += 1
        reasons.append("extreme_amount_zscore")

    # Rule 5: Large amount (>50k KES) on a brand new device
    if row['is_large_amount'] and row['is_new_device']:
        score += 1
        reasons.append("large_amount_new_device")

    # Rule 6: Sending to many different receivers rapidly coz scammer is trying to cash out quickly
    if row['unique_receivers_last_1hr'] >= 4:
        score += 1
        reasons.append("many_unique_receivers")

    return score, reasons


def get_unlabeled_features(engine):
    """
    Fetches all feature rows where is_fraud is still NULL (never labeled).
    We use IS NULL rather than = FALSE because we only set the column when we label it coz unlabeled rows have NULL, not FALSE.
    """
    sql = text("""
        SELECT * FROM features
        WHERE fraud_reasons IS NULL
        ORDER BY created_at ASC
    """)
    with engine.connect() as conn:
        result = conn.execute(sql)
        return [dict(row._mapping) for row in result]


def save_label(engine, transaction_id: str, is_fraud: bool,
               fraud_score: int, fraud_reasons: list[str]):
    """
    Updates the features row with the computed fraud label.
    We use UPDATE not INSERT because the row already exists
    """
    sql = text("""
        UPDATE features
        SET is_fraud = :is_fraud,
            fraud_score = :fraud_score,
            fraud_reasons = :fraud_reasons
        WHERE transaction_id = :transaction_id
    """)
    with engine.connect() as conn:
        conn.execute(sql, {
            'transaction_id': transaction_id,
            'is_fraud': is_fraud,
            'fraud_score': fraud_score,
            # Join the reasons list into a comma-separated string for storage
            'fraud_reasons': ', '.join(fraud_reasons) if fraud_reasons else 'none'
        })
        conn.commit()


def main():
    print("Starting label generator...")
    engine = get_db_engine()

    rows = get_unlabeled_features(engine)
    print(f"Found {len(rows)} unlabeled feature rows.")

    if not rows:
        print("Nothing to label. Run the feature pipeline first.")
        return

    fraud_count = 0
    legit_count = 0

    for row in rows:
        score, reasons = apply_fraud_rules(row)

        # Fraud threshold: 2 or more rules must fire
        is_fraud = score >= 2

        save_label(engine, row['transaction_id'], is_fraud, score, reasons)

        if is_fraud:
            fraud_count += 1
            print(f"[FRAUD]  {row['transaction_id']} | "
                  f"score={score} | reasons: {', '.join(reasons)}")
        else:
            legit_count += 1
            print(f"[LEGIT]  {row['transaction_id']} | "
                  f"score={score}")

    total = fraud_count + legit_count
    fraud_rate = (fraud_count / total * 100) if total > 0 else 0

    print(f"\nLabeling complete.")
    print(f"Total: {total} | Fraud: {fraud_count} | "
          f"Legitimate: {legit_count} | Fraud rate: {fraud_rate:.1f}%")


if __name__ == "__main__":
    main()