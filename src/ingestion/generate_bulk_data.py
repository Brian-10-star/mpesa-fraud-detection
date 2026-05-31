# generate_bulk_data.py
# Synthetic data generator for bulk seeding the frauddb database.
# Generates 10,000 realistic M-Pesa transactions, computes all 20 features, applies fraud labeling rules, and writes everything directly to PostgreSQL.

import os
import sys
import uuid
import random
from datetime import datetime, timedelta
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
load_dotenv()

# The same data pools from producer.py so generated data looks identical

CUSTOMER_NAMES = [
    "Michael Kinyua", "Amina Wanjiku", "Peter Otieno", "Grace Njeri",
    "Stacy Wairimu", "Florence Hamisi", "Samuel Kamau", "Fatuma Achieng",
    "David Mwangi", "Joyce Waithera", "Abdulah Hassan", "James Kipchoge",
    "Mary Wambui", "Hassan Abdi", "Lilian Adhiambo", "Timothy Mutiso",
    "Kevin Mutua", "Esther Chebet", "Moses Odhiambo", "Purity Karimi",
    "Christopher Njoroge", "Vincent Omondi", "Beatrice Nkirote",
    "Emmanuel Waweru", "Agnes Mumbi", "Christine Ndungu", "Dennis Kiptoo",
    "Alice Wanjiru", "Brian Mwangi", "Carolyn Njeri", "Anthony Kimani",
    "Faith Wambui", "Joseph Mwangi", "Francisca Achieng", "Mark Kiprono",
    "Ruth Njeri", "Eric Mwangi", "Jane Wairimu", "Simon Karanja",
    "Nancy Wambui", "Paul Mwangi", "Catherine Njeri", "Andrew Kipchoge",
    "Rose Wanjiku", "Daniel Mwangi", "Martha Njeri", "Kevin Otieno",
    "Alice Wairimu", "Grace Achieng", "Maryanne Mwendwa", "Valentine Wambui",
    "Abdul Majid", "Jimmy Kiprop", "Lilian Adhiambo", "Timothy Mutiso"
]

LOCATIONS = [
    "Nairobi CBD", "Westlands", "Kibera", "Kasarani", "Embakasi", "Kiambu",
    "Limuru", "Juja", "Thika", "Mombasa", "Kisumu", "Nakuru", "Eldoret",
    "Lamu", "Githunguri", "Narok", "Busia", "Embu", "Machakos", "Nyeri",
    "Meru", "Kakamega", "Garissa", "Homabay", "Kitale", "Kisii", "Vihiga",
    "Bungoma", "Murang'a", "Kericho", "Kajiado", "Siaya", "Ahero", "Chuka",
    "Turkana", "Marsabit", "Kapenguria", "Kitui", "Lodwar", "Ruiru",
    "Runda", "Karen", "Syokimau", "Ngong", "Dandora", "Mathare", "Githurai",
    "Kahawa Sukari", "Kitusuru", "Kileleshwa", "Naivasha", "Nanyuki",
    "Nyahururu", "Moyale", "Iten", "Kapsabet", "Bomet", "Maralal"
]

TRANSACTION_TYPES = [
    "Send Money", "Buy Goods", "Pay Bill", "Withdraw",
    "Pochi la Biashara", "Airtime Purchase", "Lipa na Mpesa",
]

TYPE_WEIGHTS = [0.30, 0.25, 0.15, 0.10, 0.08, 0.07, 0.05]

AMOUNT_RANGES = {
    "Send Money":        (50, 70000),
    "Buy Goods":         (20, 15000),
    "Pay Bill":          (100, 50000),
    "Withdraw":          (100, 70000),
    "Pochi la Biashara": (10, 5000),
    "Airtime Purchase":  (5, 1000),
    "Lipa na Mpesa":     (50, 30000),
}

# Database connection

def get_engine():
    url = (
        f"postgresql+psycopg2://{os.getenv('DB_USER')}:{os.getenv('DB_PASSWORD')}"
        f"@{os.getenv('DB_HOST')}:{os.getenv('DB_PORT')}/{os.getenv('DB_NAME')}"
    )
    return create_engine(url)


# Transaction generation
# Timestamps are spread across the past 30 days so velocity features (txn_count_last_10min, txn_count_last_1hr) have realistic variation instead of all 10,000 rows landing at the same second.

def generate_timestamp(base_time: datetime, index: int, total: int) -> datetime:
    """
    Spreads transactions evenly across the past 30 days.
    index/total gives a fraction between 0 and 1.
    Multiplying by 30 days worth of seconds gives an offset.
    We subtract so timestamps go from 30 days ago up to now.
    """
    thirty_days_in_seconds = 30 * 24 * 3600
    offset_seconds = (index / total) * thirty_days_in_seconds
    return base_time - timedelta(seconds=thirty_days_in_seconds - offset_seconds)


def generate_transaction(index: int, total: int, base_time: datetime) -> dict:
    """
    Builds one raw transaction as a plain dict matching the raw_transactions schema.
    """
    txn_type = random.choices(TRANSACTION_TYPES, weights=TYPE_WEIGHTS)[0]
    amount = round(random.uniform(*AMOUNT_RANGES[txn_type]), 2)
    balance_before = round(random.uniform(amount, 150000), 2)
    balance_after = round(balance_before - amount, 2)

    sender_name = random.choice(CUSTOMER_NAMES)
    receiver_name = random.choice(CUSTOMER_NAMES)
    while receiver_name == sender_name:
        receiver_name = random.choice(CUSTOMER_NAMES)

    timestamp = generate_timestamp(base_time, index, total)

    return {
        'transaction_id': f"TXN-{uuid.uuid4().hex[:12].upper()}",
        'transaction_type': txn_type,
        'sender_phone': f"07{random.randint(10,99)}{random.randint(100000,999999)}",
        'receiver_phone': f"07{random.randint(10,99)}{random.randint(100000,999999)}",
        'sender_name': sender_name,
        'receiver_name': receiver_name,
        'amount': amount,
        'sender_balance_before': balance_before,
        'sender_balance_after': balance_after,
        'location': random.choice(LOCATIONS),
        'device_fingerprint': f"DEV-{uuid.uuid4().hex[:12].upper()}",
        'timestamp': timestamp,
    }


# Insert raw transaction into PostgreSQL
# ON CONFLICT DO NOTHING skips any duplicate transaction_id safely

def insert_raw_transaction(conn, txn: dict):
    sql = text("""
        INSERT INTO raw_transactions (
            transaction_id, transaction_type, sender_phone, receiver_phone,
            sender_name, receiver_name, amount, sender_balance_before,
            sender_balance_after, location, device_fingerprint, timestamp
        ) VALUES (
            :transaction_id, :transaction_type, :sender_phone, :receiver_phone,
            :sender_name, :receiver_name, :amount, :sender_balance_before,
            :sender_balance_after, :location, :device_fingerprint, :timestamp
        )
        ON CONFLICT (transaction_id) DO NOTHING
    """)
    conn.execute(sql, txn)


# Feature computation
# These mirror the logic in the five feature modules but run inline so it does not need to call feature_pipeline.py as a subprocess.
# All lookups query raw_transactions using the same transaction's timestamp as the reference point

def compute_features(txn: dict, conn) -> dict:
    """
    Computes all 20 features for one transaction.
    Uses the transaction's own timestamp as the reference point for all window-based queries (velocity, amount history, behavioral history).
    """
    ts = txn['timestamp']
    sender = txn['sender_name']
    receiver = txn['receiver_name']
    amount = float(txn['amount'])
    device = txn['device_fingerprint']
    location = txn['location']

    # Temporal features (4 features)
    # These come purely from the timestamp so no database lookup needed
    hour = ts.hour
    day_of_week = ts.weekday()       # 0=Monday, 6=Sunday
    is_night = hour < 6 or hour >= 22
    is_weekend = day_of_week >= 5
    is_month_start = ts.day <= 3
    is_month_end = ts.day >= 28

    # Velocity features (4 features)
    # Count how many transactions this sender made in the last 10 min and 1 hr
    # We look back from this transaction's timestamp, not from now()
    ten_min_ago = ts - timedelta(minutes=10)
    one_hr_ago = ts - timedelta(hours=1)

    r = conn.execute(text("""
        SELECT
            COUNT(*) FILTER (WHERE timestamp >= :ten_min_ago) AS count_10min,
            COUNT(*) FILTER (WHERE timestamp >= :one_hr_ago)  AS count_1hr,
            COALESCE(SUM(amount) FILTER (WHERE timestamp >= :ten_min_ago), 0) AS sum_10min,
            COALESCE(SUM(amount) FILTER (WHERE timestamp >= :one_hr_ago), 0)  AS sum_1hr
        FROM raw_transactions
        WHERE sender_name = :sender
          AND timestamp < :ts
    """), {'sender': sender, 'ts': ts,
           'ten_min_ago': ten_min_ago, 'one_hr_ago': one_hr_ago}).fetchone()

    txn_count_last_10min = int(r[0])
    txn_count_last_1hr = int(r[1])
    txn_sum_last_10min = float(r[2])
    txn_sum_last_1hr = float(r[3])

    # Amount features (3 features)
    # Compare this transaction's amount against this sender's historical average
    r2 = conn.execute(text("""
        SELECT AVG(amount), STDDEV(amount)
        FROM raw_transactions
        WHERE sender_name = :sender
          AND timestamp < :ts
    """), {'sender': sender, 'ts': ts}).fetchone()

    sender_mean = float(r2[0]) if r2[0] else amount
    sender_std = float(r2[1]) if r2[1] else 1.0

    # z-score: how many standard deviations above the mean is this amount?
    # A z-score > 3 means statistically extreme
    amount_zscore = round((amount - sender_mean) / sender_std, 4) if sender_std > 0 else 0.0
    # ratio: how many times the sender's average is this amount?
    amount_vs_sender_mean = round(amount / sender_mean, 4) if sender_mean > 0 else 1.0
    # Large amount threshold: KES 50,000
    is_large_amount = amount > 50000

    # Behavioral features (3 features)
    # Check if this device or location has been seen before for this sender
    r3 = conn.execute(text("""
        SELECT
            COUNT(*) FILTER (WHERE device_fingerprint = :device) AS device_count,
            COUNT(*) FILTER (WHERE location = :location)          AS location_count,
            COUNT(DISTINCT receiver_name) FILTER (WHERE timestamp >= :one_hr_ago) AS unique_recv
        FROM raw_transactions
        WHERE sender_name = :sender
          AND timestamp < :ts
    """), {'sender': sender, 'device': device, 'location': location,
           'ts': ts, 'one_hr_ago': one_hr_ago}).fetchone()

    is_new_device = int(r3[0]) == 0
    is_new_location = int(r3[1]) == 0
    unique_receivers_last_1hr = int(r3[2])

    # Type frequency feature (1 feature)
    # How often does this sender use this transaction type? (fraction 0-1)
    r4 = conn.execute(text("""
        SELECT
            COUNT(*) FILTER (WHERE transaction_type = :txn_type) AS type_count,
            COUNT(*) AS total_count
        FROM raw_transactions
        WHERE sender_name = :sender
          AND timestamp < :ts
    """), {'sender': sender, 'txn_type': txn['transaction_type'], 'ts': ts}).fetchone()

    total_count = int(r4[1])
    type_frequency = round(int(r4[0]) / total_count, 4) if total_count > 0 else 0.0

    return {
        'transaction_id': txn['transaction_id'],
        'amount': amount,
        'transaction_type': txn['transaction_type'],
        'hour_of_day': hour,
        'day_of_week': day_of_week,
        'is_night': is_night,
        'is_weekend': is_weekend,
        'is_month_start': is_month_start,
        'is_month_end': is_month_end,
        'txn_count_last_10min': txn_count_last_10min,
        'txn_count_last_1hr': txn_count_last_1hr,
        'txn_sum_last_10min': txn_sum_last_10min,
        'txn_sum_last_1hr': txn_sum_last_1hr,
        'amount_zscore': amount_zscore,
        'amount_vs_sender_mean': amount_vs_sender_mean,
        'is_large_amount': is_large_amount,
        'is_new_device': is_new_device,
        'is_new_location': is_new_location,
        'unique_receivers_last_1hr': unique_receivers_last_1hr,
        'type_frequency': type_frequency,
    }


# Fraud labeling using exact same rules as label_generator.py
# Kept here so the generator is self-contained and runs in one pass

def apply_fraud_rules(features: dict) -> tuple:
    """
    Same six rules as label_generator.py.
    Returns (is_fraud, fraud_score, fraud_reasons_string).
    """
    score = 0
    reasons = []

    if features['is_night'] and features['amount_vs_sender_mean'] > 3.0:
        score += 1
        reasons.append("night_high_amount")

    if features['txn_count_last_10min'] >= 5:
        score += 1
        reasons.append("high_velocity_10min")

    if features['is_new_device'] and features['is_new_location']:
        score += 1
        reasons.append("new_device_and_location")

    if features['amount_zscore'] > 3.0:
        score += 1
        reasons.append("extreme_amount_zscore")

    if features['is_large_amount'] and features['is_new_device']:
        score += 1
        reasons.append("large_amount_new_device")

    if features['unique_receivers_last_1hr'] >= 4:
        score += 1
        reasons.append("many_unique_receivers")

    is_fraud = score >= 2
    reasons_str = ', '.join(reasons) if reasons else 'none'
    return is_fraud, score, reasons_str


# Insert computed features + labels into the features table

def insert_features(conn, features: dict, is_fraud: bool,
                    fraud_score: int, fraud_reasons: str):
    sql = text("""
        INSERT INTO features (
            transaction_id, amount, transaction_type,
            hour_of_day, day_of_week, is_night, is_weekend,
            is_month_start, is_month_end,
            txn_count_last_10min, txn_count_last_1hr,
            txn_sum_last_10min, txn_sum_last_1hr,
            amount_zscore, amount_vs_sender_mean, is_large_amount,
            is_new_device, is_new_location, unique_receivers_last_1hr,
            type_frequency, is_fraud, fraud_score, fraud_reasons
        ) VALUES (
            :transaction_id, :amount, :transaction_type,
            :hour_of_day, :day_of_week, :is_night, :is_weekend,
            :is_month_start, :is_month_end,
            :txn_count_last_10min, :txn_count_last_1hr,
            :txn_sum_last_10min, :txn_sum_last_1hr,
            :amount_zscore, :amount_vs_sender_mean, :is_large_amount,
            :is_new_device, :is_new_location, :unique_receivers_last_1hr,
            :type_frequency, :is_fraud, :fraud_score, :fraud_reasons
        )
        ON CONFLICT (transaction_id) DO NOTHING
    """)
    conn.execute(sql, {**features,
                       'is_fraud': is_fraud,
                       'fraud_score': fraud_score,
                       'fraud_reasons': fraud_reasons})


# Main: orchestrates the full generation run

def main():
    TARGET = 10000
    BATCH_SIZE = 500  # Commit to database every 500 rows to avoid one giant transaction

    print(f"Synthetic data generator starting...")
    print(f"Target: {TARGET} transactions")
    print(f"Batch size: {BATCH_SIZE}")
    print("-" * 50)

    engine = get_engine()
    base_time = datetime.now()

    fraud_count = 0
    legit_count = 0
    failed_count = 0

    # It processes in batches. Each batch opens one database connection, inserts BATCH_SIZE rows, then commits. This is much faster than opening a new connection for every single row.
    for batch_start in range(0, TARGET, BATCH_SIZE):
        batch_end = min(batch_start + BATCH_SIZE, TARGET)
        batch_num = (batch_start // BATCH_SIZE) + 1
        total_batches = (TARGET + BATCH_SIZE - 1) // BATCH_SIZE

        print(f"Batch {batch_num}/{total_batches} "
              f"(rows {batch_start + 1} to {batch_end})...")

        with engine.begin() as conn:
            # engine.begin() opens a connection and starts a transaction.
            # If anything inside fails, the whole batch rolls back automatically.
            # On success it commits automatically at the end of the with block.
            for i in range(batch_start, batch_end):
                try:
                    # Step 1: Generate the raw transaction dict
                    txn = generate_transaction(i, TARGET, base_time)

                    # Step 2: Insert into raw_transactions
                    insert_raw_transaction(conn, txn)

                    # Step 3: Compute all 20 features using historical data already in the database (earlier rows in this same batch are visible because they share the same transaction)                   
                    features = compute_features(txn, conn)

                    # Step 4: Apply fraud rules and get label
                    is_fraud, fraud_score, fraud_reasons = apply_fraud_rules(features)

                    # Step 5: Insert features + label into features table
                    insert_features(conn, features, is_fraud, fraud_score, fraud_reasons)

                    if is_fraud:
                        fraud_count += 1
                    else:
                        legit_count += 1

                except Exception as e:
                    failed_count += 1
                    print(f"  [FAILED] row {i + 1}: {e}")

        total_so_far = fraud_count + legit_count
        fraud_rate = (fraud_count / total_so_far * 100) if total_so_far > 0 else 0
        print(f"  Done. Running total: {total_so_far} | "
              f"Fraud: {fraud_count} ({fraud_rate:.1f}%) | "
              f"Failed: {failed_count}")

    # Final summary
    total = fraud_count + legit_count
    fraud_rate = (fraud_count / total * 100) if total > 0 else 0

    print("\n" + "=" * 50)
    print("Generation complete.")
    print(f"Total inserted:  {total}")
    print(f"Fraud:           {fraud_count} ({fraud_rate:.1f}%)")
    print(f"Legitimate:      {legit_count}")
    print(f"Failed:          {failed_count}")
    print("=" * 50)
    print("\nNext steps:")
    print("  1. Run train.py to retrain models on the new data")
    print("  2. Update README figures with new counts")


if __name__ == "__main__":
    main()