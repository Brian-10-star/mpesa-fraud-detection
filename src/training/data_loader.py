# data_loader.py
# Loads the labeled feature data from PostgreSQL into pandas DataFrames
# ready for ML training. Separates features (X) from labels (y).
# Also handles the train/test split — we train on 80% of data and
# evaluate on the remaining 20% the model has never seen.

import pandas as pd
from sqlalchemy import create_engine, text
from dotenv import load_dotenv
from sklearn.model_selection import train_test_split
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
load_dotenv()


def get_db_engine():
    url = (
        f"postgresql+psycopg2://{os.getenv('DB_USER')}:{os.getenv('DB_PASSWORD')}"
        f"@{os.getenv('DB_HOST')}:{os.getenv('DB_PORT')}/{os.getenv('DB_NAME')}"
    )
    return create_engine(url)


# These are the exact columns the ML model will use as input.
# We exclude: id, transaction_id (identifiers not useful for ML),
# created_at (timestamp of insertion), fraud_reasons (text, not numeric),
# fraud_score (derived from rules, would leak the label), is_fraud (the label itself)
FEATURE_COLUMNS = [
    'amount',
    'hour_of_day',
    'day_of_week',
    'is_night',
    'is_weekend',
    'is_month_start',
    'is_month_end',
    'txn_count_last_10min',
    'txn_count_last_1hr',
    'txn_sum_last_10min',
    'txn_sum_last_1hr',
    'amount_zscore',
    'amount_vs_sender_mean',
    'is_large_amount',
    'is_new_device',
    'is_new_location',
    'unique_receivers_last_1hr',
    'type_frequency',
]

LABEL_COLUMN = 'is_fraud'


def load_features(engine) -> pd.DataFrame:
    """
    Loads all labeled rows from the features table.
    We only load rows where fraud_reasons IS NOT NULL —
    meaning the label generator has already processed them.
    """
    sql = text("""
        SELECT * FROM features
        WHERE fraud_reasons IS NOT NULL
        ORDER BY created_at ASC
    """)
    with engine.connect() as conn:
        df = pd.read_sql(sql, conn)

    print(f"Loaded {len(df)} labeled rows from features table.")
    print(f"Fraud: {df[LABEL_COLUMN].sum()} | "
          f"Legitimate: {(~df[LABEL_COLUMN]).sum()} | "
          f"Fraud rate: {df[LABEL_COLUMN].mean()*100:.1f}%")
    return df


def prepare_data(df: pd.DataFrame):
    """
    Splits the DataFrame into:
    - X: the feature matrix (inputs to the model)
    - y: the label vector (what we want the model to predict)

    Then splits into train/test sets.
    test_size=0.2 means 20% goes to testing, 80% to training.
    stratify=y ensures both splits have the same fraud ratio —
    without this, we might get all fraud in training and none in test.
    random_state=42 makes the split reproducible — same split every run.

    Boolean columns need to be converted to int (0/1) because
    scikit-learn models expect numeric input only.
    """
    # Convert boolean columns to integers
    bool_cols = ['is_night', 'is_weekend', 'is_month_start', 'is_month_end',
                 'is_large_amount', 'is_new_device', 'is_new_location']
    for col in bool_cols:
        df[col] = df[col].astype(int)

    X = df[FEATURE_COLUMNS]
    y = df[LABEL_COLUMN].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    print(f"Training set: {len(X_train)} rows | Test set: {len(X_test)} rows")
    return X_train, X_test, y_train, y_test