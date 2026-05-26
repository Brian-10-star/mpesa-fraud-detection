# test_features.py
# Tests for the feature engineering pipeline.
# Each test function name starts with "test_" pytest automatically discovers and runs any function with this naming convention.
# We test that each feature module:
# 1. Returns a dictionary (correct type)
# 2. Contains the expected keys
# 3. Returns values in valid ranges

# We use a mock transaction; a hardcoded dict that looks like a real transaction but doesn't require Kafka or PostgreSQL to create.
# For database-dependent features (velocity, amount, behavioral), we use a real DB connection to frauddb.

import pytest
import sys
import os
from datetime import datetime

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.features.temporal_features import extract_temporal_features
from src.features.amount_features import extract_amount_features
from src.features.velocity_features import extract_velocity_features
from src.features.behavioral_features import extract_behavioral_features


# Mock transaction
# This is a fake transaction used across all tests.
# Using a fixed timestamp makes tests deterministic — same result every run.
MOCK_TXN = {
    'transaction_id': 'TXN-TEST-001',
    'transaction_type': 'Send Money',
    'sender_phone': '0700000001',
    'receiver_phone': '0700000002',
    'sender_name': 'Test Sender',
    'receiver_name': 'Test Receiver',
    'amount': 5000.0,
    'sender_balance_before': 20000.0,
    'sender_balance_after': 15000.0,
    'location': 'Nairobi CBD',
    'device_fingerprint': 'DEV-TEST-001',
    'timestamp': '2026-05-19T14:30:00'
}

NIGHT_TXN = {**MOCK_TXN, 'timestamp': '2026-05-19T03:00:00'}
WEEKEND_TXN = {**MOCK_TXN, 'timestamp': '2026-05-17T10:00:00'}  # Sunday


# Temporal feature tests

def test_temporal_returns_dict():
    """extract_temporal_features must return a dictionary."""
    result = extract_temporal_features(MOCK_TXN)
    assert isinstance(result, dict)


def test_temporal_has_required_keys():
    """All six temporal features must be present in the result."""
    result = extract_temporal_features(MOCK_TXN)
    required = ['hour_of_day', 'day_of_week', 'is_night',
                'is_weekend', 'is_month_start', 'is_month_end']
    for key in required:
        assert key in result, f"Missing key: {key}"


def test_hour_of_day_valid_range():
    """hour_of_day must be between 0 and 23 inclusive."""
    result = extract_temporal_features(MOCK_TXN)
    assert 0 <= result['hour_of_day'] <= 23


def test_day_of_week_valid_range():
    """day_of_week must be between 0 (Monday) and 6 (Sunday)."""
    result = extract_temporal_features(MOCK_TXN)
    assert 0 <= result['day_of_week'] <= 6


def test_is_night_true_at_3am():
    """Transactions at 3am must be flagged as night."""
    result = extract_temporal_features(NIGHT_TXN)
    assert result['is_night'] is True


def test_is_night_false_at_2pm():
    """Transactions at 2pm must NOT be flagged as night."""
    result = extract_temporal_features(MOCK_TXN)
    assert result['is_night'] is False


def test_is_weekend_sunday():
    """Transactions on Sunday must be flagged as weekend."""
    result = extract_temporal_features(WEEKEND_TXN)
    assert result['is_weekend'] is True


def test_temporal_booleans_are_bool_type():
    """Boolean features must be Python bool type, not int or string."""
    result = extract_temporal_features(MOCK_TXN)
    for key in ['is_night', 'is_weekend', 'is_month_start', 'is_month_end']:
        assert isinstance(result[key], bool), f"{key} is not bool"


# Amount feature tests

@pytest.fixture
def db_engine():
    """
    pytest fixture creates a database engine shared across tests.
    A fixture is a reusable setup function. Any test that declares 'db_engine' as a parameter automatically gets this engine injected.
    scope='module' means the engine is created once per test file, not once per test.
    """
    from sqlalchemy import create_engine
    from dotenv import load_dotenv
    load_dotenv()
    url = (
        f"postgresql+psycopg2://{os.getenv('DB_USER')}:{os.getenv('DB_PASSWORD')}"
        f"@{os.getenv('DB_HOST')}:{os.getenv('DB_PORT')}/{os.getenv('DB_NAME')}"
    )
    return create_engine(url)


def test_amount_returns_dict(db_engine):
    """extract_amount_features must return a dictionary."""
    result = extract_amount_features(MOCK_TXN, db_engine)
    assert isinstance(result, dict)


def test_amount_has_required_keys(db_engine):
    """All three amount features must be present."""
    result = extract_amount_features(MOCK_TXN, db_engine)
    required = ['amount_zscore', 'amount_vs_sender_mean', 'is_large_amount']
    for key in required:
        assert key in result, f"Missing key: {key}"


def test_amount_vs_sender_mean_positive(db_engine):
    """amount_vs_sender_mean must always be positive."""
    result = extract_amount_features(MOCK_TXN, db_engine)
    assert result['amount_vs_sender_mean'] > 0


def test_large_amount_flag(db_engine):
    """Amounts above 50000 must be flagged as large."""
    large_txn = {**MOCK_TXN, 'amount': 60000.0,
                 'sender_balance_before': 100000.0,
                 'sender_balance_after': 40000.0}
    result = extract_amount_features(large_txn, db_engine)
    assert result['is_large_amount'] is True


def test_small_amount_not_flagged(db_engine):
    """Amounts below 50000 must NOT be flagged as large."""
    result = extract_amount_features(MOCK_TXN, db_engine)
    assert result['is_large_amount'] is False