# test_api.py
# Tests the FastAPI endpoints using httpx which is an HTTP client library.
# These are integration tests which test the full request/response cycle including routing, validation, and response formatting.
# IMPORTANT: These tests require the API to be running.
# Start it first: uvicorn src.api.main:app --host 0.0.0.0 --port 8000
# We use httpx.Client to send real HTTP requests to the running API and assert on the response status codes and JSON structure.

import pytest
import httpx
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

BASE_URL = "http://127.0.0.1:8000"

# Sample legitimate transaction
LEGIT_TRANSACTION = {
    "transaction_id": "TXN-PYTEST-001",
    "transaction_type": "Buy Goods",
    "sender_phone": "0712345678",
    "receiver_phone": "0798765432",
    "sender_name": "Test User",
    "receiver_name": "Test Shop",
    "amount": 500.0,
    "sender_balance_before": 10000.0,
    "sender_balance_after": 9500.0,
    "location": "Westlands",
    "device_fingerprint": "DEV-PYTEST-001",
    "timestamp": "2026-05-19T14:30:00"
}

# Sample fraudulent transaction
FRAUD_TRANSACTION = {
    "transaction_id": "TXN-PYTEST-002",
    "transaction_type": "Send Money",
    "sender_phone": "0799999999",
    "receiver_phone": "0788888888",
    "sender_name": "Suspicious User",
    "receiver_name": "Unknown Receiver",
    "amount": 70000.0,
    "sender_balance_before": 75000.0,
    "sender_balance_after": 5000.0,
    "location": "Garissa",
    "device_fingerprint": "DEV-BRAND-NEW-XYZ",
    "timestamp": "2026-05-19T03:15:00"
}


@pytest.fixture(scope='module')
def client():
    """
    Creates an httpx client for all tests in this file. httpx.Client is like requests.Session which reuses the TCP connection across multiple requests for efficiency.
    """
    with httpx.Client(base_url=BASE_URL, timeout=30.0) as c:
        yield c


def test_root_endpoint(client):
    """GET / must return 200 and contain service name."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "service" in data
    assert "M-Pesa" in data["service"]


def test_health_endpoint(client):
    """GET /health must return 200 and model_loaded=true."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["model_loaded"] is True
    assert data["status"] == "healthy"


def test_model_info_endpoint(client):
    """GET /model-info must return model name and version."""
    response = client.get("/model-info")
    assert response.status_code == 200
    data = response.json()
    assert "model_name" in data
    assert data["model_name"] == "mpesa-fraud-detector"


def test_metrics_endpoint(client):
    """GET /metrics must return prediction statistics."""
    response = client.get("/metrics")
    assert response.status_code == 200
    data = response.json()
    assert "total_predictions" in data
    assert "fraud_detected" in data
    assert "fraud_rate" in data
    assert data["total_predictions"] >= 0


def test_predict_legitimate_transaction(client):
    """POST /predict must return a valid response for a legitimate transaction."""
    response = client.post("/predict", json=LEGIT_TRANSACTION)
    assert response.status_code == 200
    data = response.json()
    assert "transaction_id" in data
    assert "fraud_probability" in data
    assert "is_fraud" in data
    assert "message" in data
    assert 0.0 <= data["fraud_probability"] <= 1.0
    assert data["transaction_id"] == "TXN-PYTEST-001"


def test_predict_fraud_transaction(client):
    """POST /predict must flag a clearly fraudulent transaction."""
    response = client.post("/predict", json=FRAUD_TRANSACTION)
    assert response.status_code == 200
    data = response.json()
    assert data["is_fraud"] is True
    assert data["fraud_probability"] >= 0.5


def test_predict_missing_field(client):
    """POST /predict must return 422 when a required field is missing."""
    incomplete = {k: v for k, v in LEGIT_TRANSACTION.items()
                  if k != 'amount'}
    response = client.post("/predict", json=incomplete)
    # 422 Unprocessable Entity — Pydantic validation failed
    assert response.status_code == 422


def test_predict_invalid_type(client):
    """POST /predict must return 422 when amount is wrong type."""
    bad_txn = {**LEGIT_TRANSACTION,
               "transaction_id": "TXN-PYTEST-003",
               "amount": "not_a_number"}
    response = client.post("/predict", json=bad_txn)
    assert response.status_code == 422