# schemas.py
# Defines the data shapes for API requests and responses using Pydantic.
# Pydantic validates incoming data automatically where if a required field is missing or the wrong type, FastAPI returns a clear error message before the code even runs to prevents bad data from reaching the model.

from pydantic import BaseModel
from typing import Optional
from datetime import datetime


class TransactionRequest(BaseModel):
    """
    The shape of a transaction sent to POST /predict.
    """
    transaction_id: str
    transaction_type: str
    sender_phone: str
    receiver_phone: str
    sender_name: str
    receiver_name: str
    amount: float
    sender_balance_before: float
    sender_balance_after: float
    location: str
    device_fingerprint: str
    timestamp: str


class PredictionResponse(BaseModel):
    """
    The shape of what POST /predict returns.
    fraud_probability: 0.0 to 1.0 shows how likely this is fraud
    is_fraud: True if probability >= 0.5
    model_version: which model made this prediction
    """
    transaction_id: str
    fraud_probability: float
    is_fraud: bool
    model_version: str
    message: str


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_version: str


class MetricsResponse(BaseModel):
    # All-time counts
    total_predictions: int
    fraud_detected: int
    legitimate: int
    fraud_rate: float

    # Last 24 hours
    fraud_rate_24h: float
    total_last_24h: int
    fraud_last_24h: int

    # Latency percentiles in milliseconds computed from live requests only.
    # None when no latency data exists yet, for example right after startup.
    p50_latency_ms: Optional[float]
    p95_latency_ms: Optional[float]
    p99_latency_ms: Optional[float]

    # Predictions per minute over the last hour
    throughput_per_minute: float


class ModelInfoResponse(BaseModel):
    model_name: str
    model_version: str
    tracking_uri: str