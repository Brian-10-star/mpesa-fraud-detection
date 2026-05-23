# schemas.py
# Defines the data shapes for API requests and responses using Pydantic.
# Pydantic validates incoming data automatically — if a required field
# is missing or the wrong type, FastAPI returns a clear error message
# before our code even runs. This prevents bad data from reaching the model.

from pydantic import BaseModel
from typing import Optional
from datetime import datetime


class TransactionRequest(BaseModel):
    """
    The shape of a transaction sent to POST /predict.
    Every field here must be present in the request body.
    The types are enforced — sending a string for amount raises an error.
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
    fraud_probability: 0.0 to 1.0 — how likely this is fraud
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
    total_predictions: int
    fraud_detected: int
    legitimate: int
    fraud_rate: float


class ModelInfoResponse(BaseModel):
    model_name: str
    model_version: str
    tracking_uri: str