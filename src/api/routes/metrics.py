# routes/metrics.py
# GET /metrics — returns prediction statistics from the database.
# Shows how many transactions have been scored, how many flagged as fraud.

from fastapi import APIRouter
from src.api.schemas import MetricsResponse
from src.api.prediction_logger import get_prediction_metrics

router = APIRouter()

@router.get("/metrics", response_model=MetricsResponse)
def get_metrics():
    metrics = get_prediction_metrics()
    return MetricsResponse(**metrics)