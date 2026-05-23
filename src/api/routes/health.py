# routes/health.py
# GET /health — returns whether the API is running and model is loaded.
# Load balancers and monitoring tools call this endpoint to check
# if the service is healthy before routing traffic to it.

from fastapi import APIRouter
from src.api.schemas import HealthResponse
from src.api.model_loader import is_model_loaded, get_model_version, store

router = APIRouter()

@router.get("/health", response_model=HealthResponse)
def health_check():
    return HealthResponse(
        status="healthy" if is_model_loaded() else "degraded",
        model_loaded=is_model_loaded(),
        model_version=get_model_version()
    )