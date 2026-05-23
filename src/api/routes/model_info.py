# routes/model_info.py
# GET /model-info — returns metadata about the loaded model.
# Useful for debugging — quickly check which model version is serving.

from fastapi import APIRouter
from src.api.schemas import ModelInfoResponse
from src.api.model_loader import get_model_version, store
from dotenv import load_dotenv
import os

load_dotenv()
router = APIRouter()

@router.get("/model-info", response_model=ModelInfoResponse)
def model_info():
    return ModelInfoResponse(
        model_name=store.model_name,
        model_version=get_model_version(),
        tracking_uri=os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
    )