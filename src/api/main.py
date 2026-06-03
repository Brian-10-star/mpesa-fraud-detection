# main.py
# FastAPI application entry point.
# Creates the app, loads the model at startup, and registers all routes.

from fastapi import FastAPI
from contextlib import asynccontextmanager
from src.api.model_loader import load_model
from src.api.routes import predict, health, metrics, model_info
from src.api.logger import get_logger
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

logger = get_logger(__name__, service="fastapi")

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Runs on startup
    logger.info("api_starting")
    load_model()
    logger.info("api_ready")
    yield
    # Runs on shutdown: cleanup if needed
    logger.info("api_shutting_down")


app = FastAPI(
    title="M-Pesa Fraud Detection API",
    description="Real-time fraud detection for M-Pesa transactions",
    version="1.0.0",
    lifespan=lifespan
)

# Register all route modules
app.include_router(predict.router)
app.include_router(health.router)
app.include_router(metrics.router)
app.include_router(model_info.router)


@app.get("/")
def root():
    return {
        "service": "M-Pesa Fraud Detection API",
        "version": "1.0.0",
        "endpoints": ["/predict", "/health", "/metrics", "/model-info", "/docs"]
    }