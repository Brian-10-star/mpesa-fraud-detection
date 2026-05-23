# main.py
# FastAPI application entry point.
# Creates the app, loads the model at startup, and registers all routes.
#
# @app.on_event("startup") runs load_model() automatically when the
# API server starts — so the model is ready before the first request arrives.
# This is FastAPI's lifecycle hook system.

from fastapi import FastAPI
from contextlib import asynccontextmanager
from src.api.model_loader import load_model
from src.api.routes import predict, health, metrics, model_info
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Runs on startup — load model before accepting requests
    print("Loading ML model...")
    load_model()
    print("API ready.")
    yield
    # Runs on shutdown — cleanup if needed
    print("API shutting down.")


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