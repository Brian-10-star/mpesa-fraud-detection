# model_loader.py
# Loads the trained ML model from MLflow at API startup.
# Loads it only once when the API starts and not on every request.
# Loading a model takes about 1 second; doing it per request would make the API too slow for real-time fraud detection.
# The loaded model is stored in a global ModelStore object that all route handlers share.

import mlflow
import mlflow.sklearn
from dotenv import load_dotenv
from src.api.logger import get_logger
import os

load_dotenv()

logger = get_logger(__name__, service="fastapi")

class ModelStore:
    model = None
    version = "unknown"
    model_name = "mpesa-fraud-detector"

store = ModelStore()


def load_model():
    """
    Loads the latest version of the registered model from MLflow.
    
    models:/{model_name}/latest loads whichever version was registered most recently so retraining and registering a new version automatically gets picked up on next API restart.
    """
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI",
                                       "http://127.0.0.1:5000"))

    model_uri = f"models:/{store.model_name}/latest"
    logger.info("loading_model", extra={"model_uri": model_uri})

    store.model = mlflow.sklearn.load_model(model_uri)
    store.version = "latest"

    logger.info("model_loaded", extra={
        "model_name": store.model_name,
        "model_version": store.version
    })


def get_model():        
    return store.model


def get_model_version():   
    return store.version


def is_model_loaded():
    return store.model is not None