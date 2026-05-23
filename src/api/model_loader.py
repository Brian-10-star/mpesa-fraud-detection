# model_loader.py
# Loads the trained ML model from MLflow at API startup.
# We load it ONCE when the API starts — not on every request.
# Loading a model takes ~1 second; doing it per request would make
# the API too slow for real-time fraud detection.
#
# The loaded model is stored in a global ModelStore object that
# all route handlers share.

import mlflow
import mlflow.sklearn
from dotenv import load_dotenv
import os

load_dotenv()

# Global store — holds the model and its version after loading
class ModelStore:
    model = None
    version = "unknown"
    model_name = "mpesa-fraud-detector"

store = ModelStore()


def load_model():
    """
    Loads the latest version of the registered model from MLflow.
    
    models:/{model_name}/latest loads whichever version was registered
    most recently — so retraining and registering a new version
    automatically gets picked up on next API restart.
    """
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI",
                                       "http://127.0.0.1:5000"))

    model_uri = f"models:/{store.model_name}/latest"
    print(f"Loading model from: {model_uri}")

    store.model = mlflow.sklearn.load_model(model_uri)
    store.version = "latest"
    print(f"Model loaded successfully: {store.model_name} ({store.version})")


def get_model():
    """Returns the loaded model. Called by route handlers."""
    return store.model


def get_model_version():
    """Returns the model version string."""
    return store.version


def is_model_loaded():
    """Returns True if model has been loaded successfully."""
    return store.model is not None