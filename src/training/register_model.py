# register_model.py
# Takes the best performing model and registers it in MLflow's
# model registry with the stage "Staging".
# In a real company, a data scientist would review it and promote
# it to "Production" after manual validation.

import mlflow
from mlflow.tracking import MlflowClient


def register_best_model(best_run_id: str, model_name: str):
    """
    Registers a trained model from an MLflow run into the model registry.

    best_run_id: the MLflow run ID of the winning model
    model_name: what to call it in the registry e.g. "mpesa-fraud-detector"

    mlflow.register_model() takes a model URI and a registry name.
    The URI format "runs:/<run_id>/model" tells MLflow where to find
    the model artifact from a specific training run.
    """
    client = MlflowClient()

    model_uri = f"runs:/{best_run_id}/model"
    print(f"\nRegistering model from run: {best_run_id}")

    registered = mlflow.register_model(
        model_uri=model_uri,
        name=model_name
    )

    print(f"Model registered: {registered.name} v{registered.version}")
    return registered