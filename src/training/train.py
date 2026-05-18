# train.py
# The main training script. Orchestrates the full ML pipeline:
# 1. Load labeled data from PostgreSQL
# 2. Apply SMOTE to balance fraud vs legitimate
# 3. Train each model and log everything to MLflow
# 4. Pick the best model by recall score
# 5. Register the best model in MLflow model registry
#
# Run this file to train all models and track experiments.

import os
import sys
import mlflow
import mlflow.sklearn
from imblearn.over_sampling import SMOTE
from dotenv import load_dotenv

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
load_dotenv()

from src.training.data_loader import get_db_engine, load_features, prepare_data
from src.training.models import get_models
from src.training.evaluate import evaluate_model, print_metrics
from src.training.register_model import register_best_model


def main():
    print("Starting ML training pipeline...")
    print("="*50)

    # ── Step 1: Load data ─────────────────────────────────────────────────
    engine = get_db_engine()
    df = load_features(engine)
    X_train, X_test, y_train, y_test = prepare_data(df)

    # ── Step 2: Apply SMOTE to training data only ─────────────────────────
    # IMPORTANT: SMOTE is applied ONLY to training data, never to test data.
    # Applying it to test data would artificially inflate our metrics.
    # SMOTE creates synthetic fraud examples by interpolating between
    # existing fraud examples — it doesn't just duplicate them.
    print("\nApplying SMOTE to balance training data...")
    smote = SMOTE(random_state=42)
    X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
    print(f"Before SMOTE: {y_train.sum()} fraud, {(y_train==0).sum()} legit")
    print(f"After SMOTE:  {y_train_balanced.sum()} fraud, "
          f"{(y_train_balanced==0).sum()} legit")

    # ── Step 3: Configure MLflow ──────────────────────────────────────────
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"))
    mlflow.set_experiment("mpesa-fraud-detection")
    # set_experiment creates the experiment if it doesn't exist,
    # or uses the existing one if it does.

    models = get_models()
    best_recall = 0.0
    best_run_id = None
    best_model_name = None

    # ── Step 4: Train each model ──────────────────────────────────────────
    for model_config in models:
        name = model_config['name']
        model = model_config['model']
        params = model_config['params']

        print(f"\nTraining {name}...")

        # mlflow.start_run() opens a new experiment run.
        # Everything inside this block gets recorded to MLflow.
        # run_name makes it easy to identify in the dashboard.
        with mlflow.start_run(run_name=name):

            # Log hyperparameters — what settings we used
            mlflow.log_params(params)

            # Train the model on SMOTE-balanced data
            model.fit(X_train_balanced, y_train_balanced)

            # Evaluate on the ORIGINAL unbalanced test set —
            # this reflects real-world performance
            metrics = evaluate_model(model, X_test, y_test)
            print_metrics(name, metrics)

            # Log metrics to MLflow
            mlflow.log_metrics(metrics)

            # Save the model artifact to MLflow
            # This stores the actual trained model file so we can
            # load and use it later without retraining
            mlflow.sklearn.log_model(model, "model")

            # Track the best model by recall
            if metrics['recall'] > best_recall:
                best_recall = metrics['recall']
                best_run_id = mlflow.active_run().info.run_id
                best_model_name = name

    # ── Step 5: Register the best model ──────────────────────────────────
    print(f"\n{'='*50}")
    print(f"Best model: {best_model_name} with recall={best_recall:.4f}")

    if best_run_id:
        register_best_model(best_run_id, "mpesa-fraud-detector")

    print("\nTraining pipeline complete.")
    print(f"View experiments at: http://localhost:5000")


if __name__ == "__main__":
    main()