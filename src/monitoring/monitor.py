# monitor.py
# Main monitoring pipeline. Runs drift detection and triggers retraining automatically when significant drift is detected.
# This is the MLOps loop: detect drift, retrain, serve new model.

import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from src.monitoring.drift_detector import (
    get_db_engine, load_reference_data,
    load_current_data, run_drift_detection, detect_and_alert
)
from src.monitoring.retraining_trigger import evaluate_and_trigger
from src.api.logger import get_logger

logger = get_logger(__name__, service="monitoring")


def main():
    logger.info("drift_monitoring_started")

    engine = get_db_engine()

    # Load both datasets
    reference = load_reference_data(engine)
    current = load_current_data(engine)

    if len(current) == 0:
        logger.warning("no_current_data", extra={
            "hint": "Run batch_scorer.py first"
        })
        return

    if len(reference) == 0:
        logger.warning("no_reference_data", extra={
            "hint": "Run feature_pipeline.py first"
        })
        return

    logger.info("running_drift_detection", extra={
        "reference_rows": len(reference),
        "current_rows": len(current),
        "features_checked": len(reference.columns)
    })

    # Run drift detection across all 18 features
    drift_results, report_path = run_drift_detection(reference, current)

    # Log alerts for features that exceed drift thresholds
    alerted = detect_and_alert(engine, drift_results)

    logger.info("drift_detection_complete", extra={
        "features_checked": len(drift_results),
        "drift_alerts_logged": alerted,
        "report_path": report_path
    })

    # Evaluate drift results and trigger retraining if needed.
    # This call logs the decision to retraining_log whether or not retraining actually runs, giving a full audit trail.
    retrained = evaluate_and_trigger(drift_results)

    if retrained:
        logger.warning("retraining_complete_restart_required", extra={
            "message": (
                "New model registered in MLflow. "
                "Restart FastAPI container to load the new model version: "
                "docker-compose restart fastapi"
            )
        })

    logger.info("monitoring_pipeline_complete", extra={
        "drift_alerts": alerted,
        "retraining_triggered": retrained
    })


if __name__ == "__main__":
    main()