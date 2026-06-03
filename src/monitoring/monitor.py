# monitor.py
# Main monitoring pipeline. Loads reference and current data, runs drift detection across all features, saves an HTML report, and logs drift alerts to PostgreSQL.

import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from src.monitoring.drift_detector import (
    get_db_engine, load_reference_data,
    load_current_data, run_drift_detection, detect_and_alert
)
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

    # Run drift detection
    drift_results, report_path = run_drift_detection(reference, current)
    # Log alerts for drifted features
    alerted = detect_and_alert(engine, drift_results)

    print(f"\nDrift results per feature:")
    print("-" * 40)

    logger.info("drift_monitoring_complete", extra={
        "features_checked": len(drift_results),
        "drift_alerts_logged": alerted,
        "report_path": report_path
    })


if __name__ == "__main__":
    main()