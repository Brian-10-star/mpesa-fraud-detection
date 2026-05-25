# monitor.py
# Main monitoring platform. Running this file to:
# 1. Load reference data (training features) and current data (recent predictions)
# 2. Run Evidently drift detection across all 18 features
# 3. Save an HTML drift report to data/reports/
# 4. Log any significant drift alerts to PostgreSQL

import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from src.monitoring.drift_detector import (
    get_db_engine, load_reference_data,
    load_current_data, run_drift_detection, detect_and_alert
)


def main():
    print("Starting drift monitoring...")
    print("=" * 50)

    engine = get_db_engine()

    # Load both datasets
    reference = load_reference_data(engine)
    current = load_current_data(engine)

    if len(current) == 0:
        print("No current data found. Run batch_scorer.py first.")
        return

    if len(reference) == 0:
        print("No reference data found. Run feature_pipeline.py first.")
        return

    print(f"\nRunning Evidently drift detection across {len(reference.columns)} features...")

    # Run drift detection
    drift_results, report_path = run_drift_detection(reference, current)

    print(f"\nDrift results per feature:")
    print("-" * 40)

    # Log alerts for drifted features
    alerted = detect_and_alert(engine, drift_results)

    print(f"\n{'='*50}")
    print(f"Monitoring complete.")
    print(f"Features checked: {len(drift_results)}")
    print(f"Drift alerts logged: {alerted}")
    print(f"HTML report: {report_path}")
    print(f"View alerts in DB: SELECT * FROM drift_alerts ORDER BY detected_at DESC;")


if __name__ == "__main__":
    main()