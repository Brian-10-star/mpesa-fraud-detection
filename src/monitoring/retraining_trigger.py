# retraining_trigger.py
# Automated retraining trigger for the MLOps loop where it reads drift results from the current monitoring run and decides whether the model needs retraining based on how many features show HIGH drift simultaneously.

# The MLOps loop this implements:
# 1. Data comes in through Kafka and lands in raw_transactions
# 2. Features are computed and stored in features table
# 3. monitor.py runs drift detection comparing training distribution to recent prediction distribution
# 4. If enough features have drifted significantly, this module triggers train.py as a subprocess
# 5. train.py retrains all three models, picks the best by recall and registers it in MLflow as a new version
# 6. The FastAPI model_loader picks up the new version on next restart

import os
import sys
import subprocess
from sqlalchemy import create_engine, text
from dotenv import load_dotenv
from datetime import datetime

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
load_dotenv()

from src.api.logger import get_logger

logger = get_logger(__name__, service="monitoring")

HIGH_DRIFT_THRESHOLD = 3

def get_db_engine():
    url = (
        f"postgresql+psycopg2://{os.getenv('DB_USER')}:{os.getenv('DB_PASSWORD')}"
        f"@{os.getenv('DB_HOST')}:{os.getenv('DB_PORT')}/{os.getenv('DB_NAME')}"
    )
    return create_engine(url)


def log_retraining_event(engine, trigger_reason: str, 
                         high_drift_features: list,
                         status: str, notes: str = ""):
    """
    Writes one retraining event to retraining_log.
    status is either 'triggered', 'completed', 'failed', or 'skipped'.
    This table gives a full audit trail of when and why retraining ran.
    """
    sql = text("""
        INSERT INTO retraining_log (
            triggered_at, trigger_reason, high_drift_features,
            high_drift_count, status, notes
        ) VALUES (
            :triggered_at, :trigger_reason, :high_drift_features,
            :high_drift_count, :status, :notes
        )
    """)
    with engine.connect() as conn:
        conn.execute(sql, {
            'triggered_at': datetime.now(),
            'trigger_reason': trigger_reason,
            'high_drift_features': ', '.join(high_drift_features),
            'high_drift_count': len(high_drift_features),
            'status': status,
            'notes': notes
        })
        conn.commit()


def should_retrain(drift_results: dict) -> tuple:
    """
    Evaluates drift results and decides whether retraining is needed.
    Returns a tuple of (should_retrain: bool, high_drift_features: list).

    We collect all features with HIGH drift level (score >= 0.25).
    If the count meets or exceeds HIGH_DRIFT_THRESHOLD, retraining fires.
    """
    from src.monitoring.alert_logger import get_alert_level

    high_drift_features = [
        feature for feature, score in drift_results.items()
        if get_alert_level(score) == "HIGH"
    ]

    trigger = len(high_drift_features) >= HIGH_DRIFT_THRESHOLD
    return trigger, high_drift_features


def run_retraining(engine, high_drift_features: list) -> bool:
    """
    Runs train.py as a subprocess and logs the outcome.
    Returns True if training completed successfully and False otherwise.

    subprocess.run() blocks until train.py finishes.
    capture_output=True captures stdout and stderr so we can log them.
    sys.executable uses the same Python interpreter that is running this script, ensuring the same virtual environment is active.
    """
    trigger_reason = (
        f"{len(high_drift_features)} features exceeded HIGH drift threshold "
        f"of {HIGH_DRIFT_THRESHOLD}."
    )

    logger.warning("retraining_triggered", extra={
        "high_drift_count": len(high_drift_features),
        "high_drift_features": high_drift_features,
        "trigger_reason": trigger_reason
    })

    log_retraining_event(
        engine=engine,
        trigger_reason=trigger_reason,
        high_drift_features=high_drift_features,
        status="triggered",
        notes="Retraining subprocess starting."
    )

    try:
        # Build the path to train.py relative to the project root
        train_script = os.path.join(
            os.path.dirname(__file__), '..', '..', 'src', 'training', 'train.py'
        )
        train_script = os.path.abspath(train_script)

        logger.info("retraining_subprocess_starting", extra={
            "script": train_script
        })

        subprocess_env = {**os.environ, "PYTHONUTF8": "1"}

        result = subprocess.run(
            [sys.executable, train_script],
            capture_output=True,
            text=True,
            timeout=600,
            env=subprocess_env,
            encoding="utf-8"
        )

        if result.returncode == 0:
            logger.info("retraining_completed", extra={
                "returncode": result.returncode
            })
            log_retraining_event(
                engine=engine,
                trigger_reason=trigger_reason,
                high_drift_features=high_drift_features,
                status="completed",
                notes=result.stdout[-500:] if result.stdout else ""
                # Stores only the last 500 characters of stdout to avoid filling the database with training logs
            )
            return True

        else:
            logger.error("retraining_failed", extra={
                "returncode": result.returncode,
                "stderr": result.stderr[-500:] if result.stderr else "",
                "stdout": result.stdout[-500:] if result.stdout else ""
            })
            log_retraining_event(
                engine=engine,
                trigger_reason=trigger_reason,
                high_drift_features=high_drift_features,
                status="failed",
                notes=result.stderr[-500:] if result.stderr else ""
            )
            return False

    except subprocess.TimeoutExpired:
        logger.error("retraining_timeout", extra={
            "timeout_seconds": 600
        })
        log_retraining_event(
            engine=engine,
            trigger_reason=trigger_reason,
            high_drift_features=high_drift_features,
            status="failed",
            notes="Training subprocess timed out after 600 seconds."
        )
        return False

    except Exception as e:
        logger.error("retraining_error", extra={"error": str(e)})
        log_retraining_event(
            engine=engine,
            trigger_reason=trigger_reason,
            high_drift_features=high_drift_features,
            status="failed",
            notes=str(e)
        )
        return False


def evaluate_and_trigger(drift_results: dict) -> bool:
    """
    Main entry point called by monitor.py after drift detection.
    Evaluates drift results, triggers retraining if needed, and returns True if retraining was triggered.
    """
    engine = get_db_engine()
    trigger, high_drift_features = should_retrain(drift_results)

    if not trigger:
        logger.info("retraining_not_needed", extra={
            "high_drift_count": len(high_drift_features),
            "threshold": HIGH_DRIFT_THRESHOLD
        })
        log_retraining_event(
            engine=engine,
            trigger_reason="Drift check passed. No retraining needed.",
            high_drift_features=high_drift_features,
            status="skipped",
            notes=f"High drift features: {len(high_drift_features)}, "
                  f"threshold: {HIGH_DRIFT_THRESHOLD}"
        )
        return False

    success = run_retraining(engine, high_drift_features)
    return success