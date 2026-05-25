# alert_logger.py
# Saves drift alerts to the drift_alerts table in frauddb.
# Called by drift_detector.py whenever a feature shows significant drift.

from sqlalchemy import create_engine, text
from dotenv import load_dotenv
import os

load_dotenv()


def get_db_engine():
    url = (
        f"postgresql+psycopg2://{os.getenv('DB_USER')}:{os.getenv('DB_PASSWORD')}"
        f"@{os.getenv('DB_HOST')}:{os.getenv('DB_PORT')}/{os.getenv('DB_NAME')}"
    )
    return create_engine(url)


engine = get_db_engine()


def log_drift_alert(feature_name: str, drift_score: float, alert_level: str):
    """
    Inserts one drift alert into the drift_alerts table.

    feature_name: which feature drifted e.g. 'amount'
    drift_score: the statistical distance score (0.0 = no drift, 1.0 = complete drift)
    alert_level: 'LOW', 'MEDIUM', or 'HIGH' based on drift_score thresholds
    """
    sql = text("""
        INSERT INTO drift_alerts (feature_name, drift_score, alert_level)
        VALUES (:feature_name, :drift_score, :alert_level)
    """)
    with engine.connect() as conn:
        conn.execute(sql, {
            'feature_name': feature_name,
            'drift_score': float(round(drift_score, 4)),
            'alert_level': alert_level
        })
        conn.commit()


def get_alert_level(drift_score: float) -> str:
    """
    Converts a numeric drift score into a human-readable alert level.
    These thresholds are industry-standard starting points:
    - Below 0.1: normal statistical variation, not worth alerting
    - 0.1 to 0.25: worth watching, may need retraining soon
    - Above 0.25: significant drift, retraining recommended
    """
    if drift_score >= 0.25:
        return "HIGH"
    elif drift_score >= 0.1:
        return "MEDIUM"
    else:
        return "LOW"