# drift_detector.py
# Custom drift detector using scipy statistical tests.
# Replaces Evidently AI due to Python 3.13 / Pydantic v2 incompatibility.
#
# Uses the same statistical tests Evidently uses internally:
# - Kolmogorov-Smirnov (KS) test for numerical features
# - Chi-squared test for boolean/categorical features
#
# KS test: measures the maximum distance between two cumulative
# distribution functions. p-value < 0.05 means significant drift.
# drift_score = 1 - p_value (higher = more drift)
#
# Chi-squared test: measures whether the frequency distribution
# of categories has changed. Same p-value interpretation.

import pandas as pd
from sqlalchemy import create_engine, text
from scipy import stats
from datetime import datetime
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from dotenv import load_dotenv
load_dotenv()

from src.monitoring.alert_logger import log_drift_alert, get_alert_level

NUMERICAL_FEATURES = [
    'amount', 'hour_of_day', 'day_of_week',
    'txn_count_last_10min', 'txn_count_last_1hr',
    'txn_sum_last_10min', 'txn_sum_last_1hr',
    'amount_zscore', 'amount_vs_sender_mean',
    'unique_receivers_last_1hr', 'type_frequency'
]

BOOLEAN_FEATURES = [
    'is_night', 'is_weekend', 'is_month_start', 'is_month_end',
    'is_large_amount', 'is_new_device', 'is_new_location'
]

ALL_FEATURES = NUMERICAL_FEATURES + BOOLEAN_FEATURES


def get_db_engine():
    url = (
        f"postgresql+psycopg2://{os.getenv('DB_USER')}:{os.getenv('DB_PASSWORD')}"
        f"@{os.getenv('DB_HOST')}:{os.getenv('DB_PORT')}/{os.getenv('DB_NAME')}"
    )
    return create_engine(url)


def load_reference_data(engine) -> pd.DataFrame:
    """Loads earliest 70% of labeled features as reference (training window)."""
    sql = text("""
        SELECT * FROM features
        WHERE fraud_reasons IS NOT NULL
        ORDER BY created_at ASC
    """)
    with engine.connect() as conn:
        df = pd.read_sql(sql, conn)

    cutoff = int(len(df) * 0.7)
    df = df.iloc[:cutoff]

    for col in BOOLEAN_FEATURES:
        df[col] = df[col].astype(int)

    print(f"Reference data: {len(df)} rows")
    return df[ALL_FEATURES]


def load_current_data(engine) -> pd.DataFrame:
    """Loads recent scored transactions as current data."""
    sql = text("""
        SELECT f.* FROM features f
        INNER JOIN prediction_log p ON f.transaction_id = p.transaction_id
        ORDER BY p.predicted_at DESC
        LIMIT 200
    """)
    with engine.connect() as conn:
        df = pd.read_sql(sql, conn)

    for col in BOOLEAN_FEATURES:
        df[col] = df[col].astype(int)

    print(f"Current data: {len(df)} rows")
    return df[ALL_FEATURES]


def compute_drift_score(reference: pd.Series, current: pd.Series,
                        is_boolean: bool = False) -> float:
    """
    Computes drift score for one feature.
    Returns a score between 0.0 (no drift) and 1.0 (maximum drift).

    For numerical: uses KS test. ks_stat is the drift score directly —
    it represents the maximum difference between the two distributions.

    For boolean: uses chi-squared test on value counts.
    We convert p-value to drift score: score = 1 - p_value
    so higher score = more drift (consistent with KS interpretation).
    """
    if is_boolean:
        # Count occurrences of 0 and 1 in each dataset
        ref_counts = reference.value_counts().reindex([0, 1], fill_value=1)
        cur_counts = current.value_counts().reindex([0, 1], fill_value=1)
        try:
            _, p_value = stats.chisquare(cur_counts, f_exp=ref_counts *
                                         (len(current) / len(reference)))
            return round(max(0.0, 1.0 - p_value), 4)
        except Exception:
            return 0.0
    else:
        # KS statistic directly measures distributional distance
        ks_stat, _ = stats.ks_2samp(reference.dropna(), current.dropna())
        return round(float(ks_stat), 4)


def generate_html_report(drift_results: dict, reference: pd.DataFrame,
                         current: pd.DataFrame) -> str:
    """
    Generates a clean HTML drift report showing drift scores per feature.
    Color coded: green = OK, orange = MEDIUM, red = HIGH drift.
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    os.makedirs("data/reports", exist_ok=True)
    report_path = f"data/reports/drift_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"

    rows = ""
    for feature, score in sorted(drift_results.items(), key=lambda x: x[1], reverse=True):
        level = get_alert_level(score)
        color = {"HIGH": "#ff4444", "MEDIUM": "#ff8800", "LOW": "#22aa22"}[level]
        ref_mean = round(reference[feature].mean(), 4) if feature in reference else "N/A"
        cur_mean = round(current[feature].mean(), 4) if feature in current else "N/A"
        rows += f"""
        <tr>
            <td>{feature}</td>
            <td>{ref_mean}</td>
            <td>{cur_mean}</td>
            <td>{score:.4f}</td>
            <td style="color:{color}; font-weight:bold">{level}</td>
        </tr>"""

    drifted = sum(1 for s in drift_results.values() if s >= 0.1)
    html = f"""<!DOCTYPE html>
<html>
<head>
    <title>M-Pesa Fraud Detection — Drift Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; background: #f5f5f5; }}
        h1 {{ color: #1F3864; }}
        h3 {{ color: #2E75B6; }}
        table {{ border-collapse: collapse; width: 100%; background: white; }}
        th {{ background: #1F3864; color: white; padding: 10px; text-align: left; }}
        td {{ padding: 8px 10px; border-bottom: 1px solid #ddd; }}
        tr:hover {{ background: #f0f4ff; }}
        .summary {{ background: white; padding: 20px; margin-bottom: 20px;
                    border-left: 5px solid #1F3864; }}
    </style>
</head>
<body>
    <h1>M-Pesa Fraud Detection — Data Drift Report</h1>
    <div class="summary">
        <h3>Summary</h3>
        <p><strong>Generated:</strong> {timestamp}</p>
        <p><strong>Reference rows:</strong> {len(reference)}</p>
        <p><strong>Current rows:</strong> {len(current)}</p>
        <p><strong>Features checked:</strong> {len(drift_results)}</p>
        <p><strong>Features with drift (score &gt;= 0.1):</strong> {drifted}</p>
    </div>
    <table>
        <tr>
            <th>Feature</th>
            <th>Reference Mean</th>
            <th>Current Mean</th>
            <th>Drift Score</th>
            <th>Alert Level</th>
        </tr>
        {rows}
    </table>
</body>
</html>"""

    with open(report_path, "w", encoding="utf-8") as f:
        f.write(html)

    return report_path


def run_drift_detection(reference: pd.DataFrame,
                        current: pd.DataFrame) -> tuple:
    """Runs drift detection on all features and returns results."""
    drift_results = {}

    for feature in NUMERICAL_FEATURES:
        if feature in reference.columns and feature in current.columns:
            drift_results[feature] = compute_drift_score(
                reference[feature], current[feature], is_boolean=False)

    for feature in BOOLEAN_FEATURES:
        if feature in reference.columns and feature in current.columns:
            drift_results[feature] = compute_drift_score(
                reference[feature], current[feature], is_boolean=True)

    report_path = generate_html_report(drift_results, reference, current)
    return drift_results, report_path


def detect_and_alert(engine, drift_results: dict) -> int:
    """Logs alerts for features with significant drift."""
    alerted = 0
    for feature, score in drift_results.items():
        level = get_alert_level(score)
        if level in ["MEDIUM", "HIGH"]:
            log_drift_alert(feature, score, level)
            alerted += 1
            print(f"[{level} DRIFT] {feature}: score={score:.4f}")
        else:
            print(f"[OK]           {feature}: score={score:.4f}")
    return alerted