# model_metrics.py
# Displays model performance metrics and prediction volume over time.
# Shows aggregate stats and a bar chart of predictions per hour
# so you can see when fraud detection is most active.

import streamlit as st
import pandas as pd
from sqlalchemy import create_engine, text
from dotenv import load_dotenv
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
load_dotenv()


def get_db_engine():
    url = (
        f"postgresql+psycopg2://{os.getenv('DB_USER')}:{os.getenv('DB_PASSWORD')}"
        f"@{os.getenv('DB_HOST')}:{os.getenv('DB_PORT')}/{os.getenv('DB_NAME')}"
    )
    return create_engine(url)


def load_metrics(engine) -> dict:
    """Loads aggregate prediction statistics."""
    sql = text("""
        SELECT
            COUNT(*) as total,
            COALESCE(SUM(CASE WHEN is_fraud THEN 1 ELSE 0 END), 0) as fraud_count,
            COALESCE(AVG(fraud_probability), 0) as avg_probability,
            MAX(predicted_at) as last_prediction
        FROM prediction_log
    """)
    with engine.connect() as conn:
        row = conn.execute(sql).fetchone()
    return {
        'total': int(row.total),
        'fraud': int(row.fraud_count),
        'legit': int(row.total) - int(row.fraud_count),
        'avg_prob': float(row.avg_probability),
        'last_prediction': row.last_prediction
    }


def load_hourly_volume(engine) -> pd.DataFrame:
    """Loads prediction count per hour for the bar chart."""
    sql = text("""
        SELECT
            DATE_TRUNC('hour', predicted_at) as hour,
            COUNT(*) as total,
            SUM(CASE WHEN is_fraud THEN 1 ELSE 0 END) as fraud_count
        FROM prediction_log
        GROUP BY DATE_TRUNC('hour', predicted_at)
        ORDER BY hour DESC
        LIMIT 24
    """)
    with engine.connect() as conn:
        return pd.read_sql(sql, conn)


def load_type_breakdown(engine) -> pd.DataFrame:
    """Loads fraud rate by transaction type."""
    sql = text("""
        SELECT
            transaction_type,
            COUNT(*) as total,
            SUM(CASE WHEN is_fraud THEN 1 ELSE 0 END) as fraud_count,
            ROUND(AVG(CASE WHEN is_fraud THEN 1.0 ELSE 0.0 END) * 100, 1)
                as fraud_rate_pct
        FROM prediction_log
        GROUP BY transaction_type
        ORDER BY fraud_rate_pct DESC
    """)
    with engine.connect() as conn:
        return pd.read_sql(sql, conn)


def render():
    st.header("Model Metrics")
    st.caption("Aggregate prediction statistics from the fraud detection model")

    engine = get_db_engine()

    try:
        metrics = load_metrics(engine)

        # Top-level KPI metrics
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Predictions", f"{metrics['total']:,}")
        col2.metric("Fraud Detected", f"{metrics['fraud']:,}")
        col3.metric("Legitimate", f"{metrics['legit']:,}")
        fraud_rate = (metrics['fraud'] / metrics['total'] * 100
                      if metrics['total'] > 0 else 0)
        col4.metric("Fraud Rate", f"{fraud_rate:.1f}%")

        st.divider()

        col_left, col_right = st.columns(2)

        with col_left:
            st.subheader("Prediction Volume by Hour")
            hourly = load_hourly_volume(engine)
            if not hourly.empty:
                hourly['hour'] = pd.to_datetime(
                    hourly['hour']).dt.strftime("%H:%M")
                hourly = hourly.set_index('hour')
                st.bar_chart(hourly[['total', 'fraud_count']])
            else:
                st.info("No hourly data yet.")

        with col_right:
            st.subheader("Fraud Rate by Transaction Type")
            type_df = load_type_breakdown(engine)
            if not type_df.empty:
                type_df.columns = ['Type', 'Total',
                                   'Fraud', 'Fraud Rate %']
                st.dataframe(type_df, use_container_width=True,
                             hide_index=True)
            else:
                st.info("No transaction type data yet.")

        st.divider()
        st.caption(f"Average fraud probability: "
                   f"{metrics['avg_prob']:.4f} | "
                   f"Last prediction: {metrics['last_prediction']}")

    except Exception as e:
        st.error(f"Database error: {e}")