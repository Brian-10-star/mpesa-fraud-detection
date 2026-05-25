# drift_report.py
# Displays the latest drift alerts from the drift_alerts table
# and shows a summary of which features are drifting.
# Links to the latest HTML drift report for full details.

import streamlit as st
import pandas as pd
from sqlalchemy import create_engine, text
from dotenv import load_dotenv
import os
import sys
import glob

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
load_dotenv()


def get_db_engine():
    url = (
        f"postgresql+psycopg2://{os.getenv('DB_USER')}:{os.getenv('DB_PASSWORD')}"
        f"@{os.getenv('DB_HOST')}:{os.getenv('DB_PORT')}/{os.getenv('DB_NAME')}"
    )
    return create_engine(url)


def load_drift_alerts(engine) -> pd.DataFrame:
    """Loads all drift alerts from the database."""
    sql = text("""
        SELECT
            feature_name,
            drift_score,
            alert_level,
            detected_at
        FROM drift_alerts
        ORDER BY detected_at DESC
        LIMIT 100
    """)
    with engine.connect() as conn:
        return pd.read_sql(sql, conn)


def render():
    st.header("Drift Report")
    st.caption("Data drift alerts logged by the monitoring pipeline")

    engine = get_db_engine()

    try:
        df = load_drift_alerts(engine)

        if df.empty:
            st.info("No drift alerts yet. Run monitor.py to generate alerts.")
            return

        # Summary counts
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Alerts", len(df))
        col2.metric("HIGH Drift",
                    int((df['alert_level'] == 'HIGH').sum()))
        col3.metric("MEDIUM Drift",
                    int((df['alert_level'] == 'MEDIUM').sum()))

        st.divider()

        # Color code by alert level
        def color_alert(val):
            if val == 'HIGH':
                return 'background-color: #ffe6e6; color: red; font-weight: bold'
            elif val == 'MEDIUM':
                return 'background-color: #fff3e6; color: orange; font-weight: bold'
            return ''

        df['detected_at'] = pd.to_datetime(
            df['detected_at']).dt.strftime("%Y-%m-%d %H:%M:%S")
        df['drift_score'] = df['drift_score'].apply(lambda x: f"{x:.4f}")
        df.columns = ['Feature', 'Drift Score', 'Alert Level', 'Detected At']

        styled = df.style.applymap(color_alert, subset=['Alert Level'])
        st.dataframe(styled, use_container_width=True, hide_index=True)

        st.divider()

        # Show latest HTML report if available
        st.subheader("Latest HTML Report")
        reports = sorted(glob.glob("data/reports/drift_report_*.html"),
                         reverse=True)
        if reports:
            latest = reports[0]
            st.success(f"Latest report: `{latest}`")
            st.info("Open this file in your browser for the full "
                    "interactive drift report.")
        else:
            st.info("No HTML reports found. Run monitor.py first.")

    except Exception as e:
        st.error(f"Database error: {e}")