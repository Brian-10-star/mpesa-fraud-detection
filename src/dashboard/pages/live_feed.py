# live_feed.py
# Displays a live table of recent fraud predictions.
# Shows the most recent 50 predictions from prediction_log,
# highlighting fraud cases in red and legitimate ones in green.
# Refreshes automatically every 10 seconds.

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


def load_recent_predictions(engine) -> pd.DataFrame:
    """Loads the 50 most recent predictions from prediction_log."""
    sql = text("""
        SELECT
            transaction_id,
            transaction_type,
            amount,
            fraud_probability,
            is_fraud,
            model_version,
            predicted_at
        FROM prediction_log
        ORDER BY predicted_at DESC
        LIMIT 50
    """)
    with engine.connect() as conn:
        return pd.read_sql(sql, conn)


def render():
    st.header("Live Fraud Feed")
    st.caption("Most recent 50 predictions — auto-refreshes every 10 seconds")

    engine = get_db_engine()

    try:
        df = load_recent_predictions(engine)

        if df.empty:
            st.info("No predictions yet. Run batch_scorer.py or call POST /predict.")
            return

        # Summary metrics at the top
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Shown", len(df))
        col2.metric("Fraud Flagged", int(df['is_fraud'].sum()))
        col3.metric("Legitimate", int((~df['is_fraud']).sum()))
        col4.metric("Fraud Rate",
                    f"{df['is_fraud'].mean()*100:.1f}%")

        st.divider()

        # Color-code rows: red for fraud, green for legitimate
        # Streamlit's dataframe styling uses pandas Styler
        def highlight_fraud(row):
            if row['is_fraud']:
                return ['background-color: #ffe6e6'] * len(row)
            else:
                return ['background-color: #e6ffe6'] * len(row)

        # Format columns for display
        df['amount'] = df['amount'].apply(lambda x: f"KES {x:,.2f}")
        df['fraud_probability'] = df['fraud_probability'].apply(
            lambda x: f"{x:.4f}")
        df['is_fraud'] = df['is_fraud'].apply(
            lambda x: "FRAUD" if x else "Legitimate")
        df['predicted_at'] = pd.to_datetime(
            df['predicted_at']).dt.strftime("%Y-%m-%d %H:%M:%S")

        df.columns = ['Transaction ID', 'Type', 'Amount',
                      'Fraud Probability', 'Verdict', 'Model',
                      'Predicted At']

        st.dataframe(df, use_container_width=True, hide_index=True)

    except Exception as e:
        st.error(f"Database error: {e}")