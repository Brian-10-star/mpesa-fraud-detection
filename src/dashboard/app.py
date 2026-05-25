# app.py
# Main Streamlit dashboard entry point.
# Uses Streamlit's multipage app pattern — each page is a separate
# module in src/dashboard/pages/.
#
# Run with: streamlit run src/dashboard/app.py
#
# st.set_page_config must be the FIRST Streamlit command called.
# st.sidebar builds the left navigation panel.
# Each page's render() function is called based on user selection.

import streamlit as st
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from src.dashboard.pages import live_feed, model_metrics, drift_report

st.set_page_config(
    page_title="M-Pesa Fraud Detection",
    page_icon="🔍",
    layout="wide",       # wide layout uses full browser width
    initial_sidebar_state="expanded"
)

# Sidebar navigation
st.sidebar.title("M-Pesa Fraud Detection")
st.sidebar.caption("Real-time ML monitoring dashboard")
st.sidebar.divider()

page = st.sidebar.radio(
    "Navigate",
    ["Live Fraud Feed", "Model Metrics", "Drift Report"],
    index=0
)

st.sidebar.divider()
st.sidebar.caption("Built by Brian Mbugua Chira")
st.sidebar.caption("Stack: Kafka → PostgreSQL → scikit-learn → FastAPI → Streamlit")

# Page routing
# Calls the render() function of whichever page the user selected.
if page == "Live Fraud Feed":
    live_feed.render()
elif page == "Model Metrics":
    model_metrics.render()
elif page == "Drift Report":
    drift_report.render()