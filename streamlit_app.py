import streamlit as st
import pandas as pd
import datetime
import holidays
import plotly.express as px

# App Title
st.set_page_config(page_title="🔌 Energy Forecasting Dashboard", layout="wide")
st.title("🔌 Energy Forecasting Dashboard")
st.subheader("🔍 Explore & Predict Demand")

# Sidebar
st.sidebar.markdown("### 📊 Select Input Mode:")
input_mode = st.sidebar.radio("", ["Live API", "Upload CSV"])

uploaded_file = None
if input_mode == "Upload CSV":
    uploaded_file = st.sidebar.file_uploader("📁 Upload your CSV file (optional)", type=["csv"])
    if uploaded_file:
        st.success(f"✅ File Uploaded Successfully: {uploaded_file.name}")

# Date selection
start_date = st.sidebar.date_input("📅 Start Date", datetime.date(2015, 1, 1))
end_date = st.sidebar.date_input("📅 End Date", datetime.date(2024, 1, 15))

if start_date > end_date:
    st.error("❗ End date must be after start date.")
    st.stop()

# Read uploaded data
df = None
if uploaded_file:
    df = pd.read_csv(uploaded_file, parse_dates=True)
    st.subheader("📂 Data Preview")
    st.dataframe(df.head(), use_container_width=True)

# Demand Forecast Section Placeholder
st.subheader("🔮 Demand Forecast Output")
st.info("Forecasting module will be shown here based on your uploaded or live data.")

# ----- HOLIDAY CALENDAR SECTION -----
st.subheader("🏖️ Holiday Calendar")

try:
    # You can change the 'state' parameter to appropriate 2-letter Indian state code (e.g., 'DL', 'MH', etc.)
    india_holidays = holidays
