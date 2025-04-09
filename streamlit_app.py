import streamlit as st
import pandas as pd
import datetime
import holidays

# --------------- Page Config ---------------
st.set_page_config(page_title="Energy Demand Forecasting", layout="wide")

# --------------- Style ---------------
st.markdown("""
    <style>
    .sidebar .sidebar-content {
        background-color: #f0f2f6;
    }
    h1, h2, h3 {
        color: #2c3e50;
        font-family: 'Helvetica Neue', sans-serif;
    }
    </style>
""", unsafe_allow_html=True)

# --------------- Sidebar ---------------
with st.sidebar:
    st.title("🔌 Energy Forecasting Dashboard")
    st.subheader("🔍 Explore & Predict Demand")

    data_mode = st.radio("📊 Select Input Mode:", ["Live API", "Upload CSV"])
    uploaded_file = st.file_uploader("📁 Upload your CSV file (optional)", type=["csv"])
    start_date = st.date_input("📅 Start Date", value=datetime.date(2024, 1, 1))
    end_date = st.date_input("📅 End Date", value=datetime.date(2024, 1, 15))

# --------------- Main Content ---------------
st.title("📈 Energy Demand Forecasting")

# --------------- Holiday Info ---------------
st.header("🏖️ Holiday Calendar")

in_holidays = holidays.India(state='TN')
holiday_dates_observed = [
    date for date in in_holidays
    if isinstance(date, datetime.date) and start_date <= date <= end_date
]

if holiday_dates_observed:
    for d in holiday_dates_observed:
        st.markdown(f"- {d.strftime('%A, %d %B %Y')}")
else:
    st.info("No holidays in the selected range.")

# --------------- Weather Snapshot (Placeholder) ---------------
st.header("🌦️ Weather Snapshot (Example Data)")

# Replace this section with your Visual Crossing API logic
example_weather = {
    "Temperature": "30°C",
    "Humidity": "60%",
    "Wind Speed": "15 km/h",
    "Condition": "Partly Cloudy"
}
for key, val in example_weather.items():
    st.write(f"**{key}**: {val}")

# --------------- Data Handling ---------------
st.header("📂 Data Preview")

if uploaded_file:
    try:
        df_uploaded = pd.read_csv(uploaded_file)
        st.success("✅ File Uploaded Successfully!")
        st.dataframe(df_uploaded.head())
    except Exception as e:
        st.error(f"❌ Error reading file: {e}")
elif data_mode == "Live API":
    st.info("Live API selected. Waiting for implementation or inputs...")
else:
    st.warning("Please upload a file or choose a valid data mode.")

# --------------- Forecast Placeholder ---------------
st.header("🔮 Demand Forecast Output")
st.info("Forecasting module will be shown here based on your uploaded or live data.")
