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
    # Replace 'TN' with correct 2-letter Indian state code if needed
    india_holidays = holidays.India(years=range(start_date.year, end_date.year + 1), state='TN')
    holiday_list = [
        {"Date": date, "Holiday": india_holidays[date]}
        for date in sorted(india_holidays)
        if start_date <= date <= end_date
    ]

    if holiday_list:
        df_holidays = pd.DataFrame(holiday_list)
        if len(df_holidays) > 7:
            st.dataframe(df_holidays.head(7), use_container_width=True)
            with st.expander("📅 Show All Holidays"):
                st.dataframe(df_holidays, use_container_width=True)
        else:
            st.dataframe(df_holidays, use_container_width=True)
    else:
        st.info("No holidays in the selected range.")

except Exception as e:
    st.warning("⚠️ Couldn't fetch holiday data. Try changing the state or date range.")

# ----- WEATHER SNAPSHOT SECTION -----
st.subheader("🌦️ Weather Snapshot (Example Data)")

# You can replace this with real weather API integration
weather_data = {
    "Temperature (°C)": 30,
    "Humidity (%)": 60,
    "Wind Speed (km/h)": 15,
    "Condition": "Partly Cloudy"
}

weather_df = pd.DataFrame(list(weather_data.items()), columns=["Metric", "Value"])
st.table(weather_df)

# Optional: Visualize weather trends if available
# Replace this with actual time series data for real forecasting
weather_trend = pd.DataFrame({
    "Date": pd.date_range(start=start_date, end=end_date, freq='MS'),
    "Temperature (°C)": [28 + i % 5 for i in range(len(pd.date_range(start=start_date, end=end_date, freq='MS')))]
})
fig = px.line(weather_trend, x="Date", y="Temperature (°C)", title="📈 Average Monthly Temperature Trend")
st.plotly_chart(fig, use_container_width=True)
