import streamlit as st
import pandas as pd
import datetime
import holidays
import requests
import matplotlib.pyplot as plt

# Set page config
st.set_page_config(page_title="Energy Demand Forecasting", layout="wide")

# Sidebar
with st.sidebar:
    st.title("🔌 Energy Forecasting Dashboard")
    st.subheader("🔍 Explore & Predict Demand")

    data_mode = st.radio("📊 Select Input Mode:", ["Live API", "Upload CSV"])
    uploaded_file = st.file_uploader("📁 Upload your CSV file (optional)", type=["csv"])

    start_date = st.date_input("📅 Start Date", value=datetime.date(2015, 1, 1))
    end_date = st.date_input("📅 End Date", value=datetime.date(2024, 1, 15))

st.title("📈 Energy Demand Forecasting")

# ===========================
# 🎉 1. Holiday Calendar Fix
# ===========================
st.header("🏖️ Holiday Calendar")

try:
    india_holidays = holidays.India(years=range(start_date.year, end_date.year + 1), state='TN')
    holiday_dates = [date for date in india_holidays if start_date <= date <= end_date]

    if holiday_dates:
        for date in sorted(holiday_dates):
            st.markdown(f"- {date.strftime('%A, %d %B %Y')}: {india_holidays[date]}")
    else:
        st.info("No holidays in the selected range.")
except Exception as e:
    st.warning("Couldn't fetch holiday data. Try another state/year range.")

# ================================
# 📂 2. CSV Upload + Visualization
# ================================
st.header("📂 Data Preview")

if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file, parse_dates=True)
        # Assume date column is named 'Date' or similar
        date_col = next((col for col in df.columns if 'date' in col.lower()), None)
        if date_col:
            df[date_col] = pd.to_datetime(df[date_col])
            df = df[(df[date_col] >= pd.to_datetime(start_date)) & (df[date_col] <= pd.to_datetime(end_date))]
            df = df.sort_values(by=date_col)

            st.success("✅ File Uploaded and Filtered Successfully!")
            st.dataframe(df.head())

            # Plot daily demand if a column exists
            demand_col = next((col for col in df.columns if 'demand' in col.lower()), None)
            if demand_col:
                st.line_chart(df.set_index(date_col)[demand_col])
            else:
                st.warning("No demand column found for visualization.")
        else:
            st.warning("No date column found in uploaded file.")
    except Exception as e:
        st.error(f"Error processing CSV: {e}")
else:
    st.info("Please upload a CSV or use the live API.")

# ==========================================
# 🌤️ 3. Weather Visualization (Full Range)
# ==========================================
st.header("🌦️ Weather Overview (Historical)")

def get_weather_data(start_date, end_date, location="Tamil Nadu, India"):
    # You should plug in your actual API key below
    api_key = "YOUR_VISUAL_CROSSING_API_KEY"
    base_url = f"https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/timeline/{location}/{start_date}/{end_date}"
    params = {
        "unitGroup": "metric",
        "include": "days",
        "key": api_key,
        "contentType": "json"
    }

    try:
        response = requests.get(base_url, params=params)
        data = response.json()

        days = data.get("days", [])
        weather_df = pd.DataFrame(days)
        weather_df["datetime"] = pd.to_datetime(weather_df["datetime"])
        return weather_df
    except Exception as e:
        st.warning("⚠️ Could not fetch weather data.")
        return None

# Fetch and visualize weather
weather_df = get_weather_data(start_date, end_date)
if weather_df is not None:
    st.line_chart(weather_df.set_index("datetime")[["tempmax", "tempmin"]])
    st.bar_chart(weather_df.set_index("datetime")["precip"])
else:
    st.info("Weather data will be shown once API is active.")

# ===========================
# 🔮 4. Forecast Placeholder
# ===========================
st.header("🔮 Demand Forecast Output")

if uploaded_file:
    st.success("Ready to run model forecast (LSTM integration pending).")
    # Add your model's prediction results here when ready.
    # e.g., df_pred = your_model.predict(df)
    # st.line_chart(df_pred)
else:
    st.info("Upload data to enable forecasting.")
