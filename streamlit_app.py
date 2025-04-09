import streamlit as st
import requests
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import numpy as np
import datetime

# ------------------------------------
# 🎛️ App Setup
# ------------------------------------
st.set_page_config(page_title="🔋 Electricity Dashboard", layout="wide")

st.title("🔌 Per Capita Electricity Consumption & External Factors (India)")
st.markdown("""
A research dashboard for analyzing **state-wise electricity consumption trends** in India, enriched with **weather** and **holiday** insights to simulate external influence on demand patterns.

**Data Sources:**
- Consumption: [CEA API](https://cea.nic.in/api/percapitalConsumtion.php)
- Weather: [Visual Crossing](https://www.visualcrossing.com/)
- Holidays: Integrated calendar
""")

# ------------------------------------
# 📥 Fetch Live CEA Data
# ------------------------------------
@st.cache_data
def fetch_cea_data():
    url = "https://cea.nic.in/api/percapitalConsumtion.php"
    response = requests.get(url)
    if response.status_code == 200:
        return pd.DataFrame(response.json())
    else:
        st.error("Failed to fetch CEA data.")
        return pd.DataFrame()

df = fetch_cea_data()
df["value"] = pd.to_numeric(df["value"], errors="coerce")
df = df.dropna()
df = df.rename(columns={"value": "PerCapitaConsumption"})

# Valid States
def get_valid_states(df):
    invalid_entries = {"N R", "W R", "S R", "E R", "N E R", "All India", "Jammu & Kashmir*", "Uttarakhand*"}
    return sorted({s.strip() for s in df["State"].unique() if s.strip() not in invalid_entries})

# ------------------------------------
# 📂 Sidebar – Data Controls
# ------------------------------------
with st.sidebar:
    st.header("📂 Data Controls")

    states = get_valid_states(df)
    years = sorted(df["Year"].unique())

    selected_states = st.multiselect("🗺️ Select States", states, default=["Delhi", "Rajasthan"])
    selected_years = st.slider("📅 Year Range", min_value=int(years[0][:4]), max_value=int(years[-1][:4]), value=(2010, 2021))

    st.markdown("---")
    st.subheader("🌦️ External Factors")

    start_date = st.date_input("Start Date", datetime.date(2021, 1, 1))
    end_date = st.date_input("End Date", datetime.date(2021, 1, 7))

    selected_weather_state = st.selectbox("Select State for Weather", selected_states)

# ------------------------------------
# 🔍 Filter Data
# ------------------------------------
df["YearNum"] = df["Year"].str[:4].astype(int)
filtered_df = df[(df["YearNum"] >= selected_years[0]) & (df["YearNum"] <= selected_years[1])]
filtered_df = filtered_df[filtered_df["State"].isin(selected_states)]

# ------------------------------------
# 📈 Consumption Over Time
# ------------------------------------
st.subheader("📈 Per Capita Electricity Consumption Trends")

fig, ax = plt.subplots(figsize=(10, 5))
for state in selected_states:
    state_data = filtered_df[filtered_df["State"] == state]
    ax.plot(state_data["YearNum"], state_data["PerCapitaConsumption"], marker='o', label=state)

ax.set_xlabel("Year")
ax.set_ylabel("Per Capita Consumption (kWh)")
ax.set_title("Per Capita Electricity Consumption Over Time")
ax.legend()
ax.grid(True)
st.pyplot(fig)

# ------------------------------------
# 🔮 Forecast Next 5 Years
# ------------------------------------
st.subheader("📉 Forecast for Next 5 Years (Linear Trend)")

future_years = np.array(range(selected_years[1] + 1, selected_years[1] + 6)).reshape(-1, 1)
for state in selected_states:
    state_df = filtered_df[filtered_df["State"] == state]
    X = state_df["YearNum"].values.reshape(-1, 1)
    y = state_df["PerCapitaConsumption"].values

    if len(X) > 1:
        model = LinearRegression()
        model.fit(X, y)
        preds = model.predict(future_years)

        forecast_df = pd.DataFrame({
            "Year": future_years.flatten(),
            "Forecasted Consumption (kWh)": preds
        })

        st.markdown(f"**🔮 {state}**")
        st.dataframe(forecast_df.set_index("Year").style.format("{:.2f}"))
    else:
        st.warning(f"Not enough data for forecasting **{state}**.")

# ------------------------------------
# 📅 Show Holidays (India)
# ------------------------------------
@st.cache_data
def fetch_indian_holidays():
    url = "https://date.nager.at/api/v3/PublicHolidays/2021/IN"
    try:
        response = requests.get(url)
        data = response.json()
        return pd.DataFrame(data)
    except:
        return pd.DataFrame()

holidays_df = fetch_indian_holidays()
holidays_df["date"] = pd.to_datetime(holidays_df["date"])
holidays_df = holidays_df[holidays_df["date"] >= pd.to_datetime(start_date)]
holidays_df = holidays_df[holidays_df["date"] <= pd.to_datetime(end_date)]

st.subheader("🎉 Public Holidays (India)")
st.dataframe(holidays_df[["date", "localName", "name", "types"]].reset_index(drop=True))

# ------------------------------------
# 🌦️ Weather Data from Visual Crossing
# ------------------------------------
st.subheader(f"🌧️ Weather Forecast: {selected_weather_state} ({start_date} to {end_date})")

@st.cache_data(show_spinner=False)
def fetch_weather(state, start, end):
    try:
        url = f"https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/timeline/{state}/{start}/{end}?unitGroup=metric&key=62P5KGYJGJNFJ8GLF3JY4YY7W&contentType=json"
        response = requests.get(url)
        data = response.json()
        daily_data = data["days"]
        return pd.DataFrame(daily_data)
    except:
        return pd.DataFrame()

weather_df = fetch_weather(selected_weather_state, start_date, end_date)

if not weather_df.empty:
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(weather_df["datetime"], weather_df["tempmax"], label="Max Temp (°C)", marker='o')
    ax.plot(weather_df["datetime"], weather_df["tempmin"], label="Min Temp (°C)", marker='o')
    ax.set_xlabel("Date")
    ax.set_ylabel("Temperature (°C)")
    ax.set_title(f"Temperature Trend in {selected_weather_state}")
    ax.legend()
    st.pyplot(fig)
else:
    st.warning("Failed to fetch weather data.")

# ------------------------------------
# 📌 Footer
# ------------------------------------
st.markdown("---")
st.markdown("Developed by **Tushar Panwar**, **Garvit Bansal** under the guidance of **Dr. Asnath Vincty**.")
