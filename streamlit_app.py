import streamlit as st
import requests
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import numpy as np
import datetime

st.set_page_config(page_title="🔋 Per Capita Electricity Consumption", layout="wide")

# Dashboard Title and Summary
st.title("🔌 Per Capita Electricity Consumption Dashboard (India)")
st.markdown("""
This dashboard visualizes **per capita electricity consumption trends** across Indian states based on live data from the [CEA API](https://cea.nic.in/api/percapitalConsumtion.php). 
It also overlays **weather** and **holiday** information to help explore their potential impact on electricity usage. You can upload your own dataset for detailed custom forecasts.
""")

# Upload File Option
uploaded_file = st.file_uploader("📁 Upload your electricity consumption CSV file (optional)", type=["csv"])

# Fetch CEA API data
@st.cache_data
def fetch_cea_data():
    url = "https://cea.nic.in/api/percapitalConsumtion.php"
    response = requests.get(url)
    if response.status_code == 200:
        return pd.DataFrame(response.json())
    else:
        st.error("Failed to fetch data from API.")
        return pd.DataFrame()

# Sidebar Section
with st.sidebar:
    st.markdown("## 🔧 Filter Options")
    
    df = fetch_cea_data()
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df = df.dropna()
    df = df.rename(columns={"value": "PerCapitaConsumption"})

    def get_valid_states(df):
        invalid_entries = {
            "N R", "W R", "S R", "E R", "N E R", "All India",
            "Jammu & Kashmir*", "Uttarakhand*"
        }
        return sorted({state.strip() for state in df["State"].unique() if state.strip() not in invalid_entries})

    years = sorted(df["Year"].unique())
    states = get_valid_states(df)

    selected_states = st.multiselect("📍 Select States", states, default=["Delhi", "Rajasthan"])
    selected_years = st.slider("📅 Select Year Range", min_value=int(years[0][:4]), max_value=int(years[-1][:4]), value=(2010, 2021))

# Filter and visualize CEA data
df["YearNum"] = df["Year"].str[:4].astype(int)
filtered_df = df[(df["YearNum"] >= selected_years[0]) & (df["YearNum"] <= selected_years[1])]
filtered_df = filtered_df[filtered_df["State"].isin(selected_states)]

# Line chart
st.subheader("📈 Per Capita Electricity Consumption Over Time")
fig, ax = plt.subplots(figsize=(10, 5))
for state in selected_states:
    state_data = filtered_df[filtered_df["State"] == state]
    ax.plot(state_data["YearNum"], state_data["PerCapitaConsumption"], marker='o', label=state)

ax.set_xlabel("Year")
ax.set_ylabel("Per Capita Consumption (kWh)")
ax.legend()
ax.grid(True)
st.pyplot(fig)

# Forecasting
st.subheader("📉 Forecast Next 5 Years (Linear Trend)")
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
            "Forecasted Consumption": preds
        })

        st.markdown(f"**{state}**:")
        st.dataframe(forecast_df.set_index("Year").style.format("{:.2f} kWh"))
    else:
        st.warning(f"Not enough data for forecasting {state}")

# Holiday Visualization
st.subheader("📅 Indian Holidays Affecting Consumption")
holidays = pd.DataFrame({
    "Holiday": ["Republic Day", "Holi", "Independence Day", "Diwali", "Christmas"],
    "Date": ["2025-01-26", "2025-03-14", "2025-08-15", "2025-10-20", "2025-12-25"]
})
holidays["Date"] = pd.to_datetime(holidays["Date"])
st.table(holidays)

# Weather Visualization using Visual Crossing
st.subheader("🌦️ Weather Data for Selected States (Visual Impact)")

def fetch_weather_data(location, start_date, end_date):
    api_key = "62P5KGYJGJNFJ8GLF3JY4YY7W"
    url = f"https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/timeline/{location}/{start_date}/{end_date}?key={api_key}&unitGroup=metric&include=days"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        df_weather = pd.DataFrame(data["days"])
        df_weather["datetime"] = pd.to_datetime(df_weather["datetime"])
        return df_weather[["datetime", "temp", "humidity", "precip"]]
    else:
        st.error("Weather data fetch failed.")
        return pd.DataFrame()

# Show weather only if states are selected
today = datetime.date.today()
week_ago = today - datetime.timedelta(days=7)
for state in selected_states[:2]:  # Show weather for up to two states
    st.markdown(f"**Weather Trends in {state} (Last 7 Days)**")
    weather_df = fetch_weather_data(state, str(week_ago), str(today))
    if not weather_df.empty:
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(weather_df["datetime"], weather_df["temp"], marker='o', label="Temp (°C)")
        ax.plot(weather_df["datetime"], weather_df["humidity"], marker='s', label="Humidity (%)")
        ax.set_ylabel("Value")
        ax.set_xlabel("Date")
        ax.legend()
        st.pyplot(fig)

# Footer
st.markdown("---")
st.markdown("👨‍💻 Made by **Tushar Panwar**, **Garvit Bansal** under the guidance of **Dr. Asnath Vincty**.")
