import streamlit as st
import requests
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import numpy as np

st.set_page_config(page_title="🔋 Per Capita Electricity Consumption", layout="wide")

st.title("🔌 Per Capita Electricity Consumption Dashboard (India)")
st.markdown("Live data fetched from [CEA API](https://cea.nic.in/api/percapitalConsumtion.php)")

# Fetch API data
@st.cache_data
def fetch_cea_data():
    url = "https://cea.nic.in/api/percapitalConsumtion.php"
    response = requests.get(url)
    if response.status_code == 200:
        return pd.DataFrame(response.json())
    else:
        st.error("Failed to fetch data from API.")
        return pd.DataFrame()

df = fetch_cea_data()

# Preprocess
df["value"] = pd.to_numeric(df["value"], errors="coerce")
df = df.dropna()
df = df.rename(columns={"value": "PerCapitaConsumption"})

# Clean state names (exclude zones & special entries)
def get_valid_states(df):
    invalid_entries = {
        "N R", "W R", "S R", "E R", "N E R", "All India",
        "Jammu & Kashmir*", "Uttarakhand*"
    }
    states = sorted({state.strip() for state in df["State"].unique() if state.strip() not in invalid_entries})
    return states

# Sidebar filters
years = sorted(df["Year"].unique())
states = get_valid_states(df)

selected_states = st.sidebar.multiselect("Select States", states, default=["Delhi", "Rajasthan"])
selected_years = st.sidebar.slider("Select Year Range", min_value=int(years[0][:4]), max_value=int(years[-1][:4]), value=(2010, 2021))

# Filter data
df["YearNum"] = df["Year"].str[:4].astype(int)
filtered_df = df[(df["YearNum"] >= selected_years[0]) & (df["YearNum"] <= selected_years[1])]
filtered_df = filtered_df[filtered_df["State"].isin(selected_states)]

# Line Chart
st.subheader("📈 Per Capita Consumption Over Time")
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

# Footer
st.markdown("---")
st.markdown("Made by **Tushar Panwar**, **Garvit Bansal** under the guidance of **Dr.Asnath Vincty**.")
