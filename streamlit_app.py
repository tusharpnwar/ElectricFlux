
# --- ENERGY DEMAND FORECASTING USING LSTM ---
import streamlit as st
import pandas as pd
import numpy as np
import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import holidays
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dropout, Dense, Input
import tensorflow.keras.backend as K

st.set_page_config(page_title="🔋 Energy Dashboard", layout="wide")
st.title("🔋 Energy Demand Forecasting using LSTM")
st.markdown("Upload your own CSV or use the default historic dataset. Select a date range for prediction visualization.")

uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
start_date = st.date_input("Start Date", datetime.date(2024, 1, 1))
end_date = st.date_input("End Date", datetime.date(2024, 1, 14))

@st.cache_data
def load_default_data():
    df = pd.read_csv("historic_demand_2009_2024.csv", index_col=0)
    df.columns = df.columns.str.lower()
    df.drop(columns=["nsl_flow", "eleclink_flow"], axis=1, inplace=True, errors="ignore")
    df.drop(index=df[df["settlement_period"] > 48].index, inplace=True)
    df.reset_index(drop=True, inplace=True)
    bank_holiday_TN = holidays.India(subdiv="TN", years=range(2009, 2024), observed=True)
    holiday_dates_observed = [pd.to_datetime(date) for date, name in bank_holiday_TN.items() if "Observed" not in name]
    df["is_holiday"] = df["settlement_date"].apply(lambda x: pd.to_datetime(x) in holiday_dates_observed)
    df["is_holiday"] = df["is_holiday"].astype(int)
    df["period_hour"] = (df["settlement_period"]).apply(lambda x: str(datetime.timedelta(hours=(x - 1) * 0.5)))
    df.loc[df["period_hour"] == "1 day, 0:00:00", "period_hour"] = "0:00:00"
    df["settlement_date"] = pd.to_datetime(df["settlement_date"] + " " + df["period_hour"])
    df.set_index("settlement_date", inplace=True)
    df.sort_index(inplace=True)
    df = df[df["tsd"] != 0.0]
    return df

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.success("Uploaded file successfully!")
    df.columns = df.columns.str.lower()
    df["settlement_date"] = pd.to_datetime(df["settlement_date"])
    df.set_index("settlement_date", inplace=True)
else:
    df = load_default_data()

def create_features(df):
    df = df.copy()
    df["day_of_month"] = df.index.day
    df["day_of_week"] = df.index.day_of_week
    df["day_of_year"] = df.index.day_of_year
    df["quarter"] = df.index.quarter
    df["month"] = df.index.month
    df["year"] = df.index.year
    df["week_of_year"] = df.index.isocalendar().week.astype("int64")
    return df

df = create_features(df)
threshold_1 = "2019-06-01"
threshold_2 = "2024-06-01"
train = df[df.index < threshold_1]
test = df[(df.index >= threshold_1) & (df.index < threshold_2)]
holdout = df[df.index >= threshold_2]

FEATURES = [
    "is_holiday", "settlement_period", "day_of_month", "day_of_week",
    "day_of_year", "quarter", "month", "year", "week_of_year"
]
TARGET = "tsd"
FEATURES_TARGET = FEATURES + [TARGET]
scaler = MinMaxScaler()
train_scaled = scaler.fit_transform(train[FEATURES_TARGET])
test_scaled = scaler.transform(test[FEATURES_TARGET])
holdout_scaled = scaler.transform(holdout[FEATURES_TARGET])
X_train = train_scaled[:, :-1].reshape(train_scaled.shape[0], 1, len(FEATURES))
y_train = train_scaled[:, -1]
X_test = test_scaled[:, :-1].reshape(test_scaled.shape[0], 1, len(FEATURES))
y_test = test_scaled[:, -1]
X_hold = holdout_scaled[:, :-1].reshape(holdout_scaled.shape[0], 1, len(FEATURES))
y_hold = holdout_scaled[:, -1]

def root_mean_squared_error(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_true - y_pred)))

model = Sequential([
    Input(shape=(X_train.shape[1], X_train.shape[2])),
    LSTM(256, return_sequences=True),
    Dropout(0.5),
    LSTM(128, return_sequences=True),
    Dropout(0.5),
    LSTM(32),
    Dropout(0.5),
    Dense(1)
])
model.compile(optimizer="adam", loss=root_mean_squared_error)
model.fit(X_train, y_train, validation_data=(X_hold, y_hold), epochs=3, batch_size=155)
pred = model.predict(X_test)
results = test.copy()
results["predicted"] = scaler.inverse_transform(np.hstack([test_scaled[:, :-1], pred]))[:, -1]
results = results[(results.index.date >= start_date) & (results.index.date <= end_date)]
st.subheader("📈 Forecast vs Actual")
st.line_chart(results[["tsd", "predicted"]])
rmse = np.sqrt(mean_squared_error(results["tsd"], results["predicted"]))
mape = np.mean(np.abs((results["tsd"] - results["predicted"]) / results["tsd"])) * 100
st.metric("RMSE (MW)", f"{rmse:.2f}")
st.metric("MAPE (%)", f"{mape:.2f}")

# --- PER CAPITA CONSUMPTION DASHBOARD ---
import requests
from sklearn.linear_model import LinearRegression

st.title("🔌 Per Capita Electricity Consumption Dashboard (India)")
st.markdown("Live data fetched from [CEA API](https://cea.nic.in/api/percapitalConsumtion.php)")

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
df["value"] = pd.to_numeric(df["value"], errors="coerce")
df = df.dropna()
df = df.rename(columns={"value": "PerCapitaConsumption"})

def get_valid_states(df):
    invalid_entries = {"N R", "W R", "S R", "E R", "N E R", "All India", "Jammu & Kashmir*", "Uttarakhand*"}
    states = sorted({state.strip() for state in df["State"].unique() if state.strip() not in invalid_entries})
    return states

years = sorted(df["Year"].unique())
states = get_valid_states(df)
selected_states = st.sidebar.multiselect("Select States", states, default=["Delhi", "Rajasthan"])
selected_years = st.sidebar.slider("Select Year Range", min_value=int(years[0][:4]), max_value=int(years[-1][:4]), value=(2010, 2021))
df["YearNum"] = df["Year"].str[:4].astype(int)
filtered_df = df[(df["YearNum"] >= selected_years[0]) & (df["YearNum"] <= selected_years[1])]
filtered_df = filtered_df[filtered_df["State"].isin(selected_states)]

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

st.markdown("---")
st.markdown("Made by **Tushar Panwar**, **Garvit Bansal** under the guidance of **Dr.Asnath Vincty**.")
