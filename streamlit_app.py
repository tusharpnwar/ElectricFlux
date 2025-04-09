import streamlit as st
import pandas as pd
import numpy as np
import datetime
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import requests
from io import StringIO

st.set_page_config(page_title="🔋 Energy Demand Forecasting", layout="wide")
st.title("🔋 Energy Demand Forecasting")

# Sidebar: Data Source
st.sidebar.header("📂 Data Options")
data_source = st.sidebar.radio("Select data source", ["Upload CSV", "Fetch Live"])

# Sidebar: Filters
st.sidebar.header("📅 Filter Options")
start_date = st.sidebar.date_input("Start date", datetime.date(2022, 1, 1))
end_date = st.sidebar.date_input("End date", datetime.date(2022, 12, 31))

state_list = ["All", "State1", "State2", "State3"]
state_disabled = data_source == "Upload CSV"
selected_state = st.sidebar.selectbox("Select State", state_list, disabled=state_disabled)

MODEL_PATH = "lstm_model.h5"

@st.cache_data
def fetch_live_data():
    # Replace with actual API or data endpoint
    url = "https://raw.githubusercontent.com/datablist/sample-csv-files/main/files/people/people-100.csv"
    response = requests.get(url)
    if response.status_code != 200:
        raise Exception("Failed to fetch live data")
    df = pd.read_csv(StringIO(response.text))

    # Fake demo transformation (replace with real processing)
    df = pd.DataFrame({
        "settlement_date": pd.date_range(start="2022-01-01", periods=100, freq="H"),
        "settlement_period": np.random.randint(1, 49, 100),
        "tsd": np.random.rand(100) * 100,
        "state": np.random.choice(["State1", "State2", "State3"], 100)
    })
    return df

def load_data(uploaded_file):
    df = pd.read_csv(uploaded_file)
    df.columns = df.columns.str.lower()
    df = df.sort_values(by=["settlement_date", "settlement_period"], ignore_index=True)
    df.drop(columns=["nsl_flow", "eleclink_flow"], errors='ignore', inplace=True)
    df = df[df["settlement_period"] <= 48].reset_index(drop=True)
    df = df[df["tsd"] != 0.0].reset_index(drop=True)

    df["period_hour"] = df["settlement_period"].apply(lambda x: str(datetime.timedelta(hours=(x - 1) * 0.5)))
    df.loc[df["period_hour"] == "1 day, 0:00:00", "period_hour"] = "0:00:00"
    df.insert(2, "period_hour", df.pop("period_hour"))

    df["settlement_date"] = pd.to_datetime(df["settlement_date"] + " " + df["period_hour"])
    df.set_index("settlement_date", inplace=True)
    df.sort_index(inplace=True)

    return df

# Load data based on selection
if data_source == "Upload CSV":
    uploaded_file = st.file_uploader("📁 Upload CSV file", type=["csv"])
    if uploaded_file is not None:
        df = load_data(uploaded_file)
    else:
        st.stop()
else:
    st.info("Fetching live data from source...")
    try:
        df = fetch_live_data()
        df["settlement_date"] = pd.to_datetime(df["settlement_date"])
        df.set_index("settlement_date", inplace=True)
    except Exception as e:
        st.error(f"Failed to fetch live data: {e}")
        st.stop()

# Apply date and state filters
df = df.loc[(df.index.date >= start_date) & (df.index.date <= end_date)]
if selected_state != "All" and "state" in df.columns:
    df = df[df["state"] == selected_state]

# Display selected filters
st.markdown(f"📍 **Date Range:** `{start_date}` to `{end_date}`")
if selected_state != "All" and not state_disabled:
    st.markdown(f"📍 **State:** `{selected_state}`")

# Display DataFrame
st.subheader("📊 Filtered Energy Demand Data")
st.dataframe(df.head())

# Prediction with LSTM model
try:
    model = load_model(MODEL_PATH)
except Exception as e:
    st.error(f"Model load error: {e}")
    st.stop()

if "tsd" not in df.columns:
    st.warning("Column 'tsd' not found in the dataset.")
    st.stop()

data_to_predict = df["tsd"].values.reshape(-1, 1)
scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(data_to_predict)

n_steps = 60
if len(data_scaled) < n_steps:
    st.warning("Not enough data for prediction.")
    st.stop()

X = [data_scaled[i - n_steps:i, 0] for i in range(n_steps, len(data_scaled))]
X = np.array(X).reshape(-1, n_steps, 1)

predicted_scaled = model.predict(X)
predicted = scaler.inverse_transform(predicted_scaled)

# Plot prediction
st.subheader("📈 Forecasted Energy Demand")
fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(df.index[n_steps:], predicted, label="Predicted TSD", color="red")
ax.plot(df.index[n_steps:], df["tsd"].values[n_steps:], label="Actual TSD", color="blue")
ax.legend()
ax.set_xlabel("Date")
ax.set_ylabel("TSD")
ax.set_title("Predicted vs Actual TSD")
st.pyplot(fig)
