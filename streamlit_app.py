import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime, timedelta
import requests
import io

# --- Load model (assuming it's in the same directory)
MODEL_PATH = "lstm_model.h5"
model = load_model(MODEL_PATH)

# --- Function to fetch live data from API (dummy URL for example)
def fetch_live_data(state: str, start_date: str, end_date: str):
    API_URL = f"https://api.example.com/load?state={state}&start={start_date}&end={end_date}"
    response = requests.get(API_URL)
    if response.status_code == 200:
        return pd.read_csv(io.StringIO(response.text))
    else:
        st.error("Failed to fetch data from API.")
        return None

# --- Prediction function using LSTM
def make_predictions(data: pd.DataFrame):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data[['Load']].values)

    x_test = []
    n_lookback = 60  # Adjust based on your model training

    for i in range(n_lookback, len(scaled_data)):
        x_test.append(scaled_data[i - n_lookback:i, 0])

    x_test = np.array(x_test)
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

    predictions = model.predict(x_test)
    predictions = scaler.inverse_transform(predictions)

    pred_df = pd.DataFrame({
        'Date': data['Date'][n_lookback:].values,
        'Predicted Load': predictions.flatten()
    })
    return pred_df

# --- Streamlit UI
st.title("Energy Demand Forecasting (LSTM Model)")

st.sidebar.header("Options")
uploaded_file = st.sidebar.file_uploader("Upload CSV File", type=["csv"])

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.success("CSV uploaded successfully!")
    st.write("### Uploaded Data", data.head())
else:
    state = st.sidebar.selectbox("Select State", ["Delhi", "Maharashtra", "Tamil Nadu", "Karnataka"])
    start_date = st.sidebar.date_input("Start Date", datetime.now() - timedelta(days=90))
    end_date = st.sidebar.date_input("End Date", datetime.now())

    if start_date >= end_date:
        st.error("Start date must be before end date.")
    else:
        data = fetch_live_data(state, start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))
        if data is not None:
            st.write("### Fetched Data", data.head())

if data is not None:
    if 'Load' not in data.columns or 'Date' not in data.columns:
        st.error("Data must contain 'Date' and 'Load' columns.")
    else:
        data['Date'] = pd.to_datetime(data['Date'])
        prediction_df = make_predictions(data)

        st.write("### Load Forecast", prediction_df.tail())

        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(data['Date'], data['Load'], label='Actual Load')
        ax.plot(prediction_df['Date'], prediction_df['Predicted Load'], label='Predicted Load')
        ax.legend()
        st.pyplot(fig)
