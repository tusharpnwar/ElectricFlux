import streamlit as st
import pandas as pd
import datetime
import holidays
import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import LSTM, Dropout, Dense, Input
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
import seaborn as sns
import matplotlib.pyplot as plt
from time import sleep

# App Title
st.set_page_config(page_title="🔌 Energy Forecasting Dashboard", layout="wide")
st.title("🔌 Energy Forecasting Dashboard")
st.subheader("🔍 Explore & Predict Demand")

# Summary about the project
st.markdown("""
### 📖 Project Summary
This Energy Forecasting Dashboard is designed to predict electricity demand based on historical data. Using a deep learning model (LSTM), the app forecasts energy consumption based on various features, including holidays, time of the day, and past electricity demand.

Key features of the app:
- **Data Upload**: Upload your CSV file containing historical energy demand data.
- **Model Training**: The app preprocesses the data, trains an LSTM model, and visualizes the results.
- **Prediction Output**: The predicted values are compared with actual values, and an RMSE metric is calculated to evaluate model performance.

**How to use**:
1. Upload your historical energy demand CSV file.
2. Choose the date range for analysis.
3. Click the **Run Model** button to process and visualize the forecast results.
""")

# Sidebar: File upload and date input
st.sidebar.markdown("### 📊 Select Input Mode:")
input_mode = st.sidebar.radio("", ["Live API", "Upload CSV"])

uploaded_file = None
if input_mode == "Upload CSV":
    uploaded_file = st.sidebar.file_uploader("📁 Upload your CSV file", type=["csv"])

# Date selection
start_date = st.sidebar.date_input("📅 Start Date", datetime.date(2015, 1, 1))
end_date = st.sidebar.date_input("📅 End Date", datetime.date(2024, 1, 15))

if start_date > end_date:
    st.error("❗ End date must be after start date.")
    st.stop()

# Button to run the uploaded file
if uploaded_file:
    run_button = st.sidebar.button("Run Model")

    if run_button:
        # Animation while the model is being processed
        with st.spinner("Processing data and training model... This may take a while..."):
            sleep(2)  # Simulate some processing time

            # ----- DATA PREPROCESSING -----
            df = pd.read_csv(uploaded_file, parse_dates=True)
            df.columns = df.columns.str.lower()
            df.drop(columns=["nsl_flow", "eleclink_flow"], axis=1, inplace=True)
            df = df[df["settlement_period"] <= 48]
            df["settlement_date"] = pd.to_datetime(df["settlement_date"])

            # Add time-based features
            df["day_of_month"] = df["settlement_date"].dt.day
            df["day_of_week"] = df["settlement_date"].dt.dayofweek
            df["month"] = df["settlement_date"].dt.month
            df["year"] = df["settlement_date"].dt.year

            # Add holiday feature (using Tamil Nadu holidays as an example)
            india_holidays = holidays.India(subdiv="TN", years=range(start_date.year, end_date.year + 1))
            holiday_dates = [date for date, _ in india_holidays.items()]
            df["is_holiday"] = df["settlement_date"].apply(lambda x: 1 if x in holiday_dates else 0)

            # Set datetime as index
            df.set_index("settlement_date", inplace=True)
            df.sort_index(inplace=True)

            # ----- MODEL TRAINING -----
            threshold_date = "06-01-2019"
            train_data = df[df.index < threshold_date]
            test_data = df[df.index >= threshold_date]

            FEATURES = ["is_holiday", "settlement_period", "day_of_month", "day_of_week", "month", "year"]
            TARGET = "tsd"
            X_train = train_data[FEATURES].values
            y_train = train_data[TARGET].values
            X_test = test_data[FEATURES].values
            y_test = test_data[TARGET].values

            # Scale the features
            scaler = MinMaxScaler(feature_range=(0, 1))
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            # Reshape input for LSTM
            X_train_scaled = X_train_scaled.reshape(X_train_scaled.shape[0], 1, X_train_scaled.shape[1])
            X_test_scaled = X_test_scaled.reshape(X_test_scaled.shape[0], 1, X_test_scaled.shape[1])

            # Define and compile LSTM model
            model = Sequential()
            model.add(Input(shape=(X_train_scaled.shape[1], X_train_scaled.shape[2])))
            model.add(LSTM(256, return_sequences=True))
            model.add(Dropout(0.5))
            model.add(LSTM(128, return_sequences=True))
            model.add(Dropout(0.5))
            model.add(LSTM(32))
            model.add(Dropout(0.5))
            model.add(Dense(1))
            model.compile(loss='mean_squared_error', optimizer="adam")

            # Train the model
            model.fit(X_train_scaled, y_train, epochs=3, batch_size=32, validation_data=(X_test_scaled, y_test), verbose=1)

            # ----- EVALUATE THE MODEL -----
            pred_lstm = model.predict(X_test_scaled)
            rmse_lstm = np.sqrt(mean_squared_error(y_test, pred_lstm))

            # Visualize Loss
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(range(1, len(model.history.history["loss"]) + 1), model.history.history["loss"], label="Training Loss")
            ax.plot(range(1, len(model.history.history["val_loss"]) + 1), model.history.history["val_loss"], label="Validation Loss")
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Loss")
            ax.set_title("Model Loss")
            ax.legend()
            st.pyplot(fig)

            # Visualize Prediction vs Actual
            result_frame = test_data[[TARGET]].copy()
            result_frame["pred_lstm"] = pred_lstm
            fig, ax = plt.subplots(figsize=(15, 5))
            ax.plot(result_frame.index, result_frame[TARGET], label="Actual", color='blue')
            ax.plot(result_frame.index, result_frame["pred_lstm"], label="Prediction", color='red')
            ax.set_title(f"LSTM Prediction vs Actual (RMSE: {rmse_lstm:.2f} MW)")
            ax.set_ylabel("Energy Demand (MW)")
            ax.set_xlabel("Date")
            ax.legend()
            st.pyplot(fig)

            # Display RMSE value
            st.write(f"Root Mean Squared Error (RMSE): {rmse_lstm:.2f} MW")

        # Show success message after completion
        st.success("Model has been successfully processed!")
