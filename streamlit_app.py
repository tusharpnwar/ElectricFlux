# streamlit_app.py

import streamlit as st
import pandas as pd
import numpy as np
import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dropout, Dense, Input
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
import requests
from sklearn.linear_model import LinearRegression

st.set_page_config(page_title="⚡ Electricity Dashboard", layout="wide")
sns.set_style("whitegrid")

st.title("⚡ Electricity Forecasting & Consumption Dashboard")

uploaded_file = st.sidebar.file_uploader("📁 Upload CSV for Load Forecasting", type=["csv"])

# Sidebar default state/year selection (for both modes)
@st.cache_data
def fetch_cea_data():
    url = "https://cea.nic.in/api/percapitalConsumtion.php"
    response = requests.get(url)
    if response.status_code == 200:
        return pd.DataFrame(response.json())
    else:
        st.error("Failed to fetch data from API.")
        return pd.DataFrame()

cea_df = fetch_cea_data()
cea_df["value"] = pd.to_numeric(cea_df["value"], errors="coerce")
cea_df = cea_df.dropna()
cea_df = cea_df.rename(columns={"value": "PerCapitaConsumption"})

def get_valid_states(df):
    invalid_entries = {
        "N R", "W R", "S R", "E R", "N E R", "All India",
        "Jammu & Kashmir*", "Uttarakhand*"
    }
    return sorted({state.strip() for state in df["State"].unique() if state.strip() not in invalid_entries})

valid_states = get_valid_states(cea_df)
years = sorted(cea_df["Year"].str[:4].dropna().astype(int).unique())

selected_states = st.sidebar.multiselect("🗺️ Select States", valid_states, default=["Delhi", "Rajasthan"])
selected_years = st.sidebar.slider("📅 Select Year Range", min_value=min(years), max_value=max(years), value=(2010, 2021))

# COMMON DATE FEATURE FUNCTION
@st.cache_data
def create_features(df):
    df["day_of_month"] = df.index.day
    df["day_of_week"] = df.index.day_of_week
    df["day_of_year"] = df.index.day_of_year
    df["quarter"] = df.index.quarter
    df["month"] = df.index.month
    df["year"] = df.index.year
    df["week_of_year"] = df.index.isocalendar().week.astype("int64")
    return df

def prepare_data(df):
    df.columns = df.columns.str.lower()
    df = df[df["settlement_period"] <= 48]
    df["period_hour"] = df["settlement_period"].apply(
        lambda x: str(datetime.timedelta(hours=(x - 1) * 0.5))
    )
    df.loc[df["period_hour"] == "1 day, 0:00:00", "period_hour"] = "0:00:00"
    df["settlement_date"] = pd.to_datetime(df["settlement_date"] + " " + df["period_hour"])
    df.set_index("settlement_date", inplace=True)
    df.sort_index(inplace=True)
    df = create_features(df)
    return df

def train_lstm(X_train, y_train, X_val, y_val, features):
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

    def root_mean_squared_error(y_true, y_pred):
        return tf.math.sqrt(tf.reduce_mean(tf.square(y_true - y_pred)))

    model.compile(loss=root_mean_squared_error, optimizer="adam")

    early_stopping = EarlyStopping(monitor='val_loss', patience=5)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3)

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=5,
        batch_size=128,
        callbacks=[early_stopping, reduce_lr],
        verbose=1
    )
    return model, history

# ------------------------- MODE: CSV UPLOADED -------------------------
if uploaded_file:
    st.markdown("### 📊 Load Forecasting using LSTM")

    df = pd.read_csv(uploaded_file)

    with st.spinner("🔄 Preprocessing data..."):
        df = prepare_data(df)
        FEATURES = [
            "settlement_period", "day_of_month", "day_of_week", "day_of_year",
            "quarter", "month", "year", "week_of_year"
        ]
        TARGET = "tsd"
        df = df[df["tsd"] != 0]
        df_model = df[FEATURES + [TARGET]]

        threshold1 = "2019-06-01"
        threshold2 = "2024-06-01"
        train = df_model[df_model.index < threshold1]
        test = df_model[(df_model.index >= threshold1) & (df_model.index < threshold2)]
        holdout = df_model[df_model.index >= threshold2]

        scaler = MinMaxScaler()
        train_scaled = scaler.fit_transform(train)
        test_scaled = scaler.transform(test)
        holdout_scaled = scaler.transform(holdout)

        X_train = train_scaled[:, :-1].reshape(train_scaled.shape[0], 1, len(FEATURES))
        y_train = train_scaled[:, -1]
        X_test = test_scaled[:, :-1].reshape(test_scaled.shape[0], 1, len(FEATURES))
        y_test = test_scaled[:, -1]
        X_holdout = holdout_scaled[:, :-1].reshape(holdout_scaled.shape[0], 1, len(FEATURES))
        y_holdout = holdout_scaled[:, -1]

    st.success("✅ Data ready for training!")

    if st.button("🚀 Train LSTM Model"):
        with st.spinner("🧠 Training model..."):
            model, history = train_lstm(X_train, y_train, X_holdout, y_holdout, FEATURES)

        st.success("🎉 Model trained successfully!")

        st.subheader("📉 Loss Curve")
        fig, ax = plt.subplots()
        ax.plot(history.history["loss"], label="Train Loss")
        ax.plot(history.history["val_loss"], label="Validation Loss")
        ax.legend()
        st.pyplot(fig)

        pred_test = model.predict(X_test).reshape(-1)
        test_scaled[:, -1] = pred_test
        pred_inverse = scaler.inverse_transform(test_scaled)[:, -1]

        y_true = test[TARGET].values
        rmse = np.sqrt(mean_squared_error(y_true, pred_inverse))
        mape = np.mean(np.abs((y_true - pred_inverse) / y_true)) * 100

        st.write(f"📌 **RMSE:** {rmse:.2f} MW")
        st.write(f"📌 **MAPE:** {mape:.2f} %")

        st.subheader("🔍 Prediction vs Actual")
        pred_df = test.copy()
        pred_df["Prediction"] = pred_inverse
        fig, ax = plt.subplots(figsize=(14, 5))
        ax.plot(pred_df.index, pred_df[TARGET], label="Actual")
        ax.plot(pred_df.index, pred_df["Prediction"], label="Prediction")
        ax.set_ylabel("MW")
        ax.legend()
        st.pyplot(fig)

        csv_download = pred_df.reset_index()[["settlement_date", "tsd", "Prediction"]]
        csv = csv_download.to_csv(index=False).encode("utf-8")
        st.download_button("📥 Download Predictions as CSV", data=csv, file_name="predictions.csv")

# ------------------------- MODE: NO CSV => API DASHBOARD -------------------------
else:
    st.markdown("### 📊 Per Capita Electricity Consumption (India)")
    st.markdown("Live data from [CEA API](https://cea.nic.in/api/percapitalConsumtion.php)")

    cea_df["YearNum"] = cea_df["Year"].str[:4].astype(int)
    filtered_df = cea_df[(cea_df["YearNum"] >= selected_years[0]) & (cea_df["YearNum"] <= selected_years[1])]
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

# Footer
st.markdown("---")
st.markdown("Made by **Tushar Panwar**, **Garvit Bansal** under the guidance of **Dr. Asnath Vincty**.")
