import streamlit as st
import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
import holidays
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dropout, Dense, Input
import tensorflow as tf
from datetime import datetime as dt, timedelta

st.set_page_config(page_title="🔋 Electricity Insights", layout="wide")
st.title("🔋 Electricity Insights Dashboard")
st.markdown("Powered by **CEA API**, **Visual Crossing Weather**, and **LSTM Demand Forecasting**")

tab1, tab2 = st.tabs(["📈 Consumption & Weather", "🔮 LSTM Demand Forecast"])

# ========== Tab 1 ==========
with tab1:
    st.header("📊 Per Capita Electricity Consumption + Weather")

    @st.cache_data
    def fetch_cea_data():
        url = "https://api.yourdomain.com/endpoint"  # Replace with actual URL
        try:
            response = requests.get(url)
            response.raise_for_status()
            data = response.json()

            for entry in data:
                if entry.get("value") == "":
                    entry["value"] = None
                else:
                    try:
                        entry["value"] = float(entry["value"])
                    except ValueError:
                        entry["value"] = None

            return pd.DataFrame(data)

        except requests.exceptions.RequestException as e:
            st.error(f"Error fetching data: {e}")
            return pd.DataFrame()
        except ValueError as e:
            st.error(f"Invalid JSON response: {e}")
            return pd.DataFrame()

    df = fetch_cea_data()
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df = df.dropna().rename(columns={"value": "PerCapitaConsumption"})

    def get_valid_states(df):
        invalid_entries = {"N R", "W R", "S R", "E R", "N E R", "All India", "Jammu & Kashmir*", "Uttarakhand*"}
        return sorted({state.strip() for state in df["State"].unique() if state.strip() not in invalid_entries})

    states = get_valid_states(df)
    years = sorted(df["Year"].unique())
    df["YearNum"] = df["Year"].str[:4].astype(int)

    selected_states = st.sidebar.multiselect("Select States", states, default=["Delhi", "Rajasthan"])
    selected_years = st.sidebar.slider("Select Year Range", min_value=int(years[0][:4]), max_value=int(years[-1][:4]), value=(2010, 2021))

    filtered_df = df[(df["YearNum"] >= selected_years[0]) & (df["YearNum"] <= selected_years[1])]
    filtered_df = filtered_df[filtered_df["State"].isin(selected_states)]

    st.subheader("🌦️ 7-Day Weather Forecast")
    today = dt.now().strftime("%Y-%m-%d")
    next_week = (dt.now() + timedelta(days=6)).strftime("%Y-%m-%d")

    for state in selected_states:
        weather_url = f"https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/timeline/{state}/{today}/{next_week}?key=62P5KGYJGJNFJ8GLF3JY4YY7W&unitGroup=metric&include=days"
        weather_response = requests.get(weather_url)

        if weather_response.status_code == 200:
            data = weather_response.json()
            forecast_df = pd.DataFrame(data["days"])[["datetime", "tempmax", "tempmin", "description"]]
            st.markdown(f"**📍 {state}**")
            st.dataframe(forecast_df.rename(columns={
                "datetime": "Date",
                "tempmax": "Max Temp (°C)",
                "tempmin": "Min Temp (°C)",
                "description": "Forecast"
            }))
        else:
            st.warning(f"Failed to fetch weather data for {state}")

    st.subheader("📈 Per Capita Electricity Consumption Over Time")
    fig, ax = plt.subplots(figsize=(10, 5))
    for state in selected_states:
        state_data = filtered_df[filtered_df["State"] == state]
        ax.plot(state_data["YearNum"], state_data["PerCapitaConsumption"], marker='o', label=state)
    ax.set_xlabel("Year")
    ax.set_ylabel("Per Capita Consumption (kWh)")
    ax.set_title("Electricity Consumption Over Years")
    ax.legend()
    ax.grid(True)
    st.pyplot(fig)

    st.subheader("🔮 Linear Forecast for Next 5 Years")
    future_years = np.array(range(selected_years[1] + 1, selected_years[1] + 6)).reshape(-1, 1)
    for state in selected_states:
        state_df = filtered_df[filtered_df["State"] == state]
        X = state_df["YearNum"].values.reshape(-1, 1)
        y = state_df["PerCapitaConsumption"].values

        st.markdown(f"**{state} Forecast:**")
        if len(X) > 1:
            model = LinearRegression()
            model.fit(X, y)
            preds = model.predict(future_years)

            forecast_df = pd.DataFrame({
                "Year": future_years.flatten(),
                "Forecasted Consumption": preds
            })
            st.dataframe(forecast_df.set_index("Year").style.format("{:.2f} kWh"))
        else:
            st.warning(f"Not enough data to forecast for {state}")

# ========== Tab 2 ==========
with tab2:
    st.header("⚡ Electricity Demand Forecast using LSTM")

    @st.cache_data
    def load_data(file):
        df = pd.read_csv(file, index_col=0)
        df.columns = df.columns.str.lower()
        df.sort_values(by=["settlement_date", "settlement_period"], inplace=True, ignore_index=True)
        df.drop(columns=["nsl_flow", "eleclink_flow"], axis=1, inplace=True, errors='ignore')
        df.drop(index=df[df["settlement_period"] > 48].index, inplace=True)
        df.reset_index(drop=True, inplace=True)
        return df

    def add_holidays(df):
        holiday_dates_observed = [
            np.datetime64(date) for date, name in sorted(
                holidays.India(subdiv="TN", years=range(2009, 2024), observed=True).items()
            ) if "Observed" not in name
        ]
        df["is_holiday"] = df["settlement_date"].apply(lambda x: pd.to_datetime(x) in holiday_dates_observed).astype(int)
        return df

    def clean_and_engineer(df):
        null_days = df.loc[df["tsd"] == 0.0, "settlement_date"].unique().tolist()
        df.drop(index=df[df["settlement_date"].isin(null_days)].index, inplace=True)
        df.reset_index(drop=True, inplace=True)

        df["period_hour"] = df["settlement_period"].apply(lambda x: str(datetime.timedelta(hours=(x - 1) * 0.5)))
        df.loc[df["period_hour"] == "1 day, 0:00:00", "period_hour"] = "0:00:00"
        df.insert(2, "period_hour", df.pop("period_hour"))

        df["settlement_date"] = pd.to_datetime(df["settlement_date"] + " " + df["period_hour"])
        df.set_index("settlement_date", inplace=True)
        df.sort_index(inplace=True)

        df["day_of_month"] = df.index.day
        df["day_of_week"] = df.index.day_of_week
        df["day_of_year"] = df.index.day_of_year
        df["quarter"] = df.index.quarter
        df["month"] = df.index.month
        df["year"] = df.index.year
        df["week_of_year"] = df.index.isocalendar().week.astype("int64")

        return df

    def scale_data(df, features, target):
        full_data = df[features + [target]]
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled = scaler.fit_transform(full_data.values)
        X = scaled[:, :-1].reshape(scaled.shape[0], 1, len(features))
        y = scaled[:, -1]
        return X, y, scaler

    def build_model(input_shape):
        model = Sequential()
        model.add(Input(shape=input_shape))
        model.add(LSTM(256, return_sequences=True))
        model.add(Dropout(0.5))
        model.add(LSTM(128, return_sequences=True))
        model.add(Dropout(0.5))
        model.add(LSTM(32))
        model.add(Dropout(0.5))
        model.add(Dense(1))
        model.compile(loss=lambda y_true, y_pred: tf.math.sqrt(tf.reduce_mean(tf.square(y_true - y_pred))), optimizer="adam")
        return model

    uploaded_file = st.sidebar.file_uploader("📄 Upload CSV File for LSTM Forecast", type=["csv"])

    if uploaded_file:
        df = load_data(uploaded_file)
        st.subheader("📊 Sample of Cleaned Dataset")
        st.write(df.sample(5))

        df = add_holidays(df)
        df = clean_and_engineer(df)

        st.subheader("📈 EDA: Distribution by Month")
        fig1, ax1 = plt.subplots(figsize=(12, 4))
        sns.boxplot(x="month", y="tsd", data=df, ax=ax1)
        st.pyplot(fig1)

        st.subheader("⚙️ Training Model")
        train_data = df[df.index < "2019-06-01"]
        test_data = df[(df.index >= "2019-06-01") & (df.index < "2024-06-01")]
        hold_out_data = df[df.index >= "2024-06-01"]

        features = [
            "is_holiday", "settlement_period", "day_of_month", "day_of_week",
            "day_of_year", "quarter", "month", "year", "week_of_year"
        ]
        target = "tsd"

        X_train, y_train, scaler = scale_data(train_data, features, target)
        X_test, y_test, _ = scale_data(test_data, features, target)

        model = build_model((X_train.shape[1], X_train.shape[2]))

        with st.spinner("⏳ Training LSTM model..."):
            history = model.fit(X_train, y_train, epochs=3, batch_size=64, validation_data=(X_test, y_test), verbose=1)
        st.success("✅ Model trained!")

        st.subheader("📉 Training & Validation Loss")
        fig2, ax2 = plt.subplots()
        ax2.plot(history.history["loss"], label="Training Loss")
        ax2.plot(history.history["val_loss"], label="Validation Loss")
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Loss (RMSE)")
        ax2.legend()
        st.pyplot(fig2)

        X_hold_out, y_hold_out, _ = scale_data(hold_out_data, features, target)
        preds_scaled = model.predict(X_hold_out)
        preds = scaler.inverse_transform(np.concatenate([X_hold_out[:, 0], preds_scaled], axis=1))[:, -1]

        hold_out_plot = hold_out_data.copy()
        hold_out_plot["Forecast"] = preds

        st.subheader("🔮 Forecast vs Actual")
        fig3, ax3 = plt.subplots(figsize=(12, 5))
        ax3.plot(hold_out_plot.index, hold_out_plot["tsd"], label="Actual")
        ax3.plot(hold_out_plot.index, hold_out_plot["Forecast"], label="Forecast")
        ax3.set_xlabel("Date")
        ax3.set_ylabel("Electricity Demand")
        ax3.legend()
        st.pyplot(fig3)
