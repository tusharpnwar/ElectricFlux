with tab1:
    st.header("📊 Per Capita Electricity Consumption + Weather")

    @st.cache_data
    def fetch_cea_data():
        url = "https://cea.nic.in/api/percapitalConsumtion.php"
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

    st.subheader("🌤️ Current Weather in Chennai")
    weather_url = "https://api.weatherstack.com/current?access_key=e9d2d3d5eded1d6fe2e85a30522ae852&query=Chennai"
    weather_response = requests.get(weather_url)

    if weather_response.status_code == 200:
        weather_data = weather_response.json()
        current = weather_data.get("current", {})
        location = weather_data.get("location", {})
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Temperature", f"{current.get('temperature', 'N/A')} °C")
            st.metric("Humidity", f"{current.get('humidity', 'N/A')}%")
            st.metric("Wind Speed", f"{current.get('wind_speed', 'N/A')} km/h")
        with col2:
            st.metric("Weather Description", current.get("weather_descriptions", ['N/A'])[0])
            st.metric("Location", f"{location.get('name', '')}, {location.get('country', '')}")
            st.metric("Observation Time", current.get("observation_time", 'N/A'))
    else:
        st.warning("Failed to fetch current weather data for Chennai.")

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
