```
# Electricity Insights Dashboard

## Overview
The **Electricity Insights Dashboard** is a data-driven platform that provides valuable insights into electricity consumption patterns, weather forecasts, and demand prediction. It uses data from various sources like the **CEA API**, **Visual Crossing Weather API**, and **LSTM (Long Short-Term Memory) models** for demand forecasting.

---

## Features
### 📊 Per Capita Electricity Consumption + Weather
This section shows the per capita electricity consumption data over the years and provides a 7-day weather forecast for selected states. You can visualize:
- The per capita consumption over time.
- The weather forecast with maximum and minimum temperatures, as well as descriptions for each day.

### 🔮 LSTM Demand Forecast
This section allows you to upload historical electricity demand data and train an LSTM model to forecast future demand. You can:
- Upload your own CSV dataset for training the model.
- Visualize distribution and trends in the electricity demand.
- Train the LSTM model and see the training/validation loss.
- View the model's predictions.

### 📈 Linear Forecast for Next 5 Days
For the selected states, a linear regression model forecasts the electricity consumption for the next 5 days based on historical data.

### Custom Sidebar Menu (Hamburger)
The sidebar includes a collapsible menu with a hamburger icon that users can click to toggle the visibility of the sidebar. The sidebar uses custom CSS and JavaScript for this interactive feature.

---

## Setup and Installation

### Clone the repository:
First, clone the repository to your local machine.

```bash
git clone https://github.com/your-username/electricity-insights-dashboard.git
cd electricity-insights-dashboard
```

### Install dependencies:
Install the required dependencies by running:

```bash
pip install -r requirements.txt
```

### Set up API keys:
For **Visual Crossing Weather API**, sign up and get an API key from [Visual Crossing](https://www.visualcrossing.com/).

Replace the placeholder API key in the code with your actual API key.

### Run the app:
To launch the dashboard locally, run the following command in your terminal:

```bash
streamlit run app.py
```

This will start a local server, and you can view the app at [http://localhost:8501](http://localhost:8501).

---

## Example of Use
1. **Data Selection**: Use the sidebar to select states and adjust the year range for electricity consumption data.
2. **Weather Forecast**: View the weather forecast for the selected states.
3. **Demand Forecasting**: Upload a CSV file with electricity demand data to train the LSTM model and view predictions.

---

## Contributing
Contributions are welcome! If you'd like to contribute, feel free to fork the repository, make your changes, and submit a pull request. Here are a few ways you can help:
- Fix bugs and improve existing features.
- Add new features and enhancements.
- Help improve documentation.

---

## License
This project is licensed under the **MIT License** - see the LICENSE file for details.

---

## Contact
For any queries, please contact the project maintainers:

- **Tushar Panwar** (21BCE1074)
```
