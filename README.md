# 🔋 Electricity Insights Dashboard

## Overview
The **Electricity Insights Dashboard** is a Streamlit-based web application that provides valuable insights into electricity consumption patterns, weather forecasts, and electricity demand forecasting using advanced machine learning models.

This dashboard leverages multiple data sources, including the **CEA API** for per capita electricity consumption data, **Visual Crossing Weather API** for weather forecasts, and **LSTM (Long Short-Term Memory)** models for predicting future electricity demand.

### Key Features:
- **Consumption & Weather Analysis**: Track per capita electricity consumption and access weather forecasts for selected states.
- **Demand Forecasting**: Use an LSTM-based model to forecast future electricity demand based on historical data.
- **Data Exploration**: Visualize trends in electricity consumption over time, with a linear regression forecast for the next 5 days.
- **LSTM Training**: Train LSTM models on electricity demand data, including data preprocessing, feature engineering, and model evaluation.

## Getting Started

### Prerequisites
To run the **Electricity Insights Dashboard** locally, you need to have Python 3.7+ installed along with the following libraries:

- Streamlit
- Requests
- Pandas
- NumPy
- Matplotlib
- Seaborn
- scikit-learn
- TensorFlow
- Holidays (Python package for holidays)
- Visual Crossing API Key (for weather data)

You can install all required libraries using `pip`:

```bash
pip install -r requirements.txt
