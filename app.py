import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.arima.model import ARIMA
from prophet import Prophet
from sklearn.metrics import mean_squared_error
from math import sqrt

# ---------------------------------------------
# Streamlit App Configuration
# ---------------------------------------------
st.set_page_config(page_title="Sales Forecasting App", layout="wide")

st.title("📊 Time Series Forecasting using Holt-Winters, ARIMA, and Prophet")
st.markdown("""
This app compares three forecasting models — **Holt-Winters**, **ARIMA**, and **Prophet** — 
to predict weekly sales (units sold) based on uploaded demand data.
""")

# ---------------------------------------------
# File Upload
# ---------------------------------------------
uploaded_file = st.file_uploader("📂 Upload your demand dataset (CSV)", type=["csv"])

if uploaded_file:
    data = pd.read_csv(uploaded_file)

    # Display raw data
    st.subheader("🔹 Raw Data Preview")
    st.dataframe(data.head())

    # Check column existence
    if 'week' not in data.columns or 'units_sold' not in data.columns:
        st.error("❌ The dataset must contain 'week' and 'units_sold' columns.")
    else:
        # Convert week to datetime
        data['week'] = pd.to_datetime(data['week'], format='%d/%m/%y', errors='coerce')
        data.set_index('week', inplace=True)
        weekly_data = data['units_sold'].resample('W').sum()

        # Train-test split
        train_data = weekly_data[:int(0.8 * len(weekly_data))]
        test_data = weekly_data[int(0.8 * len(weekly_data)):]

        # ---------------------------------------------
        # Model 1: Holt-Winters Exponential Smoothing
        # ---------------------------------------------
        with st.spinner("Training Holt-Winters model..."):
            hw_model = ExponentialSmoothing(train_data, seasonal='add', seasonal_periods=52).fit()
            hw_predictions = hw_model.predict(start=test_data.index[0], end=test_data.index[-1])
            hw_rmse = sqrt(mean_squared_error(test_data, hw_predictions))

        # ---------------------------------------------
        # Model 2: ARIMA
        # ---------------------------------------------
        with st.spinner("Training ARIMA model..."):
            arima_model = ARIMA(train_data, order=(1, 0, 0)).fit()
            arima_predictions = arima_model.predict(start=test_data.index[0], end=test_data.index[-1])
            arima_rmse = sqrt(mean_squared_error(test_data, arima_predictions))

        # ---------------------------------------------
        # Model 3: Prophet
        # ---------------------------------------------
        with st.spinner("Training Prophet model..."):
            prophet_data = weekly_data.reset_index()
            prophet_data.columns = ['ds', 'y']

            train_data_prophet = prophet_data[:int(0.8 * len(prophet_data))]
            test_data_prophet = prophet_data[int(0.8 * len(prophet_data)):]

            prophet_model = Prophet(yearly_seasonality=True)
            prophet_model.fit(train_data_prophet)
            prophet_future = prophet_model.make_future_dataframe(periods=len(test_data_prophet))
            prophet_predictions = prophet_model.predict(prophet_future)
            prophet_rmse = sqrt(mean_squared_error(
                test_data_prophet['y'],
                prophet_predictions['yhat'][-len(test_data_prophet):]
            ))

        # ---------------------------------------------
        # Results
        # ---------------------------------------------
        st.subheader("📉 Model Performance (RMSE)")
        result_df = pd.DataFrame({
            "Model": ["Holt-Winters", "ARIMA", "Prophet"],
            "RMSE": [hw_rmse, arima_rmse, prophet_rmse]
        })
        st.table(result_df)

        best_model = result_df.loc[result_df['RMSE'].idxmin(), 'Model']
        st.success(f"🏆 Best Performing Model: **{best_model}**")

        # ---------------------------------------------
        # Visualization
        # ---------------------------------------------
        st.subheader("📈 Forecast Comparison")
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(weekly_data, label='Original', color='black')
        ax.plot(hw_predictions, label='Holt-Winters', color='blue')
        ax.plot(arima_predictions, label='ARIMA', color='green')
        ax.plot(prophet_predictions['ds'], prophet_predictions['yhat'], label='Prophet', color='red')
        ax.legend()
        st.pyplot(fig)

        # Prophet Components
        with st.expander("📊 Prophet Trend & Seasonality Components"):
            fig2 = prophet_model.plot_components(prophet_predictions)
            st.pyplot(fig2)

else:
    st.info("👆 Upload a CSV file to begin. Make sure it includes 'week' and 'units_sold' columns.")
