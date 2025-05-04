import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sales_forecasting_model import run_forecasting_pipeline

st.set_page_config(page_title="Sales Forecasting", layout="wide")
st.title("📈 Sales Forecasting with Prophet & LSTM")

st.markdown("""
Upload your sales CSV file (must include `Date` and `Demand Forecast` columns). The app will:
- Train Prophet and LSTM models
- Forecast future sales
- Display evaluation metrics and visualizations
- Detect model drift
- Log performance metrics
""")

uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

if uploaded_file:
    with st.spinner("Running forecast pipeline..."):
        uploaded_path = "temp_uploaded_data.csv"
        with open(uploaded_path, "wb") as f:
            f.write(uploaded_file.read())

        model_prophet, model_lstm, forecast_prophet, forecast_lstm, test_df, mape = run_forecasting_pipeline(uploaded_path)

        st.success("✅ Forecast Complete!")

        st.subheader("Prophet Forecast vs Actual (Test Data)")
        if forecast_prophet is not None:
            forecast_filtered = forecast_prophet[forecast_prophet['ds'].isin(test_df['ds'])]
            fig, ax = plt.subplots(figsize=(12, 5))
            ax.plot(test_df['ds'], test_df['y'], label='Actual', color='black')
            ax.plot(forecast_filtered['ds'], forecast_filtered['yhat'], label='Forecasted', color='blue')
            ax.fill_between(forecast_filtered['ds'], forecast_filtered['yhat_lower'], forecast_filtered['yhat_upper'], color='blue', alpha=0.2)
            ax.set_title('Prophet Forecast vs Actual (Test Data)')
            ax.legend()
            st.pyplot(fig)
        else:
            st.error("Prophet model failed to return a forecast. Please check the model training.")


        st.subheader("LSTM Forecast vs Actual (Test Data)")
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(test_df['ds'], test_df['y'], label='Actual', color='black')
        ax.plot(forecast_lstm, label='LSTM Forecast', color='red')
        ax.set_title("LSTM Forecast vs Actual")
        ax.legend()
        st.pyplot(fig)
