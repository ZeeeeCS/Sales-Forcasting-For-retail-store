import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sales_forecasting_model import run_forecasting_pipeline

st.set_page_config(page_title="Sales Forecasting", layout="wide")
st.title("📈 Sales Forecasting with Prophet")

st.markdown("""
Upload your sales CSV file (must include `Date` and `Demand Forecast` columns). The app will:
- Train a Prophet model
- Forecast future sales
- Display evaluation metrics and visualizations
- Detect model drift
- Log performance metrics
- Enable feedback loop
""")

uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

if uploaded_file:
    with st.spinner("Running forecast pipeline..."):
        uploaded_path = "temp_uploaded_data.csv"
        with open(uploaded_path, "wb") as f:
            f.write(uploaded_file.read())

        model, forecast, test_df, mape, drift, persistent_drift= run_forecasting_pipeline()

        st.success("✅ Forecast Complete!")

        if persistent_drift:
            st.error(f"🚨 Persistent Model Drift Detected! MAPE = {mape:.2f}% (last 5 runs above threshold)")
        elif drift:
            st.warning(f"⚠️ Model Drift Detected! MAPE = {mape:.2f}% exceeds threshold.")
        else:
            st.info(f"📊 Model Performance: MAPE = {mape:.2f}%")

        st.subheader("Forecast vs Actual (Test Data)")
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(test_df['ds'], test_df['y'], label='Actual', color='black')
        forecast_filtered = forecast[forecast['ds'].isin(test_df['ds'])]
        ax.plot(forecast_filtered['ds'], forecast_filtered['yhat'], label='Forecast', color='blue')
        ax.fill_between(forecast_filtered['ds'], forecast_filtered['yhat_lower'], forecast_filtered['yhat_upper'], color='lightblue', alpha=0.4)
        ax.set_title("Forecast vs Actual")
        ax.legend()
        st.pyplot(fig)

        st.subheader("Forecast Components")
        st.pyplot(model.plot_components(forecast))

        st.subheader("Raw Forecast Data")
        st.dataframe(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(30))

        st.download_button("Download Forecast as CSV", forecast.to_csv(index=False), file_name="forecast_output.csv")

        st.subheader("Model Feedback")
        st.markdown("Upload actuals to help improve the model in future retraining:")

        st.subheader("📘 Past Model Metrics")
        try:
            logs = pd.read_csv("metrics_log.csv")
            st.dataframe(logs.tail(10))
        except FileNotFoundError:
            st.info("No logs yet. Upload data to start tracking.")
