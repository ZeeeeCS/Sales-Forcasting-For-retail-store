import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sales_forecasting_model import run_forecasting_pipeline

st.set_page_config(page_title="Sales Forecast Dashboard", layout="wide")

st.title("📈 Sales Forecasting with LSTM and Prophet")

uploaded_file = st.file_uploader("Upload your sales CSV file", type=["csv"])

if uploaded_file:
    with st.spinner("Running forecasting pipeline..."):
        prophet_model,prophet_forecast,test_df,lstm_model,lstm_preds,lstm_true,dates,lstm_mape,drift,persistent_drift,prophet_mape= run_forecasting_pipeline(uploaded_file)


    st.subheader("📊 Forecast Visualizations")

    # Prophet Plot
    st.markdown("### Prophet Forecast")
    fig1, ax1 = plt.subplots()
    ax1.plot(test_df["ds"], test_df["y"], label="Actual")
    ax1.plot(prophet_forecast["ds"], prophet_forecast["yhat"], label="Forecast")
    ax1.set_xlabel("Date")
    ax1.set_ylabel("Units Sold")
    ax1.set_title("Prophet: Actual vs Forecast")
    ax1.legend()
    st.pyplot(fig1)
    st.write("Prophet MAPE:", f"{prophet_mape:.2f}%")
    
    # LSTM Plot
    st.markdown("### LSTM Forecast")
    lstm_forecast_df = pd.DataFrame({"ds": dates, "yhat": lstm_preds.flatten()})
    lstm_test_df = pd.DataFrame({"ds": dates, "y": lstm_true.flatten()})
    fig2, ax2 = plt.subplots()
    ax2.plot(lstm_test_df["ds"], lstm_test_df["y"], label="Actual")
    ax2.plot(lstm_forecast_df["ds"], lstm_forecast_df["yhat"], label="Forecast")
    ax2.set_xlabel("Date")
    ax2.set_ylabel("Units Sold")
    ax2.set_title("LSTM: Actual vs Forecast")
    ax2.legend()
    st.pyplot(fig2)

    # Drift Detection Results
    st.subheader("🚨 Drift Detection")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Prophet MAPE", f"{prophet_mape:.2f}%")
    with col2:
        st.metric("LSTM MAPE", f"{lstm_mape:.2f}%", delta="Drift" if lstm_drift else "Stable")

    st.write("📌 **Drift Status**")
    st.write(f"**LSTM Drift:** {'Detected' if lstm_drift else 'Not Detected'}")
    st.write(f"**LSTM Persistent Drift:** {'Yes' if lstm_persistent_drift else 'No'}")
