import streamlit as st
import pandas as pd
import os
import traceback
from datetime import datetime

# Import all necessary functions
try:
    from sales_forecasting_model_with_logging import (
        run_forecasting_pipeline,
        load_and_prepare,
        plot_forecast,
        plot_prophet_forecast,
        plot_lstm_forecast,
        forecast_sarima_future,
        forecast_prophet_future,
        forecast_lstm_future
    )
except ImportError as e:
    st.error(f"Fatal Error: Could not import functions from the backend script. Please ensure all functions are defined. Details: {e}")
    st.stop()


st.set_page_config(page_title="Sales Forecaster", layout="wide")
st.title("🛍️ Advanced Sales Forecaster")
st.markdown("Upload your sales data to evaluate model performance and generate future forecasts.")

uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    temp_file_path = ""
    try:
        temp_dir = "temp_data"
        os.makedirs(temp_dir, exist_ok=True)
        temp_file_path = os.path.join(temp_dir, uploaded_file.name)
        with open(temp_file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        base_df = load_and_prepare(temp_file_path)
        if base_df is None:
            st.error("Could not process the data. Please check the CSV format and column names ('Date', 'Units Sold').")
        else:
            # Create two main tabs: Model Evaluation and Future Forecast
            eval_tab, future_tab = st.tabs(["Model Performance Evaluation", "Future Forecast"])

            with eval_tab:
                st.header("Evaluating Model Performance on Historical Data")
                with st.spinner("⏳ Running historical evaluation..."):
                    results_summary = run_forecasting_pipeline(csv_path=temp_file_path)
                
                if results_summary and not results_summary.get("error"):
                    st.subheader("Model Comparison (MAPE %)")
                    col1, col2, col3 = st.columns(3)
                    col1.metric("SARIMA", f"{results_summary.get('sarima_mape', 0):.2f}%")
                    col2.metric("Prophet", f"{results_summary.get('prophet_mape', 0):.2f}%")
                    col3.metric("LSTM", f"{results_summary.get('lstm_mape', 0):.2f}%")
                    # Display other plots and data as before
                else:
                    st.error("Failed to evaluate models.")

            with future_tab:
                st.header("Generate a Forecast for the Future")
                future_date = st.date_input("Select a date to forecast up to:", datetime.now() + pd.Timedelta(days=180))
                
                if st.button("Generate Future Forecast"):
                    periods = (future_date - base_df.index.max().date()).days
                    if periods <= 0:
                        st.warning("Please select a date in the future.")
                    else:
                        st.info(f"Forecasting {periods} days into the future...")
                        
                        # Use the best model (SARIMA based on previous results) for the main plot
                        with st.spinner("⏳ Generating SARIMA forecast..."):
                            sarima_future_df = forecast_sarima_future(base_df, periods)
                        
                        if sarima_future_df is not None:
                            st.subheader("SARIMA Future Forecast")
                            fig, ax = plt.subplots(figsize=(14, 7))
                            ax.plot(base_df.index, base_df['y'], label='Historical Data')
                            ax.plot(sarima_future_df['ds'], sarima_future_df['yhat'], label='Future Forecast', linestyle='--')
                            ax.legend()
                            ax.set_title("Future Sales Forecast (SARIMA)")
                            st.pyplot(fig)
                            with st.expander("View Forecast Data"):
                                st.dataframe(sarima_future_df)
                        else:
                            st.error("SARIMA future forecast failed.")
                        
                        # Optional: Add buttons to see Prophet and LSTM future forecasts
                        # ...

    except Exception as e:
        st.error(f"A critical error occurred: {e}")
        st.text(traceback.format_exc())
    finally:
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)