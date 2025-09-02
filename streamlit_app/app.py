import streamlit as st
import pandas as pd
import os
import traceback
import sales_forecasting_model_with_logging
# --- Import functions from your model script ---
# Make sure 'sales_forecasting_model_with_logging.py' is in the same directory
try:
    from sales_forecasting_model_with_logging import (
        run_forecasting_pipeline,
        load_and_prepare,  # We need this to get the base df for plotting
        plot_prophet_forecast,
        plot_lstm_forecast
    )
except ImportError:
    st.error("Fatal Error: The 'sales_forecasting_model_with_logging.py' file was not found. Please ensure it's in the same directory as this Streamlit app.")
    st.stop()


# --- Page Configuration ---
st.set_page_config(page_title="Sales Forecaster", layout="wide")

# --- Main App ---
st.title("🛍️ Sales Forecaster")
st.markdown("Upload your sales data in CSV format to generate forecasts using Prophet and LSTM models.")

uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    temp_file_path = ""
    try:
        # Save the uploaded file temporarily so our pipeline can read it from a path
        temp_dir = "temp_data"
        os.makedirs(temp_dir, exist_ok=True)
        temp_file_path = os.path.join(temp_dir, uploaded_file.name)
        with open(temp_file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        # Prepare the data once for plotting purposes
        # This ensures our plots have the correct historical data to show
        base_df = load_and_prepare(temp_file_path)
        if base_df is None:
            st.error("Could not process the uploaded data. Please check the file for correct 'Date' and 'Units Sold' columns.")
        else:
            experiment_name = f"StreamlitRun_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}"
            
            with st.spinner("⏳ Running forecast models... This may take a few minutes."):
                results_summary = run_forecasting_pipeline(
                    csv_path=temp_file_path,
                    experiment_name=experiment_name
                )

            st.header("📊 Forecasting Results")

            if results_summary and not results_summary.get("error"):
                st.success("Forecasting pipeline completed successfully!")

                # --- Display Key Metrics ---
                st.subheader("Model Performance (MAPE %)")
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Prophet (Base)", f"{results_summary.get('prophet_original_mape', 0):.2f}%")
                with col2:
                    st.metric("LSTM (Base)", f"{results_summary.get('lstm_original_mape', 0):.2f}%")
                with col3:
                    st.metric("Prophet (Tuned)", f"{results_summary.get('prophet_hyperparam_mape', 0):.2f}%")
                with col4:
                    st.metric("LSTM (Tuned)", f"{results_summary.get('lstm_hyperparam_mape', 0):.2f}%")
                
                st.markdown("---")

                # --- Display Plots and Data ---
                prophet_fcst_df = results_summary.get("prophet_hyperparam_forecast_df")
                if prophet_fcst_df is not None and not prophet_fcst_df.empty:
                    st.subheader("Prophet Forecast (Hyperparameter Tuned)")
                    prophet_fig = plot_prophet_forecast(base_df, prophet_fcst_df, "Prophet Forecast")
                    st.pyplot(prophet_fig, use_container_width=True)
                    with st.expander("View Prophet Forecast Data"):
                        st.dataframe(prophet_fcst_df[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].round(2))

                lstm_preds = results_summary.get("lstm_hyperparam_predictions")
                lstm_actuals = results_summary.get("lstm_hyperparam_actuals")
                lstm_dates = results_summary.get("lstm_hyperparam_dates")
                if lstm_preds is not None and lstm_actuals is not None and lstm_dates is not None:
                    st.subheader("LSTM Forecast (Hyperparameter Tuned)")
                    lstm_fig = plot_lstm_forecast(base_df, lstm_preds, lstm_actuals, lstm_dates, "LSTM Forecast")
                    st.pyplot(lstm_fig, use_container_width=True)
                    with st.expander("View LSTM Forecast Data"):
                        df_to_show = pd.DataFrame({
                            'Date': lstm_dates,
                            'Actual': lstm_actuals.flatten(),
                            'Forecast': lstm_preds.flatten()
                        })
                        st.dataframe(df_to_show.round(2))
                
                st.markdown("---")

                # --- Display Drift and MLflow Info ---
                st.subheader("Additional Information")
                drift_col, mlflow_col = st.columns(2)
                with drift_col:
                    st.markdown(f"**Drift Detected on LSTM:** `{results_summary.get('drift_detected_on_hp_lstm', 'N/A')}`")
                    st.markdown(f"**Persistent Drift Check:** `{results_summary.get('persistent_drift_on_hp_lstm', 'N/A')}`")
                with mlflow_col:
                    st.info(f"**MLflow Experiment ID:** `{results_summary.get('mlflow_experiment_id', 'N/A')}`")
                    st.info(f"**MLflow Run ID:** `{results_summary.get('mlflow_run_id', 'N/A')}`")

            elif results_summary and results_summary.get("error"):
                st.error(f"Forecasting pipeline failed: {results_summary['error']}")
                if results_summary.get("mlflow_run_id"):
                    st.info(f"MLflow Run ID (for debugging failed run): {results_summary['mlflow_run_id']}")
            else:
                st.error("An unknown error occurred during the forecasting process.")

    except Exception as e:
        st.error(f"A critical error occurred in the application: {e}")
        st.text(traceback.format_exc()) # Show full traceback for debugging

    finally:
        # Clean up the temporary file regardless of success or failure
        if os.path.exists(temp_file_path):
            try:
                os.remove(temp_file_path)
            except Exception as e_clean:
                st.warning(f"Could not remove temporary file {temp_file_path}: {e_clean}")

else:
    st.info("Awaiting CSV file upload to begin...")