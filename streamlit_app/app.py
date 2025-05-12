import streamlit as st
import pandas as pd
import numpy as np # Needed if handling numpy arrays from results
import os
import sys
import time # For timestamp

# --- Import the pipeline function ---
# Option 1: If you renamed the file
from sales_forecasting_model_with_logging import run_forecasting_pipeline
# Option 2: If you kept the original name
# from sales_forecasting_model_with_logging import run_forecasting_pipeline
# ---

import mlflow # Keep for potential future use, but not strictly needed for display now


# --- Page Config ---
st.set_page_config(layout="wide") # Use wider layout

NGROK_BASE_URL = "http://127.0.0.1:5000"


st.title("Retail Store Sales Forecaster")
st.write("Upload your daily sales data (CSV format) with 'Date' and 'Units Sold' columns.")

uploaded_file = st.file_uploader("Choose CSV File", type="csv", label_visibility="collapsed")

if uploaded_file is not None:
    st.write("File Uploaded! Processing...")

    # Use BytesIO object directly if load_and_prepare handles it, otherwise save temp file
    # Using temp file approach as it's currently in the pipeline code example:
    temp_dir = "temp_data"
    if not os.path.exists(temp_dir):
        try:
            os.makedirs(temp_dir)
        except OSError as e:
            st.error(f"Could not create temporary directory: {e}")
            st.stop() # Stop execution if we can't save the file

    temp_file_path = os.path.join(temp_dir, f"upload_{int(time.time())}_{uploaded_file.name}")

    try:
        with open(temp_file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.write(f"Processing file: {uploaded_file.name}")

        # Use a unique experiment name for each Streamlit session or run
        # Using timestamp is simple for uniqueness
        experiment_name = f"Streamlit_UserRun_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}"

        results_summary = None # Initialize
        with st.spinner("Running forecast models... This may take several minutes."):
            results_summary = run_forecasting_pipeline(
                csv_path=temp_file_path, # Pass path to temp file
                experiment_name=experiment_name
            )

        # --- Display Results ---
        st.subheader("Forecasting Results Summary")

        if results_summary and results_summary.get("error") is None:

            col1, col2 = st.columns(2) # Create columns for layout

            with col1:
                st.metric("Prophet (Original) MAPE", f"{results_summary.get('prophet_original_mape', 'N/A'):.2f}%" if pd.notna(results_summary.get('prophet_original_mape')) else "N/A")
                st.metric("LSTM (Original) MAPE", f"{results_summary.get('lstm_original_mape', 'N/A'):.2f}%" if pd.notna(results_summary.get('lstm_original_mape')) else "N/A")

            with col2:
                st.metric("Prophet (Hyperparameter) MAPE", f"{results_summary.get('prophet_hyperparam_mape', 'N/A'):.2f}%" if pd.notna(results_summary.get('prophet_hyperparam_mape')) else "N/A")
                st.metric("LSTM (Hyperparameter) MAPE", f"{results_summary.get('lstm_hyperparam_mape', 'N/A'):.2f}%" if pd.notna(results_summary.get('lstm_hyperparam_mape')) else "N/A")

            st.markdown("---") # Separator

            # Display Drift Detection Info
            st.write(f"Drift Detected on LSTM (Hyperparameter): **{results_summary.get('drift_detected_on_hp_lstm', 'N/A')}**")
            st.caption(f"Persistent Drift Check on LSTM (Hyperparameter): {results_summary.get('persistent_drift_on_hp_lstm', 'N/A')} (Note: may require persistent storage)")

            # Display Forecast DataFrames and Plots (Example for Prophet HP)
            st.subheader("Forecast Visualization (Prophet - Hyperparameter)")
            prophet_fcst_df_hp = results_summary.get("prophet_hyperparam_forecast_df")
            if prophet_fcst_df_hp is not None and not prophet_fcst_df_hp.empty:
                 try:
                     # Ensure 'ds' is datetime if needed for plotting
                     prophet_fcst_df_hp['ds'] = pd.to_datetime(prophet_fcst_df_hp['ds'])
                     # Plot forecast with uncertainty intervals
                     fig = go.Figure()
                     fig.add_trace(go.Scatter(x=prophet_fcst_df_hp['ds'], y=prophet_fcst_df_hp['yhat'], mode='lines', name='Forecast (yhat)'))
                     fig.add_trace(go.Scatter(x=prophet_fcst_df_hp['ds'], y=prophet_fcst_df_hp['yhat_upper'], mode='lines', name='Upper CI', line=dict(width=0)))
                     fig.add_trace(go.Scatter(x=prophet_fcst_df_hp['ds'], y=prophet_fcst_df_hp['yhat_lower'], mode='lines', name='Lower CI', line=dict(width=0), fill='tonexty', fillcolor='rgba(0,100,80,0.2)'))
                     fig.update_layout(title="Prophet Forecast with Confidence Interval", xaxis_title="Date", yaxis_title="Units Sold")
                     st.plotly_chart(fig, use_container_width=True)

                     st.dataframe(prophet_fcst_df_hp[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].round(2))
                 except Exception as plot_e:
                     st.warning(f"Could not plot Prophet forecast: {plot_e}")
                     st.dataframe(prophet_fcst_df_hp.round(2)) # Show data anyway
            else:
                st.write("Prophet (Hyperparameter) forecast data not available.")

            # Display Forecast DataFrames and Plots (Example for LSTM HP)
            st.subheader("Forecast Visualization (LSTM - Hyperparameter)")
            lstm_preds = results_summary.get("lstm_hyperparam_predictions")
            lstm_dates = results_summary.get("lstm_hyperparam_dates")
            if lstm_preds is not None and lstm_dates is not None and len(lstm_dates) == len(lstm_preds):
                 try:
                     lstm_df_display = pd.DataFrame({'Date': lstm_dates, 'Forecast': lstm_preds})
                     st.line_chart(lstm_df_display.set_index('Date')['Forecast'])
                     st.dataframe(lstm_df_display.round(2))
                 except Exception as plot_e:
                      st.warning(f"Could not plot LSTM forecast: {plot_e}")
                      st.dataframe(pd.DataFrame({'Date': lstm_dates, 'Forecast': lstm_preds}).round(2))
            else:
                 st.write("LSTM (Hyperparameter) forecast data not available or lengths mismatch.")


            st.success("Forecasting pipeline completed successfully!")
            if results_summary.get("mlflow_run_id"):
                run_id = results_summary['mlflow_run_id']
                # IMPORTANT: Ensure your pipeline function returns 'mlflow_experiment_id'
                exp_id = results_summary.get('mlflow_experiment_id', None) # Get experiment ID

                st.info(f"MLflow Run ID: `{run_id}`") # Display Run ID clearly
                if exp_id:
                    st.info(f"MLflow Experiment ID: `{exp_id}`") # Display Exp ID

                # Check if a valid ngrok/MLflow base URL was provided by the user
                if NGROK_BASE_URL and NGROK_BASE_URL != "PASTE_YOUR_PUBLIC_NGROK_URL_HERE" and exp_id:
                    # Construct the URL to the specific run in the MLflow UI
                    mlflow_run_url = f"{NGROK_BASE_URL.rstrip('/')}/#/experiments/{exp_id}/runs/{run_id}"
                    st.markdown(f"🔗 **[View Full Run Details in MLflow UI]({mlflow_run_url})**")
                    st.caption(f"(MLflow UI via: {NGROK_BASE_URL})")
                else:
                    # If no valid base URL or no experiment ID, just show the caption
                    st.caption("(Set NGROK_BASE_URL in app.py and ensure pipeline returns Experiment ID to enable direct link)")
            else:
                 st.warning("MLflow Run ID was not found in pipeline results.")
            # -------------------------------------------------

        elif results_summary and results_summary.get("error"): # <--- This line likely already exists
            st.error(f"Forecasting pipeline failed: {results_summary['error']}") # <--- This line likely already exists
            # ... rest of your error handling ...
                # --- Optional: Link to Remote MLflow UI ---
                # MLFLOW_SERVER_URL = os.environ.get("MLFLOW_TRACKING_URI") # Example: Get from env var
                # if MLFLOW_SERVER_URL and exp_id and run_id:
                #      mlflow_run_link = f"{MLFLOW_SERVER_URL.rstrip('/')}/#/experiments/{exp_id}/runs/{run_id}"
                #      st.markdown(f"[View Run Details in MLflow UI]({mlflow_run_link})", unsafe_allow_html=True)
                # ------------------------------------------

        elif results_summary and results_summary.get("error"):
            st.error(f"Forecasting pipeline failed: {results_summary['error']}")
            if results_summary.get("mlflow_run_id"):
                st.info(f"MLflow Run ID (failed): {results_summary['mlflow_run_id']}")
        else:
            st.error("Something went wrong during the forecasting process. No summary returned.")


    except Exception as e:
        st.error(f"An error occurred in the Streamlit app: {e}")
        import traceback
        st.text(traceback.format_exc()) # Show full traceback for debugging
    finally:
        # Clean up the temporary file regardless of success/failure
        if 'temp_file_path' in locals() and os.path.exists(temp_file_path):
             try:
                 os.remove(temp_file_path)
                 st.write(f"Cleaned up temporary file: {temp_file_path}")
             except Exception as e_clean:
                 st.warning(f"Could not remove temporary file {temp_file_path}: {e_clean}")

else:
    st.info("Upload a CSV file to start forecasting.")