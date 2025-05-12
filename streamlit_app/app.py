# app.py (FOR DEPLOYMENT ON STREAMLIT CLOUD / GITHUB PAGES)
import streamlit as st
import pandas as pd
import numpy as np # Needed if handling numpy arrays from results
import os
import sys
<<<<<<< HEAD
import traceback
=======
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
>>>>>>> 8351801cfc5a8af9fdcec1cd1f0a58ce971e0d80

# --- Make sure this import is correct based on your filename ---
# Ensure 'sales_forecasting_model_with_logging.py' is in the same directory
# or the path is correctly handled for your GitHub repository structure.
from sales_forecasting_model_with_logging import run_forecasting_pipeline

st.set_page_config(layout="wide")
st.title("Retail Store Sales Forecaster")
<<<<<<< HEAD
st.markdown("Upload a CSV file with 'Date' and 'Units Sold' columns to generate forecasts.")

uploaded_file = st.file_uploader("Choose a CSV file", type="csv", help="CSV must contain 'Date' and 'Units Sold' columns.")

if uploaded_file is not None:
    st.info(f"File Uploaded: **{uploaded_file.name}**. Processing...")

    col1, col2 = st.columns([1, 2]) # Use columns for better layout

    with col1:
        # --- Debugging Block to Preview Uploaded Data ---
        try:
            st.write("--- Uploaded Data Sample (First 5 Rows) ---")
            # Try reading the first few rows to check format
            debug_df = pd.read_csv(uploaded_file, nrows=5)
            st.dataframe(debug_df)
            cols_found = debug_df.columns.tolist()
            st.write("**Columns found in uploaded file:**", cols_found)
            required_cols = ['Date', 'Units Sold']
            missing_cols = [col for col in required_cols if col not in cols_found]
            if missing_cols:
                 st.error(f"Error: Missing required columns! Expected 'Date' and 'Units Sold', but found {cols_found}. Please check your CSV file format.")
                 st.stop() # Stop execution here if columns are missing
            else:
                 st.success("Required columns ('Date', 'Units Sold') found.")
            # Reset buffer position for the actual processing by the pipeline
            uploaded_file.seek(0)
        except Exception as e_debug:
            st.error(f"Error reading the uploaded CSV for preview: {e_debug}")
            st.text(traceback.format_exc())
            st.stop() # Stop execution here if basic reading fails
        # --- End Debugging Block ---

    with col2:
        try:
            # Save the uploaded file temporarily (Streamlit Cloud provides an ephemeral filesystem)
            temp_dir = "temp_data" # This will be created within the app's container
            if not os.path.exists(temp_dir):
                os.makedirs(temp_dir)

            temp_file_path = os.path.join(temp_dir, "uploaded_data.csv") # Use a consistent name
            with open(temp_file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            st.write(f"Processing file...") # Simplified message

            # Define a unique experiment name for each run
            experiment_name = f"StreamlitCloud_UserRun_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}"

            results_summary = None # Initialize
            with st.spinner("Running forecast models... This may take a few minutes."):
                # Call the imported function from your modeling script
                results_summary = run_forecasting_pipeline(
                    csv_path=temp_file_path,
                    experiment_name=experiment_name
                )

            # --- Display Results ---
            st.subheader("Forecasting Results Summary:")

            if results_summary:
                st.write(f"**Pipeline Status:** {results_summary.get('pipeline_status', 'Unknown')}")

                if results_summary.get("pipeline_status") == "completed":
                    st.success("Forecasting pipeline completed successfully!")
                    results_table_data = {
                        "Model": ["Prophet (Original)", "LSTM (Original)", "Prophet (Hyperparam)", "LSTM (Hyperparam)"],
                        "MAPE (%)": [
                            f"{results_summary.get('prophet_original_mape', float('inf')):.2f}",
                            f"{results_summary.get('lstm_original_mape', float('inf')):.2f}",
                            f"{results_summary.get('prophet_hyperparam_mape', float('inf')):.2f}",
                            f"{results_summary.get('lstm_hyperparam_mape', float('inf')):.2f}"
                        ]
                    }
                    st.table(pd.DataFrame(results_table_data))
=======
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
                exp_id = results_summary.get('mlflow_experiment_id', None)
                st.info(f"MLflow Run ID: {run_id}")
                if exp_id:
                     st.info(f"MLflow Experiment ID: {exp_id}")
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
>>>>>>> 8351801cfc5a8af9fdcec1cd1f0a58ce971e0d80

                    drift_hp = results_summary.get('drift_detected_on_hp_lstm')
                    persist_drift_hp = results_summary.get('persistent_drift_on_hp_lstm')
                    st.write("**Drift Detection (LSTM Hyperparameter Model):**")
                    st.write(f"*   Immediate Drift Detected (>20% MAPE): `{drift_hp}`")
                    st.write(f"*   Persistent Drift Detected (Recent >20% MAPE): `{persist_drift_hp}`")

                    # MLflow Run ID - useful if you have a REMOTE MLflow server configured.
                    # If not, this ID points to an ephemeral run on Streamlit Cloud.
                    if "mlflow_run_id" in results_summary and results_summary["mlflow_run_id"]:
                        st.info(f"MLflow Run ID (for developer tracking if remote server is set up): `{results_summary['mlflow_run_id']}`")
                    else:
                        st.warning("MLflow Run ID not available in results.")

                elif results_summary.get("pipeline_status") == "failed_data_load":
                     st.error(f"Forecasting Failed: {results_summary.get('error_message', 'Could not load or prepare the uploaded data.')} Please check CSV format and required columns.")
                else: # Other failure
                     st.error(f"Forecasting Failed: {results_summary.get('error_message', 'An unexpected error occurred in the pipeline.')}")
                     if "mlflow_run_id" in results_summary and results_summary["mlflow_run_id"]:
                         st.info(f"MLflow Run ID for debugging (if remote server is set up): `{results_summary['mlflow_run_id']}`")
            else:
                st.error("Something went wrong during the forecasting process. No summary dictionary returned from pipeline.")

            # Clean up the temporary file
            if os.path.exists(temp_file_path):
                try:
                    os.remove(temp_file_path)
                except Exception as e_clean:
                    st.warning(f"Could not remove temporary file {temp_file_path}: {e_clean}")

        except Exception as e_app:
            st.error(f"An error occurred in the Streamlit application: {e_app}")
            st.text(traceback.format_exc())
else:

    st.info("Please upload a CSV file containing 'Date' and 'Units Sold' columns to start forecasting.")

    st.info("Upload a CSV file to start forecasting.")
>>>>>>> 8351801cfc5a8af9fdcec1cd1f0a58ce971e0d80
