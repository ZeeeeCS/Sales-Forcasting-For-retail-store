# app.py (FOR DEPLOYMENT ON STREAMLIT CLOUD / GITHUB PAGES)
import streamlit as st
import pandas as pd
import os
import sys
import traceback

# --- Make sure this import is correct based on your filename ---
# Ensure 'sales_forecasting_model_with_logging.py' is in the same directory
# or the path is correctly handled for your GitHub repository structure.
from sales_forecasting_model_with_logging import run_forecasting_pipeline

st.set_page_config(layout="wide")
st.title("Retail Store Sales Forecaster")
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