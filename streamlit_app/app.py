import streamlit as st
import pandas as pd
import os
import sys # To add to sys.path if your model script is in a different directory
from sales_forecasting_model_with_logging import run_forecasting_pipeline # Your main modeling script, possibly renamed

st.title("Retail Store Sales Forecaster")

uploaded_file = st.file_uploader("Choose a CSV file for sales data", type="csv")

if uploaded_file is not None:
    st.write("File Uploaded! Processing...")
    try:
        # Save the uploaded file temporarily to pass its path to your pipeline
        # (Your pipeline function currently expects a file path)
        temp_dir = "temp_data"
        if not os.path.exists(temp_dir):
            os.makedirs(temp_dir)
        
        temp_file_path = os.path.join(temp_dir, uploaded_file.name)
        with open(temp_file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        st.write(f"Temporary file saved at: {temp_file_path}")
        experiment_name = f"Streamlit_UserRun_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}"
        
        with st.spinner("Running forecast models... This may take a few minutes."):
            results_summary = run_forecasting_pipeline(
                csv_path=temp_file_path,
                experiment_name=experiment_name # Each user run gets a new MLflow experiment or run
            )

        if results_summary:
            st.subheader("Forecasting Results Summary:")
            
            # Display metrics from the summary
            if "prophet_original_mape" in results_summary:
                st.write(f"Prophet (Original) MAPE: {results_summary['prophet_original_mape']:.2f}%")
            if "lstm_original_mape" in results_summary:
                st.write(f"LSTM (Original) MAPE: {results_summary['lstm_original_mape']:.2f}%")
            if "prophet_hyperparam_mape" in results_summary:
                st.write(f"Prophet (Hyperparam) MAPE: {results_summary['prophet_hyperparam_mape']:.2f}%")
            if "lstm_hyperparam_mape" in results_summary:
                st.write(f"LSTM (Hyperparam) MAPE: {results_summary['lstm_hyperparam_mape']:.2f}%")

            st.success("Forecasting pipeline completed!")
            if "mlflow_run_id" in results_summary:
                st.info(f"MLflow Run ID for this forecast: {results_summary['mlflow_run_id']}")
                # If you have a publicly accessible MLflow UI, you could construct a link here.
                # local_mlflow_link = "http://localhost:5000/#/experiments/.../runs/..." # Needs logic
        else:
            st.error("Something went wrong during the forecasting process. No summary returned.")
            
        # Clean up the temporary file
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)

    except Exception as e:
        st.error(f"An error occurred: {e}")
        import traceback
        st.text(traceback.format_exc())

else:
    st.info("Please upload a CSV file to start forecasting.")
