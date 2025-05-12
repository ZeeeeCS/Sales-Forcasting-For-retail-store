import streamlit as st
import pandas as pd
import os
import sys # To add to sys.path if your model script is in a different directory

# --- IMPORTANT: Add the directory of your model script to Python's path ---
# This assumes 'sales_forecasting_model_with_logging.py' is in the same directory as 'streamlit_app.py'
# or in a subdirectory. Adjust as needed if your structure is different.
# If they are in the same directory, this might not be strictly necessary,
# but it's good practice if you plan to organize into modules.

# Assuming your structure is:
# Sales-Forcasting-For-retail-store/
# |--- streamlit_app/
#      |--- streamlit_app.py  (This file)
#      |--- sales_forecasting_model_with_logging.py
# If so, they are in the same directory from Python's perspective when streamlit_app.py is run.

# If sales_forecasting_model_with_logging.py was, for example, one level up:
# SCRIPT_DIR = os.path.dirname(os.path.abspath(file))
# sys.path.append(os.path.dirname(SCRIPT_DIR))

# Let's assume sales_forecasting_model_with_logging.py is in the same directory for now
# and you'll rename it to something like 'forecasting_pipeline.py'
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

        # --- Call your existing pipeline function ---
        # Modify your run_forecasting_pipeline to return key results for Streamlit
        # For example, it could return a dictionary with metrics, paths to plot images,
        # and a DataFrame of predictions.

        # Let's assume your run_forecasting_pipeline is in forecasting_pipeline.py
        # And it's set up to use a unique experiment name per run or session
        experiment_name = f"Streamlit_UserRun_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}"
        
        with st.spinner("Running forecast models... This may take a few minutes."):
            # IMPORTANT: The run_forecasting_pipeline in your current script has ngrok startup
            # logic in its main block. That main block will NOT run when imported.
            # You need to ensure that the core modeling logic is within the
            # run_forecasting_pipeline function itself, and it doesn't try to start ngrok.
            # MLflow logging within the function is fine.
            
            # For now, we'll call the pipeline. It will log to MLflow locally.
            # The MLflow UI access is a separate concern from the Streamlit app's core function.
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

            # You would ideally have your pipeline return actual forecast DataFrames or plot objects
            # For example, if run_lstm_model_with_hyperparams returned predictions:
            # df_predictions = results_summary.get("lstm_hp_predictions_df")
            # if df_predictions is not None:
            #     st.subheader("LSTM (Hyperparameter) Forecast:")
            #     st.line_chart(df_predictions.set_index('ds')['yhat']) # Example
            #     st.dataframe(df_predictions)

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
