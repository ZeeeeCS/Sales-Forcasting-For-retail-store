# app.py
import streamlit as st
import pandas as pd
import mlflow
import os
from sales_forecasting_model import run_forecasting_pipeline
import sqlite3
from pathlib import Path

# Configure MLflow to use local SQLite and file storage
MLFLOW_DIR = "./mlflow_local/"
Path(MLFLOW_DIR).mkdir(exist_ok=True)

mlflow.set_tracking_uri(f"sqlite:///{MLFLOW_DIR}/mlflow.db")
mlflow.set_registry_uri(f"file://{MLFLOW_DIR}/artifacts")

def main():
    st.title("📈 Local MLflow Sales Forecasting")
    st.markdown("""
    **Local MLflow Configuration:**  
    - SQLite backend store  
    - Local file artifact storage  
    - No cloud dependencies  
    """)

    # Sidebar controls
    with st.sidebar:
        st.header("⚙️ Settings")
        uploaded_file = st.file_uploader("Upload CSV data", type=["csv"])
        run_button = st.button("Run Forecasting")

    # Main content area
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        with st.expander("📊 Data Preview"):
            st.dataframe(df.head(3))

    if run_button and uploaded_file:
        with st.spinner("Running local MLflow pipeline..."):
            try:
                # Save temporary file
                temp_path = f"{MLFLOW_DIR}/temp_data.csv"
                uploaded_file.seek(0)
                with open(temp_path, "wb") as f:
                    f.write(uploaded_file.getvalue())

                # Start MLflow run
                with mlflow.start_run():
                    # Run pipeline
                    results = run_forecasting_pipeline(temp_path)
                    
                    # Log metrics
                    if results:
                        mlflow.log_metrics({
                            "prophet_mape": results['prophet_hyperparam_mape'],
                            "lstm_mape": results['lstm_hyperparam_mape']
                        })
                        
                        # Log artifacts
                        mlflow.log_artifact(temp_path)
                        
                        # Show results
                        st.success("Pipeline completed!")
                        st.json(results)

                        # Display local MLflow info
                        st.subheader("📂 Local MLflow Storage")
                        st.write(f"Database: `{MLFLOW_DIR}mlflow.db`")
                        st.write(f"Artifacts: `{MLFLOW_DIR}artifacts/`")

            except Exception as e:
                st.error(f"Error: {str(e)}")
            finally:
                if os.path.exists(temp_path):
                    os.remove(temp_path)

if __name__ == "__main__":
    main()