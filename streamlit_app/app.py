# app.py
import streamlit as st
import pandas as pd
import os
from pathlib import Path
from sales_forecasting_model import run_forecasting_pipeline

# Configure MLflow with SQLite backend
MLFLOW_DIR = "./mlflow_local/"
mlflow_enabled = True

try:
    import mlflow
    Path(MLFLOW_DIR).mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(f"sqlite:///{MLFLOW_DIR}/mlflow.db")
    mlflow.set_registry_uri(f"file://{MLFLOW_DIR}/artifacts")
except Exception as e:
    mlflow_enabled = False

def main():
    st.title("🔮 Sales Forecasting Dashboard")
    st.markdown("Sales forecasting with Prophet & LSTM models")
    
    with st.sidebar:
        uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
        run_button = st.button("Run Pipeline")
        
    if uploaded_file:
        try:
            df_preview = pd.read_csv(uploaded_file)
            st.dataframe(df_preview.head())
        except Exception as e:
            st.error(f"Error reading file: {e}")

    if run_button and uploaded_file:
        with st.spinner("Running pipeline..."):
            temp_path = f"./temp_{uploaded_file.name}"
            try:
                with open(temp_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                
                if mlflow_enabled and mlflow.active_run():
                    mlflow.end_run()
                
                results = run_forecasting_pipeline(temp_path) if mlflow_enabled else None
                
                if results:
                    st.success("Pipeline completed!")
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Prophet MAPE", f"{results['prophet_original_mape']:.2f}%")
                        st.metric("LSTM MAPE", f"{results['lstm_original_mape']:.2f}%")
                    with col2:
                        st.metric("Tuned Prophet", f"{results['prophet_hyperparam_mape']:.2f}%")
                        st.metric("Tuned LSTM", f"{results['lstm_hyperparam_mape']:.2f}%")
                    
                    if mlflow_enabled:
                        st.write(f"MLflow Run ID: {results['mlflow_run_id']}")
            finally:
                if os.path.exists(temp_path):
                    os.remove(temp_path)

if __name__ == "__main__":
    main()