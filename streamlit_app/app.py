# app.py
import streamlit as st
import pandas as pd
import os
from pathlib import Path
from sales_forecasting_model_with_logging import run_forecasting_pipeline

# Configure MLflow with SQLite backend
MLFLOW_DIR = 'D:\studying\DEPI\the final DEPI\Sales-Forcasting-For-retail-store\mlflow_local'
mlflow_enabled = True

try:
    import mlflow
    # Create MLflow directory structure
    Path(MLFLOW_DIR).mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(f"sqlite:///{MLFLOW_DIR}/mlflow.db")
    mlflow.set_registry_uri(f"file://{MLFLOW_DIR}/artifacts")
except ImportError:
    mlflow_enabled = False
except Exception as e:
    mlflow_enabled = False
    st.sidebar.error(f"MLflow initialization failed: {str(e)}")

def main():
    st.title("🔮 Sales Forecasting Dashboard")
    st.markdown("""
    This app runs sales forecasting using both Prophet and LSTM models, 
    tracks experiments with MLflow (when available), and monitors for data drift.
    """)

    # Sidebar controls
    with st.sidebar:
        st.header("⚙️ Settings")
        uploaded_file = st.file_uploader("Upload CSV data", type=["csv"])
        run_button = st.button("Run Forecasting Pipeline")

        if not mlflow_enabled:
            st.warning("MLflow tracking disabled - results not logged")

    # Main content area
    if uploaded_file is not None:
        try:
            df_preview = pd.read_csv(uploaded_file)
            st.subheader("📊 Data Preview")
            st.dataframe(df_preview.head(), use_container_width=True)
        except Exception as e:
            st.error(f"Error reading file: {e}")

    if run_button and uploaded_file is not None:
        with st.spinner("🚀 Running forecasting pipeline..."):
            temp_path = None
            try:
                # Save uploaded file temporarily
                temp_path = f"./temp_{uploaded_file.name}"
                with open(temp_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                
                # Handle MLflow runs if enabled
                if mlflow_enabled:
                    if mlflow.active_run():
                        mlflow.end_run()
                
                # Run pipeline
                results = run_forecasting_pipeline(temp_path) if mlflow_enabled else None
                
                if results:
                    st.success("✅ Pipeline completed successfully!")
                    st.subheader("📈 Results Summary")

                    # Create columns for metrics display
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.metric("Prophet (Original) MAPE", 
                                f"{results.get('prophet_original_mape', 0):.2f}%")
                        st.metric("LSTM (Original) MAPE", 
                                f"{results.get('lstm_original_mape', 0):.2f}%")
                    
                    with col2:
                        st.metric("Prophet (Tuned) MAPE", 
                                f"{results.get('prophet_hyperparam_mape', 0):.2f}%")
                        st.metric("LSTM (Tuned) MAPE", 
                                f"{results.get('lstm_hyperparam_mape', 0):.2f}%")

                    # Drift detection section
                    st.subheader("🚨 Drift Detection")
                    drift_col1, drift_col2 = st.columns(2)
                    
                    with drift_col1:
                        st.write("**Original Models**")
                        st.markdown(f"""
                        - Immediate Drift: {'⚠️ Detected' if results.get('drift_original_lstm', False) else '✅ Normal'}
                        - Persistent Drift: {'🔴 Detected' if results.get('persistent_drift_original_lstm', False) else '🟢 Normal'}
                        """)
                    
                    with drift_col2:
                        st.write("**Tuned Models**")
                        st.markdown(f"""
                        - Immediate Drift: {'⚠️ Detected' if results.get('drift_detected_on_hp_lstm', False) else '✅ Normal'}
                        - Persistent Drift: {'🔴 Detected' if results.get('persistent_drift_on_hp_lstm', False) else '🟢 Normal'}
                        """)

                    # MLflow info if enabled
                    if mlflow_enabled:
                        st.subheader("🔍 MLflow Tracking")
                        st.write(f"Local storage: `{MLFLOW_DIR}`")
                        st.markdown(f"""
                        - Database: `{MLFLOW_DIR}mlflow.db`
                        - Artifacts: `{MLFLOW_DIR}artifacts/`
                        """)

            except Exception as e:
                st.error(f"❌ Pipeline failed: {e}")
            finally:
                if temp_path and os.path.exists(temp_path):
                    os.remove(temp_path)

    elif run_button and not uploaded_file:
        st.warning("⚠️ Please upload a CSV file first!")

if __name__ == "__main__":
    main()