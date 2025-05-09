# app.py
import streamlit as st
import pandas as pd
import mlflow
import os
import subprocess
import signal
from pyngrok import ngrok, conf
from time import sleep
from sales_forecasting_model_with_logging import run_forecasting_pipeline

# Configuration
MLFLOW_DIR = os.path.abspath("./mlruns")
MLFLOW_PORT = 8080
NGROK_AUTH_TOKEN = "2wsCDg9OuRuTH6byPWcr3berIkS_bjXjwzFDutiN3Fvxarm1"  # Replace with your actual token

# Setup directories
os.makedirs(MLFLOW_DIR, exist_ok=True)

# Configure MLflow
mlflow.set_tracking_uri(f"file://{MLFLOW_DIR}")

def start_mlflow():
    """Start MLflow UI as background process"""
    try:
        return subprocess.Popen(
            ["mlflow", "ui", "--port", str(MLFLOW_PORT), "--backend-store-uri", MLFLOW_DIR],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            preexec_fn=os.setsid
        )
    except Exception as e:
        st.error(f"Failed to start MLflow: {str(e)}")
        return None

def start_ngrok_tunnel(port):
    """Start ngrok tunnel with error handling"""
    try:
        conf.get_default().config_path = "./ngrok.yml"
        conf.get_default().auth_token = NGROK_AUTH_TOKEN
        return ngrok.connect(port, "http")
    except Exception as e:
        st.error(f"Failed to start ngrok tunnel: {str(e)}")
        return None

def cleanup():
    """Clean up background processes"""
    if "mlflow_process" in st.session_state and st.session_state.mlflow_process:
        os.killpg(os.getpgid(st.session_state.mlflow_process.pid), signal.SIGTERM)
    if "ngrok_tunnel" in st.session_state and st.session_state.ngrok_tunnel:
        ngrok.kill()

def main():
    st.title("🔮 Sales Forecasting Dashboard")
    
    # Initialize services
    if "services_initialized" not in st.session_state:
        st.session_state.mlflow_process = start_mlflow()
        sleep(5)  # Give MLflow time to start
        
        if st.session_state.mlflow_process:
            st.session_state.ngrok_tunnel = start_ngrok_tunnel(MLFLOW_PORT)
            if st.session_state.ngrok_tunnel:
                public_url = st.session_state.ngrok_tunnel.public_url
                st.markdown(f"""
                **MLflow UI**: [Open Tracking Dashboard]({public_url})
                """)
            else:
                st.warning("Could not establish ngrok tunnel")
        else:
            st.warning("MLflow tracking disabled - results not logged")
        
        st.session_state.services_initialized = True

    # Sidebar controls
    with st.sidebar:
        st.header("⚙️ Settings")
        uploaded_file = st.file_uploader("Upload CSV data", type=["csv"])
        run_button = st.button("Run Forecasting Pipeline")

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
                
                # Run pipeline
                with mlflow.start_run():
                    results = run_forecasting_pipeline(temp_path)
                
                if results:
                    st.success("✅ Pipeline completed successfully!")
                    st.subheader("📈 Results Summary")

                    # Metrics display
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Prophet (Original) MAPE", f"{results.get('prophet_original_mape', 0):.2f}%")
                        st.metric("LSTM (Original) MAPE", f"{results.get('lstm_original_mape', 0):.2f}%")
                    with col2:
                        st.metric("Prophet (Tuned) MAPE", f"{results.get('prophet_hyperparam_mape', 0):.2f}%")
                        st.metric("LSTM (Tuned) MAPE", f"{results.get('lstm_hyperparam_mape', 0):.2f}%")

                    # Drift detection
                    st.subheader("🚨 Drift Detection")
                    cols = st.columns(2)
                    with cols[0]:
                        st.markdown("**Original Models**")
                        st.write(f"Immediate Drift: {'⚠️' if results.get('drift_original_lstm') else '✅'}")
                        st.write(f"Persistent Drift: {'🔴' if results.get('persistent_drift_original_lstm') else '🟢'}")
                    with cols[1]:
                        st.markdown("**Tuned Models**")
                        st.write(f"Immediate Drift: {'⚠️' if results.get('drift_detected_on_hp_lstm') else '✅'}")
                        st.write(f"Persistent Drift: {'🔴' if results.get('persistent_drift_on_hp_lstm') else '🟢'}")

            except Exception as e:
                st.error(f"❌ Pipeline failed: {e}")
            finally:
                if temp_path and os.path.exists(temp_path):
                    os.remove(temp_path)

    elif run_button and not uploaded_file:
        st.warning("⚠️ Please upload a CSV file first!")

if __name__ == "__main__":
    try:
        main()
    finally:
        cleanup()