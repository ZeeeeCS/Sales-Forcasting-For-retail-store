# app.py
import streamlit as st
import pandas as pd
import mlflow
from sales_forecasting_model import run_forecasting_pipeline
import hashlib
import time
from threading import Thread
import os

# Configuration
MLFLOW_TRACKING_URI = "http://localhost:5000"
CACHE_DIR = "./model_cache/"
os.makedirs(CACHE_DIR, exist_ok=True)

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

def main():
    st.title("⚡ Optimized Sales Forecasting Dashboard")
    st.markdown("""
    Optimized version with model caching and parallel execution
    """)

    # Session state initialization
    if 'run_in_progress' not in st.session_state:
        st.session_state.run_in_progress = False
    if 'latest_results' not in st.session_state:
        st.session_state.latest_results = None

    # Sidebar controls
    with st.sidebar:
        st.header("⚙️ Settings")
        uploaded_file = st.file_uploader("Upload CSV data", type=["csv"])
        
        st.subheader("Model Selection")
        use_prophet = st.checkbox("Use Prophet", True)
        use_lstm = st.checkbox("Use LSTM", True)
        
        st.subheader("Performance Options")
        quick_mode = st.checkbox("Quick Mode (Reduced accuracy)", True)
        enable_caching = st.checkbox("Enable Model Caching", True)
        
        run_button = st.button("Run Pipeline")

    # Main content area
    if uploaded_file is not None:
        with st.expander("📊 Data Preview", expanded=False):
            try:
                df_preview = pd.read_csv(uploaded_file)
                st.dataframe(df_preview.head(), use_container_width=True)
                st.write(f"Total records: {len(df_preview)}")
            except Exception as e:
                st.error(f"Error reading file: {e}")

    # Pipeline execution logic
    def run_pipeline(file_path, use_prophet, use_lstm, quick_mode, enable_cache):
        try:
            start_time = time.time()
            
            # Generate data hash for caching
            with open(file_path, "rb") as f:
                data_hash = hashlib.md5(f.read()).hexdigest()
            
            # Load cached results if available
            cache_path = os.path.join(CACHE_DIR, f"{data_hash}.pkl")
            if enable_cache and os.path.exists(cache_path):
                st.session_state.latest_results = pd.read_pickle(cache_path)
                st.toast("Loaded from cache!")
                return

            # Configure pipeline parameters
            params = {
                'prophet_params': {
                    'changepoint_prior_scale': 0.1,
                    'seasonality_prior_scale': 10.0,
                    'holidays_prior_scale': 5.0
                },
                'lstm_params': {
                    'units': 128 if not quick_mode else 64,
                    'num_layers': 2 if not quick_mode else 1,
                    'epochs': 30 if not quick_mode else 15,
                    'batch_size': 32
                }
            }

            # Run pipeline
            results = run_forecasting_pipeline(file_path)
            
            # Cache results
            if enable_cache:
                pd.to_pickle(results, cache_path)
            
            st.session_state.latest_results = results
            st.toast(f"Pipeline completed in {time.time()-start_time:.1f}s")

        except Exception as e:
            st.error(f"Pipeline failed: {str(e)}")
        finally:
            st.session_state.run_in_progress = False
            if os.path.exists(file_path):
                os.remove(file_path)

    # Handle run button click
    if run_button and not st.session_state.run_in_progress:
        if uploaded_file is None:
            st.warning("Please upload a CSV file first!")
            return

        # Save uploaded file
        temp_path = f"./temp_{int(time.time())}_{uploaded_file.name}"
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        # Start pipeline in background thread
        st.session_state.run_in_progress = True
        thread = Thread(target=run_pipeline, args=(
            temp_path,
            use_prophet,
            use_lstm,
            quick_mode,
            enable_caching
        ))
        thread.start()

    # Show loading state
    if st.session_state.run_in_progress:
        with st.container():
            st.spinner("Optimized pipeline running in background...")
            progress_bar = st.progress(0)
            for i in range(100):
                time.sleep(0.1)
                progress_bar.progress(i + 1)
    
    # Display results when available
    if st.session_state.latest_results:
        st.subheader("📈 Latest Results")
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Prophet MAPE", 
                     f"{st.session_state.latest_results['prophet_hyperparam_mape']:.1f}%")
            st.metric("LSTM MAPE", 
                     f"{st.session_state.latest_results['lstm_hyperparam_mape']:.1f}%")
        
        with col2:
            st.metric("Drift Detected", 
                     "✅ Yes" if st.session_state.latest_results['drift_detected_on_hp_lstm'] else "❌ No")
            st.metric("Persistent Drift", 
                     "✅ Yes" if st.session_state.latest_results['persistent_drift_on_hp_lstm'] else "❌ No")
        
        st.markdown(f"""
        **MLflow Tracking:**  
        Run ID: `{st.session_state.latest_results['mlflow_run_id']}`  
        [Open MLflow UI]({MLFLOW_TRACKING_URI})
        """)

if __name__ == "__main__":
    main()