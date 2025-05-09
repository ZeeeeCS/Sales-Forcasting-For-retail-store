# app.py
import streamlit as st
import pandas as pd
from sales_forecasting_model_with_logging import run_forecasting_pipeline
import mlflow

# Configure MLflow tracking URI (adjust if needed)
mlflow.set_tracking_uri("http://localhost:5000")

def main():
    st.title("🔮 Sales Forecasting Dashboard")
    st.markdown("""
    This app runs sales forecasting using both Prophet and LSTM models, 
    tracks experiments with MLflow, and monitors for data drift.
    """)

    # Sidebar controls
    with st.sidebar:
        st.header("⚙️ Settings")
        uploaded_file = st.file_uploader("Upload CSV data", type=["csv"])
        drift_threshold = st.number_input("Drift Threshold (MAPE %)", 
                                        min_value=5.0, max_value=100.0, 
                                        value=20.0, step=1.0)
        recent_checks = st.number_input("Persistent Drift Checks", 
                                      min_value=3, max_value=10, 
                                      value=5, step=1)
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
            try:
                # Save uploaded file temporarily
                temp_path = f"./temp_{uploaded_file.name}"
                with open(temp_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                
                # Run pipeline
                results = run_forecasting_pipeline(temp_path)
                
                if results:
                    st.success("✅ Pipeline completed successfully!")
                    st.subheader("📈 Results Summary")

                    # Create columns for metrics display
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.metric("Prophet (Original) MAPE", 
                                f"{results['prophet_original_mape']:.2f}%")
                        st.metric("LSTM (Original) MAPE", 
                                f"{results['lstm_original_mape']:.2f}%")
                    
                    with col2:
                        st.metric("Prophet (Tuned) MAPE", 
                                f"{results['prophet_hyperparam_mape']:.2f}%")
                        st.metric("LSTM (Tuned) MAPE", 
                                f"{results['lstm_hyperparam_mape']:.2f}%")

                    # Drift detection section
                    st.subheader("🚨 Drift Detection")
                    drift_col1, drift_col2 = st.columns(2)
                    
                    with drift_col1:
                        st.write("**Original Models**")
                        st.markdown(f"""
                        - Immediate Drift: {'⚠️ Detected' if results['drift_original_lstm'] else '✅ Normal'}
                        - Persistent Drift: {'🔴 Detected' if results['persistent_drift_original_lstm'] else '🟢 Normal'}
                        """)
                    
                    with drift_col2:
                        st.write("**Tuned Models**")
                        st.markdown(f"""
                        - Immediate Drift: {'⚠️ Detected' if results['drift_detected_on_hp_lstm'] else '✅ Normal'}
                        - Persistent Drift: {'🔴 Detected' if results['persistent_drift_on_hp_lstm'] else '🟢 Normal'}
                        """)

                    # MLflow info
                    st.subheader("🔍 MLflow Tracking")
                    st.write(f"Run ID: `{results['mlflow_run_id']}`")
                    st.markdown("[Open MLflow UI](http://localhost:5000) to see detailed metrics and models")

            except Exception as e:
                st.error(f"❌ Pipeline failed: {e}")
            finally:
                # Clean up temporary file
                import os
                if os.path.exists(temp_path):
                    os.remove(temp_path)

    elif run_button and not uploaded_file:
        st.warning("⚠️ Please upload a CSV file first!")

if __name__ == "__main__":
    main()