import streamlit as st
import pandas as pd
import os
import traceback

from sales_forecasting_model_with_logging import run_forecasting_pipeline,load_and_prepare,plot_forecast,plot_prophet_forecast



# --- Page Configuration ---
st.set_page_config(page_title="Sales Forecaster", layout="wide")

# --- Main App ---
st.title("🛍️ Advanced Sales Forecaster")
st.markdown("Upload your sales data to generate and compare forecasts from **SARIMA**, **Prophet**, and **LSTM** models.")

uploaded_file = st.file_uploader("Choose a CSV file (must contain 'Date' and 'Units Sold' columns)", type="csv")

if uploaded_file is not None:
    temp_file_path = ""
    try:
        # Save the uploaded file temporarily so our pipeline can read it from a path
        temp_dir = "temp_data"
        os.makedirs(temp_dir, exist_ok=True)
        temp_file_path = os.path.join(temp_dir, uploaded_file.name)
        with open(temp_file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        # Prepare the data once for plotting and pipeline
        base_df = load_and_prepare(temp_file_path)
        if base_df is None:
            st.error("Could not process the uploaded data. Please check the file format and column names ('Date', 'Units Sold').")
        else:
            experiment_name = f"StreamlitRun_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}"
            
            with st.spinner("⏳ Running forecasting pipeline... This may take a few minutes."):
                results_summary = run_forecasting_pipeline(
                    csv_path=temp_file_path,
                    experiment_name=experiment_name
                )

            st.header("📊 Forecasting Results Dashboard")

            if results_summary and not results_summary.get("error"):
                st.success("Forecasting pipeline completed successfully!")

                # --- Display Key Metrics ---
                st.subheader("Model Performance Comparison (MAPE %)")
                st.markdown("_(Lower is better)_")
                col1, col2, col3 = st.columns(3)
                col1.metric("SARIMA", f"{results_summary.get('sarima_mape', 0):.2f}%")
                col2.metric("Prophet", f"{results_summary.get('prophet_mape', 0):.2f}%")
                col3.metric("LSTM", f"{results_summary.get('lstm_mape', 0):.2f}%")
                
                st.markdown("---")

                # --- Create Tabs for each model's detailed results ---
                sarima_tab, prophet_tab, lstm_tab = st.tabs(["SARIMA Forecast", "Prophet Forecast", "LSTM Forecast"])

                with sarima_tab:
                    st.subheader("SARIMA Model Forecast")
                    sarima_preds = results_summary.get("sarima_predictions")
                    sarima_actuals = results_summary.get("sarima_actuals")
                    if sarima_preds is not None and sarima_actuals is not None:
                        fig = plot_forecast(base_df, sarima_preds, sarima_actuals, "SARIMA Forecast")
                        st.pyplot(fig, use_container_width=True)
                        with st.expander("View SARIMA Forecast Data"):
                            df_to_show = pd.DataFrame({'Date': sarima_actuals.index, 'Actual': sarima_actuals.values, 'Forecast': sarima_preds.values})
                            st.dataframe(df_to_show.round(2))
                    else:
                        st.warning("SARIMA model results are not available.")

                with prophet_tab:
                    st.subheader("Prophet Model Forecast")
                    prophet_fcst = results_summary.get("prophet_forecast_df")
                    if prophet_fcst is not None and not prophet_fcst.empty:
                        fig = plot_prophet_forecast(base_df, prophet_fcst, "Prophet Forecast with Features")
                        st.pyplot(fig, use_container_width=True)
                        with st.expander("View Prophet Forecast Data"):
                            st.dataframe(prophet_fcst[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].round(2))
                    else:
                        st.warning("Prophet model results are not available.")

                with lstm_tab:
                    st.subheader("LSTM Model Forecast")
                    lstm_preds = results_summary.get("lstm_predictions")
                    lstm_actuals = results_summary.get("lstm_actuals")
                    lstm_dates = results_summary.get("lstm_dates")
                    if lstm_preds is not None and lstm_actuals is not None and lstm_dates is not None:
                        # Create a temporary test series for the generic plot function
                        lstm_test_series = pd.Series(lstm_actuals, index=lstm_dates)
                        fig = plot_forecast(base_df, lstm_preds, lstm_test_series, "LSTM Forecast with Differencing")
                        st.pyplot(fig, use_container_width=True)
                        with st.expander("View LSTM Forecast Data"):
                            df_to_show = pd.DataFrame({'Date': lstm_dates, 'Actual': lstm_actuals.flatten(), 'Forecast': lstm_preds.flatten()})
                            st.dataframe(df_to_show.round(2))
                    else:
                        st.warning("LSTM model results are not available.")
                
                st.info(f"**MLflow Run ID for this session:** `{results_summary.get('mlflow_run_id', 'N/A')}`")

            elif results_summary and results_summary.get("error"):
                st.error(f"Forecasting pipeline failed: {results_summary['error']}")
                if results_summary.get("mlflow_run_id"):
                    st.info(f"MLflow Run ID (for debugging failed run): {results_summary['mlflow_run_id']}")
            else:
                st.error("An unknown error occurred during the forecasting process.")

    except Exception as e:
        st.error(f"A critical error occurred in the application: {e}")
        st.text(traceback.format_exc())

    finally:
        # Clean up the temporary file
        if os.path.exists(temp_file_path):
            try:
                os.remove(temp_file_path)
            except Exception as e_clean:
                st.warning(f"Could not remove temporary file {temp_file_path}: {e_clean}")

else:
    st.info("Awaiting CSV file upload to begin...")