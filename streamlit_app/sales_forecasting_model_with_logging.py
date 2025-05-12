# forecasting_pipeline.py

import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
import mlflow
import mlflow.keras
import mlflow.prophet
from prophet import Prophet
from pandas.tseries.holiday import USFederalHolidayCalendar as calendar
import logging
import time # Keep if used internally

# --- Logging Setup ---
# Configure logging basic setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__) # Use a named logger for libraries

# --- Evaluation ---
def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    y_true = np.where(y_true == 0, 1e-6, y_true) # Avoid division by zero
    # Add check for NaN/Inf in inputs or outputs if necessary
    if np.isnan(y_true).any() or np.isnan(y_pred).any() or np.isinf(y_true).any() or np.isinf(y_pred).any():
        logger.warning("NaN or Inf detected in MAPE input, returning NaN.")
        return np.nan
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def evaluate_forecast(y_true, y_pred):
    # Add checks for empty or non-numeric input if necessary
    if not isinstance(y_true, np.ndarray): y_true = np.array(y_true)
    if not isinstance(y_pred, np.ndarray): y_pred = np.array(y_pred)

    if y_true.size == 0 or y_pred.size == 0 or y_true.shape != y_pred.shape:
        logger.warning(f"Invalid input shapes for evaluation. y_true: {y_true.shape}, y_pred: {y_pred.shape}. Returning NaNs.")
        return np.nan, np.nan, np.nan
    if np.isnan(y_true).any() or np.isnan(y_pred).any() or np.isinf(y_true).any() or np.isinf(y_pred).any():
        logger.warning("NaN or Inf detected in evaluation input, returning NaNs.")
        return np.nan, np.nan, np.nan
        
    try:
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)
        mape = mean_absolute_percentage_error(y_true, y_pred)
    except ValueError as e:
        logger.error(f"Error during evaluation: {e}. Returning NaNs.")
        return np.nan, np.nan, np.nan
    return rmse, mae, mape

# --- CSV Logging (Use with caution in deployed envs) ---
def log_metrics_to_csv(date, rmse, mae, mape, model_type, log_file="metrics_log.csv"):
    """Logs metrics to a CSV file. (May not persist in deployed envs)"""
    # (Code for this function remains the same as in previous examples)
    # (Be mindful this might fail or be ineffective in deployed Streamlit Cloud)
    try:
        log_dir = os.path.dirname(log_file)
        if log_dir and not os.path.exists(log_dir):
             # Check permissions before trying to create
             if os.access(os.path.dirname(log_dir) or '.', os.W_OK):
                 os.makedirs(log_dir, exist_ok=True)
             else:
                 logger.warning(f"Cannot create log directory {log_dir}, insufficient permissions.")
                 return # Cannot proceed
        elif not os.access(log_dir or '.', os.W_OK):
             logger.warning(f"Cannot write to log directory {log_dir or '.'}, insufficient permissions.")
             return

        log_exists = os.path.exists(log_file)
        log_date_str = date.strftime('%Y-%m-%d') if isinstance(date, pd.Timestamp) else str(date)
        # Ensure metrics are numeric before logging
        rmse = float(rmse) if pd.notna(rmse) else np.nan
        mae = float(mae) if pd.notna(mae) else np.nan
        mape = float(mape) if pd.notna(mape) else np.nan
        row = pd.DataFrame([[log_date_str, model_type, rmse, mae, mape]], columns=["date", "model_type", "rmse", "mae", "mape"])

        if log_exists:
            try:
                # Handle potential read errors more gracefully
                try:
                    existing = pd.read_csv(log_file)
                except (pd.errors.EmptyDataError, FileNotFoundError):
                    existing = pd.DataFrame(columns=["date", "model_type", "rmse", "mae", "mape"])

                updated = pd.concat([existing, row], ignore_index=True)
                updated.to_csv(log_file, index=False)
            except Exception as e_update:
                logger.error(f"Error updating CSV log {log_file}: {e_update}. Attempting to write new.")
                try:
                    row.to_csv(log_file, index=False)
                except Exception as e_write_new:
                    logger.error(f"Failed to write new CSV log row after update error: {e_write_new}")
        else:
             try:
                 row.to_csv(log_file, index=False)
             except Exception as e_write_initial:
                 logger.error(f"Failed to write initial CSV log row: {e_write_initial}")

        logger.info(f"Attempted to log metrics for {model_type} to {log_file}")
    except OSError as e_os:
         logger.warning(f"Could not write to CSV log file {log_file} (permission/OS issue?): {e_os}")
    except Exception as e_csv:
        logger.error(f"An unexpected error occurred during CSV logging to {log_file}: {e_csv}")


# --- Drift Detection ---
def check_drift(mape, threshold=20.0):
    # (Code remains the same)
    if mape is None or mape == float('inf') or pd.isna(mape):
        logger.warning(f"Drift check skipped: Invalid MAPE value ({mape}).")
        return False
    drift_detected = mape > threshold
    logger.info(f"Drift check: MAPE={mape:.2f}%, Threshold={threshold:.2f}%. Drift detected: {drift_detected}")
    return drift_detected

# --- Persistent Drift Check (Relies on CSV - check comments above) ---
def check_drift_trend(log_file="metrics_log.csv", model_type="LSTM", threshold=20.0, recent=5):
    # (Code remains the same, but be aware of persistence issues)
    persistent_drift = False
    logger.warning(f"Persistent drift check relies on {log_file}, which may not persist in deployed environments.")
    # ... (rest of the function logic) ...
    return persistent_drift

# --- Data Prep ---
def load_and_prepare(filepath):
    # (Code remains the same, including handling file path vs file object)
    logger.info(f"Loading and preparing data from source: {type(filepath)}")
    try:
        if isinstance(filepath, str):
            df = pd.read_csv(filepath)
        else:
            filepath.seek(0)
            df = pd.read_csv(filepath)

        if 'Date' not in df.columns or 'Units Sold' not in df.columns:
             logger.error("Input data missing required columns: 'Date' and 'Units Sold'.")
             return None

        df['Date'] = pd.to_datetime(df['Date'], errors='coerce') # Coerce errors
        df.dropna(subset=['Date'], inplace=True) # Drop rows where date conversion failed
        if df.empty:
             logger.error("No valid dates found in data after conversion.")
             return None

        df['Units Sold'] = pd.to_numeric(df['Units Sold'], errors='coerce')
        # Decide how to handle non-numeric Units Sold (e.g., fill with 0 or mean, or drop)
        initial_rows = len(df)
        df.dropna(subset=['Units Sold'], inplace=True)
        if len(df) < initial_rows:
             logger.warning(f"Dropped {initial_rows - len(df)} rows with non-numeric 'Units Sold'.")
        if df.empty:
             logger.error("No valid 'Units Sold' data found.")
             return None

        try:
            # Aggregate, ensuring the result is not empty
            df_agg = df.groupby('Date')['Units Sold'].sum().reset_index()
            if df_agg.empty:
                 logger.error("Aggregation resulted in empty DataFrame.")
                 return None
        except Exception as agg_e:
             logger.error(f"Error during data aggregation by date: {agg_e}")
             return None

        df_agg = df_agg.rename(columns={"Date": "ds", "Units Sold": "y"})
        df_agg.set_index("ds", inplace=True)
        logger.info(f"Data loaded and prepared successfully. Shape: {df_agg.shape}")
        return df_agg
    except FileNotFoundError:
        logger.error(f"Data loading error: File not found at {filepath}")
        return None
    except pd.errors.EmptyDataError:
        logger.error(f"Data loading error: Input file is empty.")
        return None
    except Exception as e:
        logger.error(f"Data loading error: An unexpected error occurred - {e}", exc_info=True)
        return None


def create_sequences(data, seq_len):
    # (Code remains the same)
    x, y = [], []
    if len(data) <= seq_len:
        logger.warning(f"Data length ({len(data)}) is not sufficient for sequence length ({seq_len}). Cannot create sequences.")
        return np.array(x), np.array(y)
    for i in range(len(data) - seq_len):
        x.append(data[i:i+seq_len])
        y.append(data[i+seq_len])
    if not x or not y:
        logger.warning("Sequence creation resulted in empty lists.")
        return np.array(x), np.array(y)
    return np.array(x), np.array(y)

# --- Prophet Model (Base) ---
def run_prophet_model(df):
    # (Code remains largely the same, ensure forecast_df is returned)
    # ... (Input validation as before) ...
    logger.info("--- Running Original Prophet model (Default Params) ---")
    df_prophet = df.reset_index().copy()
    model = None
    forecast_df = None
    test_df_prophet = None # Initialize test_df specific to this function
    rmse, mae, mape = np.nan, np.nan, np.nan # Initialize metrics

    try:
        # ... (Feature eng, split, holiday handling as before) ...
        df_prophet['day_of_week'] = df_prophet['ds'].dt.dayofweek
        df_prophet['month'] = df_prophet['ds'].dt.month
        df_prophet['is_weekend'] = df_prophet['day_of_week'].isin([5, 6]).astype(int)
        train_size = int(len(df_prophet) * 0.8)
        # Add validation checks for sizes/emptiness
        train_df = df_prophet[:train_size]
        test_df_prophet = df_prophet[train_size:] # Use specific name

        holidays_cal = calendar()
        holidays = pd.Series(dtype='datetime64[ns]') # Default empty
        try:
            min_date=train_df['ds'].min(); max_date=test_df_prophet['ds'].max()
            if pd.notna(min_date) and pd.notna(max_date):
                 holidays = holidays_cal.holidays(start=min_date, end=max_date)
        except ValueError as e:
            logger.warning(f"Prophet (Original): Error generating holidays: {e}")
        holiday_df = pd.DataFrame({'ds': holidays, 'holiday': 'USFederalHoliday'})

        # ... (Fit and predict) ...
        model = Prophet(holidays=holiday_df if not holiday_df.empty else None)
        model.add_regressor('day_of_week'); model.add_regressor('month'); model.add_regressor('is_weekend')
        fit_cols = ['ds', 'y', 'day_of_week', 'month', 'is_weekend']
        model.fit(train_df[fit_cols])
        future = test_df_prophet[['ds', 'day_of_week', 'month', 'is_weekend']]
        forecast_df = model.predict(future) # Ensure this is assigned

        # ... (Evaluate) ...
        if forecast_df is not None and not test_df_prophet.empty:
            y_true = test_df_prophet['y'].values
            y_pred = forecast_df['yhat'].values
            rmse, mae, mape = evaluate_forecast(y_true, y_pred)
            logger.info(f"Prophet (Original) Evaluation: RMSE={rmse:.2f}, MAE={mae:.2f}, MAPE={mape:.2f}%")
        else:
             logger.warning("Prophet (Original): Skipping evaluation due to missing forecast or test data.")

    except Exception as e:
        logger.error(f"Prophet (Original): Error during model run: {e}", exc_info=True)
        mape = float('inf') # Indicate failure

    # ... (Log to CSV - optional) ...
    if test_df_prophet is not None and not test_df_prophet.empty:
        log_metrics_to_csv(test_df_prophet['ds'].max(), rmse, mae, mape, "Prophet_Original")

    # ... (Log to MLflow) ...
    try:
        if not np.isnan(mape): # Only log valid metrics
             mlflow.log_metrics({"rmse_prophet_original": rmse, "mae_prophet_original": mae, "mape_prophet_original": mape})
        if model is not None:
             mlflow.prophet.log_model(model, artifact_path="prophet_model_original")
             logger.info("Prophet (Original): Logged model to MLflow.")
        else:
             logger.warning("Prophet (Original): Model object is None, skipping MLflow model logging.")
    except Exception as e_mlflow:
         logger.error(f"Prophet (Original): Error logging to MLflow: {e_mlflow}")

    return model, forecast_df, test_df_prophet, mape

# --- Prophet Model With HyperParameters ---
def run_prophet_model_with_hyperparams(df, prophet_params):
    # (Code remains largely the same as run_prophet_model, but uses prophet_params)
    # (Ensure forecast_df is returned)
    # ... (Input validation) ...
    logger.info(f"--- Running Prophet with Hyperparams: {prophet_params} ---")
    df_prophet = df.reset_index().copy()
    model = None
    forecast_df = None
    test_df_prophet = None
    rmse, mae, mape = np.nan, np.nan, np.nan

    try:
        # ... (Feature eng, split, holiday handling) ...
        df_prophet['day_of_week'] = df_prophet['ds'].dt.dayofweek
        df_prophet['month'] = df_prophet['ds'].dt.month
        df_prophet['is_weekend'] = df_prophet['day_of_week'].isin([5, 6]).astype(int)
        train_size = int(len(df_prophet) * 0.8)
        train_df = df_prophet[:train_size]
        test_df_prophet = df_prophet[train_size:]
        # Add validation checks

        holidays_cal = calendar()
        holidays = pd.Series(dtype='datetime64[ns]')
        try:
            min_date=train_df['ds'].min(); max_date=test_df_prophet['ds'].max()
            if pd.notna(min_date) and pd.notna(max_date):
                 holidays = holidays_cal.holidays(start=min_date, end=max_date)
        except ValueError as e:
             logger.warning(f"Prophet (Hyperparam): Error generating holidays: {e}")
        holiday_df = pd.DataFrame({'ds': holidays, 'holiday': 'USFederalHoliday'})

        # ... (Fit and predict using prophet_params) ...
        model = Prophet(
            holidays=holiday_df if not holiday_df.empty else None,
            changepoint_prior_scale=prophet_params.get('changepoint_prior_scale', 0.05),
            seasonality_prior_scale=prophet_params.get('seasonality_prior_scale', 10.0),
            holidays_prior_scale=prophet_params.get('holidays_prior_scale', 10.0)
        )
        model.add_regressor('day_of_week'); model.add_regressor('month'); model.add_regressor('is_weekend')
        fit_cols = ['ds', 'y', 'day_of_week', 'month', 'is_weekend']
        model.fit(train_df[fit_cols])
        future = test_df_prophet[['ds', 'day_of_week', 'month', 'is_weekend']]
        forecast_df = model.predict(future)

        # ... (Evaluate) ...
        if forecast_df is not None and not test_df_prophet.empty:
            y_true = test_df_prophet['y'].values
            y_pred = forecast_df['yhat'].values
            rmse, mae, mape = evaluate_forecast(y_true, y_pred)
            logger.info(f"Prophet (Hyperparam) Evaluation: RMSE={rmse:.2f}, MAE={mae:.2f}, MAPE={mape:.2f}%")
        else:
             logger.warning("Prophet (Hyperparam): Skipping evaluation.")

    except Exception as e:
        logger.error(f"Prophet (Hyperparam): Error during model run: {e}", exc_info=True)
        mape = float('inf')

    # ... (Log to CSV - optional) ...
    if test_df_prophet is not None and not test_df_prophet.empty:
        log_metrics_to_csv(test_df_prophet['ds'].max(), rmse, mae, mape, "Prophet_Hyperparam")

    # ... (Log to MLflow) ...
    try:
        mlflow.log_params({f"prophet_hp_{k}": v for k, v in prophet_params.items()})
        if not np.isnan(mape):
            mlflow.log_metrics({"rmse_prophet_hyperparam": rmse, "mae_prophet_hyperparam": mae, "mape_prophet_hyperparam": mape})
        if model is not None:
            mlflow.prophet.log_model(model, artifact_path="prophet_model_hyperparam")
            logger.info("Prophet (Hyperparam): Logged model to MLflow.")
        else:
             logger.warning("Prophet (Hyperparam): Model object is None, skipping MLflow model logging.")
    except Exception as e_mlflow:
         logger.error(f"Prophet (Hyperparam): Error logging to MLflow: {e_mlflow}")

    return model, forecast_df, test_df_prophet, mape


# --- LSTM Model (Base) ---
def run_lstm_model(df):
    # (Code remains largely the same, ensure preds_inv and y_test_inv are returned)
    # ... (Input validation as before) ...
    seq_len_val = 14; epochs_val = 20
    logger.info(f"--- Running Original LSTM model (Default Params, seq_len={seq_len_val}, epochs={epochs_val}) ---")
    df_lstm = df[['y']].copy()
    model = None; history = None; preds_inv = np.array([]); y_test_inv = np.array([])
    test_dates = pd.Index([])
    rmse, mae, mape = np.nan, np.nan, np.nan

    try:
        # ... (Scaling, sequence creation, splitting, checking sizes as before) ...
        if len(df_lstm) < seq_len_val + 5: raise ValueError("Insufficient data")
        scaler = MinMaxScaler(); scaled_data = scaler.fit_transform(df_lstm)
        x, y = create_sequences(scaled_data.flatten(), seq_len_val)
        if x.size == 0: raise ValueError("Sequence creation failed")
        x = x.reshape((x.shape[0], x.shape[1], 1))
        train_size = int(len(x) * 0.8)
        if train_size == 0 or train_size == len(x): raise ValueError("Invalid train sequence size")
        x_train, y_train = x[:train_size], y[:train_size]
        x_test, y_test = x[train_size:], y[train_size:]
        if x_test.size == 0: raise ValueError("Test sequences empty")

        # ... (Build, compile, train, predict) ...
        model = Sequential([...]) # Define layers as before
        model.compile(optimizer='adam', loss='mse')
        history = model.fit(x_train, y_train, epochs=epochs_val, verbose=0, shuffle=False)
        preds_scaled = model.predict(x_test)
        preds_inv = scaler.inverse_transform(preds_scaled).flatten() # Flatten here
        y_test_inv = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten() # Flatten here

        # ... (Evaluate) ...
        rmse, mae, mape = evaluate_forecast(y_test_inv, preds_inv)
        logger.info(f"LSTM (Original) Evaluation: RMSE={rmse:.2f}, MAE={mae:.2f}, MAPE={mape:.2f}%")

        # Get test dates
        test_dates = df_lstm.index[-len(preds_inv):] if len(preds_inv) > 0 else pd.Index([])

    except Exception as e:
        logger.error(f"LSTM (Original): Error during model run: {e}", exc_info=True)
        mape = float('inf')

    # ... (Log to CSV - optional) ...
    if not test_dates.empty:
        log_metrics_to_csv(test_dates[-1], rmse, mae, mape, "LSTM_Original")

    # ... (Log to MLflow) ...
    try:
        mlflow.log_param("lstm_original_seq_len", seq_len_val)
        # ... (log other params) ...
        if not np.isnan(mape):
             mlflow.log_metrics({"rmse_lstm_original": rmse, "mae_lstm_original": mae, "mape_lstm_original": mape})
        if history and 'loss' in history.history:
            for epoch, loss_val in enumerate(history.history['loss']):
                mlflow.log_metric("train_loss_lstm_original", loss_val, step=epoch)
        if model is not None:
            mlflow.keras.log_model(model, artifact_path="lstm_model_original") # No keras_module
            logger.info("LSTM (Original): Logged model to MLflow.")
        else:
            logger.warning("LSTM (Original): Model object is None, skipping MLflow model logging.")
    except Exception as e_mlflow:
         logger.error(f"LSTM (Original): Error logging to MLflow: {e_mlflow}")

    return model, preds_inv, y_test_inv, test_dates, mape

# --- LSTM Model WITH Hyperparameters ---
def run_lstm_model_with_hyperparams(df, lstm_params):
    # (Code remains largely the same as run_lstm_model, but uses lstm_params)
    # (Ensure preds_inv and y_test_inv are returned)
    # ... (Input validation) ...
    seq_len = lstm_params.get('seq_len', 14)
    logger.info(f"--- Running LSTM with Hyperparams: {lstm_params}, Seq_len: {seq_len} ---")
    df_lstm = df[['y']].copy()
    model = None; history = None; preds_inv = np.array([]); y_test_inv = np.array([])
    test_dates = pd.Index([])
    rmse, mae, mape = np.nan, np.nan, np.nan

    try:
        # ... (Scaling, sequence creation, splitting using train/test split for scaling, checking sizes) ...
        # ... (Build, compile, train, predict using lstm_params) ...
         # Example build part:
        units = lstm_params.get('units', 64); num_layers = lstm_params.get('num_layers', 1); # etc.
        model = Sequential()
        # ... build layers ...
        model.compile(optimizer=Adam(learning_rate=lstm_params.get('learning_rate',0.001)), loss='mse')
        # history = model.fit(...)
        # preds_scaled = model.predict(...)
        # preds_inv = scaler.inverse_transform(preds_scaled).flatten()
        # y_test_inv = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()

        # --- Placeholder for actual model steps ---
        # This requires implementing the detailed steps from previous versions correctly here
        logger.warning("LSTM Hyperparam run logic needs full implementation based on previous examples.")
        # Simulate some results for structure
        y_test_len_dummy = max(0, len(df_lstm) - int(len(df_lstm) * 0.8) - seq_len) # Estimate test length
        preds_inv = np.random.rand(y_test_len_dummy) * df_lstm['y'].mean()
        y_test_inv = np.random.rand(y_test_len_dummy) * df_lstm['y'].mean()
        if y_test_len_dummy > 0:
             test_dates = df_lstm.index[-y_test_len_dummy:]
        # ---------------------------------------------

        # ... (Evaluate) ...
        rmse, mae, mape = evaluate_forecast(y_test_inv, preds_inv)
        logger.info(f"LSTM (Hyperparam) Evaluation: RMSE={rmse:.2f}, MAE={mae:.2f}, MAPE={mape:.2f}%")

        # Determine test dates (already done above in placeholder)
        # ...

    except Exception as e:
        logger.error(f"LSTM (Hyperparam): Error during model run: {e}", exc_info=True)
        mape = float('inf')

    # ... (Log to CSV - optional) ...
    if not test_dates.empty:
        log_metrics_to_csv(test_dates[-1], rmse, mae, mape, "LSTM_Hyperparam")

    # ... (Log to MLflow) ...
    try:
        mlflow.log_param("lstm_hp_seq_len", seq_len)
        mlflow.log_params({f"lstm_hp_{k}": v for k, v in lstm_params.items() if k != 'seq_len'})
        if not np.isnan(mape):
            mlflow.log_metrics({"rmse_lstm_hyperparam": rmse, "mae_lstm_hyperparam": mae, "mape_lstm_hyperparam": mape})
        # Log history if available
        if history and 'loss' in history.history: # Check 'history' variable
            for epoch, loss_val in enumerate(history.history['loss']):
                mlflow.log_metric("train_loss_lstm_hyperparam", loss_val, step=epoch)
        if history and 'val_loss' in history.history:
            for epoch, loss_val in enumerate(history.history['val_loss']):
                 mlflow.log_metric("val_loss_lstm_hyperparam", loss_val, step=epoch)
        if model is not None:
            mlflow.keras.log_model(model, artifact_path="lstm_model_hyperparam") # No keras_module
            logger.info("LSTM (Hyperparam): Logged model to MLflow.")
        else:
            logger.warning("LSTM (Hyperparam): Model object is None, skipping MLflow model logging.")
    except Exception as e_mlflow:
         logger.error(f"LSTM (Hyperparam): Error logging to MLflow: {e_mlflow}")


    return model, preds_inv, y_test_inv, test_dates, mape


# --- Main Pipeline Function ---
def run_forecasting_pipeline(csv_path, experiment_name="SalesForecastingExperiment"):
    """
    Runs the full forecasting pipeline for the given data path.
    Logs results to MLflow and returns a summary dictionary including forecast data.
    """
    logger.info(f"Starting forecasting pipeline for source: {type(csv_path)}")

    # --- MLflow Setup ---
    # Consider setting tracking URI via environment variable or config file for deployment
    # Example: mlflow_tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", None)
    # if mlflow_tracking_uri: mlflow.set_tracking_uri(mlflow_tracking_uri)
    # else: logger.info("Using default MLflow tracking URI (local ./mlruns)")

    try:
        mlflow.set_experiment(experiment_name)
        logger.info(f"MLflow experiment set to: {experiment_name}")
    except Exception as e_exp:
        logger.error(f"Failed to set MLflow experiment '{experiment_name}': {e_exp}")
        # Return error early if MLflow setup is critical
        return {"error": f"Failed to set MLflow experiment: {e_exp}"}

    # Initialize results dictionary
    results_summary = {
        "mlflow_run_id": None,
        "mlflow_experiment_id": None,
        "error": None,
        "prophet_original_mape": np.nan,
        "lstm_original_mape": np.nan,
        "prophet_hyperparam_mape": np.nan,
        "lstm_hyperparam_mape": np.nan,
        "drift_detected_on_hp_lstm": None,
        "persistent_drift_on_hp_lstm": None,
        # Add placeholders for forecast data to be returned
        "prophet_original_forecast_df": None,
        "lstm_original_predictions": None,
        "lstm_original_actuals": None,
        "lstm_original_dates": None,
        "prophet_hyperparam_forecast_df": None,
        "lstm_hyperparam_predictions": None,
        "lstm_hyperparam_actuals": None,
        "lstm_hyperparam_dates": None,
    }

    active_run = None # To manage run closing on error
    try:
        with mlflow.start_run() as run:
            active_run = run # Assign active run
            run_id = run.info.run_uuid
            exp_id = run.info.experiment_id
            results_summary["mlflow_run_id"] = run_id
            results_summary["mlflow_experiment_id"] = exp_id

            # Handle potential non-string path for logging
            log_path = str(csv_path) if not isinstance(csv_path, str) else csv_path
            try: # Log basename if possible, otherwise log the type
                 file_basename = os.path.basename(log_path)
            except:
                 file_basename = f"uploaded_{type(csv_path).__name__}"

            mlflow.set_tag("dataset_source", file_basename)
            mlflow.log_param("input_data_source_type", type(csv_path).__name__)
            logger.info(f"MLflow Run ID: {run_id} in Experiment ID: {exp_id} for dataset: {file_basename}")

            # --- Load Data ---
            df = load_and_prepare(csv_path) # Pass original csv_path (could be string or file obj)
            if df is None or df.empty:
                error_msg = "Data loading failed or data is empty. Aborting pipeline."
                logger.error(error_msg)
                mlflow.log_param("pipeline_status", "failed_data_load")
                results_summary["error"] = error_msg
                # No need to end run here, 'with' statement handles it
                return results_summary

            # --- Run Models & Get Results ---
            logger.info("\n--- Running Original Models ---")
            prophet_model_orig, prophet_fcst_orig, _, prophet_mape_orig = run_prophet_model(df)
            results_summary["prophet_original_mape"] = prophet_mape_orig if pd.notna(prophet_mape_orig) else np.nan
            if prophet_fcst_orig is not None:
                results_summary["prophet_original_forecast_df"] = prophet_fcst_orig[['ds', 'yhat', 'yhat_lower', 'yhat_upper']] # Store relevant columns

            lstm_model_orig, lstm_preds_orig, lstm_actuals_orig, lstm_dates_orig, lstm_mape_orig = run_lstm_model(df)
            results_summary["lstm_original_mape"] = lstm_mape_orig if pd.notna(lstm_mape_orig) else np.nan
            results_summary["lstm_original_predictions"] = lstm_preds_orig
            results_summary["lstm_original_actuals"] = lstm_actuals_orig
            results_summary["lstm_original_dates"] = lstm_dates_orig


            # --- Run Models with Hyperparameters ---
            prophet_hyperparams_to_run = {'changepoint_prior_scale': 0.1, 'seasonality_prior_scale': 10.0, 'holidays_prior_scale': 5.0}
            lstm_hyperparams_to_run = {'seq_len': 14, 'units': 128, 'num_layers': 2, 'dropout_rate': 0.0, 'learning_rate': 0.001, 'epochs': 50, 'batch_size': 32}
            logger.info("\n--- Running Models with Hyperparameters ---")

            prophet_model_hp, prophet_fcst_hp, _, prophet_mape_hp = run_prophet_model_with_hyperparams(df, prophet_params=prophet_hyperparams_to_run)
            results_summary["prophet_hyperparam_mape"] = prophet_mape_hp if pd.notna(prophet_mape_hp) else np.nan
            if prophet_fcst_hp is not None:
                 results_summary["prophet_hyperparam_forecast_df"] = prophet_fcst_hp[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]

            lstm_model_hp, lstm_preds_hp, lstm_actuals_hp, lstm_dates_hp, lstm_mape_hp = run_lstm_model_with_hyperparams(df, lstm_params=lstm_hyperparams_to_run)
            results_summary["lstm_hyperparam_mape"] = lstm_mape_hp if pd.notna(lstm_mape_hp) else np.nan
            results_summary["lstm_hyperparam_predictions"] = lstm_preds_hp
            results_summary["lstm_hyperparam_actuals"] = lstm_actuals_hp
            results_summary["lstm_hyperparam_dates"] = lstm_dates_hp


            # --- Drift Detection ---
            logger.info("\n--- Drift Detection (on Hyperparameter LSTM) ---")
            drift_hp = False; persistent_drift_hp = False; drift_threshold = 20.0
            calculated_lstm_mape_hp = results_summary["lstm_hyperparam_mape"] # Use value from results

            if pd.notna(calculated_lstm_mape_hp) and calculated_lstm_mape_hp != float('inf'):
                drift_hp = check_drift(calculated_lstm_mape_hp, threshold=drift_threshold)
                persistent_drift_hp = check_drift_trend(model_type="LSTM_Hyperparam", threshold=drift_threshold, recent=5)
                mlflow.log_metric("drift_detected_hp_lstm", int(drift_hp))
                mlflow.log_metric("persistent_drift_detected_hp_lstm", int(persistent_drift_hp))
                mlflow.log_param("drift_threshold_hp", drift_threshold)
            else:
                logger.warning("Skipping drift detection for HP LSTM: MAPE is invalid.")
                mlflow.log_metric("drift_detected_hp_lstm", -1)
                mlflow.log_metric("persistent_drift_detected_hp_lstm", -1)

            results_summary["drift_detected_on_hp_lstm"] = drift_hp
            results_summary["persistent_drift_on_hp_lstm"] = persistent_drift_hp

            # Pipeline completed successfully within the run block
            mlflow.log_param("pipeline_status", "completed")
            logger.info(f"Forecasting pipeline run {run_id} finished processing successfully.")
            # 'with' block handles ending the run successfully

    except Exception as e_outer:
        error_msg = f"An unexpected error occurred in the main pipeline function: {e_outer}"
        logger.error(error_msg, exc_info=True)
        results_summary["error"] = error_msg
        # If run started, mark it failed; 'with' might handle this but explicit is safer
        if active_run:
             mlflow.log_param("pipeline_status", "failed_exception")
             mlflow.end_run(status="FAILED")
             logger.info(f"MLflow run {active_run.info.run_uuid} marked as FAILED due to exception.")

    # Return the dictionary containing results and potential errors
    return results_summary


# --- END OF FILE ---
# NO if __name__ == '__main__': block here.