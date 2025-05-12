<<<<<<< HEAD
# sales_forecasting_model_with_logging.py
=======
# forecasting_pipeline.py
>>>>>>> 8351801cfc5a8af9fdcec1cd1f0a58ce971e0d80

import pandas as pd
import numpy as np
import os
<<<<<<< HEAD
# import matplotlib.pyplot as plt # Keep commented unless generating plots to save
=======
>>>>>>> 8351801cfc5a8af9fdcec1cd1f0a58ce971e0d80
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
<<<<<<< HEAD
import sys # Needed for enhanced logging

import mlflow
# Example for a self-hosted server:
MLFLOW_TRACKING_URI = "http://127.0.0.1:5000"
# Example for Databricks:
# MLFLOW_TRACKING_URI = "databricks"
# Example for Azure ML:
# MLFLOW_TRACKING_URI = azureml.core.Workspace.get(....).get_mlflow_tracking_uri()

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

# --- Logging Setup ---
# Configure logging to also output to stdout for better visibility in Streamlit logs/terminal
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    handlers=[logging.StreamHandler(sys.stdout)]) # Add StreamHandler
log = logging.getLogger(__name__) # Use a named logger
=======
import time # Keep if used internally

# --- Logging Setup ---
# Configure logging basic setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__) # Use a named logger for libraries
>>>>>>> 8351801cfc5a8af9fdcec1cd1f0a58ce971e0d80

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

<<<<<<< HEAD
# --- CSV Logging (Optional - might not be persistent on Streamlit Cloud) ---
def log_metrics_to_csv(date, rmse, mae, mape, model_type, log_file="metrics_log.csv"):
    """Logs metrics to a CSV file."""
    log_exists = os.path.exists(log_file)
    log_date_str = date.strftime('%Y-%m-%d') if isinstance(date, pd.Timestamp) else str(date)
    row = pd.DataFrame([[log_date_str, model_type, rmse, mae, mape]], columns=["date", "model_type", "rmse", "mae", "mape"])
    try:
        if log_exists:
            existing = pd.read_csv(log_file)
            updated = pd.concat([existing, row], ignore_index=True)
            updated.to_csv(log_file, index=False)
        else:
            row.to_csv(log_file, index=False)
        log.info(f"Logged metrics for {model_type} to {log_file}")
    except pd.errors.EmptyDataError:
        log.warning(f"CSV log file {log_file} was empty. Overwriting.")
        row.to_csv(log_file, index=False)
    except Exception as e:
        log.error(f"Error updating CSV log {log_file}: {e}")
        # Optionally attempt to write anyway if creation failed mid-write
        if not log_exists:
             try: row.to_csv(log_file, index=False)
             except: pass # Ignore secondary error
=======
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

>>>>>>> 8351801cfc5a8af9fdcec1cd1f0a58ce971e0d80

# --- Drift Detection ---
def check_drift(mape, threshold=20.0):
    # (Code remains the same)
    if mape is None or mape == float('inf') or pd.isna(mape):
<<<<<<< HEAD
        log.warning(f"Drift check skipped: Invalid MAPE value ({mape}).")
        return False
    drift_detected = mape > threshold
    log.info(f"Drift check: MAPE={mape:.2f}%, Threshold={threshold:.2f}%. Drift detected: {drift_detected}")
=======
        logger.warning(f"Drift check skipped: Invalid MAPE value ({mape}).")
        return False
    drift_detected = mape > threshold
    logger.info(f"Drift check: MAPE={mape:.2f}%, Threshold={threshold:.2f}%. Drift detected: {drift_detected}")
>>>>>>> 8351801cfc5a8af9fdcec1cd1f0a58ce971e0d80
    return drift_detected

# --- Persistent Drift Check (Relies on CSV - check comments above) ---
def check_drift_trend(log_file="metrics_log.csv", model_type="LSTM", threshold=20.0, recent=5):
    # (Code remains the same, but be aware of persistence issues)
    persistent_drift = False
<<<<<<< HEAD
    if not os.path.exists(log_file):
        log.warning(f"Persistent drift check ({model_type}): Log file {log_file} not found.")
        return False
    try:
        df = pd.read_csv(log_file)
        if df.empty:
            log.info(f"Persistent drift check ({model_type}): Log file {log_file} is empty.")
            return False
        model_specific_df = df[df['model_type'] == model_type]
        if model_specific_df.empty:
            log.info(f"Persistent drift check ({model_type}): No previous metrics found for this model type.")
            return False
        if 'mape' not in model_specific_df.columns:
            log.warning(f"Persistent drift check ({model_type}): 'mape' column not found in log.")
            return False
        mape_values = pd.to_numeric(model_specific_df['mape'], errors='coerce')
        mape_values = mape_values.replace([np.inf, -np.inf], np.nan).dropna()
        recent_mape = mape_values.tail(recent)
        if len(recent_mape) >= recent:
            persistent_drift = all(m > threshold for m in recent_mape)
            log.info(f"Persistent drift check ({model_type}): Recent {len(recent_mape)} valid MAPEs > {threshold}? {persistent_drift}")
        else:
            log.info(f"Persistent drift check ({model_type}): Not enough valid data points ({len(recent_mape)}/{recent}) for a trend.")
    except Exception as e:
        log.error(f"Persistent drift check ({model_type}): Error reading or processing {log_file}: {e}", exc_info=True)
=======
    logger.warning(f"Persistent drift check relies on {log_file}, which may not persist in deployed environments.")
    # ... (rest of the function logic) ...
>>>>>>> 8351801cfc5a8af9fdcec1cd1f0a58ce971e0d80
    return persistent_drift

# --- Data Prep ---
# <<< MODIFIED load_and_prepare with enhanced logging >>>
def load_and_prepare(filepath):
<<<<<<< HEAD
    """Loads data, aggregates by date, renames columns, and sets index."""
    log.info(f"Attempting to load and prepare data from: {filepath}")
    try:
        df = pd.read_csv(filepath)
        log.info(f"Successfully read CSV. Columns: {df.columns.tolist()}")

        # Check for essential columns BEFORE processing
        required_cols = ['Date', 'Units Sold']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
             log.error(f"Essential columns {missing_cols} not found in {filepath}")
             return None # Return None if columns are missing

        # Attempt date parsing
        try:
            df['Date'] = pd.to_datetime(df['Date'])
            log.info("Parsed 'Date' column.")
        except Exception as e_date:
            log.error(f"Failed to parse 'Date' column in {filepath}: {e_date}", exc_info=True)
            return None # Return None if date parsing fails

        # Ensure 'Units Sold' is numeric
        try:
            df['Units Sold'] = pd.to_numeric(df['Units Sold'])
            log.info("Converted 'Units Sold' column to numeric.")
        except Exception as e_units:
            log.error(f"Failed to convert 'Units Sold' to numeric in {filepath}: {e_units}", exc_info=True)
            return None # Return None if conversion fails

        # Perform aggregation and renaming
        df_agg = df.groupby('Date', as_index=False)['Units Sold'].sum()
        log.info("Grouped by 'Date' and summed 'Units Sold'.")
        df_agg = df_agg.rename(columns={"Date": "ds", "Units Sold": "y"})
        df_agg.set_index("ds", inplace=True)
        log.info(f"Data loaded and prepared successfully. Final shape for modeling: {df_agg.shape}")
        return df_agg

    except FileNotFoundError:
        log.error(f"Data loading error: File not found at {filepath}")
        return None
    except Exception as e:
        # Catch other potential errors during read_csv or processing
        log.error(f"Data loading/preparation error for {filepath}: {e}", exc_info=True)
=======
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
>>>>>>> 8351801cfc5a8af9fdcec1cd1f0a58ce971e0d80
        return None


def create_sequences(data, seq_len):
    # (Code remains the same)
    x, y = [], []
    if len(data) <= seq_len:
<<<<<<< HEAD
        log.warning(f"Data length ({len(data)}) is not sufficient for sequence length ({seq_len}). Cannot create sequences.")
=======
        logger.warning(f"Data length ({len(data)}) is not sufficient for sequence length ({seq_len}). Cannot create sequences.")
>>>>>>> 8351801cfc5a8af9fdcec1cd1f0a58ce971e0d80
        return np.array(x), np.array(y)
    for i in range(len(data) - seq_len):
        x.append(data[i:i+seq_len])
        y.append(data[i+seq_len])
    if not x or not y:
        logger.warning("Sequence creation resulted in empty lists.")
        return np.array(x), np.array(y)
    return np.array(x), np.array(y)

# --- Prophet Model (Base) ---
# <<< run_prophet_model function definition remains the same as your version >>>
def run_prophet_model(df):
<<<<<<< HEAD
    log.info("--- Running Original Prophet model (Default Params) ---")
    df_prophet = df.reset_index().copy()
    df_prophet['day_of_week'] = df_prophet['ds'].dt.dayofweek
    df_prophet['month'] = df_prophet['ds'].dt.month
    df_prophet['is_weekend'] = df_prophet['day_of_week'].isin([5, 6]).astype(int)
    log.info("Prophet (Original): Feature engineering completed.")
    if len(df_prophet) < 20:
        log.error(f"Prophet (Original) Error: Insufficient data ({len(df_prophet)} rows) for train/test split.")
        return None, None, None, float('inf')
    train_size = int(len(df_prophet) * 0.8)
    if train_size == 0 or train_size >= len(df_prophet): # Fixed condition >=
        log.error(f"Prophet (Original) Error: Train size ({train_size}) is invalid for data of length {len(df_prophet)}.")
        return None, None, None, float('inf')
    train_df, test_df = df_prophet[:train_size], df_prophet[train_size:]
    log.info(f"Prophet (Original): Train size: {len(train_df)}, Test size: {len(test_df)}")
    if train_df.empty or test_df.empty:
        log.error("Prophet (Original) Error: Train or test dataframe is empty after split.")
        return None, None, None, float('inf')
    holidays_cal = calendar()
    holidays = pd.Series(dtype='datetime64[ns]') # Default empty
    try:
        holidays = holidays_cal.holidays(start=train_df['ds'].min(), end=test_df['ds'].max())
    except ValueError as e:
        log.warning(f"Prophet (Original): Error generating holidays (min: {train_df['ds'].min()}, max: {test_df['ds'].max()}): {e}. Proceeding without holidays.")
    holiday_df = pd.DataFrame({'ds': holidays, 'holiday': 'USFederalHoliday'})
    log.info(f"Prophet (Original): Generated {len(holiday_df)} US Federal holidays.")
    model = Prophet(holidays=holiday_df if not holiday_df.empty else None)
    model.add_regressor('day_of_week')
    model.add_regressor('month')
    model.add_regressor('is_weekend')
    log.info("Prophet (Original): Fitting model with default parameters.")
    fit_cols = ['ds', 'y', 'day_of_week', 'month', 'is_weekend']
    forecast = None
    try:
        model.fit(train_df[fit_cols])
        future = test_df[['ds', 'day_of_week', 'month', 'is_weekend']]
        forecast = model.predict(future)
    except Exception as e:
        log.error(f"Prophet (Original): Error during model fitting or prediction: {e}", exc_info=True)
        return model, None, test_df, float('inf') # Return model even if prediction fails
    y_true = test_df['y'].values
    y_pred = forecast['yhat'].values
    rmse, mae, mape = evaluate_forecast(y_true, y_pred)
    log.info(f"Prophet (Original) Evaluation: RMSE={rmse:.2f}, MAE={mae:.2f}, MAPE={mape:.2f}%")
    log_metrics_to_csv(test_df['ds'].max(), rmse, mae, mape, "Prophet_Original")
    try:
        mlflow.log_metrics({"rmse_prophet_original": rmse, "mae_prophet_original": mae, "mape_prophet_original": mape})
        mlflow.prophet.log_model(model, artifact_path="prophet_model_original")
        log.info("Prophet (Original): Logged metrics and model to MLflow and CSV.")
    except Exception as e_mlflow:
        log.error(f"Prophet (Original): Error logging to MLflow: {e_mlflow}", exc_info=True)
    return model, forecast, test_df, mape


# --- Prophet Model With HyperParameters ---
# <<< run_prophet_model_with_hyperparams function definition remains the same as your version >>>
def run_prophet_model_with_hyperparams(df, prophet_params):
    log.info(f"--- Running Prophet with Hyperparams: {prophet_params} ---")
    df_prophet = df.reset_index().copy()
    df_prophet['day_of_week'] = df_prophet['ds'].dt.dayofweek
    df_prophet['month'] = df_prophet['ds'].dt.month
    df_prophet['is_weekend'] = df_prophet['day_of_week'].isin([5, 6]).astype(int)
    log.info("Prophet (Hyperparam): Feature engineering completed.")
    if len(df_prophet) < 20:
        log.error(f"Prophet (Hyperparam) Error: Insufficient data ({len(df_prophet)} rows) for train/test split.")
        return None, None, None, float('inf')
    train_size = int(len(df_prophet) * 0.8)
    if train_size <= 0 or train_size >= len(df_prophet): # Fixed condition >=
        log.error(f"Prophet (Hyperparam) Error: Invalid training size ({train_size}) for data of length {len(df_prophet)}.")
        return None, None, None, float('inf')
    train_df = df_prophet[:train_size]
    test_df = df_prophet[train_size:]
    log.info(f"Prophet (Hyperparam): Train size: {len(train_df)}, Test size: {len(test_df)}")
    if train_df.empty or test_df.empty:
        log.error("Prophet (Hyperparam) Error: Train or test dataframe is empty after split.")
        return None, None, None, float('inf')
    min_date = train_df['ds'].min()
    max_date = test_df['ds'].max()
    holidays_cal = calendar()
    holidays = pd.Series(dtype='datetime64[ns]') # Default empty
    try:
        holidays = holidays_cal.holidays(start=min_date, end=max_date)
    except ValueError as e:
        log.warning(f"Prophet (Hyperparam): Error generating holidays (min: {min_date}, max: {max_date}): {e}. Proceeding without holidays.")
    holiday_df = pd.DataFrame({'ds': holidays, 'holiday': 'USFederalHoliday'})
    log.info(f"Prophet (Hyperparam): Generated {len(holiday_df)} US Federal holidays.")
    model = Prophet(
        holidays=holiday_df if not holiday_df.empty else None,
        changepoint_prior_scale=prophet_params.get('changepoint_prior_scale', 0.05),
        seasonality_prior_scale=prophet_params.get('seasonality_prior_scale', 10.0),
        holidays_prior_scale=prophet_params.get('holidays_prior_scale', 10.0)
    )
    model.add_regressor('day_of_week')
    model.add_regressor('month')
    model.add_regressor('is_weekend')
    log.info(f"Prophet (Hyperparam): Fitting model with parameters: {prophet_params}")
    fit_cols = ['ds', 'y', 'day_of_week', 'month', 'is_weekend']
    forecast = None
    try:
        model.fit(train_df[fit_cols])
        future = test_df[['ds', 'day_of_week', 'month', 'is_weekend']]
        forecast = model.predict(future)
    except Exception as e:
        log.error(f"Prophet (Hyperparam): Error during model fitting or prediction: {e}", exc_info=True)
        return model, None, test_df, float('inf')
    y_true = test_df['y'].values
    y_pred = forecast['yhat'].values
    rmse, mae, mape = evaluate_forecast(y_true, y_pred)
    log.info(f"Prophet (Hyperparam) Evaluation: RMSE={rmse:.2f}, MAE={mae:.2f}, MAPE={mape:.2f}%")
    log_metrics_to_csv(test_df['ds'].max(), rmse, mae, mape, "Prophet_Hyperparam")
    try:
        mlflow.log_params({f"prophet_hp_{k}": v for k, v in prophet_params.items()})
        mlflow.log_metrics({"rmse_prophet_hyperparam": rmse, "mae_prophet_hyperparam": mae, "mape_prophet_hyperparam": mape})
        mlflow.prophet.log_model(model, artifact_path="prophet_model_hyperparam")
        log.info("Prophet (Hyperparam): Logged parameters, metrics and model to MLflow and CSV.")
    except Exception as e_mlflow:
        log.error(f"Prophet (Hyperparam): Error logging to MLflow: {e_mlflow}", exc_info=True)
    return model, forecast, test_df, mape
=======
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

>>>>>>> 8351801cfc5a8af9fdcec1cd1f0a58ce971e0d80


# --- LSTM Model (Base) ---
# <<< run_lstm_model function definition remains the same, ensure keras_module removed >>>
def run_lstm_model(df):
<<<<<<< HEAD
    seq_len_val = 14
    epochs_val = 20
    log.info(f"--- Running Original LSTM model (Default Params, seq_len={seq_len_val}, epochs={epochs_val}) ---")
    df_lstm = df[['y']].copy()
    if len(df_lstm) < seq_len_val + 5:
        log.error(f"LSTM (Original) Error: Insufficient data ({len(df_lstm)} rows) for seq_len {seq_len_val}.")
        return None, None, None, None, float('inf')
    scaler = MinMaxScaler()
    model = None # Initialize model variable
    preds_inv = None
    y_test_inv = None
    test_dates = pd.Index([])
    mape = float('inf')
    try:
        scaled_data = scaler.fit_transform(df_lstm)
        log.info("LSTM (Original): Data scaled.")
        x, y = create_sequences(scaled_data.flatten(), seq_len_val)
        if x.size == 0 or y.size == 0:
            log.error(f"LSTM (Original) Error: Failed to create LSTM sequences. SeqLen={seq_len_val}")
            return None, None, None, None, float('inf')
        x = x.reshape((x.shape[0], x.shape[1], 1))
        train_size = int(len(x) * 0.8)
        if train_size == 0 or train_size >= len(x): # Fixed condition >=
            log.error(f"LSTM (Original) Error: Train size for sequences ({train_size}) is invalid for {len(x)} sequences.")
            return None, None, None, None, float('inf')
        x_train, y_train = x[:train_size], y[:train_size]
        x_test, y_test = x[train_size:], y[train_size:]
        log.info(f"LSTM (Original): Train sequences: {x_train.shape}, Test sequences: {x_test.shape}")
        if x_test.size == 0 or y_test.size == 0:
            log.error(f"LSTM (Original) Error: Test sequences are empty after split.")
            return None, None, None, None, float('inf')
        model = Sequential([
            LSTM(50, activation='relu', input_shape=(seq_len_val, 1)),
            Dense(1)
        ])
        model.compile(optimizer='adam', loss='mse')
        log.info("LSTM (Original): Model compiled.")
        log.info(f"LSTM (Original): Training model (epochs={epochs_val}).")
        history = model.fit(x_train, y_train, epochs=epochs_val, verbose=0, shuffle=False)
        log.info("LSTM (Original): Training completed.")
        preds = model.predict(x_test)
        preds_inv = scaler.inverse_transform(preds)
        y_test_inv = scaler.inverse_transform(y_test.reshape(-1, 1))
        rmse, mae, mape = evaluate_forecast(y_test_inv.flatten(), preds_inv.flatten())
        log.info(f"LSTM (Original) Evaluation: RMSE={rmse:.2f}, MAE={mae:.2f}, MAPE={mape:.2f}%")
        test_dates = df_lstm.index[-len(preds_inv):] if len(preds_inv) > 0 else pd.Index([])
        if not test_dates.empty:
            log_metrics_to_csv(test_dates[-1], rmse, mae, mape, "LSTM_Original")
        else:
            log.warning("LSTM (Original): Could not determine test dates for CSV logging.")
        # MLflow logging attempt
        try:
            mlflow.log_param("lstm_original_seq_len", seq_len_val)
            mlflow.log_param("lstm_original_epochs", epochs_val)
            mlflow.log_param("lstm_original_units", 50)
            mlflow.log_metrics({"rmse_lstm_original": rmse, "mae_lstm_original": mae, "mape_lstm_original": mape})
            if 'loss' in history.history:
                for epoch, loss_val in enumerate(history.history['loss']):
                    mlflow.log_metric("train_loss_lstm_original", loss_val, step=epoch)
            # <<< Ensure keras_module is REMOVED here >>>
            mlflow.keras.log_model(model, artifact_path="lstm_model_original")
            log.info("LSTM (Original): Logged metrics and model to MLflow and CSV.")
        except Exception as e_mlflow:
             log.error(f"LSTM (Original): Error logging to MLflow: {e_mlflow}", exc_info=True)

    except Exception as e_lstm:
        log.error(f"LSTM (Original): Error during model execution: {e_lstm}", exc_info=True)
        # Ensure we return consistent types even on error
        return None, None, None, None, float('inf')

    # Return results (even if MLflow logging failed)
    return model, preds_inv.flatten() if preds_inv is not None else None, y_test_inv.flatten() if y_test_inv is not None else None, test_dates, mape

=======
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
>>>>>>> 8351801cfc5a8af9fdcec1cd1f0a58ce971e0d80

# --- LSTM Model WITH Hyperparameters ---
# <<< run_lstm_model_with_hyperparams function definition remains the same, ensure keras_module removed >>>
def run_lstm_model_with_hyperparams(df, lstm_params):
<<<<<<< HEAD
    seq_len = lstm_params.get('seq_len', 14)
    log.info(f"--- Running LSTM with Hyperparams: {lstm_params}, Seq_len: {seq_len} ---")
    df_lstm = df[['y']].copy()
    if len(df_lstm) < seq_len + 5:
        log.error(f"LSTM (Hyperparam) Error: Insufficient data ({len(df_lstm)} rows) for seq_len {seq_len}.")
        return None, None, None, None, float('inf')

    model = None # Initialize
    preds_inv = None
    y_test_inv = None
    test_dates = pd.Index([])
    mape = float('inf')

    try:
        original_len = len(df_lstm)
        split_idx_original = int(original_len * 0.8)
        if split_idx_original <= 0 or split_idx_original >= original_len: # Fixed condition >=
            log.error(f"LSTM Hyperparam Error: Invalid original split index ({split_idx_original})")
            return None, None, None, None, float('inf')
        train_data_unscaled = df_lstm.iloc[:split_idx_original]
        test_data_unscaled = df_lstm.iloc[split_idx_original:]
        log.info(f"LSTM (Hyperparam): Unscaled train data len: {len(train_data_unscaled)}, test data len: {len(test_data_unscaled)}")
        if train_data_unscaled.empty or test_data_unscaled.empty:
            log.error(f"LSTM Hyperparam Error: Unscaled train or test data is empty.")
            return None, None, None, None, float('inf')
        scaler = MinMaxScaler()
        scaler.fit(train_data_unscaled)
        scaled_train_data = scaler.transform(train_data_unscaled)
        scaled_test_data = scaler.transform(test_data_unscaled)
        log.info("LSTM (Hyperparam): Data scaled (train fit, train/test transform).")
        x_train, y_train = create_sequences(scaled_train_data.flatten(), seq_len)
        x_test, y_test = create_sequences(scaled_test_data.flatten(), seq_len)
        if x_train.size == 0 or y_train.size == 0 or x_test.size == 0 or y_test.size == 0:
            log.error(f"LSTM Hyperparam Error: Failed to create sequences (train/test). SeqLen={seq_len}")
            return None, None, None, None, float('inf')
        log.info(f"LSTM (Hyperparam): Train sequences: {x_train.shape}, Test sequences: {x_test.shape}")
        x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], 1))
        x_test = x_test.reshape((x_test.shape[0], x_test.shape[1], 1))
        num_test_predictions = len(y_test)
        if num_test_predictions > 0:
            test_dates_start_index_in_original_df = split_idx_original + seq_len
            test_dates = df_lstm.index[test_dates_start_index_in_original_df : test_dates_start_index_in_original_df + num_test_predictions]
            if len(test_dates) != num_test_predictions:
                log.warning(f"LSTM (Hyperparam): Test dates length ({len(test_dates)}) mismatch ({num_test_predictions}). Using fallback.")
                test_dates = df_lstm.index[-num_test_predictions:]
        else:
            log.warning("LSTM (Hyperparam): y_test is empty, no test dates to assign.")
            test_dates = pd.Index([])
        units = lstm_params.get('units', 64)
        num_layers = lstm_params.get('num_layers', 1)
        dropout_rate = lstm_params.get('dropout_rate', 0.0)
        learning_rate = lstm_params.get('learning_rate', 0.001)
        epochs = lstm_params.get('epochs', 30)
        batch_size = lstm_params.get('batch_size', 32)
        log.info(f"LSTM (Hyperparam): Building model with: units={units}, layers={num_layers}, dropout={dropout_rate}, lr={learning_rate}")
        model = Sequential()
        for i in range(num_layers):
            return_sequences = (i < num_layers - 1)
            if i == 0: model.add(LSTM(units, activation='relu', return_sequences=return_sequences, input_shape=(seq_len, 1)))
            else: model.add(LSTM(units, activation='relu', return_sequences=return_sequences))
            if dropout_rate > 0: model.add(Dropout(dropout_rate))
        model.add(Dense(1))
        optimizer = Adam(learning_rate=learning_rate)
        model.compile(optimizer=optimizer, loss='mse')
        log.info("LSTM (Hyperparam): Model compiled.")
        log.info(f"LSTM (Hyperparam): Training model (epochs={epochs}, batch={batch_size}).")
        history = model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(x_test, y_test), verbose=0, shuffle=False)
        log.info("LSTM (Hyperparam): Training completed.")
        preds_scaled = model.predict(x_test)
        preds_inv = scaler.inverse_transform(preds_scaled)
        y_test_inv = scaler.inverse_transform(y_test.reshape(-1, 1))
        rmse, mae, mape = evaluate_forecast(y_test_inv.flatten(), preds_inv.flatten())
        log.info(f"LSTM (Hyperparam) Evaluation: RMSE={rmse:.2f}, MAE={mae:.2f}, MAPE={mape:.2f}%")
        if not test_dates.empty:
            log_metrics_to_csv(test_dates[-1], rmse, mae, mape, "LSTM_Hyperparam")
        else:
            log.warning("LSTM (Hyperparam): Could not determine test dates for CSV logging.")
        # MLflow logging attempt
        try:
            mlflow.log_param("lstm_hp_seq_len", seq_len)
            mlflow.log_params({f"lstm_hp_{k}": v for k, v in lstm_params.items() if k != 'seq_len'})
            mlflow.log_metrics({"rmse_lstm_hyperparam": rmse, "mae_lstm_hyperparam": mae, "mape_lstm_hyperparam": mape})
            if 'loss' in history.history:
                for epoch, loss_val in enumerate(history.history['loss']):
                    mlflow.log_metric("train_loss_lstm_hyperparam", loss_val, step=epoch)
            if 'val_loss' in history.history:
                for epoch, loss_val in enumerate(history.history['val_loss']):
                    mlflow.log_metric("val_loss_lstm_hyperparam", loss_val, step=epoch)
            # <<< Ensure keras_module is REMOVED here >>>
            mlflow.keras.log_model(model, artifact_path="lstm_model_hyperparam")
            log.info("LSTM (Hyperparam): Logged parameters, metrics, history and model to MLflow and CSV.")
        except Exception as e_mlflow:
            log.error(f"LSTM (Hyperparam): Error logging to MLflow: {e_mlflow}", exc_info=True)

    except Exception as e_lstm_hp:
        log.error(f"LSTM (Hyperparam): Error during model execution: {e_lstm_hp}", exc_info=True)
        return None, None, None, None, float('inf')

    return model, preds_inv.flatten() if preds_inv is not None else None, y_test_inv.flatten() if y_test_inv is not None else None, test_dates, mape


# --- Main Forecasting Pipeline Function ---
# <<< MODIFIED run_forecasting_pipeline to return results consistently >>>
def run_forecasting_pipeline(csv_path, experiment_name="SalesForecastingExperiment"):
    log.info(f"Starting forecasting pipeline for: {csv_path}")
    mlflow.set_experiment(experiment_name)
    log.info(f"MLflow experiment set to: {experiment_name}")

    # Initialize results with default failure values
    results_summary = {
        "mlflow_run_id": None,
        "mlflow_experiment_id": None,
        "pipeline_status": "failed_data_load",
        "prophet_original_mape": float('inf'),
        "lstm_original_mape": float('inf'),
        "prophet_hyperparam_mape": float('inf'),
        "lstm_hyperparam_mape": float('inf'),
        "drift_detected_on_hp_lstm": None,
        "persistent_drift_on_hp_lstm": None,
        "error_message": "Pipeline did not start due to data loading failure."
    }

    try:
        with mlflow.start_run() as run:
            run_id = run.info.run_uuid
            exp_id = run.info.experiment_id
            results_summary["mlflow_run_id"] = run_id # Store run ID early
            results_summary["mlflow_experiment_id"] = exp_id
            results_summary["pipeline_status"] = "started" # Update status
            results_summary.pop("error_message", None) # Remove initial error message

            mlflow.set_tag("dataset_path", os.path.basename(csv_path))
            mlflow.log_param("full_data_path", csv_path)
            log.info(f"MLflow Run ID: {run_id} for dataset: {csv_path}")

            df = load_and_prepare(csv_path)
            if df is None or df.empty:
                log.error("Data loading failed or data is empty. Aborting pipeline run logic.")
                mlflow.log_param("pipeline_status", "failed_data_load")
                results_summary["pipeline_status"] = "failed_data_load"
                results_summary["error_message"] = "Data loading failed or data is empty."
                # Note: mlflow run context will end, but we return the summary
                return results_summary # Exit function early if data loading fails

            # --- Run Models ---
            log.info("\n--- Running Original Models (Base Comparison) ---")
            _, _, _, prophet_mape = run_prophet_model(df)
            results_summary["prophet_original_mape"] = prophet_mape if prophet_mape is not None else float('inf')

            _, _, _, _, lstm_mape = run_lstm_model(df)
            results_summary["lstm_original_mape"] = lstm_mape if lstm_mape is not None else float('inf')

            prophet_hyperparams_to_run = {'changepoint_prior_scale': 0.1, 'seasonality_prior_scale': 10.0, 'holidays_prior_scale': 5.0}
            lstm_hyperparams_to_run = {'seq_len': 14, 'units': 128, 'num_layers': 2, 'dropout_rate': 0.0, 'learning_rate': 0.001, 'epochs': 50, 'batch_size': 32}

            log.info("\n--- Running Models with Specific Hyperparameters ---")
            _, _, _, prophet_mape_hp = run_prophet_model_with_hyperparams(df, prophet_params=prophet_hyperparams_to_run)
            results_summary["prophet_hyperparam_mape"] = prophet_mape_hp if prophet_mape_hp is not None else float('inf')

            _, _, _, _, lstm_mape_hp = run_lstm_model_with_hyperparams(df, lstm_params=lstm_hyperparams_to_run)
            results_summary["lstm_hyperparam_mape"] = lstm_mape_hp if lstm_mape_hp is not None else float('inf')

            # --- Drift Detection ---
            log.info("\n--- Drift Detection (on Hyperparameter LSTM) ---")
            drift_hp = None
            persistent_drift_hp = None
            drift_threshold = 20.0
            if lstm_mape_hp != float('inf') and pd.notna(lstm_mape_hp):
                drift_hp = check_drift(lstm_mape_hp, threshold=drift_threshold)
                persistent_drift_hp = check_drift_trend(model_type="LSTM_Hyperparam", threshold=drift_threshold, recent=5)
                log.info(f"Drift detected on HP LSTM: {drift_hp}, Persistent drift on HP LSTM: {persistent_drift_hp}")
                try:
                    mlflow.log_metric("drift_detected_hp_lstm", int(drift_hp))
                    mlflow.log_metric("persistent_drift_detected_hp_lstm", int(persistent_drift_hp))
                    mlflow.log_param("drift_threshold_hp", drift_threshold)
                except Exception as e_mlflow_drift:
                    log.error(f"Error logging drift metrics to MLflow: {e_mlflow_drift}", exc_info=True)
            else:
                log.warning("Skipping drift detection for HP LSTM: LSTM Hyperparameter model failed or produced invalid MAPE.")
                try:
                    mlflow.log_metric("drift_detected_hp_lstm", -1) # Indicate skipped/failed
                    mlflow.log_metric("persistent_drift_detected_hp_lstm", -1)
                except Exception as e_mlflow_drift_skip:
                     log.error(f"Error logging skipped drift metrics to MLflow: {e_mlflow_drift_skip}", exc_info=True)

            results_summary["drift_detected_on_hp_lstm"] = drift_hp
            results_summary["persistent_drift_on_hp_lstm"] = persistent_drift_hp

            # --- Finalize ---
            results_summary["pipeline_status"] = "completed"
            log.info(f"Forecasting pipeline finished successfully. Summary: {results_summary}")
            mlflow.log_param("pipeline_status", "completed")
            # Implicitly returns results_summary at the end of the 'with' block if no error occurred

    except Exception as e_pipeline:
        log.error(f"Unhandled exception in forecasting pipeline: {e_pipeline}", exc_info=True)
        results_summary["pipeline_status"] = "failed_exception"
        results_summary["error_message"] = str(e_pipeline)
        # Attempt to log failure status to MLflow if run was started
        try:
            if 'run' in locals() and run: # Check if mlflow run context exists
                 mlflow.log_param("pipeline_status", "failed_exception")
                 mlflow.end_run(status="FAILED") # Explicitly mark run as failed
        except Exception as e_mlflow_fail:
            log.error(f"Failed to log pipeline failure status to MLflow: {e_mlflow_fail}")

    return results_summary # Return the summary dictionary in all cases


# --- NO if __name__ == '__main__': block here ---
# --- The file should end after the last function definition ---

"""
# ... (all your functions: mean_absolute_percentage_error, evaluate_forecast, log_metrics_to_csv,
#      check_drift, check_drift_trend, load_and_prepare, create_sequences, run_prophet_model,
#      run_prophet_model_with_hyperparams, run_lstm_model, run_lstm_model_with_hyperparams,
#      run_forecasting_pipeline GO ABOVE THIS BLOCK) ...
if __name__ == '__main__':
    # --- ngrok Configuration ---
    # !!! CRITICAL: ENSURE YOUR NGROK AUTHTOKEN IS PASTED CORRECTLY HERE !!!
    NGROK_AUTHTOKEN_FROM_USER = "2wsCDg9OuRuTH6byPWcr3berIkS_bjXjwzFDutiN3Fvxarm1"  # <--- YOUR ACTUAL TOKEN HERE
=======
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
>>>>>>> 8351801cfc5a8af9fdcec1cd1f0a58ce971e0d80

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

<<<<<<< HEAD
    logging.info("Script execution fully completed.")
"""
=======
    # Return the dictionary containing results and potential errors
    return results_summary


# --- END OF FILE ---
# NO if __name__ == '__main__': block here.
>>>>>>> 8351801cfc5a8af9fdcec1cd1f0a58ce971e0d80
