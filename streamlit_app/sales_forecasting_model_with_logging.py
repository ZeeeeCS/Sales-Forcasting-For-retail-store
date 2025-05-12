# sales_forecasting_model_with_logging.py
# (Or rename to forecasting_pipeline.py)

import pandas as pd
import numpy as np
import os
# import matplotlib.pyplot as plt # Keep if models generate plots you want to save/log
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
import time # Keep if used internally, otherwise can remove
import subprocess # Can likely remove unless helper functions use it
# from pyngrok import ngrok, conf # DEFINITELY REMOVE THESE - No ngrok needed here

# --- Logging Setup ---
# Configure logging basic setup - Streamlit might override this later
# Consider using logging.getLogger(__name__) if integrating into a larger app
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__) # Use a named logger

# --- Evaluation ---
def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    y_true = np.where(y_true == 0, 1e-6, y_true) # Avoid division by zero
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def evaluate_forecast(y_true, y_pred):
    # Add checks for empty or non-numeric input if necessary
    if len(y_true) == 0 or len(y_pred) == 0 or len(y_true) != len(y_pred):
        logger.warning("Invalid input for evaluation. Returning NaN.")
        return np.nan, np.nan, np.nan
    try:
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)
        mape = mean_absolute_percentage_error(y_true, y_pred)
    except ValueError as e:
        logger.error(f"Error during evaluation: {e}. Returning NaN.")
        return np.nan, np.nan, np.nan
    return rmse, mae, mape

# --- CSV Logging (Optional - Consider if needed in deployed env) ---
# Logging metrics to a local CSV might not be useful in a deployed Streamlit app
# unless you have persistent storage mapped, which is uncommon on free tiers.
# Consider logging ONLY to MLflow if deploying remotely.
def log_metrics_to_csv(date, rmse, mae, mape, model_type, log_file="metrics_log.csv"):
    """Logs metrics to a CSV file. (Use with caution in deployed envs)"""
    try:
        # Check if path is writable - might fail in deployed envs
        log_dir = os.path.dirname(log_file)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir, exist_ok=True) # Try to create dir if needed
        
        log_exists = os.path.exists(log_file)
        log_date_str = date.strftime('%Y-%m-%d') if isinstance(date, pd.Timestamp) else str(date)
        row = pd.DataFrame([[log_date_str, model_type, rmse, mae, mape]], columns=["date", "model_type", "rmse", "mae", "mape"])
        
        if log_exists:
            try:
                existing = pd.read_csv(log_file)
                updated = pd.concat([existing, row], ignore_index=True)
                updated.to_csv(log_file, index=False)
            except pd.errors.EmptyDataError:
                logger.warning(f"CSV log file {log_file} was empty. Overwriting.")
                row.to_csv(log_file, index=False)
            except Exception as e_update:
                logger.error(f"Error updating CSV log {log_file}: {e_update}. Attempting to write new.")
                row.to_csv(log_file, index=False)
        else:
            row.to_csv(log_file, index=False)
        logger.info(f"Logged metrics for {model_type} to {log_file}")
    except OSError as e_os:
         logger.warning(f"Could not write to CSV log file {log_file} (permission issue?): {e_os}")
    except Exception as e_csv:
        logger.error(f"An unexpected error occurred during CSV logging to {log_file}: {e_csv}")


# --- Drift Detection ---
def check_drift(mape, threshold=20.0):
    """Checks if the current MAPE exceeds a threshold."""
    if mape is None or mape == float('inf') or pd.isna(mape):
        logger.warning(f"Drift check skipped: Invalid MAPE value ({mape}).")
        return False
    drift_detected = mape > threshold
    logger.info(f"Drift check: MAPE={mape:.2f}%, Threshold={threshold:.2f}%. Drift detected: {drift_detected}")
    return drift_detected

# --- Persistent Drift Check (Relies on log_metrics_to_csv) ---
# This function will likely NOT work reliably in a standard ephemeral deployed environment
# because the metrics_log.csv file won't persist across sessions/restarts.
# Keep this in mind if 'persistent_drift' is critical for your deployed app.
def check_drift_trend(log_file="metrics_log.csv", model_type="LSTM", threshold=20.0, recent=5):
    """Checks if the MAPE for a model has consistently exceeded the threshold recently."""
    persistent_drift = False
    logger.warning(f"Persistent drift check relies on {log_file}, which may not persist in deployed environments.")
    if not os.path.exists(log_file):
        logger.warning(f"Persistent drift check ({model_type}): Log file {log_file} not found.")
        return False
    try:
        df = pd.read_csv(log_file)
        # ... (rest of the function remains the same) ...
        # Convert to numeric, coercing errors to NaN, then drop NaNs and Infs
        mape_values = pd.to_numeric(model_specific_df['mape'], errors='coerce')
        mape_values = mape_values.replace([np.inf, -np.inf], np.nan).dropna()

        recent_mape = mape_values.tail(recent)

        if len(recent_mape) >= recent:
            persistent_drift = all(m > threshold for m in recent_mape)
            logger.info(f"Persistent drift check ({model_type}): Recent {len(recent_mape)} valid MAPEs > {threshold}? {persistent_drift}")
        else:
            logger.info(f"Persistent drift check ({model_type}): Not enough valid data points ({len(recent_mape)}/{recent}) for a trend.")

    except FileNotFoundError: # Explicitly catch FileNotFoundError
         logger.warning(f"Persistent drift check ({model_type}): Log file {log_file} not found during read attempt.")
         return False
    except Exception as e:
        logger.error(f"Persistent drift check ({model_type}): Error reading or processing {log_file}: {e}")
        # Return current state on error, avoid returning None
    return persistent_drift

# --- Data Prep ---
def load_and_prepare(filepath):
    """Loads data, aggregates by date, renames columns, and sets index."""
    logger.info(f"Loading and preparing data from: {filepath}")
    try:
        # Check if filepath is a string (path) or a file-like object
        if isinstance(filepath, str):
            df = pd.read_csv(filepath)
        else: # Assume it's a file uploaded by Streamlit (BytesIO or similar)
             # Reset seek position in case it was read before
            filepath.seek(0)
            df = pd.read_csv(filepath)

        # Basic validation
        if 'Date' not in df.columns or 'Units Sold' not in df.columns:
             logger.error("Input data missing required columns: 'Date' and 'Units Sold'.")
             return None

        df['Date'] = pd.to_datetime(df['Date'])
        # Handle potential errors during aggregation
        try:
            df_agg = df.groupby('Date', as_index=False)['Units Sold'].sum()
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
        logger.error(f"Data loading error: Uploaded file is empty.")
        return None
    except Exception as e:
        logger.error(f"Data loading error: An unexpected error occurred - {e}", exc_info=True)
        return None


def create_sequences(data, seq_len):
    """Creates sequences for LSTM input."""
    x, y = [], []
    if len(data) <= seq_len:
        logger.warning(f"Data length ({len(data)}) is not sufficient for sequence length ({seq_len}). Cannot create sequences.")
        return np.array(x), np.array(y) # Return empty arrays
    for i in range(len(data) - seq_len):
        x.append(data[i:i+seq_len])
        y.append(data[i+seq_len])
    # Check if sequences were actually created before returning
    if not x or not y:
        logger.warning("Sequence creation resulted in empty lists.")
        return np.array(x), np.array(y)
    return np.array(x), np.array(y)

# --- Prophet Model (Base) ---
def run_prophet_model(df):
    # Add input validation
    if df is None or df.empty or 'y' not in df.columns:
         logger.error("Prophet (Original): Invalid or empty DataFrame provided.")
         return None, None, None, float('inf')
         
    logger.info("--- Running Original Prophet model (Default Params) ---")
    # ... (rest of the function as before, ensure logging uses 'logger') ...
    # Make sure to return the forecast dataframe if needed by Streamlit
    df_prophet = df.reset_index().copy()

    df_prophet['day_of_week'] = df_prophet['ds'].dt.dayofweek
    df_prophet['month'] = df_prophet['ds'].dt.month
    df_prophet['is_weekend'] = df_prophet['day_of_week'].isin([5, 6]).astype(int)
    logger.info("Prophet (Original): Feature engineering completed.")

    if len(df_prophet) < 20:
        logger.error(f"Prophet (Original) Error: Insufficient data ({len(df_prophet)} rows) for train/test split.")
        return None, None, None, float('inf')

    train_size = int(len(df_prophet) * 0.8)
    if train_size == 0 or train_size == len(df_prophet):
        logger.error(f"Prophet (Original) Error: Train size ({train_size}) is invalid for data of length {len(df_prophet)}.")
        return None, None, None, float('inf')

    train_df, test_df = df_prophet[:train_size], df_prophet[train_size:]
    logger.info(f"Prophet (Original): Train size: {len(train_df)}, Test size: {len(test_df)}")

    if train_df.empty or test_df.empty:
        logger.error("Prophet (Original) Error: Train or test dataframe is empty after split.")
        return None, None, None, float('inf')

    holidays_cal = calendar()
    try:
        min_date = train_df['ds'].min()
        max_date = test_df['ds'].max()
        if pd.isna(min_date) or pd.isna(max_date):
             raise ValueError("Min or Max date is NaT after split.")
        holidays = holidays_cal.holidays(start=min_date, end=max_date)
    except ValueError as e:
        logger.warning(f"Prophet (Original): Error generating holidays (min_date: {min_date}, max_date: {max_date}): {e}. Proceeding without holidays.")
        holidays = pd.Series(dtype='datetime64[ns]')

    holiday_df = pd.DataFrame({'ds': holidays, 'holiday': 'USFederalHoliday'})
    logger.info(f"Prophet (Original): Generated {len(holiday_df)} US Federal holidays.")

    try:
        model = Prophet(holidays=holiday_df if not holiday_df.empty else None)
        model.add_regressor('day_of_week')
        model.add_regressor('month')
        model.add_regressor('is_weekend')
        logger.info("Prophet (Original): Fitting model with default parameters.")

        fit_cols = ['ds', 'y', 'day_of_week', 'month', 'is_weekend']
        model.fit(train_df[fit_cols])

        future = test_df[['ds', 'day_of_week', 'month', 'is_weekend']]
        forecast_df = model.predict(future) # Assign to forecast_df

    except Exception as e:
        logger.error(f"Prophet (Original): Error during model fitting or prediction: {e}", exc_info=True)
        return None, None, None, float('inf')

    y_true = test_df['y'].values
    y_pred = forecast_df['yhat'].values
    rmse, mae, mape = evaluate_forecast(y_true, y_pred)
    logger.info(f"Prophet (Original) Evaluation: RMSE={rmse:.2f}, MAE={mae:.2f}, MAPE={mape:.2f}%")

    if not test_df.empty and 'ds' in test_df.columns:
        log_metrics_to_csv(test_df['ds'].max(), rmse, mae, mape, "Prophet_Original")
    else:
         logger.warning("Prophet (Original): Could not log metrics to CSV, test_df is invalid.")

    # MLflow logging
    try:
        mlflow.log_metrics({
            "rmse_prophet_original": rmse,
            "mae_prophet_original": mae,
            "mape_prophet_original": mape
        })
        # Log model - consider adding signature if possible
        mlflow.prophet.log_model(model, artifact_path="prophet_model_original")
        logger.info("Prophet (Original): Logged metrics and model to MLflow.")
    except Exception as e_mlflow:
         logger.error(f"Prophet (Original): Error logging to MLflow: {e_mlflow}")

    # Return forecast_df along with other results
    return model, forecast_df, test_df, mape


# --- Prophet Model With HyperParameters ---
def run_prophet_model_with_hyperparams(df, prophet_params):
    if df is None or df.empty or 'y' not in df.columns:
         logger.error("Prophet (Hyperparam): Invalid or empty DataFrame provided.")
         return None, None, None, float('inf')
         
    logger.info(f"--- Running Prophet with Hyperparams: {prophet_params} ---")
    # ... (rest of the function as before, ensure logging uses 'logger') ...
    # Make sure to return the forecast dataframe if needed by Streamlit
    df_prophet = df.reset_index().copy()
    # ... (feature eng, split, holidays - similar to run_prophet_model, add error handling) ...
    
    # Calculate train/test split
    train_size = int(len(df_prophet) * 0.8)
    # ... add validation for train_size, train_df, test_df as in original function ...
    train_df = df_prophet[:train_size]
    test_df = df_prophet[train_size:]

    # --- Add similar holiday generation and error handling as run_prophet_model ---
    holidays_cal = calendar()
    try:
        min_date = train_df['ds'].min()
        max_date = test_df['ds'].max()
        if pd.isna(min_date) or pd.isna(max_date):
             raise ValueError("Min or Max date is NaT after split.")
        holidays = holidays_cal.holidays(start=min_date, end=max_date)
    except ValueError as e:
        logger.warning(f"Prophet (Hyperparam): Error generating holidays (min_date: {min_date}, max_date: {max_date}): {e}. Proceeding without holidays.")
        holidays = pd.Series(dtype='datetime64[ns]')
    holiday_df = pd.DataFrame({'ds': holidays, 'holiday': 'USFederalHoliday'})
    # -----------------------------------------------------------------------------

    try:
        model = Prophet(
            holidays=holiday_df if not holiday_df.empty else None,
            changepoint_prior_scale=prophet_params.get('changepoint_prior_scale', 0.05),
            seasonality_prior_scale=prophet_params.get('seasonality_prior_scale', 10.0),
            holidays_prior_scale=prophet_params.get('holidays_prior_scale', 10.0)
        )
        # ... (add regressors) ...
        model.add_regressor('day_of_week')
        model.add_regressor('month')
        model.add_regressor('is_weekend')
        
        # ... (fit model) ...
        fit_cols = ['ds', 'y', 'day_of_week', 'month', 'is_weekend']
        model.fit(train_df[fit_cols])

        # ... (predict) ...
        future = test_df[['ds', 'day_of_week', 'month', 'is_weekend']]
        forecast_df = model.predict(future) # Assign to forecast_df

    except Exception as e:
        logger.error(f"Prophet (Hyperparam): Error during model fitting or prediction: {e}", exc_info=True)
        return None, None, None, float('inf')

    # ... (evaluate) ...
    y_true = test_df['y'].values
    y_pred = forecast_df['yhat'].values
    rmse, mae, mape = evaluate_forecast(y_true, y_pred)
    logger.info(f"Prophet (Hyperparam) Evaluation: RMSE={rmse:.2f}, MAE={mae:.2f}, MAPE={mape:.2f}%")

    # ... (log to CSV - optional) ...
    if not test_df.empty and 'ds' in test_df.columns:
        log_metrics_to_csv(test_df['ds'].max(), rmse, mae, mape, "Prophet_Hyperparam")

    # ... (log to MLflow) ...
    try:
        mlflow.log_params({f"prophet_hp_{k}": v for k, v in prophet_params.items()})
        mlflow.log_metrics({
            "rmse_prophet_hyperparam": rmse,
            "mae_prophet_hyperparam": mae,
            "mape_prophet_hyperparam": mape
        })
        mlflow.prophet.log_model(model, artifact_path="prophet_model_hyperparam")
        logger.info("Prophet (Hyperparam): Logged parameters, metrics and model to MLflow.")
    except Exception as e_mlflow:
         logger.error(f"Prophet (Hyperparam): Error logging to MLflow: {e_mlflow}")
         
    # Return forecast_df
    return model, forecast_df, test_df, mape


# --- LSTM Model (Base) ---
def run_lstm_model(df):
    if df is None or df.empty or 'y' not in df.columns:
         logger.error("LSTM (Original): Invalid or empty DataFrame provided.")
         return None, None, None, None, float('inf')
         
    seq_len_val = 14
    epochs_val = 20
    logger.info(f"--- Running Original LSTM model (Default Params, seq_len={seq_len_val}, epochs={epochs_val}) ---")
    df_lstm = df[['y']].copy()

    # ... (rest of the function as before, ensure logging uses 'logger') ...
    # Add more robust error handling for scaling, sequence creation, training, prediction
    try:
        if len(df_lstm) < seq_len_val + 5:
            logger.error(f"LSTM (Original) Error: Insufficient data ({len(df_lstm)} rows) for seq_len {seq_len_val}.")
            return None, None, None, None, float('inf')

        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(df_lstm)
        logger.info("LSTM (Original): Data scaled.")

        x, y = create_sequences(scaled_data.flatten(), seq_len_val)
        if x.size == 0 or y.size == 0:
            logger.error(f"LSTM (Original) Error: Failed to create LSTM sequences. SeqLen={seq_len_val}")
            return None, None, None, None, float('inf')

        x = x.reshape((x.shape[0], x.shape[1], 1))

        train_size = int(len(x) * 0.8)
        if train_size == 0 or train_size == len(x):
            logger.error(f"LSTM (Original) Error: Train size for sequences ({train_size}) is invalid for {len(x)} sequences.")
            return None, None, None, None, float('inf')

        x_train, y_train = x[:train_size], y[:train_size]
        x_test, y_test = x[train_size:], y[train_size:]
        logger.info(f"LSTM (Original): Train sequences: {x_train.shape}, Test sequences: {x_test.shape}")

        if x_test.size == 0 or y_test.size == 0:
            logger.error(f"LSTM (Original) Error: Test sequences are empty after split.")
            return None, None, None, None, float('inf')

        model = Sequential([
            LSTM(50, activation='relu', input_shape=(seq_len_val, 1)),
            Dense(1)
        ])
        model.compile(optimizer='adam', loss='mse')
        logger.info("LSTM (Original): Model compiled.")

        logger.info(f"LSTM (Original): Training model with default parameters (epochs={epochs_val}).")
        history = model.fit(x_train, y_train, epochs=epochs_val, verbose=0, shuffle=False)

        preds_scaled = model.predict(x_test)
        preds_inv = scaler.inverse_transform(preds_scaled)
        y_test_inv = scaler.inverse_transform(y_test.reshape(-1, 1))

    except Exception as e:
        logger.error(f"LSTM (Original): Error during model setup, training, or prediction: {e}", exc_info=True)
        return None, None, None, None, float('inf')

    # ... (evaluate) ...
    rmse, mae, mape = evaluate_forecast(y_test_inv.flatten(), preds_inv.flatten())
    logger.info(f"LSTM (Original) Evaluation: RMSE={rmse:.2f}, MAE={mae:.2f}, MAPE={mape:.2f}%")

    # ... (log to CSV - optional) ...
    test_dates = df_lstm.index[-len(preds_inv):] if len(preds_inv) > 0 else pd.Index([])
    if not test_dates.empty:
        log_metrics_to_csv(test_dates[-1], rmse, mae, mape, "LSTM_Original")

    # ... (log to MLflow) ...
    try:
        mlflow.log_param("lstm_original_seq_len", seq_len_val)
        mlflow.log_param("lstm_original_epochs", epochs_val)
        mlflow.log_param("lstm_original_units", 50)
        mlflow.log_metrics({
            "rmse_lstm_original": rmse,
            "mae_lstm_original": mae,
            "mape_lstm_original": mape
        })
        if history and 'loss' in history.history: # Check if history exists
            for epoch, loss_val in enumerate(history.history['loss']):
                mlflow.log_metric("train_loss_lstm_original", loss_val, step=epoch)
        # Log model - REMOVED keras_module
        mlflow.keras.log_model(model, artifact_path="lstm_model_original")
        logger.info("LSTM (Original): Logged metrics and model to MLflow.")
    except Exception as e_mlflow:
         logger.error(f"LSTM (Original): Error logging to MLflow: {e_mlflow}")

    # Return predictions/actuals
    return model, preds_inv.flatten(), y_test_inv.flatten(), test_dates, mape


# --- LSTM Model WITH Hyperparameters ---
def run_lstm_model_with_hyperparams(df, lstm_params):
    if df is None or df.empty or 'y' not in df.columns:
         logger.error("LSTM (Hyperparam): Invalid or empty DataFrame provided.")
         return None, None, None, None, float('inf')
         
    seq_len = lstm_params.get('seq_len', 14)
    logger.info(f"--- Running LSTM with Hyperparams: {lstm_params}, Seq_len: {seq_len} ---")
    df_lstm = df[['y']].copy()

    # ... (rest of the function as before, ensure logging uses 'logger') ...
    # Add more robust error handling for scaling, sequence creation, training, prediction
    try:
        # ... (Check data length) ...
        if len(df_lstm) < seq_len + 5:
            logger.error(f"LSTM (Hyperparam) Error: Insufficient data ({len(df_lstm)} rows) for seq_len {seq_len}.")
            return None, None, None, None, float('inf')

        # ... (Split, check splits) ...
        original_len = len(df_lstm)
        split_idx_original = int(original_len * 0.8)
        if split_idx_original <= 0 or split_idx_original >= original_len:
             logger.error(f"LSTM Hyperparam Error: Invalid original split index ({split_idx_original})")
             return None, None, None, None, float('inf')
        train_data_unscaled = df_lstm.iloc[:split_idx_original]
        test_data_unscaled = df_lstm.iloc[split_idx_original:]
        if train_data_unscaled.empty or test_data_unscaled.empty:
            logger.error(f"LSTM Hyperparam Error: Unscaled train or test data is empty.")
            return None, None, None, None, float('inf')

        # ... (Scale) ...
        scaler = MinMaxScaler()
        scaler.fit(train_data_unscaled) # Fit only on training data
        scaled_train_data = scaler.transform(train_data_unscaled)
        scaled_test_data = scaler.transform(test_data_unscaled)

        # ... (Create sequences, check sequences) ...
        x_train, y_train = create_sequences(scaled_train_data.flatten(), seq_len)
        x_test, y_test = create_sequences(scaled_test_data.flatten(), seq_len)
        if x_train.size == 0 or y_train.size == 0 or x_test.size == 0 or y_test.size == 0:
             logger.error(f"LSTM Hyperparam Error: Failed to create sequences (train/test). SeqLen={seq_len}")
             return None, None, None, None, float('inf')

        # ... (Reshape) ...
        x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], 1))
        x_test = x_test.reshape((x_test.shape[0], x_test.shape[1], 1))
        
        # ... (Build model) ...
        units = lstm_params.get('units', 64)
        num_layers = lstm_params.get('num_layers', 1)
        dropout_rate = lstm_params.get('dropout_rate', 0.0)
        learning_rate = lstm_params.get('learning_rate', 0.001)
        epochs = lstm_params.get('epochs', 30)
        batch_size = lstm_params.get('batch_size', 32)

        model = Sequential()
        # ... (model building loop) ...
        for i in range(num_layers):
            return_sequences = (i < num_layers - 1)
            if i == 0:
                 model.add(LSTM(units, activation='relu', return_sequences=return_sequences, input_shape=(seq_len, 1)))
            else:
                 model.add(LSTM(units, activation='relu', return_sequences=return_sequences))
            if dropout_rate > 0:
                model.add(Dropout(dropout_rate))
        model.add(Dense(1))

        # ... (Compile) ...
        optimizer = Adam(learning_rate=learning_rate)
        model.compile(optimizer=optimizer, loss='mse')
        
        # ... (Train) ...
        history = model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(x_test, y_test), verbose=0, shuffle=False)

        # ... (Predict) ...
        preds_scaled = model.predict(x_test)
        preds_inv = scaler.inverse_transform(preds_scaled)
        y_test_inv = scaler.inverse_transform(y_test.reshape(-1, 1))

    except Exception as e:
        logger.error(f"LSTM (Hyperparam): Error during model setup, training, or prediction: {e}", exc_info=True)
        return None, None, None, None, float('inf')


    # ... (Evaluate) ...
    rmse, mae, mape = evaluate_forecast(y_test_inv.flatten(), preds_inv.flatten())
    logger.info(f"LSTM (Hyperparam) Evaluation: RMSE={rmse:.2f}, MAE={mae:.2f}, MAPE={mape:.2f}%")

    # ... (Log to CSV - optional) ...
    # Determine test dates robustly
    num_test_predictions = len(y_test)
    test_dates_start_index_in_original_df = split_idx_original + seq_len
    if num_test_predictions > 0:
        test_dates = df_lstm.index[test_dates_start_index_in_original_df : test_dates_start_index_in_original_df + num_test_predictions]
        if len(test_dates) != num_test_predictions:
             logger.warning(f"LSTM (Hyperparam): Test dates length mismatch. Using fallback.")
             test_dates = df_lstm.index[-num_test_predictions:]
    else:
        test_dates = pd.Index([])

    if not test_dates.empty:
        log_metrics_to_csv(test_dates[-1], rmse, mae, mape, "LSTM_Hyperparam")

    # ... (Log to MLflow) ...
    try:
        mlflow.log_param("lstm_hp_seq_len", seq_len)
        mlflow.log_params({f"lstm_hp_{k}": v for k, v in lstm_params.items() if k != 'seq_len'})
        mlflow.log_metrics({
            "rmse_lstm_hyperparam": rmse,
            "mae_lstm_hyperparam": mae,
            "mape_lstm_hyperparam": mape
        })
        if history and 'loss' in history.history:
            for epoch, loss_val in enumerate(history.history['loss']):
                mlflow.log_metric("train_loss_lstm_hyperparam", loss_val, step=epoch)
        if history and 'val_loss' in history.history:
            for epoch, loss_val in enumerate(history.history['val_loss']):
                mlflow.log_metric("val_loss_lstm_hyperparam", loss_val, step=epoch)
        # Log model - REMOVED keras_module
        mlflow.keras.log_model(model, artifact_path="lstm_model_hyperparam")
        logger.info("LSTM (Hyperparam): Logged parameters, metrics, history and model to MLflow.")
    except Exception as e_mlflow:
        logger.error(f"LSTM (Hyperparam): Error logging to MLflow: {e_mlflow}")

    # Return predictions/actuals
    return model, preds_inv.flatten(), y_test_inv.flatten(), test_dates, mape


# --- Main Pipeline Function ---
def run_forecasting_pipeline(csv_path, experiment_name="SalesForecastingExperiment"):
    """
    Runs the full forecasting pipeline for the given data path.
    Logs results to MLflow and returns a summary dictionary.
    """
    logger.info(f"Starting forecasting pipeline for: {csv_path}")

    # --- MLflow Setup ---
    # Consider setting tracking URI here if using a remote server
    # Example: mlflow.set_tracking_uri("http://your-remote-server:5000")
    # Or read from environment variable: mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI"))
    try:
        mlflow.set_experiment(experiment_name)
        logger.info(f"MLflow experiment set to: {experiment_name}")
    except Exception as e_exp:
        logger.error(f"Failed to set MLflow experiment '{experiment_name}': {e_exp}")
        # Decide if you want to proceed without MLflow or return an error
        # return {"error": f"Failed to set MLflow experiment: {e_exp}"}


    # Initialize results dictionary
    results_summary = {
        "mlflow_run_id": None,
        "error": None, # To store potential errors
        # Initialize metrics to None or NaN
        "prophet_original_mape": np.nan,
        "lstm_original_mape": np.nan,
        "prophet_hyperparam_mape": np.nan,
        "lstm_hyperparam_mape": np.nan,
        "drift_detected_on_hp_lstm": None,
        "persistent_drift_on_hp_lstm": None,
    }

    try:
        with mlflow.start_run() as run:
            run_id = run.info.run_uuid
            results_summary["mlflow_run_id"] = run_id # Store run ID early
            mlflow.set_tag("dataset_path", os.path.basename(str(csv_path))) # Handle non-string paths if needed
            mlflow.log_param("full_data_path", str(csv_path))
            logger.info(f"MLflow Run ID: {run_id} for dataset: {csv_path}")

            # --- Load Data ---
            df = load_and_prepare(csv_path)
            if df is None or df.empty:
                error_msg = "Data loading failed or data is empty. Aborting pipeline."
                logger.error(error_msg)
                mlflow.log_param("pipeline_status", "failed_data_load")
                results_summary["error"] = error_msg
                return results_summary # Return immediately with error

            # --- Run Models & Get Results ---
            logger.info("\n--- Running Original Models (Base Comparison) ---")
            try:
                 _, _, _, prophet_mape = run_prophet_model(df)
                 results_summary["prophet_original_mape"] = prophet_mape if prophet_mape is not None else np.nan
            except Exception as e_prophet_orig:
                 logger.error(f"Error running original Prophet model: {e_prophet_orig}", exc_info=True)

            try:
                 _, _, _, _, lstm_mape = run_lstm_model(df)
                 results_summary["lstm_original_mape"] = lstm_mape if lstm_mape is not None else np.nan
            except Exception as e_lstm_orig:
                 logger.error(f"Error running original LSTM model: {e_lstm_orig}", exc_info=True)


            # --- Run Models with Hyperparameters ---
            # Define hyperparameters (consider making these arguments to the function later)
            prophet_hyperparams_to_run = {
                'changepoint_prior_scale': 0.1,
                'seasonality_prior_scale': 10.0,
                'holidays_prior_scale': 5.0
            }
            lstm_hyperparams_to_run = {
                'seq_len': 14, 'units': 128, 'num_layers': 2, 'dropout_rate': 0.0,
                'learning_rate': 0.001, 'epochs': 50, 'batch_size': 32
            }

            logger.info("\n--- Running Models with Specific Hyperparameters ---")
            try:
                _, _, _, prophet_mape_hp = run_prophet_model_with_hyperparams(
                    df, prophet_params=prophet_hyperparams_to_run
                )
                results_summary["prophet_hyperparam_mape"] = prophet_mape_hp if prophet_mape_hp is not None else np.nan
            except Exception as e_prophet_hp:
                logger.error(f"Error running Prophet model with hyperparams: {e_prophet_hp}", exc_info=True)

            try:
                _, _, _, _, lstm_mape_hp = run_lstm_model_with_hyperparams(
                    df, lstm_params=lstm_hyperparams_to_run,
                )
                results_summary["lstm_hyperparam_mape"] = lstm_mape_hp if lstm_mape_hp is not None else np.nan
            except Exception as e_lstm_hp:
                 logger.error(f"Error running LSTM model with hyperparams: {e_lstm_hp}", exc_info=True)


            # --- Drift Detection ---
            # Use the calculated lstm_mape_hp if available
            calculated_lstm_mape_hp = results_summary["lstm_hyperparam_mape"]
            logger.info("\n--- Drift Detection (on Hyperparameter LSTM) ---")
            drift_hp = False
            persistent_drift_hp = False
            drift_threshold = 20.0

            if not pd.isna(calculated_lstm_mape_hp) and calculated_lstm_mape_hp != float('inf'):
                try:
                    drift_hp = check_drift(calculated_lstm_mape_hp, threshold=drift_threshold)
                    # Persistent drift relies on CSV, may not work reliably in deployed env
                    persistent_drift_hp = check_drift_trend(model_type="LSTM_Hyperparam", threshold=drift_threshold, recent=5)
                    logger.info(f"Drift detected on HP LSTM: {drift_hp}, Persistent drift on HP LSTM: {persistent_drift_hp}")

                    mlflow.log_metric("drift_detected_hp_lstm", int(drift_hp))
                    mlflow.log_metric("persistent_drift_detected_hp_lstm", int(persistent_drift_hp))
                    mlflow.log_param("drift_threshold_hp", drift_threshold)
                except Exception as e_drift:
                     logger.error(f"Error during drift detection: {e_drift}")
            else:
                logger.warning("Skipping drift detection for HP LSTM: MAPE is invalid.")
                mlflow.log_metric("drift_detected_hp_lstm", -1) # Indicate skipped/invalid
                mlflow.log_metric("persistent_drift_detected_hp_lstm", -1)

            results_summary["drift_detected_on_hp_lstm"] = drift_hp
            results_summary["persistent_drift_on_hp_lstm"] = persistent_drift_hp

            # Pipeline completed successfully (even if some models failed internally)
            mlflow.log_param("pipeline_status", "completed")
            logger.info(f"Forecasting pipeline finished processing. Returning summary.")
            return results_summary

    except Exception as e_outer:
        # Catch errors occurring outside the mlflow run block (e.g., setting experiment)
        # Or potentially errors within the block if not caught internally
        error_msg = f"An unexpected error occurred in the main pipeline function: {e_outer}"
        logger.error(error_msg, exc_info=True)
        results_summary["error"] = error_msg
        # Attempt to log failure status if a run was started
        if 'run' in locals() and run:
             mlflow.log_param("pipeline_status", "failed_outer_exception")
             mlflow.end_run(status="FAILED") # Explicitly fail the run
        return results_summary


# --- FILE SHOULD END HERE ---
# NO if __name__ == '__main__': block that starts ngrok/mlflow ui.