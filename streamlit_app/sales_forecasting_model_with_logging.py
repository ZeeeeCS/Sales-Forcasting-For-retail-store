# sales_forecasting_model_with_logging.py

import pandas as pd
import numpy as np
import os
# import matplotlib.pyplot as plt # Keep commented unless generating plots to save
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

# --- Evaluation ---
def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    y_true = np.where(y_true == 0, 1e-6, y_true) # Avoid division by zero
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def evaluate_forecast(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    mape = mean_absolute_percentage_error(y_true, y_pred)
    return rmse, mae, mape

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

# --- Drift Detection ---
def check_drift(mape, threshold=20.0):
    """Checks if the current MAPE exceeds a threshold."""
    if mape is None or mape == float('inf') or pd.isna(mape):
        log.warning(f"Drift check skipped: Invalid MAPE value ({mape}).")
        return False
    drift_detected = mape > threshold
    log.info(f"Drift check: MAPE={mape:.2f}%, Threshold={threshold:.2f}%. Drift detected: {drift_detected}")
    return drift_detected

def check_drift_trend(log_file="metrics_log.csv", model_type="LSTM", threshold=20.0, recent=5):
    """Checks if the MAPE for a model has consistently exceeded the threshold recently."""
    persistent_drift = False
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
    return persistent_drift

# --- Data Prep ---
# <<< MODIFIED load_and_prepare with enhanced logging >>>
def load_and_prepare(filepath):
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
        return None

def create_sequences(data, seq_len):
    """Creates sequences for LSTM input."""
    x, y = [], []
    if len(data) <= seq_len:
        log.warning(f"Data length ({len(data)}) is not sufficient for sequence length ({seq_len}). Cannot create sequences.")
        return np.array(x), np.array(y)
    for i in range(len(data) - seq_len):
        x.append(data[i:i+seq_len])
        y.append(data[i+seq_len])
    return np.array(x), np.array(y)

# --- Prophet Model (Base) ---
# <<< run_prophet_model function definition remains the same as your version >>>
def run_prophet_model(df):
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


# --- LSTM Model (Base) ---
# <<< run_lstm_model function definition remains the same, ensure keras_module removed >>>
def run_lstm_model(df):
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


# --- LSTM Model WITH Hyperparameters ---
# <<< run_lstm_model_with_hyperparams function definition remains the same, ensure keras_module removed >>>
def run_lstm_model_with_hyperparams(df, lstm_params):
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
# In sales_forecasting_model_with_logging.py

def run_forecasting_pipeline(csv_path, experiment_name="SalesForecastingExperiment"):
    logging.info(f"Starting forecasting pipeline for: {csv_path}")
    mlflow.set_experiment(experiment_name)
    logging.info(f"MLflow experiment set to: {experiment_name}")

    # --- Initialize variables BEFORE the try block ---
    run_id = None
    exp_id = None
    results_summary = None # Use this to track success/failure/results

    try:
        with mlflow.start_run() as run:
            # --- Assign run_id and exp_id *immediately* after starting run ---
            run_id = run.info.run_uuid
            exp_id = run.info.experiment_id
            logging.info(f"MLflow Run started: {run_id} (Experiment: {exp_id}) for dataset: {csv_path}")

            # --- Log basic params ---
            mlflow.set_tag("dataset_path", os.path.basename(csv_path))
            mlflow.log_param("full_data_path", csv_path)

            # --- Load Data ---
            df = load_and_prepare(csv_path)
            if df is None or df.empty:
                error_message = "Data loading failed or data is empty."
                logging.error(f"{error_message} Aborting pipeline for run {run_id}.")
                mlflow.log_param("pipeline_status", "failed_data_load")
                mlflow.set_tag("pipeline_error", error_message) # Add tag for error
                # Return error status, including IDs if the run started
                return {
                    "error": error_message,
                    "mlflow_run_id": run_id,
                    "mlflow_experiment_id": exp_id
                }

            # --- Run Models & Log (Errors inside these could still occur) ---
            logging.info(f"\n--- Running Original Models (Base Comparison) for run {run_id} ---")
            _, _, _, prophet_mape = run_prophet_model(df) # Assuming these log internally now
            if prophet_mape is None: prophet_mape = float('inf')

            _, _, _, _, lstm_mape = run_lstm_model(df)
            if lstm_mape is None: lstm_mape = float('inf')

            # Define fixed hyperparameters here or pass them in
            prophet_hyperparams_to_run = {
                'changepoint_prior_scale': 0.1, 'seasonality_prior_scale': 10.0, 'holidays_prior_scale': 5.0
            }
            lstm_hyperparams_to_run = {
                'seq_len': 14, 'units': 128, 'num_layers': 2, 'dropout_rate': 0.0,
                'learning_rate': 0.001, 'epochs': 50, 'batch_size': 32
            }

            logging.info(f"\n--- Running Models with Specific Hyperparameters for run {run_id} ---")
            # Pass run_id or let them log within the active run context
            _, _, _, prophet_mape_hp = run_prophet_model_with_hyperparams(df, prophet_hyperparams_to_run)
            prophet_mape_hp = prophet_mape_hp if prophet_mape_hp is not None else float('inf')

            _, _, _, _, lstm_mape_hp = run_lstm_model_with_hyperparams(df, lstm_hyperparams_to_run)
            lstm_mape_hp = lstm_mape_hp if lstm_mape_hp is not None else float('inf')

            # --- Drift Detection ---
            logging.info(f"\n--- Drift Detection (on Hyperparameter LSTM) for run {run_id} ---")
            drift_hp = False
            persistent_drift_hp = False
            drift_threshold = 20.0

            if lstm_mape_hp != float('inf') and pd.notna(lstm_mape_hp):
                drift_hp = check_drift(lstm_mape_hp, threshold=drift_threshold)
                persistent_drift_hp = check_drift_trend(model_type="LSTM_Hyperparam", threshold=drift_threshold, recent=5)
                mlflow.log_metric("drift_detected_hp_lstm", int(drift_hp))
                mlflow.log_metric("persistent_drift_detected_hp_lstm", int(persistent_drift_hp))
                mlflow.log_param("drift_threshold_hp", drift_threshold)
            else:
                logging.warning(f"Skipping drift detection for HP LSTM for run {run_id}: Invalid MAPE.")
                mlflow.log_metric("drift_detected_hp_lstm", -1)
                mlflow.log_metric("persistent_drift_detected_hp_lstm", -1)

            # --- Prepare SUCCESS results dictionary *INSIDE* the 'with' block ---
            results_summary = {
                "mlflow_run_id": run_id, # Now run_id is guaranteed to exist IF this point is reached
                "mlflow_experiment_id": exp_id, # exp_id should also exist
                "prophet_original_mape": prophet_mape,
                "lstm_original_mape": lstm_mape,
                "prophet_hyperparam_mape": prophet_mape_hp,
                "lstm_hyperparam_mape": lstm_mape_hp,
                "drift_detected_on_hp_lstm": drift_hp,
                "persistent_drift_on_hp_lstm": persistent_drift_hp,
                "error": None # Explicitly set error to None on success
            }
            logging.info(f"Forecasting pipeline finished successfully for run {run_id}. Summary: {results_summary}")
            mlflow.log_param("pipeline_status", "completed")
            # --- Return results ---
            return results_summary

    except Exception as e:
        # --- Catch ANY exception during the 'with mlflow.start_run...' block ---
        error_message = f"An error occurred during pipeline execution: {e}"
        logging.error(error_message, exc_info=True) # Log full traceback

        # Try to update MLflow run status if the run was started (run_id exists)
        if run_id:
            try:
                # Use client if context is potentially broken, or just log tags/params
                mlflow.log_param("pipeline_status", "failed_runtime_error")
                mlflow.set_tag("pipeline_error", str(e)) # Log error message as tag
            except Exception as mlflow_err:
                logging.error(f"Could not update MLflow run status after error for run {run_id}: {mlflow_err}")

        # Return error dictionary, including run_id and exp_id if they were captured
        return {
            "error": error_message,
            "mlflow_run_id": run_id, # Return ID if available
            "mlflow_experiment_id": exp_id # Return ID if available
        }

    # --- Fallback Return (Should ideally not be reached if try/except is structured well) ---
    # If execution somehow exits the try block without returning (e.g., complex control flow not shown)
    # Return the last known state of results_summary or an error
    if results_summary is None:
         return {
             "error": "Pipeline execution finished in an unexpected state.",
             "mlflow_run_id": run_id,
             "mlflow_experiment_id": exp_id
             }
    # This return is likely redundant now but safe to keep if results_summary was assigned
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

    MLFLOW_UI_PORT = 5000 # Port for MLflow UI

    public_url = None
    mlflow_ui_process = None

    # Check if the user has provided a token
    if not NGROK_AUTHTOKEN_FROM_USER or NGROK_AUTHTOKEN_FROM_USER == "2wsCDg9OuRuTH6byPWcr3berIkS_bjXjwzFDutiN3Fvxarm1": # Check if it's empty or the placeholder
        logging.warning("NGROK_AUTHTOKEN_FROM_USER is effectively not set or is the placeholder. ngrok will not be started.")
        print("WARNING: ngrok authtoken is not properly set in the script. MLflow UI will only be accessible locally if you start it manually.")
    else:
        # If we are in this 'else' block, it means NGROK_AUTHTOKEN_FROM_USER should be your actual token.
        logging.info(f"Attempting to configure ngrok with token: {NGROK_AUTHTOKEN_FROM_USER[:5]}... (partially hidden)")
        try:
            conf.get_default().auth_token = NGROK_AUTHTOKEN_FROM_USER # Use the token variable
            logging.info("ngrok authtoken configured successfully with pyngrok.")

            # Ensure mlruns directory exists for MLflow UI
            if not os.path.exists("mlruns"):
                os.makedirs("mlruns")
                logging.info("Created 'mlruns' directory.")

            # Start MLflow UI in the background
            mlflow_command = [
                "mlflow", "ui",
                "--backend-store-uri", f"file:{os.path.join(os.getcwd(), 'mlruns')}",
                "--port", str(MLFLOW_UI_PORT),
                "--host", "0.0.0.0"
            ]
            logging.info(f"Starting MLflow UI with command: {' '.join(mlflow_command)}")
            mlflow_ui_process = subprocess.Popen(mlflow_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            logging.info(f"MLflow UI process started with PID: {mlflow_ui_process.pid if mlflow_ui_process else 'N/A'}")

            logging.info(f"Waiting for MLflow UI to start on port {MLFLOW_UI_PORT}...")
            time.sleep(5) # Give MLflow UI a moment to start

            logging.info("Attempting to start ngrok tunnel...")
            public_url = ngrok.connect(MLFLOW_UI_PORT, "http")
            logging.info(f"ngrok tunnel established. MLflow UI accessible at: {public_url}")
            print(f"MLflow UI is running.")
            print(f"  Local access: http://localhost:{MLFLOW_UI_PORT}")
            print(f"  Publicly via ngrok: {public_url}")

        except Exception as e:
            logging.error(f"Error starting MLflow UI or ngrok: {e}", exc_info=True)
            print(f"ERROR: Could not start MLflow UI with ngrok: {e}")
            if mlflow_ui_process:
                try:
                    mlflow_ui_process.terminate()
                    mlflow_ui_process.wait(timeout=2) # Short timeout
                except: # Broad except to ensure it doesn't crash shutdown
                    if mlflow_ui_process.poll() is None: # If still running
                        mlflow_ui_process.kill()
                        mlflow_ui_process.wait()

            public_url = None # Ensure public_url is None if ngrok fails


    # --- Forecasting Pipeline Execution ---
    actual_csv_path = "retail_store_inventory.csv" # Your dataset
    actual_experiment_name = "RetailStoreForecasting_ActualData_Ngrok" # Experiment name

    summary_actual = None # Initialize summary
    if os.path.exists(actual_csv_path):
        logging.info(f"Running forecasting pipeline for: {actual_csv_path}")
        try:
            summary_actual = run_forecasting_pipeline(actual_csv_path, experiment_name=actual_experiment_name)
            if summary_actual:
                print(f"\nSummary for {actual_csv_path}:")
                for key, value in summary_actual.items():
                    if isinstance(value, float) and not pd.isna(value):
                        print(f"  {key}: {value:.2f}")
                    else:
                        print(f"  {key}: {value}")
            else:
                print(f"Pipeline execution for {actual_csv_path} completed but returned no summary.")
                logging.warning(f"Pipeline execution for {actual_csv_path} returned no summary.")
        except Exception as e_pipeline:
            logging.error(f"Error during forecasting pipeline execution for {actual_csv_path}: {e_pipeline}", exc_info=True)
            print(f"ERROR during forecasting pipeline: {e_pipeline}")
    else:
        logging.error(f"Dataset not found: {actual_csv_path}")
        print(f"ERROR: Dataset not found - {actual_csv_path}")

    # --- Keep alive for ngrok and Cleanup ---
    print("\nForecasting pipeline finished processing.") # Clarified message

    if public_url: # If ngrok was successfully started
        print(f"MLflow UI remains accessible at: {public_url} (and http://localhost:{MLFLOW_UI_PORT})")
        print("Press Ctrl+C in this terminal to stop the script, ngrok tunnel, and MLflow UI process.")
        try:
            while True:
                time.sleep(1) # Keep the main thread alive
        except KeyboardInterrupt:
            logging.info("KeyboardInterrupt received. Shutting down...")
            print("\nShutting down ngrok and MLflow UI...")
        finally:
            logging.info("Starting cleanup...")
            if public_url:
                try:
                    active_tunnels = ngrok.get_tunnels()
                    for tunnel in active_tunnels:
                        if tunnel.public_url == public_url or tunnel.public_url.replace("https://", "http://") == public_url: # ngrok might return https
                            ngrok.disconnect(tunnel.public_url)
                            logging.info(f"ngrok tunnel {tunnel.public_url} disconnected.")
                            break # Assuming only one tunnel was started for this port
                except Exception as e_ngrok_disc:
                    logging.error(f"Error disconnecting ngrok tunnel: {e_ngrok_disc}")
                try:
                    ngrok.kill() # Kills all ngrok processes started by this pyngrok instance
                    logging.info("All ngrok processes (for this pyngrok instance) killed.")
                except Exception as e_ngrok_kill:
                    logging.error(f"Error killing ngrok processes: {e_ngrok_kill}")

            if mlflow_ui_process:
                logging.info(f"Terminating MLflow UI process (PID: {mlflow_ui_process.pid})...")
                try:
                    mlflow_ui_process.terminate() # Send SIGTERM
                    mlflow_ui_process.wait(timeout=5) # Wait for it to terminate
                    if mlflow_ui_process.poll() is None: # If still running after timeout
                        logging.warning("MLflow UI process did not terminate gracefully, killing.")
                        mlflow_ui_process.kill() # Send SIGKILL
                        mlflow_ui_process.wait()
                        logging.info("MLflow UI process killed.")
                    else:
                        logging.info("MLflow UI process terminated successfully.")
                except Exception as e_mlflow_term:
                    logging.error(f"Error terminating MLflow UI process: {e_mlflow_term}")
            print("Shutdown complete.")
    else:
        # This block executes if public_url is None (ngrok didn't start or failed)
        if not NGROK_AUTHTOKEN_FROM_USER or NGROK_AUTHTOKEN_FROM_USER == "2wsCDg9OuRuTH6byPWcr3berIkS_bjXjwzFDutiN3Fvxarm1":
            # Message already printed at the beginning
            pass
        else: # Token was provided, but ngrok/UI failed for other reasons
            print("An ngrok authtoken was provided, but ngrok or MLflow UI did not start correctly. Please check logs above for specific errors (e.g., port conflicts, ngrok service issues).")
        print("Script finished. If you started MLflow UI manually, you'll need to stop it manually.")

    logging.info("Script execution fully completed.")
"""