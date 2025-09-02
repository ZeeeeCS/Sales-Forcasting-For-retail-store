import pandas as pd
import numpy as np
import os
import logging
import time
import subprocess
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from prophet import Prophet
from pandas.tseries.holiday import USFederalHolidayCalendar as calendar

import mlflow
import mlflow.keras
import mlflow.prophet

# Attempt to import ngrok, handle if not installed
try:
    from pyngrok import ngrok, conf
    NGROK_AVAILABLE = True
except ImportError:
    NGROK_AVAILABLE = False
    print("WARNING: 'pyngrok' is not installed. ngrok functionality will be disabled.")
    print("Install it with: pip install pyngrok")


# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

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

def log_metrics_to_csv(date, rmse, mae, mape, model_type, log_file="metrics_log.csv"):
    """Logs metrics to a CSV file."""
    try:
        log_dir = os.path.dirname(log_file)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir, exist_ok=True)
            
        log_exists = os.path.exists(log_file)
        log_date_str = date.strftime('%Y-%m-%d') if isinstance(date, pd.Timestamp) else str(date)
        
        row_data = {
            "date": [log_date_str],
            "model_type": [model_type],
            "rmse": [float(rmse) if pd.notna(rmse) else np.nan],
            "mae": [float(mae) if pd.notna(mae) else np.nan],
            "mape": [float(mape) if pd.notna(mape) else np.nan]
        }
        row = pd.DataFrame(row_data)

        if log_exists:
            existing_df = pd.read_csv(log_file)
            updated_df = pd.concat([existing_df, row], ignore_index=True)
            updated_df.to_csv(log_file, index=False)
        else:
            row.to_csv(log_file, index=False)
        logger.info(f"Successfully logged metrics for {model_type} to {log_file}")
    except Exception as e:
        logger.error(f"Failed to log metrics to {log_file}: {e}")

# --- Drift Detection ---
def check_drift(mape, threshold=20.0):
    """Checks if the current MAPE exceeds a threshold."""
    if mape is None or not np.isfinite(mape):
        logging.warning(f"Drift check skipped: Invalid MAPE value ({mape}).")
        return False
    drift_detected = mape > threshold
    logging.info(f"Drift check: MAPE={mape:.2f}%, Threshold={threshold:.2f}%. Drift detected: {drift_detected}")
    return drift_detected

def check_drift_trend(log_file="metrics_log.csv", model_type="LSTM", threshold=20.0, recent=5):
    """Checks if the MAPE for a model has consistently exceeded the threshold recently."""
    if not os.path.exists(log_file):
        logging.warning(f"Persistent drift check ({model_type}): Log file {log_file} not found.")
        return False
    try:
        df = pd.read_csv(log_file)
        model_specific_df = df[df['model_type'] == model_type]
        if len(model_specific_df) < recent:
            logging.info(f"Persistent drift check ({model_type}): Not enough data points ({len(model_specific_df)}/{recent}) for a trend.")
            return False

        mape_values = pd.to_numeric(model_specific_df['mape'], errors='coerce').dropna()
        recent_mape = mape_values.tail(recent)

        if len(recent_mape) < recent:
             logging.info(f"Persistent drift check ({model_type}): Not enough valid recent data points ({len(recent_mape)}/{recent}) for a trend.")
             return False
             
        persistent_drift = all(m > threshold for m in recent_mape)
        logging.info(f"Persistent drift check ({model_type}): Recent {len(recent_mape)} MAPEs > {threshold}? {persistent_drift}")
        return persistent_drift
    except Exception as e:
        logging.error(f"Persistent drift check ({model_type}): Error processing {log_file}: {e}")
        return False

# --- Data Prep ---
def load_and_prepare(filepath):
    """Loads data, aggregates by date, renames columns, and sets index."""
    logging.info(f"Loading and preparing data from: {filepath}")
    try:
        # Check if filepath is a file-like object (from Streamlit upload) or a path string
        if hasattr(filepath, 'seek'):
            filepath.seek(0)
        df = pd.read_csv(filepath)
        
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        df.dropna(subset=['Date'], inplace=True)
        df['Units Sold'] = pd.to_numeric(df['Units Sold'], errors='coerce')
        df.dropna(subset=['Units Sold'], inplace=True)
        
        if df.empty:
            logger.error("Data is empty after cleaning 'Date' and 'Units Sold'.")
            return None

        df_agg = df.groupby('Date')['Units Sold'].sum().reset_index()
        df_agg = df_agg.rename(columns={"Date": "ds", "Units Sold": "y"})
        df_agg.set_index("ds", inplace=True)
        logger.info(f"Data loaded and prepared successfully. Shape: {df_agg.shape}")
        return df_agg
    except FileNotFoundError:
        logging.error(f"Data loading error: File not found at {filepath}")
        return None
    except Exception as e:
        logging.error(f"Data loading error: {e}")
        return None

def create_sequences(data, seq_len):
    """Creates sequences for LSTM input."""
    x, y = [], []
    if len(data) <= seq_len:
        logging.warning(f"Data length ({len(data)}) is not sufficient for sequence length ({seq_len}).")
        return np.array(x), np.array(y)
    for i in range(len(data) - seq_len):
        x.append(data[i:i + seq_len])
        y.append(data[i + seq_len])
    return np.array(x), np.array(y)

# --- Prophet Model (Base) ---
def run_prophet_model(df):
    logging.info("--- Running Original Prophet model (Default Params) ---")
    df_prophet = df.reset_index().copy()

    if len(df_prophet) < 20:
        logging.error(f"Prophet (Original) Error: Insufficient data ({len(df_prophet)} rows).")
        return None, None, None, float('inf')

    train_size = int(len(df_prophet) * 0.8)
    train_df, test_df = df_prophet[:train_size], df_prophet[train_size:]
    
    holidays_cal = calendar()
    holidays_df = pd.DataFrame({
        'ds': holidays_cal.holidays(start=df_prophet['ds'].min(), end=df_prophet['ds'].max()),
        'holiday': 'USFederalHoliday'
    })

    model = Prophet(holidays=holidays_df)
    try:
        model.fit(train_df)
        forecast_df = model.predict(test_df[['ds']])
        
        y_true = test_df['y'].values
        y_pred = forecast_df['yhat'].values
        rmse, mae, mape = evaluate_forecast(y_true, y_pred)
        logging.info(f"Prophet (Original) Evaluation: RMSE={rmse:.2f}, MAE={mae:.2f}, MAPE={mape:.2f}%")

        mlflow.log_metrics({"rmse_prophet_original": rmse, "mae_prophet_original": mae, "mape_prophet_original": mape})
        mlflow.prophet.log_model(model, artifact_path="prophet_model_original")
        log_metrics_to_csv(test_df['ds'].max(), rmse, mae, mape, "Prophet_Original")
        
        return model, forecast_df, test_df, mape
    except Exception as e:
        logging.error(f"Prophet (Original): Error during model fitting or prediction: {e}")
        return None, None, None, float('inf')

# --- Prophet Model With HyperParameters ---
def run_prophet_model_with_hyperparams(df, prophet_params):
    logging.info(f"--- Running Prophet with Hyperparams: {prophet_params} ---")
    df_prophet = df.reset_index().copy()

    if len(df_prophet) < 20:
        return None, None, None, float('inf')

    train_size = int(len(df_prophet) * 0.8)
    train_df, test_df = df_prophet[:train_size], df_prophet[train_size:]
    
    holidays_cal = calendar()
    holidays_df = pd.DataFrame({
        'ds': holidays_cal.holidays(start=df_prophet['ds'].min(), end=df_prophet['ds'].max()),
        'holiday': 'USFederalHoliday'
    })
        
    model = Prophet(holidays=holidays_df, **prophet_params)
    try:
        model.fit(train_df)
        forecast_df = model.predict(test_df[['ds']])
        
        y_true = test_df['y'].values
        y_pred = forecast_df['yhat'].values
        rmse, mae, mape = evaluate_forecast(y_true, y_pred)
        logging.info(f"Prophet (Hyperparam) Evaluation: RMSE={rmse:.2f}, MAE={mae:.2f}, MAPE={mape:.2f}%")
        
        mlflow.log_params({f"prophet_hp_{k}": v for k, v in prophet_params.items()})
        mlflow.log_metrics({"rmse_prophet_hyperparam": rmse, "mae_prophet_hyperparam": mae, "mape_prophet_hyperparam": mape})
        mlflow.prophet.log_model(model, artifact_path="prophet_model_hyperparam")
        log_metrics_to_csv(test_df['ds'].max(), rmse, mae, mape, "Prophet_Hyperparam")

        return model, forecast_df, test_df, mape
    except Exception as e:
        logging.error(f"Prophet (Hyperparam): Error during model fitting or prediction: {e}")
        return None, None, None, float('inf')

# --- LSTM Model (Base) ---
def run_lstm_model(df):
    seq_len = 14
    epochs = 20
    logging.info(f"--- Running Original LSTM model (seq_len={seq_len}, epochs={epochs}) ---")
    df_lstm = df[['y']].copy()

    if len(df_lstm) < seq_len + 5:
        return None, None, None, None, float('inf')

    # **CRITICAL FIX: Prevent Data Leakage**
    # 1. Split data BEFORE scaling
    train_size_idx = int(len(df_lstm) * 0.8)
    train_data = df_lstm[:train_size_idx]
    test_data = df_lstm[train_size_idx:]

    # 2. Fit scaler ONLY on training data
    scaler = MinMaxScaler()
    scaler.fit(train_data)

    # 3. Transform both train and test data
    scaled_train = scaler.transform(train_data)
    scaled_test = scaler.transform(test_data)

    # 4. Create sequences from scaled data
    x_train, y_train = create_sequences(scaled_train, seq_len)
    x_test, y_test = create_sequences(scaled_test, seq_len)

    if x_train.size == 0 or x_test.size == 0:
        logging.error(f"LSTM (Original) Error: Failed to create train/test sequences.")
        return None, None, None, None, float('inf')

    x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], 1))
    x_test = x_test.reshape((x_test.shape[0], x_test.shape[1], 1))
        
    model = Sequential([
        LSTM(50, activation='relu', input_shape=(seq_len, 1)),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    model.fit(x_train, y_train, epochs=epochs, verbose=0, shuffle=False)
    
    preds_scaled = model.predict(x_test)
    preds_inv = scaler.inverse_transform(preds_scaled)
    y_test_inv = scaler.inverse_transform(y_test.reshape(-1, 1))

    rmse, mae, mape = evaluate_forecast(y_test_inv, preds_inv)
    logging.info(f"LSTM (Original) Evaluation: RMSE={rmse:.2f}, MAE={mae:.2f}, MAPE={mape:.2f}%")
    
    # Correctly determine test dates based on the test set's start and sequence length
    test_dates = df_lstm.index[train_size_idx + seq_len : train_size_idx + seq_len + len(y_test_inv)]
    
    try:
        mlflow.log_param("lstm_original_seq_len", seq_len)
        mlflow.log_param("lstm_original_epochs", epochs)
        mlflow.log_metrics({"rmse_lstm_original": rmse, "mae_lstm_original": mae, "mape_lstm_original": mape})
        mlflow.keras.log_model(model, artifact_path="lstm_model_original")
        if not test_dates.empty:
            log_metrics_to_csv(test_dates.max(), rmse, mae, mape, "LSTM_Original")
    except Exception as e_mlflow:
        logger.error(f"LSTM (Original): Error logging to MLflow: {e_mlflow}")
        
    return model, preds_inv, y_test_inv, test_dates, mape

# --- LSTM Model WITH Hyperparameters ---
def run_lstm_model_with_hyperparams(df, lstm_params):
    seq_len = lstm_params.get('seq_len', 14)
    logging.info(f"--- Running LSTM with Hyperparams: {lstm_params} ---")
    df_lstm = df[['y']].copy()

    if len(df_lstm) < seq_len + 5:
        return None, None, None, None, float('inf')

    train_size = int(len(df_lstm) * 0.8)
    train_data = df_lstm[:train_size]
    test_data = df_lstm[train_size:]
    
    scaler = MinMaxScaler()
    scaler.fit(train_data) # Fit ONLY on training data
    scaled_train = scaler.transform(train_data)
    scaled_test = scaler.transform(test_data)

    x_train, y_train = create_sequences(scaled_train, seq_len)
    x_test, y_test = create_sequences(scaled_test, seq_len)

    if x_train.size == 0 or x_test.size == 0:
        return None, None, None, None, float('inf')

    x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], 1))
    x_test = x_test.reshape((x_test.shape[0], x_test.shape[1], 1))
    
    units = lstm_params.get('units', 64)
    num_layers = lstm_params.get('num_layers', 1)
    dropout_rate = lstm_params.get('dropout_rate', 0.0)
    learning_rate = lstm_params.get('learning_rate', 0.001)
    epochs = lstm_params.get('epochs', 30)
    batch_size = lstm_params.get('batch_size', 32)

    model = Sequential()
    for i in range(num_layers):
        return_sequences = (i < num_layers - 1)
        if i == 0:
            model.add(LSTM(units, activation='relu', return_sequences=return_sequences, input_shape=(seq_len, 1)))
        else:
            model.add(LSTM(units, activation='relu', return_sequences=return_sequences))
        if dropout_rate > 0:
            model.add(Dropout(dropout_rate))
    model.add(Dense(1))

    model.compile(optimizer=Adam(learning_rate=learning_rate), loss='mse')
    model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(x_test, y_test), verbose=0, shuffle=False)
    
    preds_scaled = model.predict(x_test)
    preds_inv = scaler.inverse_transform(preds_scaled)
    y_test_inv = scaler.inverse_transform(y_test.reshape(-1, 1))

    rmse, mae, mape = evaluate_forecast(y_test_inv, preds_inv)
    logging.info(f"LSTM (Hyperparam) Evaluation: RMSE={rmse:.2f}, MAE={mae:.2f}, MAPE={mape:.2f}%")
    
    test_dates = df_lstm.index[train_size + seq_len : train_size + seq_len + len(y_test_inv)]

    try:
        mlflow.log_params({f"lstm_hp_{k}": v for k, v in lstm_params.items()})
        mlflow.log_metrics({"rmse_lstm_hyperparam": rmse, "mae_lstm_hyperparam": mae, "mape_lstm_hyperparam": mape})
        mlflow.keras.log_model(model, artifact_path="lstm_model_hyperparam")
        if not test_dates.empty:
            log_metrics_to_csv(test_dates.max(), rmse, mae, mape, "LSTM_Hyperparam")
    except Exception as e_mlflow:
        logger.error(f"LSTM (Hyperparam): Error logging to MLflow: {e_mlflow}")

    return model, preds_inv, y_test_inv, test_dates, mape

# --- Plotting Functions ---
def plot_prophet_forecast(df, forecast_df, title="Prophet Forecast"):
    if df is None or forecast_df is None: return None
    fig, ax = plt.subplots(figsize=(14, 7))
    sns.set(style="whitegrid")
    ax.plot(df.index, df['y'], label='Historical Data', color='blue', alpha=0.8)
    ax.plot(forecast_df['ds'], forecast_df['yhat'], label='Forecast', color='orange')
    ax.fill_between(forecast_df['ds'], forecast_df['yhat_lower'], forecast_df['yhat_upper'], color='orange', alpha=0.2, label='Uncertainty Interval')
    ax.set_title(title, fontsize=16)
    ax.set_xlabel("Date")
    ax.set_ylabel("Units Sold")
    ax.legend()
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    fig.autofmt_xdate()
    return fig

def plot_lstm_forecast(df, preds_inv, y_test_inv, test_dates, title="LSTM Forecast"):
    if df is None or len(y_test_inv) == 0: return None
    fig, ax = plt.subplots(figsize=(14, 7))
    sns.set(style="whitegrid")
    ax.plot(df.index, df['y'], label='Historical Data', color='blue', alpha=0.8)
    ax.plot(test_dates, y_test_inv, label='Actual Test Data', color='green', marker='o', linestyle='None', markersize=5)
    ax.plot(test_dates, preds_inv, label='LSTM Predictions', color='red', marker='x', linestyle='--')
    ax.set_title(title, fontsize=16)
    ax.set_xlabel("Date")
    ax.set_ylabel("Units Sold")
    ax.legend()
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    fig.autofmt_xdate()
    return fig
    
# --- Main Pipeline Function ---
def run_forecasting_pipeline(csv_path, experiment_name="SalesForecastingExperiment"):
    mlflow.set_experiment(experiment_name)
    results_summary = {}

    with mlflow.start_run() as run:
        results_summary["mlflow_run_id"] = run.info.run_uuid
        results_summary["mlflow_experiment_id"] = run.info.experiment_id

        try:
            df = load_and_prepare(csv_path)
            if df is None:
                raise ValueError("Data loading and preparation failed. Check CSV format.")
            mlflow.set_tag("dataset_source", os.path.basename(str(csv_path)))

            _, prophet_fcst_orig, _, prophet_mape_orig = run_prophet_model(df)
            _, lstm_preds_orig, lstm_actuals_orig, lstm_dates_orig, lstm_mape_orig = run_lstm_model(df)
            
            results_summary.update({
                "prophet_original_mape": prophet_mape_orig,
                "prophet_original_forecast_df": prophet_fcst_orig,
                "lstm_original_mape": lstm_mape_orig,
                "lstm_original_predictions": lstm_preds_orig,
                "lstm_original_actuals": lstm_actuals_orig,
                "lstm_original_dates": lstm_dates_orig,
            })

            prophet_hyperparams = {'changepoint_prior_scale': 0.1, 'seasonality_prior_scale': 10.0}
            lstm_hyperparams = {'seq_len': 14, 'units': 128, 'num_layers': 2, 'dropout_rate': 0.1, 'learning_rate': 0.001, 'epochs': 50, 'batch_size': 32}
            
            _, prophet_fcst_hp, _, prophet_mape_hp = run_prophet_model_with_hyperparams(df, prophet_params=prophet_hyperparams)
            _, lstm_preds_hp, lstm_actuals_hp, lstm_dates_hp, lstm_mape_hp = run_lstm_model_with_hyperparams(df, lstm_params=lstm_hyperparams)

            results_summary.update({
                "prophet_hyperparam_mape": prophet_mape_hp,
                "prophet_hyperparam_forecast_df": prophet_fcst_hp,
                "lstm_hyperparam_mape": lstm_mape_hp,
                "lstm_hyperparam_predictions": lstm_preds_hp,
                "lstm_hyperparam_actuals": lstm_actuals_hp,
                "lstm_hyperparam_dates": lstm_dates_hp,
            })

            drift_threshold = 20.0
            drift_hp = check_drift(lstm_mape_hp, threshold=drift_threshold)
            persistent_drift_hp = check_drift_trend(model_type="LSTM_Hyperparam", threshold=drift_threshold)
            
            mlflow.log_metric("drift_detected_hp_lstm", int(drift_hp))
            mlflow.log_metric("persistent_drift_detected_hp_lstm", int(persistent_drift_hp))
            
            results_summary["drift_detected_on_hp_lstm"] = drift_hp
            results_summary["persistent_drift_on_hp_lstm"] = persistent_drift_hp
            mlflow.log_param("pipeline_status", "completed")

        except Exception as e:
            error_msg = f"An unexpected error occurred in the pipeline: {e}"
            logger.error(error_msg, exc_info=True)
            results_summary["error"] = error_msg
            mlflow.log_param("pipeline_status", "failed_exception")

    logging.info("Forecasting pipeline finished.")
    return results_summary