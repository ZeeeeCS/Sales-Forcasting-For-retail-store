# lstm_forecasting_with_mlflow_logging_added.py
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt # Kept, though not directly used for plotting in this script version
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
import mlflow
import mlflow.keras
from prophet import Prophet
from pandas.tseries.holiday import USFederalHolidayCalendar as calendar
import logging

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Evaluation ---
def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    y_true = np.where(y_true == 0, 1e-6, y_true)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def evaluate_forecast(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    mape = mean_absolute_percentage_error(y_true, y_pred)
    return rmse, mae, mape

def log_metrics_to_csv(date, rmse, mae, mape, model_type, log_file="metrics_log.csv"):
    """Logs metrics to a CSV file."""
    log_exists = os.path.exists(log_file)
    log_date_str = date.strftime('%Y-%m-%d') if isinstance(date, pd.Timestamp) else str(date)
    row = pd.DataFrame([[log_date_str, model_type, rmse, mae, mape]], columns=["date", "model_type", "rmse", "mae", "mape"])
    if log_exists:
        existing = pd.read_csv(log_file)
        updated = pd.concat([existing, row], ignore_index=True)
        updated.to_csv(log_file, index=False)
    else:
        row.to_csv(log_file, index=False)
    logging.info(f"Logged metrics for {model_type} to {log_file}")


# --- Drift Detection ---
def check_drift(mape, threshold=20.0):
    """Checks if the current MAPE exceeds a threshold."""
    if mape is None or mape == float('inf'):
        logging.warning(f"Drift check skipped: Invalid MAPE value ({mape}).")
        return False
    drift_detected = mape > threshold
    logging.info(f"Drift check: MAPE={mape:.2f}%, Threshold={threshold:.2f}%. Drift detected: {drift_detected}")
    return drift_detected

def check_drift_trend(log_file="metrics_log.csv", model_type="LSTM", threshold=20.0, recent=5):
    """Checks if the MAPE for a model has consistently exceeded the threshold recently."""
    persistent_drift = False
    if not os.path.exists(log_file):
        logging.warning(f"Persistent drift check ({model_type}): Log file {log_file} not found.")
        return False
    try:
        df = pd.read_csv(log_file)
        model_specific_df = df[df['model_type'] == model_type]
        if model_specific_df.empty:
            logging.info(f"Persistent drift check ({model_type}): No previous metrics found for this model type.")
            return False
        recent_mape = model_specific_df['mape'].tail(recent)
        if len(recent_mape) >= recent:
            persistent_drift = all(m > threshold for m in recent_mape if pd.notnull(m) and m != float('inf'))
            logging.info(f"Persistent drift check ({model_type}): Recent {len(recent_mape)} valid MAPEs > {threshold}? {persistent_drift}")
        else:
            logging.info(f"Persistent drift check ({model_type}): Not enough data points ({len(recent_mape)}/{recent}) for a trend.")
    except Exception as e:
        logging.error(f"Persistent drift check ({model_type}): Error reading or processing {log_file}: {e}")
    return persistent_drift

# --- Data Prep ---
def load_and_prepare(filepath):
    """Loads data, aggregates by date, renames columns, and sets index."""
    logging.info(f"Loading and preparing data from: {filepath}")
    try:
        df = pd.read_csv(filepath)
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.groupby('Date', as_index=False)['Units Sold'].sum()
        df = df.rename(columns={"Date": "ds", "Units Sold": "y"})
        df.set_index("ds", inplace=True)
        logging.info(f"Data loaded successfully. Shape: {df.shape}")
        return df
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
        logging.warning(f"Data length ({len(data)}) is not sufficient for sequence length ({seq_len}). Cannot create sequences.")
        return np.array(x), np.array(y)
    for i in range(len(data) - seq_len):
        x.append(data[i:i+seq_len])
        y.append(data[i+seq_len])
    return np.array(x), np.array(y)

# Prophet Model (Base)
def run_prophet_model(df):
    logging.info("--- Running Original Prophet model (Default Params) ---")
    df_prophet = df.reset_index().copy()

    df_prophet['day_of_week'] = df_prophet['ds'].dt.dayofweek
    df_prophet['month'] = df_prophet['ds'].dt.month
    df_prophet['is_weekend'] = df_prophet['day_of_week'].isin([5, 6]).astype(int)
    logging.info("Prophet (Original): Feature engineering completed.")

    train_size = int(len(df_prophet) * 0.8)
    train_df, test_df = df_prophet[:train_size], df_prophet[train_size:]
    logging.info(f"Prophet (Original): Train size: {len(train_df)}, Test size: {len(test_df)}")

    holidays_cal = calendar()
    holidays = holidays_cal.holidays(start=train_df['ds'].min(), end=test_df['ds'].max())
    holiday_df = pd.DataFrame({'ds': holidays, 'holiday': 'USFederalHoliday'})
    logging.info(f"Prophet (Original): Generated {len(holiday_df)} US Federal holidays.")

    model = Prophet(holidays=holiday_df)
    model.add_regressor('day_of_week')
    model.add_regressor('month')
    model.add_regressor('is_weekend')
    logging.info("Prophet (Original): Fitting model with default parameters.")

    fit_cols = ['ds', 'y', 'day_of_week', 'month', 'is_weekend']
    model.fit(train_df[fit_cols])

    future = test_df[['ds', 'day_of_week', 'month', 'is_weekend']]
    forecast = model.predict(future)

    y_true = test_df['y'].values
    y_pred = forecast['yhat'].values
    rmse, mae, mape = evaluate_forecast(y_true, y_pred)
    logging.info(f"Prophet (Original) Evaluation: RMSE={rmse:.2f}, MAE={mae:.2f}, MAPE={mape:.2f}%")

    log_metrics_to_csv(test_df['ds'].max(), rmse, mae, mape, "Prophet_Original")
    mlflow.log_metrics({"rmse_prophet_original": rmse, "mae_prophet_original": mae, "mape_prophet_original": mape})
    mlflow.prophet.log_model(model, "prophet_model_original")
    logging.info("Prophet (Original): Logged metrics and model to MLflow and CSV.")
    return model, forecast, test_df, mape

# Prophet Model With HyperParameters
def run_prophet_model_with_hyperparams(df, prophet_params):
    logging.info(f"--- Running Prophet with Hyperparams: {prophet_params} ---")
    df_prophet = df.reset_index().copy()

    df_prophet['day_of_week'] = df_prophet['ds'].dt.dayofweek
    df_prophet['month'] = df_prophet['ds'].dt.month
    df_prophet['is_weekend'] = df_prophet['day_of_week'].isin([5, 6]).astype(int)
    logging.info("Prophet (Hyperparam): Feature engineering completed.")

    train_size = int(len(df_prophet) * 0.8)
    if train_size <= 0 or train_size >= len(df_prophet):
         logging.error(f"Prophet Hyperparam Error: Invalid training size ({train_size})")
         return None, None, None, float('inf')
    train_df = df_prophet[:train_size]
    test_df = df_prophet[train_size:]
    logging.info(f"Prophet (Hyperparam): Train size: {len(train_df)}, Test size: {len(test_df)}")

    min_date = df_prophet['ds'].min()
    max_date = df_prophet['ds'].max()
    holidays_cal = calendar()
    holidays = holidays_cal.holidays(start=min_date, end=max_date)
    holiday_df = pd.DataFrame({'ds': holidays, 'holiday': 'USFederalHoliday'})
    logging.info(f"Prophet (Hyperparam): Generated {len(holiday_df)} US Federal holidays.")

    model = Prophet(
        holidays=holiday_df,
        changepoint_prior_scale=prophet_params.get('changepoint_prior_scale', 0.05),
        seasonality_prior_scale=prophet_params.get('seasonality_prior_scale', 10.0),
        holidays_prior_scale=prophet_params.get('holidays_prior_scale', 10.0)
    )
    model.add_regressor('day_of_week')
    model.add_regressor('month')
    model.add_regressor('is_weekend')
    logging.info(f"Prophet (Hyperparam): Fitting model with parameters: {prophet_params}")

    fit_cols = ['ds', 'y', 'day_of_week', 'month', 'is_weekend']
    model.fit(train_df[fit_cols])

    future = test_df[['ds', 'day_of_week', 'month', 'is_weekend']]
    forecast = model.predict(future)

    y_true = test_df['y'].values
    y_pred = forecast['yhat'].values
    rmse, mae, mape = evaluate_forecast(y_true, y_pred)
    logging.info(f"Prophet (Hyperparam) Evaluation: RMSE={rmse:.2f}, MAE={mae:.2f}, MAPE={mape:.2f}%")

    log_metrics_to_csv(test_df['ds'].max(), rmse, mae, mape, "Prophet_Hyperparam")
    mlflow.log_params({f"prophet_hp_{k}": v for k, v in prophet_params.items()})
    mlflow.log_metrics({
        "rmse_prophet_hyperparam": rmse,
        "mae_prophet_hyperparam": mae,
        "mape_prophet_hyperparam": mape
    })
    mlflow.prophet.log_model(model, "prophet_model_hyperparam")
    logging.info("Prophet (Hyperparam): Logged parameters, metrics and model to MLflow and CSV.")
    return model, forecast, test_df, mape

# LSTM Model (Base)
def run_lstm_model(df):
    seq_len_val = 14 
    logging.info(f"--- Running Original LSTM model (Default Params, seq_len={seq_len_val}) ---")
    df_lstm = df[['y']].copy()

    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df_lstm)
    logging.info("LSTM (Original): Data scaled.")

    seq_len = 14
    x, y = create_sequences(scaled_data.flatten(), seq_len)
    if x.size == 0 or y.size == 0:
         logging.error(f"LSTM (Original) Error: Failed to create LSTM sequences. SeqLen={seq_len}")
         return None, None, None, None, float('inf')

    x = x.reshape((x.shape[0], x.shape[1], 1))

    train_size = int(len(x) * 0.8)
    x_train, y_train = x[:train_size], y[:train_size]
    x_test, y_test = x[train_size:], y[train_size:]
    logging.info(f"LSTM (Original): Train sequences: {x_train.shape}, Test sequences: {x_test.shape}")

    model = Sequential()
    model.add(LSTM(50, activation='relu', input_shape=(seq_len, 1)))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    logging.info("LSTM (Original): Model compiled.")

    logging.info("LSTM (Original): Training model with default parameters (epochs=20).")
    history = model.fit(x_train, y_train, epochs=20, verbose=0, shuffle=False)

    preds = model.predict(x_test)
    preds_inv = scaler.inverse_transform(preds)
    y_test_inv = scaler.inverse_transform(y_test.reshape(-1, 1))

    rmse, mae, mape = evaluate_forecast(y_test_inv.flatten(), preds_inv.flatten())
    logging.info(f"LSTM (Original) Evaluation: RMSE={rmse:.2f}, MAE={mae:.2f}, MAPE={mape:.2f}%")

    test_dates = df_lstm.index[-len(preds_inv):] if len(preds_inv) > 0 else pd.Index([])
    if not test_dates.empty:
        log_metrics_to_csv(test_dates[-1], rmse, mae, mape, "LSTM_Original")
    else:
        logging.warning("LSTM (Original): Could not determine test dates for CSV logging.")

    mlflow.log_param("lstm_original_seq_len", seq_len)
    mlflow.log_metrics({"rmse_lstm_original": rmse, "mae_lstm_original": mae, "mape_lstm_original": mape})
    if 'loss' in history.history:
        for epoch, loss_val in enumerate(history.history['loss']):
            mlflow.log_metric("train_loss_lstm_original", loss_val, step=epoch)
    mlflow.keras.log_model(model, "lstm_model_original")
    logging.info("LSTM (Original): Logged metrics and model to MLflow and CSV.")
    return model, preds_inv.flatten(), y_test_inv.flatten(), test_dates, mape

# LSTM Model WITH Hyperparameters
def run_lstm_model_with_hyperparams(df, lstm_params):
    seq_len=14
    logging.info(f"--- Running LSTM with Hyperparams: {lstm_params}, Seq_len: {seq_len} ---")
    df_lstm = df[['y']].copy()

    original_len = len(df_lstm)
    split_idx_original = int(original_len * 0.8)
    if split_idx_original <= 0 or split_idx_original >= original_len:
         logging.error(f"LSTM Hyperparam Error: Invalid original split index ({split_idx_original})")
         return None, None, None, None, float('inf') 

    train_data_unscaled = df_lstm.iloc[:split_idx_original]
    test_data_unscaled = df_lstm.iloc[split_idx_original:]
    logging.info(f"LSTM (Hyperparam): Unscaled train data len: {len(train_data_unscaled)}, test data len: {len(test_data_unscaled)}")

    scaler = MinMaxScaler()
    scaler.fit(train_data_unscaled)
    scaled_train_data = scaler.transform(train_data_unscaled)
    scaled_test_data = scaler.transform(test_data_unscaled)
    logging.info("LSTM (Hyperparam): Data scaled (train fit, train/test transform).")

    x_train, y_train = create_sequences(scaled_train_data.flatten(), seq_len)
    x_test, y_test = create_sequences(scaled_test_data.flatten(), seq_len)

    if x_train.size == 0 or y_train.size == 0 or x_test.size == 0 or y_test.size == 0:
         logging.error(f"LSTM Hyperparam Error: Failed to create sequences (train/test). SeqLen={seq_len}")
         return None, None, None, None, float('inf')

    logging.info(f"LSTM (Hyperparam): Train sequences: {x_train.shape}, Test sequences: {x_test.shape}")
    x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], 1))
    x_test = x_test.reshape((x_test.shape[0], x_test.shape[1], 1))

    test_dates_start_index = split_idx_original + seq_len
    test_dates_end_idx = test_dates_start_index + len(y_test)
    test_dates = df_lstm.index[test_dates_start_index:test_dates_end_idx]

    if len(test_dates) != len(y_test):
        logging.warning(f"Mismatch in LSTM Hyperparam date calculation. Dates len: {len(test_dates)}, y_test len: {len(y_test)}. Adjusting...")
        test_dates = df_lstm.index[-len(y_test):] if len(y_test) > 0 else pd.Index([])
        if len(test_dates) != len(y_test) and len(y_test) > 0 :
             logging.error("LSTM Hyperparam Error: Date correction failed. Test dates length still doesn't match y_test length.")
    elif len(y_test) == 0:
        logging.warning("LSTM (Hyperparam): y_test is empty, no test dates to assign.")
        test_dates = pd.Index([])


    units = lstm_params.get('units', 64)
    num_layers = lstm_params.get('num_layers', 1)
    dropout_rate = lstm_params.get('dropout_rate', 0.0)
    learning_rate = lstm_params.get('learning_rate', 0.001)
    epochs = lstm_params.get('epochs', 30)
    batch_size = lstm_params.get('batch_size', 32)
    logging.info(f"LSTM (Hyperparam): Building model with: units={units}, layers={num_layers}, dropout={dropout_rate}, lr={learning_rate}")

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

    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='mse')
    logging.info("LSTM (Hyperparam): Model compiled.")

    logging.info(f"LSTM (Hyperparam): Training model for {epochs} epochs, batch_size={batch_size}.")
    history = model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(x_test, y_test), verbose=0, shuffle=False)
    logging.info("LSTM (Hyperparam): Training completed.")

    preds_scaled = model.predict(x_test)
    preds_inv = scaler.inverse_transform(preds_scaled)
    y_test_inv = scaler.inverse_transform(y_test.reshape(-1, 1))

    rmse, mae, mape = evaluate_forecast(y_test_inv.flatten(), preds_inv.flatten())
    logging.info(f"LSTM (Hyperparam) Evaluation: RMSE={rmse:.2f}, MAE={mae:.2f}, MAPE={mape:.2f}%")

    if not test_dates.empty:
        log_metrics_to_csv(test_dates[-1], rmse, mae, mape, "LSTM_Hyperparam")
    else:
        logging.warning("LSTM (Hyperparam): Could not determine test dates for CSV logging.")

    mlflow.log_param("lstm_hp_seq_len", seq_len)
    mlflow.log_params({f"lstm_hp_{k}": v for k, v in lstm_params.items()})
    mlflow.log_metrics({
        "rmse_lstm_hyperparam": rmse,
        "mae_lstm_hyperparam": mae,
        "mape_lstm_hyperparam": mape
    })
    if 'loss' in history.history:
        for epoch, loss_val in enumerate(history.history['loss']):
            mlflow.log_metric("train_loss_lstm_hyperparam", loss_val, step=epoch)
    if 'val_loss' in history.history:
            for epoch, loss_val in enumerate(history.history['val_loss']):
                mlflow.log_metric("val_loss_lstm_hyperparam", loss_val, step=epoch)
    mlflow.keras.log_model(model, "lstm_model_hyperparam")
    logging.info("LSTM (Hyperparam): Logged parameters, metrics, history and model to MLflow and CSV.")
    return model, preds_inv.flatten(), y_test_inv.flatten(), test_dates, mape

# --- Main Entrypoint ---
def run_forecasting_pipeline(csv_path):
    logging.info(f"Starting forecasting pipeline for: {csv_path}")
    mlflow.set_experiment("SalesForecastingExperiment")
    logging.info(f"MLflow experiment set to: SalesForecastingExperiment")
    with mlflow.start_run() as run:
        run_id = run.info.run_uuid
        mlflow.log_param("data_path", csv_path)
        logging.info(f"MLflow Run ID: {run_id}")

        df = load_and_prepare(csv_path)
        if df is None or df.empty:
            logging.error("Data loading failed or data is empty. Aborting pipeline.")
            return None

        logging.info("\n--- Running Original Models (Base Comparison) ---")
        _, _, _, prophet_mape = run_prophet_model(df)
        if prophet_mape is None: prophet_mape = float('inf')

        _, _, _, _, lstm_mape = run_lstm_model(df)
        if lstm_mape is None: lstm_mape = float('inf')

        drift_original = check_drift(lstm_mape)
        persistent_drift_original = check_drift_trend(model_type="LSTM_Original")

        prophet_hyperparams_to_run = {
            'changepoint_prior_scale': 0.1,
            'seasonality_prior_scale': 10.0,
            'holidays_prior_scale': 5.0
        }
        lstm_hyperparams_to_run = {
            'units': 128, 'num_layers': 2, 'dropout_rate': 0.0,
            'learning_rate': 0.001, 'epochs': 50, 'batch_size': 32
        }

        logging.info("\n--- Running Models with Specific Hyperparameters ---")

        _, _, _, prophet_mape_hp = run_prophet_model_with_hyperparams(
            df,
            prophet_params=prophet_hyperparams_to_run
        )
        prophet_mape_hp = prophet_mape_hp if prophet_mape_hp is not None else float('inf')

        _, _, _, _, lstm_mape_hp = run_lstm_model_with_hyperparams(
            df,
            lstm_params=lstm_hyperparams_to_run,
        )
        lstm_mape_hp = lstm_mape_hp if lstm_mape_hp is not None else float('inf')

        logging.info("\n--- Drift Detection (on Hyperparameter LSTM) ---")
        drift_hp = False
        persistent_drift_hp = False

        if lstm_mape_hp != float('inf'):
            drift_threshold = 20.0
            drift_hp = check_drift(lstm_mape_hp, threshold=drift_threshold)
            persistent_drift_hp = check_drift_trend(model_type="LSTM_Hyperparam", threshold=drift_threshold, recent=5)
            logging.info(f"Drift detected on HP LSTM: {drift_hp}, Persistent drift on HP LSTM: {persistent_drift_hp}")

            mlflow.log_metric("drift_detected_hp_lstm", int(drift_hp))
            mlflow.log_metric("persistent_drift_detected_hp_lstm", int(persistent_drift_hp))
            mlflow.log_param("drift_threshold_hp", drift_threshold)
        else:
            logging.warning("Skipping drift detection for HP LSTM: LSTM Hyperparameter model failed or produced invalid MAPE.")

        results_summary = {
            "mlflow_run_id": run_id,
            "prophet_original_mape": prophet_mape,
            "lstm_original_mape": lstm_mape,
            "prophet_hyperparam_mape": prophet_mape_hp,
            "lstm_hyperparam_mape": lstm_mape_hp,
            "drift_original_lstm": drift_original,
            "persistent_drift_original_lstm": persistent_drift_original,
            "drift_detected_on_hp_lstm": drift_hp,
            "persistent_drift_on_hp_lstm": persistent_drift_hp,
        }
        logging.info(f"Forecasting pipeline finished. Summary: {results_summary}")

    return results_summary
