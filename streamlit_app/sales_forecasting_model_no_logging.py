# lstm_forecasting_with_mlflow_minimal.py
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
import mlflow
import mlflow.keras
from prophet import Prophet
from pandas.tseries.holiday import USFederalHolidayCalendar as calendar

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
    log_exists = os.path.exists(log_file)
    row = pd.DataFrame([[date, model_type, rmse, mae, mape]], columns=["date", "model_type", "rmse", "mae", "mape"])
    if log_exists:
        existing = pd.read_csv(log_file)
        updated = pd.concat([existing, row], ignore_index=True)
        updated.to_csv(log_file, index=False)
    else:
        row.to_csv(log_file, index=False)


# --- Drift Detection ---
def check_drift(mape, threshold=20.0):
    """Checks if the current MAPE exceeds a threshold."""
    drift_detected = mape > threshold
    return drift_detected

def check_drift_trend(log_file="metrics_log.csv", model_type="LSTM", threshold=20.0, recent=5):
    """Checks if the MAPE for a model has consistently exceeded the threshold recently."""
    persistent_drift = False
    if os.path.exists(log_file):
        df = pd.read_csv(log_file)
        recent_mape = df[df['model_type'] == model_type]['mape'].tail(recent)
        if len(recent_mape) >= recent:
            persistent_drift = all(m > threshold for m in recent_mape)
    return persistent_drift

# --- Data Prep ---
def load_and_prepare(filepath):
    """Loads data, aggregates by date, renames columns, and sets index."""
    df = pd.read_csv(filepath)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.groupby('Date', as_index=False)['Units Sold'].sum()
    df = df.rename(columns={"Date": "ds", "Units Sold": "y"})
    df.set_index("ds", inplace=True)
    return df

def create_sequences(data, seq_len):
    """Creates sequences for LSTM input."""
    x, y = [], []
    if len(data) <= seq_len:
        return np.array(x), np.array(y)
    for i in range(len(data) - seq_len):
        x.append(data[i:i+seq_len])
        y.append(data[i+seq_len])
    return np.array(x), np.array(y)

# --- Prophet Model ---
def run_prophet_model(df):
    df_prophet = df.reset_index().copy()

    # Feature Engineering
    df_prophet['day_of_week'] = df_prophet['ds'].dt.dayofweek
    df_prophet['month'] = df_prophet['ds'].dt.month
    df_prophet['is_weekend'] = df_prophet['day_of_week'].isin([5, 6]).astype(int)

    # Split Data
    train_size = int(len(df_prophet) * 0.8)
    train_df, test_df = df_prophet[:train_size], df_prophet[train_size:]

    # Add Holidays
    holidays_cal = calendar()
    holidays = holidays_cal.holidays(start=train_df['ds'].min(), end=test_df['ds'].max())
    holiday_df = pd.DataFrame({'ds': holidays, 'holiday': 'USFederalHoliday'})

    # Initialize and Train Prophet Model
    model = Prophet(holidays=holiday_df,)
    model.add_regressor('day_of_week')
    model.add_regressor('month')
    model.add_regressor('is_weekend')

    fit_cols = ['ds', 'y', 'day_of_week', 'month', 'is_weekend']
    model.fit(train_df[fit_cols])

    # Create future dataframe and predict
    future = test_df[['ds', 'day_of_week', 'month', 'is_weekend']]
    forecast = model.predict(future)

    # Evaluate
    y_true = test_df['y'].values
    y_pred = forecast['yhat'].values
    rmse, mae, mape = evaluate_forecast(y_true, y_pred)

    # Log to CSV
    log_metrics(test_df['ds'].max(), rmse, mae, mape, "Prophet")
    mlflow.log_metrics({"rmse_prophet": rmse, "mae_prophet": mae, "mape_prophet": mape})
    print(f"Original Prophet Eval: RMSE={rmse:.2f}, MAE={mae:.2f}, MAPE={mape:.2f}%")
    return model, forecast, test_df, mape

# Prophet Model With HyperParameters
def run_prophet_model_with_hyperparams(df, prophet_params):
    print(f"--- Running Prophet with Hyperparams: {prophet_params} ---")
    df_prophet = df.reset_index().copy()

    # Feature Engineering
    df_prophet['day_of_week'] = df_prophet['ds'].dt.dayofweek
    df_prophet['month'] = df_prophet['ds'].dt.month
    df_prophet['is_weekend'] = df_prophet['day_of_week'].isin([5, 6]).astype(int)

    # Split Data
    train_size = int(len(df_prophet) * 0.8)
    if train_size <= 0 or train_size >= len(df_prophet):
         print(f"Prophet Hyperparam Error: Invalid training size ({train_size})")
         return None, None, None, None, None, None
    train_df = df_prophet[:train_size]
    test_df = df_prophet[train_size:]

    # Holidays
    min_date = df_prophet['ds'].min()
    max_date = df_prophet['ds'].max()
    holidays_cal = calendar()
    holidays = holidays_cal.holidays(start=min_date, end=max_date)
    holiday_df = pd.DataFrame({'ds': holidays, 'holiday': 'USFederalHoliday'})

    # Initialize Model WITH HYPERPARAMS
    model = Prophet(
        holidays=holiday_df,
        changepoint_prior_scale=prophet_params.get('changepoint_prior_scale', 0.05),
        seasonality_prior_scale=prophet_params.get('seasonality_prior_scale', 10.0),
        holidays_prior_scale=prophet_params.get('holidays_prior_scale', 10.0)
    )
    model.add_regressor('day_of_week')
    model.add_regressor('month')
    model.add_regressor('is_weekend')

    # Fit
    fit_cols = ['ds', 'y', 'day_of_week', 'month', 'is_weekend']
    model.fit(train_df[fit_cols])

    # Predict
    future = test_df[['ds', 'day_of_week', 'month', 'is_weekend']]
    forecast = model.predict(future)

    # Evaluate
    y_true = test_df['y'].values
    y_pred = forecast['yhat'].values
    rmse, mae, mape = evaluate_forecast(y_true, y_pred)
    print(f"Prophet Hyperparam Eval: RMSE={rmse:.2f}, MAE={mae:.2f}, MAPE={mape:.2f}%")

    # Log (MLflow & CSV)
    log_metrics_to_csv(test_df['ds'].max(), rmse, mae, mape, "Prophet_Hyperparam")
    mlflow.log_params({f"prophet_hp_{k}": v for k, v in prophet_params.items()})
    mlflow.log_metrics({
        "rmse_prophet_hyperparam": rmse,
        "mae_prophet_hyperparam": mae,
        "mape_prophet_hyperparam": mape
    })
    mlflow.prophet.log_model(model, "prophet_model_hyperparam")

    return model, forecast, test_df, rmse, mae, mape

# --- LSTM Model ---
def run_lstm_model(df):
    df_lstm = df[['y']].copy()

    # Scale Data
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df_lstm)

    # Create Sequences
    seq_len = 14
    x, y = create_sequences(scaled_data.flatten(), seq_len)
    if x.size == 0 or y.size == 0:
         raise ValueError("Failed to create LSTM sequences, Not enough data for sequence length(14).")

    # Reshape for LSTM
    x = x.reshape((x.shape[0], x.shape[1], 1))

    # Split Data
    train_size = int(len(x) * 0.8)
    x_train, y_train = x[:train_size], y[:train_size]
    x_test, y_test = x[train_size:], y[train_size:]

    # Build LSTM Model
    model = Sequential()
    model.add(LSTM(50, activation='relu', input_shape=(seq_len, 1)))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    model.fit(x_train, y_train, epochs=20, verbose=0)

    # Predict and Inverse Transform
    preds = model.predict(x_test)
    preds_inv = scaler.inverse_transform(preds)
    y_test_inv = scaler.inverse_transform(y_test.reshape(-1, 1))

    rmse, mae, mape = evaluate_forecast(y_test_inv.flatten(), preds_inv.flatten())
    print(f"Original LSTM Eval: RMSE={rmse:.2f}, MAE={mae:.2f}, MAPE={mape:.2f}%")

    log_metrics(df_lstm.index[-1], rmse, mae, mape, "LSTM")
    mlflow.log_metrics({"rmse_lstm": rmse, "mae_lstm": mae, "mape_lstm": mape})
    mlflow.keras.log_model(model, "lstm_model")

    return model, preds_inv.flatten(), y_test_inv.flatten(), df_lstm.index[-len(preds_inv):], rmse, mae, mape




# LSTM Model WITH Hyperparameters
def run_lstm_model_with_hyperparams(df, lstm_params):
    df_lstm = df[['y']].copy()

    # Scale Data (Fit on Train Only)
    original_len = len(df_lstm)
    split_idx_original = int(original_len * 0.8)
    if split_idx_original <= 0 or split_idx_original >= original_len:
         print(f"LSTM Hyperparam Error: Invalid original split index ({split_idx_original})")
         return None, None, None, None, None, None, None

    train_data_unscaled = df_lstm.iloc[:split_idx_original]
    test_data_unscaled = df_lstm.iloc[split_idx_original:]

    scaler = MinMaxScaler()
    scaler.fit(train_data_unscaled)
    scaled_train_data = scaler.transform(train_data_unscaled)
    scaled_test_data = scaler.transform(test_data_unscaled)

    # Create Sequences for Train and Test Separately
    seq_len = 14
    x_train, y_train = create_sequences(scaled_train_data.flatten(), seq_len)
    x_test, y_test = create_sequences(scaled_test_data.flatten(), seq_len)

    if x_train.size == 0 or y_train.size == 0 or x_test.size == 0 or y_test.size == 0:
         print(f"LSTM Hyperparam Error: Failed to create sequences (train/test). SeqLen={seq_len}")
         return None, None, None, None, None, None, None

    # Reshape
    x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], 1))
    x_test = x_test.reshape((x_test.shape[0], x_test.shape[1], 1))

    # Get corresponding dates for the test set predictions
    test_dates_start_index = split_idx_original + seq_len
    test_dates = df_lstm.index[test_dates_start_index : test_dates_start_index + len(y_test)]

    if len(test_dates) != len(y_test):
        print(f"Warning: Mismatch in LSTM Hyperparam date calculation. Dates len: {len(test_dates)}, y_test len: {len(y_test)}")
        test_dates = df_lstm.index[-len(y_test):]
        if len(test_dates) != len(y_test):
             print("LSTM Hyperparam Error: Date correction failed.")
             return None, None, None, None, None, None, None

    # Build Model WITH HYPERPARAMS
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

    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='mse')
    model.summary()

    # Train Model
    history = model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(x_test, y_test), verbose=0, shuffle=False)

    # Predict
    preds_scaled = model.predict(x_test)
    preds_inv = scaler.inverse_transform(preds_scaled)

    # Inverse transform actuals
    y_test_scaled_2d = y_test.reshape(-1, 1)
    y_test_inv = scaler.inverse_transform(y_test_scaled_2d)

    # Evaluate
    rmse, mae, mape = evaluate_forecast(y_test_inv.flatten(), preds_inv.flatten())
    print(f"LSTM Hyperparam Eval: RMSE={rmse:.2f}, MAE={mae:.2f}, MAPE={mape:.2f}%")

    # Log (MLflow & CSV)
    if not test_dates.empty:
        log_metrics_to_csv(test_dates[-1], rmse, mae, mape, "LSTM_Hyperparam")
    mlflow.log_param("lstm_hp_seq_len", seq_len)
    mlflow.log_params({f"lstm_hp_{k}": v for k, v in lstm_params.items()})
    mlflow.log_metrics({
        "rmse_lstm_hyperparam": rmse,
        "mae_lstm_hyperparam": mae,
        "mape_lstm_hyperparam": mape
    })
    # Log loss history
    if 'loss' in history.history:
        for epoch, loss in enumerate(history.history['loss']):
            mlflow.log_metric("train_loss_lstm_hyperparam", loss, step=epoch)
    if 'val_loss' in history.history:
            for epoch, loss in enumerate(history.history['val_loss']):
                mlflow.log_metric("val_loss_lstm_hyperparam", loss, step=epoch)

    mlflow.keras.log_model(model, "lstm_model_hyperparam")

    return model, preds_inv.flatten(), y_test_inv.flatten(), test_dates, rmse, mae, mape

# --- Main Entrypoint ---
def run_forecasting_pipeline(csv_path):
    mlflow.set_experiment("SalesForecastingExperiment")
    with mlflow.start_run():
        mlflow.log_param("data_path", csv_path)
        run_id = mlflow.active_run().info.run_uuid
        print(f"MLflow Run ID: {run_id}")

        # --- Load Data ---
        df = load_and_prepare(csv_path)
        if df is None:
            print("Data loading failed.")
            mlflow.end_run(status="FAILED")
            return None

        # Original Models (LSTM, Prophet)
        prophet_model, prophet_forecast, test_df, prophet_mape = run_prophet_model(df)
        lstm_model, lstm_preds, lstm_true, dates, lstm_mape = run_lstm_model(df)
        # Drift Check For Original Models
        drift = check_drift(lstm_mape)
        persistent_drift = check_drift_trend()

        # Models Hyperparameters
        prophet_hyperparams_to_run = {
            'changepoint_prior_scale': 0.1,
            'seasonality_prior_scale': 10.0,
            'holidays_prior_scale': 5.0
        }
        lstm_hyperparams_to_run = {
            'units': 128,
            'num_layers': 2,
            'dropout_rate': 0.0,
            'learning_rate': 0.001,
            'epochs': 50,
            'batch_size': 32
        }
        lstm_seq_len_to_run = 14

                # --- Run New Models WITH Hyperparameters ---
        print("\n Running Models with Specific Hyperparameters")
        prophet_model_hp, prophet_forecast_hp, test_df_hp, prophet_mape_hp = run_prophet_model_with_hyperparams(
          df,
          prophet_params=prophet_hyperparams_to_run
        )
        lstm_model_hp, lstm_preds_hp, lstm_true_hp, lstm_dates_hp, lstm_mape_hp = run_lstm_model_with_hyperparams(
            df,
            seq_len=lstm_seq_len_to_run,
            lstm_params=lstm_hyperparams_to_run
        )

        # handle None cases and fallback to inf
        prophet_hp_mape = prophet_mape_hp if prophet_mape_hp is not None else float('inf')
        lstm_hp_mape = lstm_mape_hp if lstm_mape_hp is not None else float('inf')

        # Drift Detection (Based on the LSTM run WITH hyperparameters)
        print("\n Drift Detection (on Hyperparam LSTM)")
        drift = False
        persistent_drift = False

        if lstm_hp_mape != float('inf'):
            drift_threshold = 20.0
            drift = check_drift(lstm_hp_mape, threshold=drift_threshold)
            # Use the specific model_type logged by the hyperparameter function
            persistent_drift = check_drift_trend(model_type="LSTM_Hyperparam", threshold=drift_threshold, recent=5)
            print(f"Drift detected: {drift}, Persistent drift: {persistent_drift}")

            # Log drift results specifically for the hyperparameter model run
            mlflow.log_metric("drift_detected_hp_lstm", int(drift))
            mlflow.log_metric("persistent_drift_detected_hp_lstm", int(persistent_drift))
            mlflow.log_param("drift_threshold", drift_threshold)
        else:
            print("Skipping drift detection: LSTM Hyperparameter model failed or produced invalid MAPE.")

        results_summary = {
            "prophet_original_mape": prophet_mape,
            "lstm_original_mape": lstm_mape,
            "prophet_hyperparam_mape": prophet_hp_mape,
            "lstm_hyperparam_mape": lstm_hp_mape,
            "drift_detected_on_hp_lstm": drift,
            "persistent_drift_on_hp_lstm": persistent_drift,
        }
    # Return the summary dictionary
    return results_summary

if __name__ == "__main__":
    # Main Function
    pass