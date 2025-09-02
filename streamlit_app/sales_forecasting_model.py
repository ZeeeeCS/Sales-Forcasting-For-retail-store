# lstm_forecasting_with_mlflow.py
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from keras.models import Sequential
from keras.layers import LSTM, Dense
import mlflow
import mlflow.keras
from prophet import Prophet
from pandas.tseries.holiday import USFederalHolidayCalendar as calendar

# --- Evaluation ---
def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def evaluate_forecast(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    mape = mean_absolute_percentage_error(y_true, y_pred)
    return rmse, mae, mape

def log_metrics(date, rmse, mae, mape, model_type, log_file="metrics_log.csv"):
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
    return mape > threshold

def check_drift_trend(log_file="metrics_log.csv", threshold=20.0, recent=5):
    if os.path.exists(log_file):
        df = pd.read_csv(log_file)
        recent_mape = df[df['model_type'] == 'LSTM']['mape'].tail(recent)
        if len(recent_mape) >= recent:
            return all(m > threshold for m in recent_mape)
    return False

# --- Data Prep ---
def load_and_prepare(filepath):
    df = pd.read_csv(filepath)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.groupby('Date', as_index=False)['Units Sold'].sum()
    df = df.rename(columns={"Date": "ds", "Units Sold": "y"})
    df.set_index("ds", inplace=True)
    return df

def create_sequences(data, seq_len):
    x, y = [], []
    for i in range(len(data) - seq_len):
        x.append(data[i:i+seq_len])
        y.append(data[i+seq_len])
    return np.array(x), np.array(y)

# --- Prophet Model ---
def run_prophet_model(df):
    df_prophet = df.reset_index()
    df_prophet['day_of_week'] = df_prophet['ds'].dt.dayofweek
    df_prophet['month'] = df_prophet['ds'].dt.month
    df_prophet['is_weekend'] = df_prophet['day_of_week'].isin([5, 6]).astype(int)
    train_size = int(len(df_prophet) * 0.8)
    train_df, test_df = df_prophet[:train_size], df_prophet[train_size:]

    holidays = calendar().holidays(start=train_df['ds'].min(), end=test_df['ds'].max())
    holiday_df = pd.DataFrame({'ds': holidays, 'holiday': 'USFederalHoliday'})

    model = Prophet(holidays=holiday_df)
    model.add_regressor('day_of_week')
    model.add_regressor('month')
    model.add_regressor('is_weekend')
    model.fit(train_df[['ds', 'y', 'day_of_week', 'month', 'is_weekend']])

    future = test_df[['ds', 'day_of_week', 'month', 'is_weekend']]
    forecast = model.predict(future)
    y_true = test_df['y'].values
    y_pred = forecast['yhat'].values
    rmse, mae, mape = evaluate_forecast(y_true, y_pred)

    log_metrics(test_df['ds'].max(), rmse, mae, mape, "Prophet")
    mlflow.log_metrics({"rmse_prophet": rmse, "mae_prophet": mae, "mape_prophet": mape})
    return model, forecast, test_df, mape


def tune_prophet(train_df, holidays, test_df):
    best_model = None
    best_mape = float('inf')
    best_forecast = None

    for cps in [0.01, 0.1, 0.5]:
        for sps in [1.0, 10.0, 20.0]:
            m = Prophet(holidays=holidays,
                       changepoint_prior_scale=cps,
                       seasonality_prior_scale=sps)
            m.add_regressor('day_of_week')
            m.add_regressor('month')
            m.add_regressor('is_weekend')
            m.fit(train_df[['ds', 'y', 'day_of_week', 'month', 'is_weekend']])

            future = pd.date_range(start=test_df['ds'].min(), periods=len(test_df), freq='D')
            future_df = pd.DataFrame({'ds': future})
            future_df['day_of_week'] = future_df['ds'].dt.dayofweek
            future_df['month'] = future_df['ds'].dt.month
            future_df['is_weekend'] = future_df['day_of_week'].isin([5, 6]).astype(int)

            forecast = m.predict(future_df)
            y_true = test_df['y'].values
            y_pred = forecast['yhat'].values
            mape = mean_absolute_percentage_error(y_true, y_pred)

            if mape < best_mape:
                best_model = m
                best_mape = mape
                best_forecast = forecast

    return best_model, best_forecast


# --- LSTM Model ---
def run_lstm_model(df):
    df_lstm = df.copy()
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df_lstm)

    seq_len = 14
    x, y = create_sequences(scaled_data, seq_len)
    x = x.reshape((x.shape[0], x.shape[1], 1))

    train_size = int(len(x) * 0.8)
    x_train, y_train = x[:train_size], y[:train_size]
    x_test, y_test = x[train_size:], y[train_size:]

    model = Sequential()
    model.add(LSTM(50, activation='relu', input_shape=(seq_len, 1)))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    model.fit(x_train, y_train, epochs=20, verbose=0)

    preds = model.predict(x_test)
    preds_inv = scaler.inverse_transform(preds)
    y_test_inv = scaler.inverse_transform(y_test.reshape(-1, 1))

    rmse, mae, mape = evaluate_forecast(y_test_inv, preds_inv)
    log_metrics(df_lstm.index[-1], rmse, mae, mape, "LSTM")
    mlflow.log_metrics({"rmse_lstm": rmse, "mae_lstm": mae, "mape_lstm": mape})
    mlflow.keras.log_model(model, "lstm_model")

    return model, preds_inv, y_test_inv, df_lstm.index[-len(preds_inv):], mape

# --- Main Entrypoint ---
def run_forecasting_pipeline(csv_path):
    mlflow.set_experiment("SalesForecastingExperiment")
    with mlflow.start_run():
        df = load_and_prepare(csv_path)
        prophet_model, prophet_forecast, test_df, prophet_mape = run_prophet_model(df)
        lstm_model, lstm_preds, lstm_true, dates, lstm_mape = run_lstm_model(df)

        drift = check_drift(lstm_mape)
        persistent_drift = check_drift_trend()

        return prophet_model,prophet_forecast,test_df,lstm_model,lstm_preds,lstm_true,dates,lstm_mape,drift,persistent_drift,prophet_mape

