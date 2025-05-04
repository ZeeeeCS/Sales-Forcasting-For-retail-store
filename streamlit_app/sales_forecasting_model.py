import mlflow
import mlflow.prophet
import mlflow.keras
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
from prophet import Prophet
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from pandas.tseries.holiday import USFederalHolidayCalendar as calendar
from sklearn.preprocessing import MinMaxScaler

import os
import warnings

warnings.filterwarnings('ignore')


def mean_absolute_percentage_error(y_true, y_pred):
    """Calculates MAPE given y_true and y_pred"""
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


# Helper functions for evaluation and logging
def evaluate_forecast(y_true, y_pred, name="Model"):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    mape = mean_absolute_percentage_error(y_true, y_pred)
    print(f"{name} RMSE: {rmse:.2f}, MAE: {mae:.2f}, MAPE: {mape:.2f}%")
    return rmse, mae, mape


def log_metrics(date, rmse, mae, mape, log_file="metrics_log.csv"):
    log_exists = os.path.exists(log_file)
    row = pd.DataFrame([[date, rmse, mae, mape]], columns=["date", "rmse", "mae", "mape"])
    if log_exists:
        existing = pd.read_csv(log_file)
        updated = pd.concat([existing, row], ignore_index=True)
        updated.to_csv(log_file, index=False)
    else:
        row.to_csv(log_file, index=False)


def check_drift(current_mape, threshold=20.0):
    return current_mape > threshold


def check_drift_trend(log_file="metrics_log.csv", threshold=20.0, recent=5):
    if os.path.exists(log_file):
        df = pd.read_csv(log_file)
        recent_mape = df['mape'].tail(recent)
        if len(recent_mape) >= recent:
            return all(mape > threshold for mape in recent_mape)
    return False


def feedback_available(feedback_path="feedback_data.csv"):
    return os.path.exists(feedback_path)


def merge_with_feedback(main_df, feedback_path="feedback_data.csv"):
    if os.path.exists(feedback_path):
        feedback_df = pd.read_csv(feedback_path)
        feedback_df['Date'] = pd.to_datetime(feedback_df['Date'])
        feedback_df = feedback_df.groupby('Date', as_index=False)['Units Sold'].sum()
        feedback_df['day_of_week'] = feedback_df['Date'].dt.dayofweek
        feedback_df['month'] = feedback_df['Date'].dt.month
        feedback_df['is_weekend'] = feedback_df['day_of_week'].isin([5, 6]).astype(int)
        feedback_df = feedback_df.rename(columns={'Date': 'ds', 'Units Sold': 'y'})
        combined = pd.concat([main_df, feedback_df], ignore_index=True).drop_duplicates(subset='ds')
        return combined.sort_values('ds')
    else:
        return main_df


# Load and prepare the data
def load_and_prepare(filepath):
    data = pd.read_csv(filepath)
    data['Date'] = pd.to_datetime(data['Date'])
    daily_df = data.groupby('Date', as_index=False)['Units Sold'].sum()
    daily_df['day_of_week'] = daily_df['Date'].dt.dayofweek
    daily_df['month'] = daily_df['Date'].dt.month
    daily_df['is_weekend'] = daily_df['day_of_week'].isin([5, 6]).astype(int)
    daily_df = daily_df.rename(columns={'Date': 'ds', 'Units Sold': 'y'})
    return daily_df


# Split the data into training and testing sets
def split_data(df, split_date='2023-10-01'):
    train = df[df['ds'] < split_date]
    test = df[df['ds'] >= split_date]
    return train, test


# Add US Holidays to the forecast
def get_us_holidays(start_date, end_date):
    cal = calendar()
    holidays = cal.holidays(start=start_date, end=end_date, return_name=True)
    holiday_df = pd.DataFrame(data=holidays, columns=['holiday'])
    return holiday_df.reset_index().rename(columns={'index': 'ds'})


# Train Prophet Model
def train_prophet(train_df, holidays=None):
    m = Prophet(holidays=holidays)
    m.add_regressor('day_of_week')
    m.add_regressor('month')
    m.add_regressor('is_weekend')
    m.fit(train_df[['ds', 'y', 'day_of_week', 'month', 'is_weekend']])
    return m


# Train LSTM Model
def train_lstm(train_df):
    scaler = MinMaxScaler(feature_range=(0, 1))
    train_scaled = scaler.fit_transform(train_df[['y']].values)

    X_train, y_train = [], []
    for i in range(60, len(train_scaled)):
        X_train.append(train_scaled[i-60:i, 0])
        y_train.append(train_scaled[i, 0])
    X_train, y_train = np.array(X_train), np.array(y_train)

    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
    model.add(LSTM(units=50, return_sequences=False))
    model.add(Dense(units=1))

    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X_train, y_train, epochs=20, batch_size=32)
    
    return model, scaler


# Forecast Future (Prophet & LSTM)
def forecast_with_model(model, horizon, start_date, df=None, lstm_model=None, scaler=None):
    future = pd.date_range(start=start_date, periods=horizon, freq='D')
    future_df = pd.DataFrame({'ds': future})
    future_df['day_of_week'] = future_df['ds'].dt.dayofweek
    future_df['month'] = future_df['ds'].dt.month
    future_df['is_weekend'] = future_df['day_of_week'].isin([5, 6]).astype(int)

    # Prophet Forecasting
    forecast_prophet = model.predict(future_df)

    # LSTM Forecasting
    future_scaled = scaler.transform(future_df[['y']].fillna(0))  # Example for LSTM usage
    forecast_lstm = lstm_model.predict(future_scaled)

    return forecast_prophet, forecast_lstm


# Main forecasting pipeline with MLflow tracking
def run_forecasting_pipeline(csv_path):
    df = load_and_prepare(csv_path)
    if feedback_available():
        df = merge_with_feedback(df)

    train_df, test_df = split_data(df)
    holidays = get_us_holidays(train_df['ds'].min(), test_df['ds'].max())

    # Prophet model
    with mlflow.start_run():
        mlflow.set_tag("model", "prophet")
        model_prophet = train_prophet(train_df, holidays)
        forecast_prophet, _ = forecast_with_model(model_prophet, len(test_df), test_df['ds'].min())

        rmse, mae, mape = evaluate_forecast(test_df['y'].values, forecast_prophet['yhat'].values, "Prophet")
        log_metrics(test_df['ds'].max(), rmse, mae, mape)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("mape", mape)

    # LSTM model
    with mlflow.start_run():
        mlflow.set_tag("model", "lstm")
        model_lstm, scaler = train_lstm(train_df)
        forecast_lstm, _ = forecast_with_model(None, len(test_df), test_df['ds'].min(), df, model_lstm, scaler)

        rmse, mae, mape = evaluate_forecast(test_df['y'].values, forecast_lstm, "LSTM")
        log_metrics(test_df['ds'].max(), rmse, mae, mape)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("mape", mape)

    return model_prophet, model_lstm, forecast_prophet, forecast_lstm, test_df, mape
