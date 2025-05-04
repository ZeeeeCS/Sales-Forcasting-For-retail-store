import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
from prophet import Prophet
from pandas.tseries.holiday import USFederalHolidayCalendar as calendar
import os
import warnings
warnings.filterwarnings('ignore')


def mean_absolute_percentage_error(y_true, y_pred):
    """Calculates MAPE given y_true and y_pred"""
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100
# ---- Helper Functions ----
def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

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

# ---- Load & Prepare Data ----
def load_and_prepare(filepath):
    data = pd.read_csv(filepath)
    data['Date'] = pd.to_datetime(data['Date'])
    daily_df = data.groupby('Date', as_index=False)['Units Sold'].sum()
    daily_df['day_of_week'] = daily_df['Date'].dt.dayofweek
    daily_df['month'] = daily_df['Date'].dt.month
    daily_df['is_weekend'] = daily_df['day_of_week'].isin([5, 6]).astype(int)
    daily_df = daily_df.rename(columns={'Date': 'ds', 'Units Sold': 'y'})
    return daily_df

# ---- Split Data ----
def split_data(df, split_date='2023-10-01'):
    train = df[df['ds'] < split_date]
    test = df[df['ds'] >= split_date]
    return train, test

# ---- Add US Holidays ----
def get_us_holidays(start_date, end_date):
    cal = calendar()
    holidays = cal.holidays(start=start_date, end=end_date, return_name=True)
    holiday_df = pd.DataFrame(data=holidays, columns=['holiday'])
    return holiday_df.reset_index().rename(columns={'index': 'ds'})

# ---- Train Prophet Model ----
def train_prophet(train_df, holidays=None):
    m = Prophet(holidays=holidays)
    m.add_regressor('day_of_week')
    m.add_regressor('month')
    m.add_regressor('is_weekend')
    m.fit(train_df[['ds', 'y', 'day_of_week', 'month', 'is_weekend']])
    return m

# ---- Forecast Future ----
def forecast_with_model(model, horizon, start_date):
    future = pd.date_range(start=start_date, periods=horizon, freq='D')
    future_df = pd.DataFrame({'ds': future})
    future_df['day_of_week'] = future_df['ds'].dt.dayofweek
    future_df['month'] = future_df['ds'].dt.month
    future_df['is_weekend'] = future_df['day_of_week'].isin([5, 6]).astype(int)
    forecast = model.predict(future_df)
    return forecast

# ---- Main Process (used in script or Streamlit) ----
def run_forecasting_pipeline(csv_path):
    df = load_and_prepare(csv_path)
    train_df, test_df = split_data(df)
    holidays = get_us_holidays(train_df['ds'].min(), test_df['ds'].max())
    model = train_prophet(train_df, holidays)
    forecast = forecast_with_model(model, len(test_df), test_df['ds'].min())
    forecast_test = forecast[forecast['ds'].isin(test_df['ds'])]

    rmse, mae, mape = evaluate_forecast(test_df['y'].values, forecast_test['yhat'].values, "Prophet + Holidays")
    log_metrics(test_df['ds'].max(), rmse, mae, mape)
    drift = check_drift(mape)
    persistent_drift = check_drift_trend()
    feedback_ready = feedback_available()

    return model, forecast, test_df, mape, drift, persistent_drift, feedback_ready

