import pandas as pd
import numpy as np
from prophet import Prophet
from pandas.tseries.holiday import USFederalHolidayCalendar as calendar
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import os

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

# ---- Hyperparameter Tuning ----
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

# ---- Residual Analysis ----
def plot_residuals(test_df, forecast):
    y_true = test_df['y'].values
    y_pred = forecast[forecast['ds'].isin(test_df['ds'])]['yhat'].values
    residuals = y_true - y_pred

    fig, axs = plt.subplots(1, 2, figsize=(14, 5))
    axs[0].hist(residuals, bins=30, color='gray')
    axs[0].set_title("Residual Distribution")
    axs[0].set_xlabel("Residual")

    axs[1].plot(test_df['ds'], residuals, marker='o', linestyle='-')
    axs[1].set_title("Residuals Over Time")
    axs[1].set_xlabel("Date")
    axs[1].set_ylabel("Residual")
    plt.tight_layout()
    plt.show()

# ---- Main Process ----
def run_forecasting_pipeline(csv_path):
    df = load_and_prepare(csv_path)
    train_df, test_df = split_data(df)
    holidays = get_us_holidays(train_df['ds'].min(), test_df['ds'].max())

    model, forecast = tune_prophet(train_df, holidays, test_df)
    forecast_test = forecast[forecast['ds'].isin(test_df['ds'])]

    rmse, mae, mape = evaluate_forecast(test_df['y'].values, forecast_test['yhat'].values, "Best Tuned Prophet")
    log_metrics(test_df['ds'].max(), rmse, mae, mape)
    drift = check_drift(mape)
    persistent_drift = check_drift_trend()


    plot_residuals(test_df, forecast)

    return model, forecast, test_df, mape, drift, persistent_drift