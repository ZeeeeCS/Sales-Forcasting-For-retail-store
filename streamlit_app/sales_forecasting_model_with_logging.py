import pandas as pd
import numpy as np
import os
import logging
import time
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.statespace.sarimax import SARIMAX

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.optimizers import Adam
from prophet import Prophet

import mlflow
import mlflow.keras
import mlflow.prophet
import mlflow.statsmodels

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Evaluation & Helpers ---
# In sales_forecasting_model_with_logging.py

def evaluate_forecast(y_true, y_pred, model_name=""):
    """Evaluates forecast and logs results, handling division-by-zero for MAPE."""
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    
    # FIX: Replace actual values of 0 with a very small number to prevent division by zero
    y_true_safe = np.where(y_true == 0, 1e-6, y_true)
    mape = np.mean(np.abs((y_true - y_pred) / y_true_safe)) * 100
    
    logger.info(f"{model_name} Evaluation: RMSE={rmse:.2f}, MAE={mae:.2f}, MAPE={mape:.2f}%")
    return rmse, mae, mape


def create_time_features(df):
    """Creates time series features from a datetime index."""
    df_feat = df.copy()
    df_feat['dayofweek'] = df_feat.index.dayofweek
    df_feat['quarter'] = df_feat.index.quarter
    df_feat['month'] = df_feat.index.month
    df_feat['year'] = df_feat.index.year
    df_feat['dayofyear'] = df_feat.index.dayofyear
    return df_feat

def create_sequences(data, seq_len):
    """Creates sequences for LSTM input."""
    x, y = [], []
    for i in range(len(data) - seq_len):
        x.append(data[i:i + seq_len])
        y.append(data[i + seq_len])
    return np.array(x), np.array(y)

# --- Model Implementations ---

def run_sarima_model(df):
    """Trains and evaluates a SARIMA model."""
    logging.info("--- Running SARIMA model ---")
    df_sarima = df['y'].copy()
    
    train_size = int(len(df_sarima) * 0.8)
    train, test = df_sarima[:train_size], df_sarima[train_size:]

    try:
        # Standard seasonal parameters for daily data (s=7 for weekly seasonality)
        # These (p,d,q)(P,D,Q,s) orders are a common starting point.
        model = SARIMAX(train, order=(1, 1, 1), seasonal_order=(1, 1, 0, 7))
        results = model.fit(disp=False)
        
        predictions = results.get_prediction(start=len(train), end=len(train) + len(test) - 1, dynamic=False)
        y_pred = predictions.predicted_mean
        
        rmse, mae, mape = evaluate_forecast(test.values, y_pred.values, "SARIMA")
        
        mlflow.log_params({"sarima_order": "(1,1,1)", "sarima_seasonal_order": "(1,1,0,7)"})
        mlflow.log_metrics({"rmse_sarima": rmse, "mae_sarima": mae, "mape_sarima": mape})
        mlflow.statsmodels.log_model(results, artifact_path="sarima-model")
        
        return results, y_pred, test, mape
    except Exception as e:
        logging.error(f"SARIMA model failed: {e}")
        return None, None, None, float('inf')


def run_prophet_model(df_features):
    """Trains and evaluates a Prophet model using extra features."""
    logging.info("--- Running Prophet model with Features ---")
    
    df_prophet = df_features.reset_index().copy()
    feature_names = ['dayofweek', 'quarter', 'month', 'year', 'dayofyear']
    
    train_size = int(len(df_prophet) * 0.8)
    train_df, test_df = df_prophet[:train_size], df_prophet[train_size:]

    try:
        model = Prophet()
        for feature in feature_names:
            model.add_regressor(feature)
        
        model.fit(train_df)
        
        future = test_df.drop(columns=['y'])
        forecast_df = model.predict(future)
        
        y_true = test_df['y'].values
        y_pred = forecast_df['yhat'].values
        rmse, mae, mape = evaluate_forecast(y_true, y_pred, "Prophet")

        mlflow.log_metrics({"rmse_prophet": rmse, "mae_prophet": mae, "mape_prophet": mape})
        mlflow.prophet.log_model(model, artifact_path="prophet-model")

        return model, forecast_df, test_df, mape
    except Exception as e:
        logging.error(f"Prophet model failed: {e}")
        return None, None, None, float('inf')


# In sales_forecasting_model_with_logging.py

# In sales_forecasting_model_with_logging.py

def run_lstm_model(df, use_differencing=True):
    """Trains and evaluates a univariate LSTM model, with an option for differencing."""
    logging.info(f"--- Running LSTM model (Differencing: {use_differencing}) ---")
    seq_len = 14
    df_lstm = df[['y']].copy()
    
    original_indices = df_lstm.index # Store original indices before differencing

    if use_differencing:
        df_lstm['y'] = df_lstm['y'].diff()
        # FIX: Drop the NaN value created by the differencing operation
        df_lstm.dropna(inplace=True)

    if len(df_lstm) < seq_len + 5: # Check length after potentially dropping a row
        logging.error("LSTM Error: Not enough data points after differencing.")
        return None, None, None, None, float('inf')

    train_size = int(len(df_lstm) * 0.8)
    # The rest of the function remains the same...
    train_data, test_data = df_lstm[:train_size], df_lstm[train_size:]
    scaler = MinMaxScaler()
    scaler.fit(train_data)
    scaled_train = scaler.transform(train_data)
    scaled_test = scaler.transform(test_data)
    x_train, y_train = create_sequences(scaled_train, seq_len)
    x_test, y_test = create_sequences(scaled_test, seq_len)
    if x_train.size == 0 or x_test.size == 0:
        logging.error("LSTM Error: Not enough data to create train/test sequences.")
        return None, None, None, None, float('inf')
    x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], 1))
    x_test = x_test.reshape((x_test.shape[0], x_test.shape[1], 1))
    try:
        model = Sequential([
            LSTM(64, activation='relu', input_shape=(seq_len, 1), return_sequences=True),
            LSTM(32, activation='relu'),
            Dense(1)
        ])
        optimizer = Adam(learning_rate=0.001) 
        model.compile(optimizer, loss='mse')
        model.fit(x_train, y_train, epochs=20, verbose=0, shuffle=False)
        preds_scaled = model.predict(x_test)
        preds_inv_diff = scaler.inverse_transform(preds_scaled)

        # Correctly align with the original dataframe for inverse differencing
        last_train_value_index = df.index.get_loc(train_data.index[-1])
        last_train_value = df['y'].iloc[last_train_value_index]

        y_test_inv = df['y'][df.index.isin(test_data.index[seq_len:])].values

        if use_differencing:
            predictions_cumulative = np.cumsum(np.insert(preds_inv_diff.flatten(), 0, last_train_value))
            preds_inv = predictions_cumulative[1:]
        else:
            preds_inv = preds_inv_diff.flatten()

        preds_inv[preds_inv < 0] = 0
        min_len = min(len(y_test_inv), len(preds_inv))
        y_test_aligned = y_test_inv[:min_len]
        preds_aligned = preds_inv[:min_len]

        test_dates = test_data.index[seq_len:seq_len + min_len]
        
        rmse, mae, mape = evaluate_forecast(y_test_aligned, preds_aligned, "LSTM")

        mlflow.log_param("lstm_seq_len", seq_len)
        mlflow.log_param("lstm_use_differencing", use_differencing)
        mlflow.log_metrics({"rmse_lstm": rmse, "mae_lstm": mae, "mape_lstm": mape})
        mlflow.keras.log_model(model, artifact_path="lstm-model")
        return model, preds_aligned, y_test_aligned, test_dates, mape
    except Exception as e:
        logging.error(f"LSTM model failed: {e}")
        return None, None, None, None, float('inf')
    
def load_and_prepare(filepath):
    """Loads data, strips column whitespace, aggregates, and sets index."""
    logging.info(f"Loading and preparing data from: {filepath}")
    try:
        if hasattr(filepath, 'seek'):
            filepath.seek(0)
        df = pd.read_csv(filepath)

        # FIX: Strip whitespace from column names to handle minor formatting errors
        df.columns = df.columns.str.strip()

        # Check if the required columns exist after stripping whitespace
        if 'Date' not in df.columns or 'Units Sold' not in df.columns:
            logger.error("Data loading error: CSV must contain 'Date' and 'Units Sold' columns.")
            return None

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
    except Exception as e:
        logging.error(f"Data loading error: {e}")
        return None
# --- Main Pipeline Function ---
def run_forecasting_pipeline(csv_path, experiment_name="SalesForecastingExperiment"):
    mlflow.set_experiment(experiment_name)
    results_summary = {}

    with mlflow.start_run() as run:
        results_summary["mlflow_run_id"] = run.info.run_id
        results_summary["mlflow_experiment_id"] = run.info.experiment_id

        try:
            base_df = load_and_prepare(csv_path)
            if base_df is None:
                raise ValueError("Data loading failed.")
            
            # 1. Create features
            df_features = create_time_features(base_df)
            mlflow.set_tag("dataset_source", os.path.basename(str(csv_path)))

            # 2. Run models
            _, sarima_preds, sarima_actuals, sarima_mape = run_sarima_model(base_df)
            _, prophet_fcst, _, prophet_mape = run_prophet_model(df_features)
            _, lstm_preds, lstm_actuals, lstm_dates, lstm_mape = run_lstm_model(base_df)

            # 3. Compile results
            results_summary.update({
                "sarima_mape": sarima_mape,
                "sarima_predictions": sarima_preds,
                "sarima_actuals": sarima_actuals,
                "prophet_mape": prophet_mape,
                "prophet_forecast_df": prophet_fcst,
                "lstm_mape": lstm_mape,
                "lstm_predictions": lstm_preds,
                "lstm_actuals": lstm_actuals,
                "lstm_dates": lstm_dates,
            })
            mlflow.log_param("pipeline_status", "completed")

        except Exception as e:
            error_msg = f"Pipeline failed: {e}"
            logger.error(error_msg, exc_info=True)
            results_summary["error"] = error_msg
            mlflow.log_param("pipeline_status", "failed_exception")

    logging.info("Forecasting pipeline finished.")
    return results_summary

# --- Plotting Functions ---
def plot_forecast(df, y_pred, y_test, title="Forecast"):
    """Generic plotting function for test vs predictions."""
    fig, ax = plt.subplots(figsize=(14, 7))
    sns.set(style="whitegrid")
    ax.plot(df.index, df['y'], label='Historical Data', color='blue', alpha=0.8)
    ax.plot(y_test.index, y_test, label='Actual Test Data', color='green', marker='o', linestyle='None', markersize=5)
    ax.plot(y_test.index, y_pred, label='Model Predictions', color='red', marker='x', linestyle='--')
    ax.set_title(title, fontsize=16)
    ax.legend()
    fig.autofmt_xdate()
    return fig

def plot_prophet_forecast(df, forecast_df, title="Prophet Forecast"):
    """Plots Prophet forecast specifically."""
    fig, ax = plt.subplots(figsize=(14, 7))
    sns.set(style="whitegrid")
    # Plotting historical data from the original df
    test_start_date = forecast_df['ds'].min()
    ax.plot(df.index, df['y'], label='Historical Data', color='blue', alpha=0.8)
    # Plotting actuals for the test period
    actuals_test = df[df.index >= test_start_date]
    ax.plot(actuals_test.index, actuals_test['y'], label='Actual Test Data', color='green', marker='o', linestyle='None', markersize=5)
    # Plotting prophet forecast
    ax.plot(forecast_df['ds'], forecast_df['yhat'], label='Forecast', color='orange', linestyle='--')
    ax.fill_between(forecast_df['ds'], forecast_df['yhat_lower'], forecast_df['yhat_upper'], color='orange', alpha=0.2, label='Uncertainty Interval')
    ax.set_title(title, fontsize=16)
    ax.legend()
    fig.autofmt_xdate()
    return fig
def plot_lstm_forecast(df, preds_inv, y_test_inv, test_dates, title="LSTM Forecast"):
    """Plots the original data and LSTM's forecast."""
    fig, ax = plt.subplots(figsize=(14, 7))
    sns.set(style="whitegrid")
    ax.plot(df.index, df['y'], label='Historical Data', color='blue', alpha=0.8)
    ax.plot(test_dates, y_test_inv, label='Actual Test Data', color='green', marker='o', linestyle='None', markersize=5)
    ax.plot(test_dates, preds_inv, label='LSTM Predictions', color='red', marker='x', linestyle='--')
    ax.set_title(title, fontsize=16)
    ax.set_xlabel("Date")
    ax.set_ylabel("Units Sold")
    ax.legend()
    fig.autofmt_xdate()
    return fig