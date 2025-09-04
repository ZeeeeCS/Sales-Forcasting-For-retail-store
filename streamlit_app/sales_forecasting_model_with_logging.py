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
def evaluate_forecast(y_true, y_pred, model_name=""):
    """Evaluates forecast and logs results."""
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
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