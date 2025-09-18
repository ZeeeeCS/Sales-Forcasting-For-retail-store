# Sales-Forcasting-For-retail-store

### 🎯 Overview

This project provides a **Streamlit** dashboard for forecasting daily retail sales. It allows users to upload their own sales data in a CSV format and instantly generate sales forecasts using two powerful machine learning models: **Prophet** (by Meta) , **Sarima** and a **Long Short-Term Memory (LSTM)** neural network. The dashboard evaluates the performance of each model and provides key metrics to help you choose the best one for your data.

### 🚀 Features

  * **Interactive UI**: A user-friendly web interface built with **Streamlit** for easy file uploads and result viewing.
  * **Dual-Model Forecasting**: Runs two distinct models—a time-series decomposition model (**Prophet**) and a deep learning model (**LSTM**) for comparative analysis.
  * **Performance Metrics**: Automatically calculates and displays key metrics like **Mean Absolute Percentage Error (MAPE)**, **Root Mean Squared Error (RMSE)**, and **Mean Absolute Error (MAE)**.
  * **Data Drift Detection**: Implements a basic drift detection mechanism to alert you if the model's performance on new data degrades beyond a set threshold.
  * **MLflow Integration**: Logs all model runs, parameters, and metrics to an **MLflow** experiment for easy tracking, comparison, and reproducibility.
  * **Visualizations**: Generates informative plots showing the original data, forecast, and uncertainty intervals for a clear visual understanding of the model's predictions,
    
   Model Comparison: Visualizes the performance of SARIMA, Prophet, and LSTM against historical test data.

   Future Forecast: After identifying the best model (SARIMA), it generates and plots a forecast for the next six months, showing the expected trend and seasonality.

   Uncertainty Intervals: Includes uncertainty intervals in the plots to provide a realistic range of potential future outcomes.

-----

### 📦 Prerequisites

Ensure you have **Python 3.8** or newer installed.

You can install all the necessary libraries using `pip`:

```bash
pip install streamlit pandas numpy scikit-learn tensorflow keras prophet mlflow matplotlib seaborn
```

-----

### ⚙️ How to Run

1.  **Clone the Repository (or save the files):**
    Ensure you have `app.py` and `forecasting_pipeline.py` in the same directory.

2.  **Open your Terminal or Command Prompt:**
    Navigate to the project directory where your files are saved.

3.  **Run the Streamlit App:**
    Execute the following command:

    ```bash
    streamlit run app.py
    ```

4.  **Access the Dashboard:**
    Your web browser will automatically open a new tab with the dashboard. If it doesn't, navigate to the local URL displayed in your terminal (usually `http://localhost:8501`).

-----

### 📂 Input Data Format

The application expects a CSV file with two required columns:

  * **`Date`**: The date of the sales record. This column should be in a recognized date format (e.g., `YYYY-MM-DD`).
  * **`Units Sold`**: The numerical value of units sold on that date.

| Date       | Units Sold |
| :--------- | :--------- |
| 2023-01-01 | 150        |
| 2023-01-02 | 165        |
| 2023-01-03 | 142        |
| ...        | ...        |

-----

### 🧠 Model Explanations

#### 1\. Prophet Model

Prophet is a forecasting procedure developed by Meta. It's designed to handle time-series data with strong seasonal effects and historical trends. Prophet works by decomposing the time series into three main components:

  * **Trend**: Models a non-periodic change in the value of the time series.
  * **Seasonality**: Models periodic changes (e.g., weekly, yearly).
  * **Holidays**: Accounts for the impact of holidays.

The dashboard runs two versions: one with default parameters and another with a customized set of **hyperparameters** for optimized performance.

#### 2\. LSTM Model

**Long Short-Term Memory (LSTM)** is a type of recurrent neural network (**RNN**) well-suited for learning from sequence data, such as time series. LSTMs are capable of learning long-term dependencies in the data, making them highly effective for forecasting.

This model processes the data by creating sequences and uses a deep learning approach to learn complex, non-linear patterns.

-----

### 🛠️ Key Technical Components

  * **`app.py`**: The main **Streamlit** application file. This file manages the user interface, handles file uploads, calls the forecasting pipeline, and displays the results and visualizations.
  * **`forecasting_pipeline.py`**: A modular Python script containing the core logic for data preparation, model training (Prophet and LSTM), evaluation, and logging. This separation of concerns keeps the `app.py` file clean and focused on presentation.
  * **MLflow**: An open-source platform for managing the machine learning lifecycle. It automatically logs each model run, allowing you to compare the performance of different models and hyperparameters. The MLflow tracking URI is set locally by default, creating an `mlruns` directory in your project folder. You can view your experiments by running `mlflow ui` in your terminal.
