import numpy as np
import pandas as pd
import yfinance as yf
from io import BytesIO
import matplotlib.pyplot as plt
import seaborn as sns
from kitetrader import Kite 
import pyotp
import pandas_ta as ta
import datetime as dt
from datetime import datetime, timedelta
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout

def compute_accuracy(actual, predicted):
    mse = mean_squared_error(actual, predicted)
    rmse = np.sqrt(mse)
    mape = mean_absolute_percentage_error(actual, predicted) * 100
    return mse, rmse, mape

def fetch_stock_data(ticker, start_date, end_date):
    """
    Fetch stock data from Yahoo Finance
    """
    stock_data = yf.download(ticker, start=start_date, end=end_date)
    return stock_data['Close'].values.reshape(-1, 1), stock_data.index


def prepare_data(data, time_steps):
    """
    Prepare data for LSTM model
    """
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)

    X, y = [], []
    for i in range(len(scaled_data) - time_steps):
        X.append(scaled_data[i:i+time_steps])
        y.append(scaled_data[i+time_steps])

    return np.array(X), np.array(y), scaler


def create_model(time_steps):
    """
    Create and compile LSTM model
    """
    model = Sequential([
        LSTM(units=50, return_sequences=True, input_shape=(time_steps, 1)),
        Dropout(0.2),
        LSTM(units=50, return_sequences=False),
        Dropout(0.2),
        Dense(units=1)
    ])

    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def generate_future_dates(last_date, num_days):
    """
    Generate future dates for predictions
    """
    future_dates = []
    current_date = last_date
    for _ in range(num_days):
        current_date += timedelta(days=1)
        # Skip weekends
        while current_date.weekday() > 4:
            current_date += timedelta(days=1)
        future_dates.append(current_date)
    return pd.DatetimeIndex(future_dates)

def predict_future(model, last_sequence, scaler, n_future):
    """
    Predict future values
    """
    future_predictions = []
    current_sequence = last_sequence.copy()

    for _ in range(n_future):
        # Get prediction for next day
        current_prediction = model.predict(current_sequence.reshape(1, *current_sequence.shape))
        future_predictions.append(current_prediction[0])

        # Update sequence for next prediction
        current_sequence = np.roll(current_sequence, -1)
        current_sequence[-1] = current_prediction

    future_predictions = np.array(future_predictions)
    future_predictions = scaler.inverse_transform(future_predictions)

    return future_predictions

def plot_predictions_with_future(dates, actual_values, train_predictions, test_predictions,
                               future_dates, future_predictions, train_size, time_steps, ticker):
    """
    Plot the actual vs predicted values including future predictions
    """
    plt.figure(figsize=(15, 8))

    # Adjust dates array to match the actual values length
    plot_dates = dates[time_steps:]

    # Plot actual values
    plt.plot(plot_dates, actual_values.flatten(),
             label='Actual Prices', color='blue', linewidth=2)

    # Plot training predictions
    train_dates = plot_dates[:train_size]
    plt.plot(train_dates, train_predictions.flatten(),
             label='Training Predictions', color='green', linestyle='--', linewidth=2)

    # Plot testing predictions
    test_dates = plot_dates[train_size:]
    plt.plot(test_dates, test_predictions.flatten(),
             label='Testing Predictions', color='red', linestyle='--', linewidth=2)

    # Plot future predictions
    plt.plot(future_dates, future_predictions.flatten(),
             label='Future Predictions', color='purple', linestyle='--', linewidth=2)

    # Add vertical lines to separate periods
    plt.axvline(x=plot_dates[train_size], color='gray', linestyle='-',
                label='Train/Test Split')
    plt.axvline(x=plot_dates[-1], color='gray', linestyle=':',
                label='Prediction Start')

    # Add confidence interval for future predictions
    future_std = np.std(test_predictions - actual_values[train_size:])
    plt.fill_between(future_dates,
                    (future_predictions - future_std * 2).flatten(),
                    (future_predictions + future_std * 2).flatten(),
                    color='purple', alpha=0.1, label='95% Confidence Interval')

    # Customize the plot
    plt.title(f'{ticker} Stock Price Prediction with Future Trend', fontsize=16)
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Stock Price (USD)', fontsize=12)
    plt.legend(loc='upper left')
    plt.grid(True)

    # Rotate x-axis labels
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def predict_stock_prices(ticker, start_date, end_date, time_steps=60, train_split=0.8, future_days=30):
    """
    Main function to predict stock prices including future trends
    """

    # Fetch data
    data, dates = fetch_stock_data(ticker, start_date, end_date)

    # Prepare data
    X, y, scaler = prepare_data(data, time_steps)

    # Split into training and testing sets
    train_size = int(len(X) * train_split)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    # Create and train model
    model = create_model(time_steps)
    history = model.fit(X_train, y_train, epochs=50, batch_size=32,
                       validation_split=0.1, verbose=1)

    # Make predictions
    train_predictions = model.predict(X_train)
    test_predictions = model.predict(X_test)

    # Predict future values
    last_sequence = X[-1]
    future_predictions = predict_future(model, last_sequence, scaler, future_days)

    # Generate future dates
    future_dates = generate_future_dates(dates[-1], future_days)

    # Inverse transform predictions
    train_predictions = scaler.inverse_transform(train_predictions)
    test_predictions = scaler.inverse_transform(test_predictions)
    actual_values = scaler.inverse_transform(y)

    # Compute accuracy
    train_mse, train_rmse, train_mape = compute_accuracy(actual_values[:train_size], train_predictions)
    test_mse, test_rmse, test_mape = compute_accuracy(actual_values[train_size:], test_predictions)

    print("\nModel Accuracy:")
    print(f"Training MSE: {train_mse:.4f}, RMSE: {train_rmse:.4f}, MAPE: {train_mape:.2f}%")
    print(f"Testing MSE: {test_mse:.4f}, RMSE: {test_rmse:.4f}, MAPE: {test_mape:.2f}%")

    # Plot results
    plot_predictions_with_future(dates, actual_values, train_predictions, test_predictions,
                               future_dates, future_predictions, train_size, time_steps, ticker)

    return {
        'train_predictions': train_predictions,
        'test_predictions': test_predictions,
        'future_predictions': future_predictions,
        'actual_values': actual_values,
        'dates': dates[time_steps:],
        'future_dates': future_dates,
        'accuracy': {
            'train_mse': train_mse, 'train_rmse': train_rmse, 'train_mape': train_mape,
            'test_mse': test_mse, 'test_rmse': test_rmse, 'test_mape': test_mape
        }
    }
if __name__ == "__main__":
    # Ask user for inputs
    ticker = input("Enter the stock ticker (e.g., ^NSEI for Nifty 50): ").strip()
    start_date = input("Enter the start date (YYYY-MM-DD): ").strip()
    end_date = input("Enter the end date (YYYY-MM-DD): ").strip()

    time_steps = 60  # Number of time steps to look back
    future_days = 30  # Number of days to predict into the future

    # Get predictions and plots
    results = predict_stock_prices(ticker, start_date, end_date, time_steps,
                                   future_days=future_days)

    # Print future predictions
    print("\nFuture Predictions:")
    for date, price in zip(results['future_dates'], results['future_predictions']):
        print(f"{date.strftime('%Y-%m-%d')}: {price[0]:.2f}")