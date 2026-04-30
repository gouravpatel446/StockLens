"""
Machine Learning Models for Stock Price Prediction - 90 Day Forecast
This module contains ML models to predict future stock prices for the next 3 months
"""

import numpy as np  # For numerical operations and arrays
from sklearn.linear_model import LinearRegression  # Our prediction model
from sklearn.preprocessing import MinMaxScaler  # For scaling data (not used here but imported)
from sklearn.metrics import mean_squared_error  # To measure prediction accuracy
import pandas as pd  # For handling data and dates


def calculate_statistics(hist):
    """
    Calculate comprehensive statistics for the stock data

    Parameters:
    -----------
    hist : pd.DataFrame
        Historical stock data

    Returns:
    --------
    dict
        Dictionary containing various statistical measures
    """
    stats = {}

    # Basic price statistics
    close_prices = hist['Close']
    stats['current_price'] = close_prices.iloc[-1]
    stats['price_range'] = {
        'high': close_prices.max(),
        'low': close_prices.min(),
        'avg': close_prices.mean()
    }

    # Returns statistics
    returns = close_prices.pct_change().dropna()
    stats['returns'] = {
        'mean_daily': returns.mean(),
        'std_daily': returns.std(),
        'annual_volatility': returns.std() * np.sqrt(252),
        'sharpe_ratio': returns.mean() / returns.std() * np.sqrt(252) if returns.std() != 0 else 0,
        'max_drawdown': (close_prices / close_prices.expanding().max() - 1).min()
    }

    # Volume statistics (if available)
    if 'Volume' in hist.columns:
        volume = hist['Volume']
        stats['volume'] = {
            'avg_daily': volume.mean(),
            'total_volume': volume.sum(),
            'volume_trend': 'increasing' if volume.tail(10).mean() > volume.tail(20).head(10).mean() else 'decreasing'
        }

    # Price momentum
    stats['momentum'] = {
        '1_week': (close_prices.iloc[-1] / close_prices.iloc[-5] - 1) * 100 if len(close_prices) >= 5 else 0,
        '1_month': (close_prices.iloc[-1] / close_prices.iloc[-20] - 1) * 100 if len(close_prices) >= 20 else 0,
        '3_month': (close_prices.iloc[-1] / close_prices.iloc[-60] - 1) * 100 if len(close_prices) >= 60 else 0
    }

    return stats


def get_technical_signals(hist):
    """
    Generate technical signals based on price action and indicators

    Parameters:
    -----------
    hist : pd.DataFrame
        Historical stock data with technical indicators

    Returns:
    --------
    dict
        Dictionary containing technical signals and their strength
    """
    signals = {}

    close_prices = hist['Close']
    current_price = close_prices.iloc[-1]

    # Trend signals
    if len(close_prices) >= 50:
        ma50 = close_prices.rolling(50).mean().iloc[-1]
        ma200 = close_prices.rolling(200).mean().iloc[-1] if len(close_prices) >= 200 else close_prices.rolling(50).mean().iloc[-20]

        signals['trend'] = {
            'direction': 'bullish' if current_price > ma50 > ma200 else 'bearish' if current_price < ma200 else 'sideways',
            'strength': 'strong' if abs(current_price - ma50) / ma50 > 0.05 else 'moderate' if abs(current_price - ma50) / ma50 > 0.02 else 'weak'
        }

    # Momentum signals
    if len(close_prices) >= 14:
        rsi = 100 - (100 / (1 + (close_prices.diff().clip(lower=0).rolling(14).mean() /
                                close_prices.diff().clip(upper=0).abs().rolling(14).mean())))
        current_rsi = rsi.iloc[-1]

        signals['rsi'] = {
            'value': current_rsi,
            'signal': 'overbought' if current_rsi > 70 else 'oversold' if current_rsi < 30 else 'neutral',
            'strength': 'strong' if current_rsi > 75 or current_rsi < 25 else 'moderate' if current_rsi > 65 or current_rsi < 35 else 'weak'
        }

    # Volatility signals
    if len(close_prices) >= 20:
        returns = close_prices.pct_change().dropna()
        volatility = returns.std() * np.sqrt(252)

        signals['volatility'] = {
            'level': volatility,
            'assessment': 'high' if volatility > 0.4 else 'moderate' if volatility > 0.2 else 'low',
            'risk': 'high' if volatility > 0.4 else 'medium' if volatility > 0.25 else 'low'
        }

    # Volume signals (if available)
    if 'Volume' in hist.columns and len(hist) >= 20:
        volume = hist['Volume']
        avg_volume = volume.rolling(20).mean().iloc[-1]
        current_volume = volume.iloc[-1]

        signals['volume'] = {
            'current': current_volume,
            'avg_20d': avg_volume,
            'signal': 'high' if current_volume > avg_volume * 1.5 else 'normal' if current_volume > avg_volume * 0.8 else 'low',
            'interpretation': 'strong interest' if current_volume > avg_volume * 1.5 else 'normal activity' if current_volume > avg_volume * 0.8 else 'weak interest'
        }

    return signals


def get_recent_ohlcv(hist, periods=10):
    """
    Get recent OHLCV data for the last N periods

    Parameters:
    -----------
    hist : pd.DataFrame
        Historical stock data
    periods : int
        Number of recent periods to return (default: 10)

    Returns:
    --------
    pd.DataFrame
        Recent OHLCV data formatted nicely
    """
    # Get the last N periods
    recent_data = hist.tail(periods).copy()

    # Format the data
    formatted_data = pd.DataFrame({
        'Date': recent_data.index.strftime('%Y-%m-%d'),
        'Open': recent_data['Open'].round(2),
        'High': recent_data['High'].round(2),
        'Low': recent_data['Low'].round(2),
        'Close': recent_data['Close'].round(2),
        'Volume': recent_data['Volume'] if 'Volume' in recent_data.columns else None
    })

    # Add change and change % columns
    if len(recent_data) > 1:
        formatted_data['Change'] = recent_data['Close'].diff().round(2)
        formatted_data['Change %'] = ((recent_data['Close'].pct_change()) * 100).round(2)
    else:
        formatted_data['Change'] = 0.0
        formatted_data['Change %'] = 0.0

    return formatted_data


def predict_linear(hist, forecast_days=90):
    """
    Predict future stock prices using simple Linear Regression for the next 90 days

    This function uses a straight line to predict where stock prices might go next!
    It's like drawing a trend line and extending it into the future for a 3-month period.

    🎯 How it works:
    - Uses historical closing prices
    - Treats each day as a number (Day 1, Day 2, Day 3...)
    - Finds the best straight line that fits the price data
    - Extends that line to predict future prices for 90 days

    📊 Parameters:
    -----------
    hist : pd.DataFrame
        Historical stock data with 'Close' column and date index
    forecast_days : int
        How many days into the future to predict (default: 90 for 3-month forecast)

    🎉 Returns:
    --------
    tuple
        (future_dates, predicted_prices, model_accuracy)
        - future_dates: List of dates for predictions
        - predicted_prices: Predicted prices for those dates
        - model_accuracy: How accurate the model is (lower is better)

    🚀 Example:
    --------
    >>> dates, prices, accuracy = predict_linear(stock_data, forecast_days=90)
    >>> print(f"Model accuracy: ₹{accuracy:.2f} RMSE")
    """

    # 🧹 Step 1: Clean and prepare the data
    print("🧹 Preparing data for prediction...")

    # Get only the closing prices and remove any missing values
    # We use .copy() to avoid changing the original data
    df = hist[['Close']].copy().dropna()

    # 🎯 Step 2: Create our X (input) and Y (target) data
    print("🎯 Setting up the prediction model...")

    # Convert dates to simple numbers (Day 0, Day 1, Day 2...)
    # This is like saying "Day 1 had this price, Day 2 had that price..."
    df['Day'] = np.arange(len(df))

    # X = Our input (the day numbers)
    # Y = What we want to predict (the closing prices)
    X = df[['Day']].values  # Shape: (number_of_days, 1)
    y = df['Close'].values  # Shape: (number_of_days,)

    # 🤖 Step 3: Create and train the Linear Regression model
    print("🤖 Training the prediction model...")

    # Create a Linear Regression model (finds the best straight line)
    model = LinearRegression()

    # Train the model on ALL our historical data
    # This finds the line that best fits our price data
    model.fit(X, y)

    # 📊 Step 4: Test how good our model is
    print("📊 Testing model accuracy...")

    # Use the last 20% of data to test our predictions
    # Split point: 80% for training, 20% for testing
    split_point = int(len(X) * 0.8)

    # Get the actual prices for the test period
    actual_test_prices = y[split_point:]

    # Predict prices for the test period
    predicted_test_prices = model.predict(X[split_point:])

    # Calculate Root Mean Square Error (RMSE)
    # This tells us how far off our predictions are (in dollars)
    # Lower RMSE = Better predictions
    rmse = np.sqrt(mean_squared_error(actual_test_prices, predicted_test_prices))

    # 🔮 Step 5: Predict the future!
    print(f" Predicting next {forecast_days} days (3-month forecast)...")

    # Get the total number of days we have data for
    last_day_number = len(df)

    # Create day numbers for future predictions
    # Example: If we have 100 days of data, and want 7 days prediction:
    # future_days = [100, 101, 102, 103, 104, 105, 106]
    future_day_numbers = np.arange(last_day_number, last_day_number + forecast_days)

    # Reshape for sklearn (needs 2D array)
    future_X = future_day_numbers.reshape(-1, 1)

    # 🎯 Make the predictions!
    future_predicted_prices = model.predict(future_X)

    # 📅 Step 6: Create future dates
    print("📅 Creating future date labels...")

    # Get the last date we have data for
    last_date = df.index[-1]

    # Create future business days (skips weekends)
    # periods = forecast_days + 1, then we take [1:] to skip the last_date
    future_dates = pd.date_range(
        start=last_date,
        periods=forecast_days + 1,
        freq='B'  # Business days only
    )[1:]  # Skip the first date (which is last_date)

    # 🎉 Step 7: Return the results
    print(" 90-day prediction complete!")
    print(f"Model accuracy: ₹{rmse:.2f} RMSE (lower is better)")
    return future_dates, future_predicted_prices, rmse


def get_prediction_summary(future_dates, future_prices, rmse):
    """
    Create a nice summary of the predictions

    📊 Parameters:
    -----------
    future_dates : pd.DatetimeIndex
        The future dates for predictions
    future_prices : np.ndarray
        The predicted prices
    rmse : float
        Model accuracy (Root Mean Square Error)

    🎉 Returns:
    --------
    dict
        Summary with predictions and accuracy info
    """

    # Create a summary dictionary
    summary = {
        'predictions': [],
        'accuracy': {
            'rmse': rmse,
            'rmse_interpretation': 'Lower is better (typical range: ₹1-50 for stocks)'
        },
        'model_type': 'Linear Regression',
        'forecast_period': len(future_dates)
    }

    # Add each prediction
    for date, price in zip(future_dates, future_prices):
        summary['predictions'].append({
            'date': date.strftime('%Y-%m-%d'),
            'predicted_price': round(price, 2)
        })

    return summary


# 🚀 Example usage (uncomment to test)
if __name__ == "__main__":
    # This would be used when testing the module directly
    print("📈 Linear Regression Stock Predictor - 90 Day Forecast")
    print("Use predict_linear(hist_data) to make predictions!")
    print("Example: dates, prices, accuracy = predict_linear(stock_data, forecast_days=90)")