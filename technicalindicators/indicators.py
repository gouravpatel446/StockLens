"""
Technical Analysis Indicators Module
Provides functions to calculate various technical indicators for stock analysis
"""

import pandas as pd  # For data manipulation
import numpy as np   # For numerical operations


def calculate_rsi(data, period=14):
    """
    Calculate Relative Strength Index (RSI)
    
    RSI measures the magnitude of recent price changes to evaluate 
    overbought or oversold conditions.
    
    Formula: RSI = 100 - (100 / (1 + RS))
    Where RS = Average Gain / Average Loss
    
    Parameters:
    -----------
    data : pd.DataFrame
        DataFrame with 'Close' column containing closing prices
    period : int
        Period for RSI calculation (default: 14)
    
    Returns:
    --------
    pd.Series
        RSI values (0-100). Values above 70 suggest overbought, 
        below 30 suggest oversold
    
    Example:
    --------
    >>> rsi = calculate_rsi(stock_data, period=14)
    """
    
    # Calculate price changes (deltas)
    delta = data['Close'].diff()
    
    # Separate gains and losses
    # Gains: positive deltas, losses: absolute value of negative deltas
    gains = delta.where(delta > 0, 0)
    losses = -delta.where(delta < 0, 0)
    
    # Calculate average gains and average losses over the period
    avg_gain = gains.rolling(window=period).mean()
    avg_loss = losses.rolling(window=period).mean()
    
    # Calculate RS (Relative Strength)
    # Handle division by zero by using numpy
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    
    # Calculate RSI using the formula
    rsi = 100 - (100 / (1 + rs))
    
    return pd.Series(rsi, index=data.index)


def calculate_macd(data, fast=12, slow=26, signal=9):
    """
    Calculate MACD (Moving Average Convergence Divergence)
    
    MACD is a trend-following momentum indicator that shows the 
    relationship between two moving averages.
    
    Formula:
    - MACD Line = 12-day EMA - 26-day EMA
    - Signal Line = 9-day EMA of MACD
    - Histogram = MACD - Signal Line
    
    Parameters:
    -----------
    data : pd.DataFrame
        DataFrame with 'Close' column containing closing prices
    fast : int
        Period for fast EMA (default: 12)
    slow : int
        Period for slow EMA (default: 26)
    signal : int
        Period for signal line EMA (default: 9)
    
    Returns:
    --------
    tuple
        (MACD line, Signal line, Histogram)
        - MACD line: Fast EMA - Slow EMA
        - Signal line: EMA of MACD
        - Histogram: Difference between MACD and Signal
    
    Example:
    --------
    >>> macd, signal, histogram = calculate_macd(stock_data)
    """
    
    # Calculate exponential moving averages (EMA)
    # Fast EMA (12-day) - more responsive to recent prices
    ema_fast = data['Close'].ewm(span=fast, adjust=False).mean()
    
    # Slow EMA (26-day) - less responsive, smoother
    ema_slow = data['Close'].ewm(span=slow, adjust=False).mean()
    
    # MACD Line: Difference between fast and slow EMA
    macd_line = ema_fast - ema_slow
    
    # Signal Line: 9-day EMA of the MACD line
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    
    # Histogram: Difference between MACD and Signal line
    # Positive histogram = bullish, Negative = bearish
    histogram = macd_line - signal_line
    
    return macd_line, signal_line, histogram


def calculate_moving_averages(data):
    """
    Calculate Simple Moving Averages (SMA)
    
    Moving averages smooth out price data to identify trends more clearly.
    
    Formula: SMA = Sum of prices over period / Number of periods
    
    Parameters:
    -----------
    data : pd.DataFrame
        DataFrame with 'Close' column containing closing prices
    
    Returns:
    --------
    pd.DataFrame
        Original data with added columns:
        - MA50: 50-day Simple Moving Average
        - MA200: 200-day Simple Moving Average
    
    Note:
    -----
    - MA50 is used for short-term trend analysis
    - MA200 is used for long-term trend analysis
    - When MA50 > MA200, it's generally considered bullish (uptrend)
    - When MA50 < MA200, it's generally considered bearish (downtrend)
    
    Example:
    --------
    >>> data_with_ma = calculate_moving_averages(stock_data)
    >>> print(data_with_ma[['Close', 'MA50', 'MA200']])
    """
    
    # Create a copy to avoid modifying the original data
    data = data.copy()
    
    # Calculate 50-day Simple Moving Average
    # Used for intermediate-term trend analysis
    data['MA50'] = data['Close'].rolling(window=50).mean()
    
    # Calculate 200-day Simple Moving Average
    # Used for long-term trend analysis
    data['MA200'] = data['Close'].rolling(window=200).mean()
    
    return data


# Example usage and indicator combinations
def get_all_indicators(data):
    """
    Calculate all technical indicators for comprehensive analysis
    
    Parameters:
    -----------
    data : pd.DataFrame
        DataFrame with 'Close' column containing closing prices
    
    Returns:
    --------
    pd.DataFrame
        DataFrame with all calculated indicators
    
    Example:
    --------
    >>> full_analysis = get_all_indicators(stock_data)
    """
    
    # Start with the original data
    result = data.copy()
    
    # Add RSI
    result['RSI'] = calculate_rsi(data, period=14)
    
    # Add MACD components
    result['MACD'], result['MACD_Signal'], result['MACD_Histogram'] = calculate_macd(data)
    
    # Add Moving Averages
    result = calculate_moving_averages(result)
    
    return result
