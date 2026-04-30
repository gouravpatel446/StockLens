"""
🧠 LSTM Stock Price Predictor — Advanced AI Forecasting
═══════════════════════════════════════════════════════════
Uses cutting-edge Long Short-Term Memory neural networks to predict future stock prices.
This AI model learns complex patterns in historical data to forecast market trends.

🚀 Features:
   • 3-Layer Stacked LSTM Architecture
   • Batch Normalization for stable training
   • Dropout regularization to prevent overfitting
   • Early stopping and adaptive learning rates
   • 90-day business day forecasts
   • Comprehensive performance metrics

⚡ Performance: Advanced AI with 'memory' that learns market patterns
"""

import random
import warnings
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error

# TensorFlow imports moved inside functions to avoid slow startup
# warnings.filterwarnings("ignore")


def _set_seeds(seed: int) -> None:
    """
    🎲 Initialize Random Seeds for Reproducible AI Training

    Ensures consistent results across training runs by controlling
    randomness in NumPy, Python's random module, and TensorFlow.
    This makes the AI predictions deterministic and reproducible.
    """
    # Import TensorFlow here to avoid slow startup
    import tensorflow as tf

    np.random.seed(seed)      # 🎯 NumPy random operations
    random.seed(seed)         # 🐍 Python's random module
    tf.random.set_seed(seed)  # 🤖 TensorFlow operations


def _build_model(lookback: int):
    """
    🏗️ Build Advanced LSTM Neural Network Architecture

    Creates a sophisticated 3-layer stacked LSTM model with:
    • Multiple LSTM layers for capturing long-term dependencies
    • Batch normalization for training stability
    • Dropout layers to prevent overfitting
    • Dense layers for final prediction refinement

    Architecture: 3×LSTM → BatchNorm → Dropout | 2×Dense → Linear Output
    Total parameters: ~250K (sophisticated but not overkill)
    """
    # Import TensorFlow here to avoid slow startup
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
    from tensorflow.keras.optimizers import Adam

    # 🧠 Build the neural network architecture
    model = Sequential([
        # 🎯 Layer 1: First LSTM layer (captures basic patterns)
        LSTM(100, return_sequences=True, input_shape=(lookback, 1)),
        BatchNormalization(),  # Stabilizes training
        Dropout(0.3),         # Prevents overfitting (30% dropout)

        # 🎯 Layer 2: Second LSTM layer (learns complex relationships)
        LSTM(100, return_sequences=True),
        BatchNormalization(),
        Dropout(0.3),

        # 🎯 Layer 3: Final LSTM layer (makes final predictions)
        LSTM(50, return_sequences=False),  # No sequences returned
        BatchNormalization(),
        Dropout(0.3),

        # 🧮 Dense layers for final processing
        Dense(50, activation="relu"),    # Feature refinement
        Dropout(0.2),                    # Light regularization
        Dense(25, activation="relu"),    # Further processing
        Dense(1),                        # 🎯 Final price prediction
    ])

    # ⚙️ Configure optimizer with gradient clipping for stability
    optimizer = Adam(learning_rate=0.001, clipvalue=1.0)

    # 🔧 Compile model with appropriate loss and metrics
    model.compile(
        optimizer=optimizer,
        loss="mean_squared_error",    # Standard for regression
        metrics=["mae"],             # Mean Absolute Error for interpretability
    )

    return model


def _make_sequences(
    scaled,
    lookback: int,
    split_ratio: float = 0.8,
):
    """
    Slice the scaled price array into overlapping (X, y) windows and
    split them into training and validation sets.

    Returns
    -------
    X_train, y_train, X_val, y_val : np.ndarray
        Arrays shaped for LSTM input: (samples, lookback, 1).
    """
    split_idx = int(len(scaled) * split_ratio)
    X_all, y_all = [], []

    for i in range(lookback, len(scaled)):
        X_all.append(scaled[i - lookback:i, 0])
        y_all.append(scaled[i, 0])

    X_all = np.array(X_all).reshape(-1, lookback, 1)
    y_all = np.array(y_all)

    # Adjust split index relative to the sequence array
    seq_split = split_idx - lookback

    X_train, y_train = X_all[:seq_split], y_all[:seq_split]
    X_val,   y_val   = X_all[seq_split:], y_all[seq_split:]

    return X_train, y_train, X_val, y_val


def predict_lstm(
    hist: pd.DataFrame,
    forecast_days: int = 90,
    lookback: int = 60,
    random_seed: int = 42,
):
    """
    🚀 Advanced LSTM Stock Price Forecasting Engine

    This function trains a sophisticated neural network to predict future stock prices
    using historical data. The AI learns complex patterns and market relationships.

    🎯 What it does:
       1. Validates and preprocesses historical stock data
       2. Builds and trains a 3-layer LSTM neural network
       3. Uses advanced training techniques (early stopping, adaptive learning)
       4. Generates 90-day business day forecasts
       5. Provides comprehensive performance metrics

    ⚡ Advanced Features:
       • Early stopping to prevent overfitting
       • Adaptive learning rate reduction
       • Validation-based training monitoring
       • Comprehensive error analysis (RMSE, MAE, MAPE)

    📊 Returns: (future_dates, predicted_prices, training_metrics)
    """
    # 🎲 Step 1: Initialize random seeds for reproducible AI results
    _set_seeds(random_seed)

    # ── 🔍 1. Validate and clean input ──────────────────────────────────────
    if "Close" not in hist.columns:
        raise ValueError("'hist' must contain a 'Close' column. Got: {list(hist.columns)}")

    df = hist[["Close"]].copy().dropna()

    # 📏 Check if we have enough data for reliable predictions
    min_required = lookback + forecast_days + 30  # buffer for validation split
    if len(df) < min_required:
        raise ValueError(
            f"Need at least {min_required} rows for reliable predictions "
            f"(lookback={lookback}, forecast_days={forecast_days}). "
            f"Received {len(df)} rows."
        )

    if len(df) < 200:
        print(
            f"⚠️  Warning: Only {len(df)} rows available. "
            "Prediction accuracy may be reduced with limited history."
        )

    print("🧠 Training advanced LSTM stock prediction model...")
    print(f"   📊 Data points: {len(df)} | Lookback window: {lookback} days")
    print(f"   🎯 Forecast horizon: {forecast_days} business days")

    # ── 🔄 2. Scale prices to [0, 1] ─────────────────────────────────────────
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(df.values)

    # ── 🏗️ 3. Build sequence datasets ────────────────────────────────────────
    X_train, y_train, X_val, y_val = _make_sequences(scaled, lookback)
    print(f"   📈 Training samples: {len(X_train)} | Validation samples: {len(X_val)}")

    # ── 🧠 4. Build advanced LSTM model ─────────────────────────────────────
    model = _build_model(lookback)

    # Import callbacks here where they're used (lazy loading)
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

    # 🛡️ Advanced training callbacks for optimal performance
    callbacks = [
        EarlyStopping(
            monitor="val_loss",        # Monitor validation loss
            patience=15,               # Wait 15 epochs for improvement
            restore_best_weights=True, # Keep best model weights
            verbose=0,                 # Silent operation
        ),
        ReduceLROnPlateau(
            monitor="val_loss",        # Monitor validation loss
            factor=0.5,               # Reduce LR by 50% when plateau
            patience=8,               # Wait 8 epochs before reducing
            min_lr=1e-6,              # Minimum learning rate
            verbose=0,                # Silent operation
        ),
    ]

    # ── 🚀 5. Train the neural network ──────────────────────────────────────
    print("   🔄 Training in progress — this may take a few minutes...")
    print("   💡 Using early stopping and adaptive learning rates for optimal training")

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=100,                  # Maximum epochs (early stopping may end sooner)
        batch_size=64,               # Process 64 samples at a time
        callbacks=callbacks,         # Advanced training callbacks
        verbose=0,                   # Silent training (less console spam)
    )

    # 📊 Extract training results
    epochs_trained = len(history.history["loss"])
    final_train_loss = history.history["loss"][-1]
    final_val_loss   = history.history["val_loss"][-1]
    final_train_mae  = history.history["mae"][-1]
    final_val_mae    = history.history["val_mae"][-1]

    print(f"   ✅ Training complete — {epochs_trained} epochs trained")
    print(f"   📉 Train loss (MSE): {final_train_loss:.4f} | Val loss: {final_val_loss:.4f}")
    print(f"   📊 Train MAE:        {final_train_mae:.4f} | Val MAE:  {final_val_mae:.4f}")

    # ── 📏 6. Evaluate on validation set ────────────────────────────────────
    val_preds_scaled = model.predict(X_val, verbose=0)
    val_preds = scaler.inverse_transform(val_preds_scaled).flatten()
    val_actual = scaler.inverse_transform(y_val.reshape(-1, 1)).flatten()

    # Calculate comprehensive error metrics
    rmse = float(np.sqrt(mean_squared_error(val_actual, val_preds)))
    mae  = float(mean_absolute_error(val_actual, val_preds))
    mape = float(np.mean(np.abs((val_actual - val_preds) / (val_actual + 1e-8))) * 100)

    print(f"   🎯 Validation Performance:")
    print(f"      RMSE: ₹{rmse:.2f} | MAE: ₹{mae:.2f} | MAPE: {mape:.1f}%")

    # ── 🔮 7. Generate future predictions ───────────────────────────────────
    print(f"   🔮 Forecasting {forecast_days} business days ahead...")

    # Start with the most recent data for prediction
    current_batch = scaled[-lookback:].reshape(1, lookback, 1)
    predictions = []

    # Iterative forecasting: predict one day, add to input, repeat
    for day in range(forecast_days):
        # Predict next day's price (scaled)
        next_scaled = float(model.predict(current_batch, verbose=0)[0, 0])
        predictions.append(next_scaled)

        # Slide the prediction window: drop oldest day, add new prediction
        current_batch = np.concatenate(
            [current_batch[:, 1:, :], np.array([[[next_scaled]]])],
            axis=1,
        )

        # Progress indicator every 30 days
        if (day + 1) % 30 == 0:
            print(f"      📅 {day + 1}/{forecast_days} days predicted...")

    # ── 💰 8. Convert predictions back to dollar prices ─────────────────────
    predicted_prices = scaler.inverse_transform(
        np.array(predictions).reshape(-1, 1)
    ).flatten()

    # ── 📅 9. Generate future business day dates ────────────────────────────
    last_date = df.index[-1]
    future_dates = pd.date_range(
        start=last_date, periods=forecast_days + 1, freq="B"
    )[1:]  # Skip the start date (already have historical data)

    # ── 📊 10. Compile comprehensive training metrics ───────────────────────
    training_metrics = {
        "epochs_trained":    epochs_trained,
        "final_train_loss":  round(final_train_loss, 6),
        "final_val_loss":    round(final_val_loss, 6),
        "final_train_mae":   round(final_train_mae, 6),
        "final_val_mae":     round(final_val_mae, 6),
        "val_rmse_usd":      round(rmse, 2),
        "val_mae_usd":       round(mae, 2),
        "val_mape_pct":      round(mape, 2),
        "loss_curve":        history.history["loss"],
        "val_loss_curve":    history.history["val_loss"],
    }

    # 🎉 Final success message
    print(
        f"   🎉 Forecast complete: {len(future_dates)} days | "
        f"₹{predicted_prices[0]:.2f} (Day 1) → ₹{predicted_prices[-1]:.2f} (Day {forecast_days})"
    )

    return future_dates, predicted_prices, training_metrics


def get_lstm_prediction_summary(
    future_dates: pd.DatetimeIndex,
    future_prices,
    training_metrics = None,
) -> dict:
    """
    📊 Generate Comprehensive LSTM Prediction Report

    Creates a detailed summary of the AI's predictions including:
    • Day-by-day price forecasts with dates
    • Model architecture and training details
    • Performance metrics and statistics
    • Statistical analysis of predicted price movements

    🎨 Returns a beautifully structured dictionary perfect for UI display
    """
    if len(future_dates) != len(future_prices):
        raise ValueError(
            f"future_dates ({len(future_dates)}) and future_prices "
            f"({len(future_prices)}) must have the same length."
        )

    # 📈 Calculate price movement statistics
    price_change    = float(future_prices[-1] - future_prices[0])
    price_change_pct = float(price_change / future_prices[0] * 100) if future_prices[0] != 0 else 0.0

    # 🏗️ Build comprehensive prediction summary
    summary = {
        "predictions": [
            {
                "date":            date.strftime("%Y-%m-%d"),
                "predicted_price": round(float(price), 2),
                "day_of_week":     date.strftime("%A"),
            }
            for date, price in zip(future_dates, future_prices)
        ],
        "model_info": {
            "type":         "🚀 Advanced Stacked LSTM Neural Network",
            "architecture": "3×LSTM → BatchNorm → Dropout | 2×Dense → Linear Output",
            "lookback_days": 60,
            "regularization": ["Dropout (0.3/0.2)", "Batch Normalization", "Early Stopping"],
            "training_features": [
                "🎯 Early stopping (patience=15)",
                "📉 Adaptive learning rate reduction",
                "⚖️  80/20 train/validation split",
                "🛡️  Gradient clipping (clipvalue=1.0)",
                "🔄 100 max epochs with smart stopping",
            ],
        },
        "statistics": {
            "forecast_start":      future_dates[0].strftime("%Y-%m-%d"),
            "forecast_end":        future_dates[-1].strftime("%Y-%m-%d"),
            "forecast_days":       len(future_dates),
            "day1_price":          round(float(future_prices[0]), 2),
            "final_price":         round(float(future_prices[-1]), 2),
            "min_price":           round(float(future_prices.min()), 2),
            "max_price":           round(float(future_prices.max()), 2),
            "predicted_change_usd": round(price_change, 2),
            "predicted_change_pct": round(price_change_pct, 2),
        },
        "performance": training_metrics or {},
    }

    return summary


# ── 🎮 CLI Demo ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("🧠 Advanced LSTM Stock Predictor — Demo Mode")
    print("═══════════════════════════════════════════════════════")
    print("🚀 Features: 3-Layer LSTM, BatchNorm, Dropout, Early Stopping")
    print("📊 Generates 90-day business day forecasts")
    print("⚡ Advanced AI with market pattern recognition")
    print()
    print("💡 Usage:")
    print("   from mlmodels import predict_lstm, get_lstm_prediction_summary")
    print("   dates, prices, metrics = predict_lstm(stock_df, forecast_days=90)")
    print("   summary = get_lstm_prediction_summary(dates, prices, metrics)")
    print()
    print("🎯 Example Output:")
    print("   Day-1 forecast: ₹150.25")
    print("   90-day range: ₹145.80 → ₹167.42")
    print("   Expected change: +11.4%")
    print("  summary = get_lstm_prediction_summary(dates, prices, metrics)")