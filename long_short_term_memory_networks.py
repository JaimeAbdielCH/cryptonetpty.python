from datetime import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from ta.momentum import RSIIndicator
from ta.volatility import BollingerBands
from binance.um_futures import UMFutures
from sklearn.model_selection import train_test_split
import tensorflow as tf



# Convert to supervised learning format
def create_sequences(data, target_idx, sequence_length=60):
    X, y = [], []
    for i in range(len(data) - sequence_length):
        X.append(data[i:i + sequence_length])
        y.append(data[i + sequence_length, target_idx])
    return np.array(X), np.array(y)


# Main function
if __name__ == "__main__":
    # Load data
    symbol = 'MOVEUSDT'
    interval = '1h'
    limit = '365'
    client = UMFutures()
    klines = client.klines(symbol=symbol, interval=interval, limit=limit)
    data = pd.DataFrame(klines, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_asset_volume', 'number_of_trades', 'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'])
    data['close'] = data['close'].astype(float)


    data['Date'] = pd.date_range('1/24/2024', '1/9/2025')
    data.set_index('Date', inplace=True)

    # Calculate technical indicators
    data['RSI'] = RSIIndicator(close=data['close']).rsi()
    bb = BollingerBands(close=data['close'])
    data['BB_Upper'] = bb.bollinger_hband()
    data['BB_Lower'] = bb.bollinger_lband()

    # Features and target
    features = ['open', 'high', 'low', 'close', 'volume', 'RSI', 'BB_Upper', 'BB_Lower']
    target = 'close'

    # Drop invalid values
    data.replace([np.inf, -np.inf], np.nan, inplace=True)
    data.dropna(inplace=True)

    # Scale features and target
    scaler_X = MinMaxScaler()
    scaled_features = scaler_X.fit_transform(data[features])

    scaler_y = MinMaxScaler()
    scaled_target = scaler_y.fit_transform(data[['close']])

    # Create sequences
    X, y = create_sequences(scaled_features, target_idx=3, sequence_length=60)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True)

    # Define the model
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(X_train.shape[1], X_train.shape[2])),
        tf.keras.layers.LSTM(50, return_sequences=True),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.LSTM(50, return_sequences=False),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(25, activation='relu'),
        tf.keras.layers.Dense(1)
    ])

    # Compile the model
    model.compile(optimizer='adam', loss='mean_squared_error')

    # Train the model
    history = model.fit(X_train, y_train, batch_size=32, epochs=20, validation_data=(X_test, y_test), callbacks=[tf.keras.callbacks.TerminateOnNaN()])

    model.save(f'{symbol}_{datetime.now().strftime("%Y-%m-%d")}.keras')
    # Plot training and validation loss
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    # # Predict
    predictions = model.predict(X_test)
    predictions = scaler_y.inverse_transform(predictions)

    # # Plot predictions vs actuals
    # actual_values = scaler_y.inverse_transform(y_test.reshape(-1, 1))
    # plt.figure(figsize=(10, 6))
    # plt.plot(actual_values, label='Actual Prices')
    # plt.plot(predictions, label='Predicted Prices')
    # plt.title(f'Predictions vs Actual Prices of {symbol}')
    # plt.xlabel('Time')
    # plt.ylabel('Price')
    # plt.legend()
    # plt.show()

    # Number of future steps to predict
    future_steps = 3

    # Take the last sequence from the test set
    last_sequence = X_test[-1]  # Shape: (sequence_length, features)
    future_predictions = []

    for _ in range(future_steps):
        # Add batch dimension (1, sequence_length, features)
        input_sequence = last_sequence[np.newaxis, :, :]
        
        # Predict the next value
        predicted_value = model.predict(input_sequence)[0, 0]
        future_predictions.append(predicted_value)
        
        # Update the sequence by appending the predicted value and removing the oldest value
        next_input = np.append(last_sequence[1:], [[predicted_value] + [0] * (last_sequence.shape[1] - 1)], axis=0)
        last_sequence = next_input

    # Inverse scale the future predictions
    future_predictions = scaler_y.inverse_transform(np.array(future_predictions).reshape(-1, 1))

    # Create a timeline for the future predictions
    last_index = data.index[-1]
    future_dates = pd.date_range(start=last_index, periods=future_steps + 1, freq='D')[1:]

    # Plot predictions vs actuals
    plt.figure(figsize=(12, 6))
    plt.plot(data.index[-len(y_test):], scaler_y.inverse_transform(y_test.reshape(-1, 1)), label='Precios Actuales')
    plt.plot(data.index[-len(predictions):], predictions, label='Precios de tf')

    # Plot future predictions
    plt.plot(future_dates, future_predictions, label='Pronóstico de tf', linestyle='--')

    # Add labels and legend
    plt.title(f'Predicciones vs Precios Actuales (con pronostico) de {symbol}')
    plt.xlabel('Día')
    plt.ylabel('Precio')
    plt.legend()
    plt.show()