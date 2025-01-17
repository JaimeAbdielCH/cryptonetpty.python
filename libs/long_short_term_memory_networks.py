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

# Convirtiendo en formato de aprendizaje supervisado.
def create_sequences(data, target_idx, sequence_length=60):
    X, y = [], []
    for i in range(len(data) - sequence_length):
        X.append(data[i:i + sequence_length])
        y.append(data[i + sequence_length, target_idx])
    return np.array(X), np.array(y)



# Funcion Principal
if __name__ == "__main__":
    symbol = 'XRPUSDT'
    interval = '1d'
    limit = 365
    client = UMFutures()
    klines = client.klines(symbol=symbol, interval=interval, limit=limit)
    data = pd.DataFrame(klines, columns=['timestamp', 'open', 'high', 'low', 
                                         'close', 'volume', 'close_time', 'quote_asset_volume', 'number_of_trades', 
                                         'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'])
    data['close'] = data['close'].astype(float)


    data['Date'] = pd.date_range('1/18/2024', '1/16/2025')
    data.set_index('Date', inplace=True)

    # Calculando indicadores tecnicos
    data['RSI'] = RSIIndicator(close=data['close']).rsi()
    bb = BollingerBands(close=data['close'])
    data['BB_Upper'] = bb.bollinger_hband()
    data['BB_Lower'] = bb.bollinger_lband()

    # Caracteristicas y objetivo 'close'
    features = ['open', 'high', 'low', 'close', 'volume', 'RSI', 'BB_Upper', 'BB_Lower']

    # Descartar valores invalidos
    data.replace([np.inf, -np.inf], np.nan, inplace=True)
    data.dropna(inplace=True)
    
    # Creando escala de los valores caracteristicos
    # y de nuestro objetivo.
    scaler_X = MinMaxScaler()
    scaled_features = scaler_X.fit_transform(data[features])

    scaler_y = MinMaxScaler()
    scaled_target = scaler_y.fit_transform(data[['close']])

    # Creando secuencia
    X, y = create_sequences(scaled_features, target_idx=3, sequence_length=60)

    # Separando valores de entrenamiento y pruebas
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True)

    # Definiendo modelo LSTM
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(X_train.shape[1], X_train.shape[2])),
        tf.keras.layers.LSTM(50, return_sequences=True),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.LSTM(50, return_sequences=False),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(25, activation='relu'),
        tf.keras.layers.Dense(1)
    ])

    # Compilando el modelo
    model.compile(optimizer='adam', loss='mean_squared_error')

    # Entrenando modelo
    history = model.fit(X_train, y_train, batch_size=32, epochs=20, validation_data=(X_test, y_test), 
                        callbacks=[tf.keras.callbacks.TerminateOnNaN()])
    # Salvando modelo
    model.save(f'{symbol}_{datetime.now().strftime("%Y-%m-%d")}.keras')
    
    # Grafico de entrenamiento y validacion
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['loss'], label='Entrenamiento negativos')
    plt.plot(history.history['val_loss'], label='Validaciones negativas')
    plt.title('Media de acercamiento del modelo')
    plt.xlabel('Epoca')
    plt.ylabel('Valores negativos')
    plt.legend()
    plt.show()

    # # Prediccion
    predictions = model.predict(X_test)
    predictions = scaler_y.inverse_transform(predictions)

    # Numero de pronosticos
    future_steps = 3

    # Tomar la ultima secuencia de las pruebas
    last_sequence = X_test[-1]  # Shape: (sequence_length, features)
    future_predictions = []

    for _ in range(future_steps):
        # Agregando la dimension del lote (1, sequence_length, features)
        input_sequence = last_sequence[np.newaxis, :, :]
        
        # Sacando pronostico
        predicted_value = model.predict(input_sequence)[0, 0]
        future_predictions.append(predicted_value)
        
        # Actualizando la secuencia, agregando el valor pronosticado y removiendo el valor mas antiguo.
        next_input = np.append(last_sequence[1:], [[predicted_value] + [0] * (last_sequence.shape[1] - 1)], axis=0)
        last_sequence = next_input

    # Invirtiendo los valores escalados
    future_predictions = scaler_y.inverse_transform(np.array(future_predictions).reshape(-1, 1))

    # Creando linea de tiempo
    last_index = data.index[-1]
    future_dates = pd.date_range(start=last_index, periods=future_steps + 1, freq='D')[1:]

    # Grafico de predicciones y valores actuales.
    plt.figure(figsize=(12, 6))
    plt.plot(data.index[-len(y_test):], scaler_y.inverse_transform(y_test.reshape(-1, 1)), label='Precios Actuales')
    plt.plot(data.index[-len(predictions):], predictions, label='Precios de tf')

    # Grafico de pronosticos
    plt.plot(future_dates, future_predictions, label='Pronóstico de tf', linestyle='--')

    # Agregando titulos de los ejes
    plt.title(f'Predicciones vs Precios Actuales (con pronostico) de {symbol}')
    plt.xlabel('Día')
    plt.ylabel('Precio')
    plt.legend()
    plt.show()
