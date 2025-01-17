import random
from binance.um_futures import UMFutures
import matplotlib.pyplot as plt
from ta.momentum import RSIIndicator
from ta.volatility import BollingerBands
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def create_sequences(data, target_idx, sequence_length=60):
    X, y = [], []
    for i in range(len(data) - sequence_length):
        X.append(data[i:i + sequence_length])
        y.append(data[i + sequence_length, target_idx])
    return np.array(X), np.array(y)

# Funcion Principal
if __name__ == "__main__":
    symbol = 'BTCUSDT'
    interval = '1d'
    limit = 150
    client = UMFutures()
    klines = client.klines(symbol=symbol, interval=interval, limit=limit)
    data = pd.DataFrame(klines, columns=['timestamp', 'open', 'high', 'low', 
                                         'close', 'volume', 'close_time', 'quote_asset_volume', 'number_of_trades', 
                                         'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'])
    data['close'] = data['close'].astype(float)
    
    features = ['open', 'high', 'low', 'close', 'volume', 'RSI', 'BB_Upper', 'BB_Lower']


    data['Date'] = pd.date_range('8/20/2024', '1/16/2025')
    data.set_index('Date', inplace=True)
    data['random'] = [random.randint(50000, 100000) for _ in range(150)]
    # Calculando indicadores tecnicos
    data['RSI'] = RSIIndicator(close=data['close']).rsi()
    bb = BollingerBands(close=data['close'])
    data['BB_Upper'] = bb.bollinger_hband()
    data['BB_Lower'] = bb.bollinger_lband()



    plt.figure(figsize=(10,6))
    plt.plot(data['close'], label='Close Price')
    plt.plot(data['random'])
    plt.show()