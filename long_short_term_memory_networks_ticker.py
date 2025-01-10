import tensorflow as tf
import numpy as np
import json
import asyncio
import websockets
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime

# Function to preprocess the tick data for prediction
def preprocess_tick_data(tick_data, scaler, sequence_length, sequence_buffer):
    """
    Processes the tick data to create a valid input sequence for the model.
    """
    # Convert the tick data to an array
    tick_array = np.array(tick_data).reshape(-1, len(tick_data))
    
    # Scale the data
    tick_scaled = scaler.transform(tick_array)
    
    # Update the sequence buffer
    if len(sequence_buffer) >= sequence_length:
        sequence_buffer.pop(0)
    sequence_buffer.append(tick_scaled[0])
    
    if len(sequence_buffer) < sequence_length:
        return None  # Not enough data to form a sequence yet
    
    # Convert to a tensor with the correct shape
    return np.array(sequence_buffer).reshape(1, sequence_length, len(tick_data))

# Main function for WebSocket and prediction
async def main():
    # Configuration
    symbol = "BTCUSDT"
    interval = "3m"
    websocket_url = f'wss://fstream.binance.com/ws/{symbol}_perpetual@continuousKline_{interval}'
    sequence_length = 60
    features = ['open', 'high', 'low', 'close', 'volume', 'RSI', 'BB_Upper', 'BB_Lower']
    model_path = f"{symbol}_{datetime.now().strftime('%Y-%m-%d')}.keras"
    sequence_buffer = []  # Buffer to hold the current sequence

    # Load the model
    model = tf.keras.models.load_model(model_path)

    # Initialize the scaler (use pre-fitted values)
    scaler = MinMaxScaler()
    # Example range values from training, adjust based on your data
    dummy_data = np.zeros((1, len(features)))
    scaler.fit(dummy_data)  # Replace with real scaler from training

    async with websockets.connect(websocket_url) as websocket:
        print(f"Connected to WebSocket: {websocket_url}")

        while True:
            try:
                # Receive data from WebSocket
                message = await websocket.recv()
                kline_data = json.loads(message)
                
                # Extract Kline data
                kline = kline_data['k']
                tick_data = [
                    float(kline['o']),  # Open
                    float(kline['h']),  # High
                    float(kline['l']),  # Low
                    float(kline['c']),  # Close
                    float(kline['v']),  # Volume
                ]
                
                # Preprocess the data for prediction
                input_sequence = preprocess_tick_data(tick_data, scaler, sequence_length, sequence_buffer)
                
                if input_sequence is None:
                    print("Waiting for more ticks to form a full sequence...")
                    continue
                
                # Predict the next close price
                prediction = model.predict(input_sequence)
                predicted_price = scaler.inverse_transform([[0]*4 + [prediction[0][0]]])[0][-1]
                
                print(f"Predicted Close Price: {predicted_price:.2f}")
            
            except Exception as e:
                print(f"Error during prediction: {e}")
                break

# Run the WebSocket connection
if __name__ == "__main__":
    asyncio.run(main())
