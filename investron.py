from tvDatafeed import TvDatafeed, Interval
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout

# Initialize TVDatafeed
tv = TvDatafeed()

# Fetch Stock Data
def fetch_stock_data(symbol, exchange, interval, n_bars=1000):
    df = tv.get_hist(symbol, exchange, interval, n_bars=n_bars)
    return df

# Add Technical Indicators
def add_indicators(df):
    df['SMA'] = df['close'].rolling(window=20).mean()
    df['EMA'] = df['close'].ewm(span=20).mean()
    df['RSI'] = compute_rsi(df['close'], window=14)
    return df

def compute_rsi(series, window):
    delta = series.diff(1)
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

# Preprocess Data for LSTM
def preprocess_data(df):
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(df[['close']].values)
    
    X, y = [], []
    for i in range(60, len(data_scaled)):
        X.append(data_scaled[i-60:i])
        y.append(data_scaled[i])
    
    return np.array(X), np.array(y), scaler

# Build LSTM Model
def build_model(input_shape):
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        LSTM(50, return_sequences=False),
        Dropout(0.2),
        Dense(25),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# Mock Trading System
class MockTrader:
    def __init__(self, initial_balance=10000):
        self.balance = initial_balance
        self.position = 0
    
    def buy(self, price, quantity):
        cost = price * quantity
        if self.balance >= cost:
            self.balance -= cost
            self.position += quantity
            print(f"Bought {quantity} shares at {price}, Balance: {self.balance}")
        else:
            print("Insufficient funds to buy.")
    
    def sell(self, price, quantity):
        if self.position >= quantity:
            self.balance += price * quantity
            self.position -= quantity
            print(f"Sold {quantity} shares at {price}, Balance: {self.balance}")
        else:
            print("Insufficient shares to sell.")
    
    def report(self):
        print(f"Final Balance: {self.balance}, Position: {self.position}")

# Main Bot Logic
def run_trading_bot(symbol="RELIANCE", exchange="NSE", interval=Interval.in_1_hour, epochs=20):
    # Fetch Data
    df = fetch_stock_data(symbol, exchange, interval)
    df = add_indicators(df)
    df.dropna(inplace=True)
    
    # Preprocess Data
    X, y, scaler = preprocess_data(df)
    X_train, y_train = X[:int(0.8*len(X))], y[:int(0.8*len(y))]
    X_test, y_test = X[int(0.8*len(X)):], y[int(0.8*len(y)):]
    
    # Build and Train Model
    model = build_model((X_train.shape[1], X_train.shape[2]))
    model.fit(X_train, y_train, batch_size=32, epochs=epochs)
    
    # Predict and Simulate Trades
    predictions = model.predict(X_test)
    predictions = scaler.inverse_transform(predictions)
    
    trader = MockTrader()
    for i in range(len(predictions)):
        if predictions[i] > df['close'].iloc[-len(predictions) + i]:  # Buy Signal
            trader.buy(df['close'].iloc[-len(predictions) + i], 1)
        else:  # Sell Signal
            trader.sell(df['close'].iloc[-len(predictions) + i], 1)
    
    trader.report()

# Run the bot
run_trading_bot()

