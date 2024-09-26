import yfinance as yf
import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt

# Define the stocks and their corresponding exchange symbols
stocks = {
    'AAPL': {'NYSE': 'AAPL', 'LSE': 'AAPL.L'},  # Example of AAPL on NYSE and AAPL.L on LSE
    'GOOGL': {'NASDAQ': 'GOOGL', 'LSE': 'GOOGL.L'},  # Google on NASDAQ and LSE
    'TSLA': {'NASDAQ': 'TSLA', 'BSE': 'TSLA.BO'}  # Tesla on NASDAQ and Bombay Stock Exchange (example)
}

# Function to get stock data from specific exchange
def get_stock_prices(stock_symbol, exchange):
    stock_data = yf.Ticker(stock_symbol)
    try:
        price = stock_data.history(period='1d')['Close'].iloc[-1]  # Most recent closing price
        print(f"Fetched {exchange} price for {stock_symbol}: {price}")  # Debugging
        return price
    except Exception as e:
        print(f"Error fetching data for {stock_symbol} on {exchange}: {e}")
        return None

# Function to identify arbitrage opportunities
def check_arbitrage(stock_prices):
    for stock, prices in stock_prices.items():
        print(f"\nChecking arbitrage for {stock}: {prices}")  # Debugging
        
        max_price = max(prices.values())
        min_price = min(prices.values())
        arbitrage_percentage = (max_price - min_price) / min_price * 100
        
        print(f"Max price: {max_price}, Min price: {min_price}, Arbitrage %: {arbitrage_percentage}")  # Debugging
        
        if arbitrage_percentage > 0.1:  # Lower threshold for testing
            print(f"Arbitrage opportunity detected for {stock}!")
            print(f"Max price: {max_price}, Min price: {min_price}, Percentage difference: {arbitrage_percentage}%\n")

# Main function to track prices
def track_arbitrage():
    stock_prices = {stock: {} for stock in stocks}
    
    for stock, exchanges in stocks.items():
        for exchange, symbol in exchanges.items():
            stock_prices[stock][exchange] = get_stock_prices(symbol, exchange)
    
    check_arbitrage(stock_prices)

# Run the arbitrage tracker once every few seconds
while True:
    track_arbitrage()
    time.sleep(60)  # Wait 1 minute before checking again
