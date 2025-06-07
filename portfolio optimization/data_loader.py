# data_loader.py

import yfinance as yf
import pandas as pd

def load_data(tickers, start_date, end_date):
    data = yf.download(tickers, start=start_date, end=end_date, auto_adjust=False)['Close']
    data = data.dropna()
    return data

def calculate_returns(price_data):
    returns = price_data.pct_change().dropna()
    return returns
