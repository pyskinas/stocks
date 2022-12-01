"""
    Preamble:
    This program is the first in a series of 3.
    1. Predict stock prices on opening monday morning
    2. Predict opening stock prices on mornings
    3. Predict 5 min interval changes in stock price in real time    
"""

# Stock Data
import yfinance as yf

# Dataftames, useful for numpy
import pandas as pd

# Matrices to represent stocks
import numpy as np

# To get today's date.
import datetime

def main():
    # Initialise the neural network
    nn = NN()
    
    # Get History
    endDate = (datetime.date.today() - datetime.timedelta(days=8)).strftime("%Y-%m-%d")
    startDate = (datetime.date.today() - datetime.timedelta(days=15)).strftime("%Y-%m-%d")
    ticker = pd.DataFrame(yf.Ticker("aapl").history(interval="1m", start=startDate, end=endDate))
    print(ticker)
    print(ticker["Datetime"])
    ticker1wk1m = np.array(ticker["Open"])
    
    # Some data can be missing, or perhaps overcrowded, so I'll fill remaining data with the last value
    N = len(ticker1wk1m)
    if N < 1950:
        pass
        
        
    pass

class NN:
    
    layers = []
    
    def __init__(self):
        layers = self.layers
        layers.append(1)
        layers.append(2)
        

main()