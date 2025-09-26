#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: safiyah
"""

import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import yfinance as yf
from scipy.stats import jarque_bera, probplot
import matplotlib.pyplot as plt


#Get the stock returns for the past 10 years
tickers = ["JPM", "NVDA"]
start_date = "2015-01-01"
end_date = "2024-12-31"

returns = pd.DataFrame()
for ticker in tickers:
    returns[ticker] = yf.download(ticker, start_date, end_date)["Close"]

#Calculating the log returns
log_returns = (np.log(returns / returns.shift(1)).dropna())*100

# Descriptive statistics
np.set_printoptions(precision=4)
statistics = pd.DataFrame()
statistics["Number of Observations"] = log_returns.count()
statistics["Mean"] = log_returns.mean()
statistics["Standard Deviation"] = log_returns.std()
statistics["Lowest Log Return"] = log_returns.min()
statistics["25% Quantile"] = log_returns.quantile(0.25)
statistics["50% Quantile"] = log_returns.quantile(0.5)
statistics["75% Quantile"] = log_returns.quantile(0.75)
statistics["Highest Log Return"] = log_returns.max()
statistics["Skewness"] = log_returns.skew()
statistics["Kurtosis"] = log_returns.kurtosis()
statistics["Jarque-Bera Test"] = log_returns.apply(lambda x: jarque_bera(x)[0])
statistics["p-value"] = log_returns.apply(lambda x: jarque_bera(x)[1])
statistics["Reject Null"] = log_returns.apply(lambda x: jarque_bera(x)[1] < 0.05)
print(statistics.transpose())

# Plot of the returns
for ticker in tickers:
    if ticker in returns.columns:
        returns[ticker].plot(figsize=(10, 6))
        plt.title(f"{ticker} Returns")
        plt.xlabel("Date")
        plt.ylabel("Returns")
        plt.show()

# Plot and QQ-plot of the log returns
for ticker in tickers:
    if ticker in log_returns.columns:
        log_returns[ticker].plot(figsize=(10, 6))
        plt.title(f"{ticker} Log Returns")
        plt.xlabel("Date")
        plt.ylabel("Log Returns")
        plt.show()
        
        probplot(log_returns[ticker], dist="norm", plot=plt)
        plt.title(f"QQ-plot: {ticker} Log Returns vs. Standard Normal")
        plt.xlabel("Standard Normal Quantiles")
        plt.ylabel("Sample Quantiles")
        plt.show()
