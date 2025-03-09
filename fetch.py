# fetch apple data from yfinance library for the past 30 days and plot it on matplotlib

import yfinance as yf
import matplotlib
import matplotlib.pyplot as plt
import datetime
import numpy as np

# Define the ticker symbol
ticker_symbol = 'SBIN.NS'

# Define the date range
end_date = datetime.datetime.now()
start_date = end_date - datetime.timedelta(days=60)

# Fetch the data
data = yf.download(ticker_symbol, start=start_date, end=end_date,interval="1h")

# Display the data
print(data.shape)

# plot = data['Open'].plot()
# plt.show()

fft_result = np.fft.fft(data['Open'])
frequency = np.fft.fftfreq(len(fft_result),d = 1)

magnitude = np.abs(fft_result)
periods = 1/frequency
plt.plot(periods,magnitude)
plt.show()
plt.plot(np.abs(fft_result))
plt.show()