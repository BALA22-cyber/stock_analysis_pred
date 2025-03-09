import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft

# Fetch stock data
ticker = "TATAMOTORS.NS"
data = yf.download(ticker, period="6mo", interval="1d")  # Daily data for 6 months
print(data.shape)
# Get closing prices
closing_prices = data['Close'].dropna().values
N = len(closing_prices)

# Apply FFT
fft_values = fft(closing_prices)
frequencies = np.fft.fftfreq(N)  # Get frequency components

# Plot FFT spectrum
plt.figure(figsize=(10,5))
plt.plot(frequencies[:N//2], np.abs(fft_values)[:N//2])
plt.xlabel("Frequency")
plt.ylabel("Amplitude")
plt.title("FFT Spectrum of Tata Motors Stock Prices")
plt.grid()
plt.show()
