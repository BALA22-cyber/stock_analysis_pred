import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft, ifft
import yfinance as yf

def clean_live_data(df):
    # Ensure the correct columns are present
    expected_columns = ["Close", "High", "Low", "Open", "Volume"]
    df = df[expected_columns] if all(col in df.columns for col in expected_columns) else df.dropna(axis=1, how='all')
    
    # Convert columns to numeric and drop NaN rows
    df = df.apply(pd.to_numeric, errors='coerce')
    df = df.dropna()
    return df

def fetch_live_data(ticker, period="5y", interval="1d"):
    data = yf.download(ticker, period=period, interval=interval)
    data.index = pd.to_datetime(data.index)
    return clean_live_data(data)

# Choose between live data fetching or file loading
use_live_data = True  # Set to False if using CSV

if use_live_data:
    ticker = "TATASTEEL.NS"
    data = fetch_live_data(ticker)
else:
    file_path = "/mnt/data/data.csv"
    data = pd.read_csv(file_path, index_col=0, parse_dates=True, skiprows=2)
    data = clean_live_data(data)

# Ensure the data has expected columns
data.columns = ["Close", "High", "Low", "Open", "Volume"]

# Get closing prices
closing_prices = data['Close'].dropna().values
N = len(closing_prices)

# Ensure sufficient data points
if N < 2:
    raise ValueError("Insufficient data for FFT analysis.")

# Apply FFT
fft_values = fft(closing_prices)
frequencies = np.fft.fftfreq(N)

# Filter out high-frequency noise (keep dominant frequencies)
cutoff = max(1, int(N * 0.05))  # Ensure at least one frequency component is retained
fft_filtered = np.copy(fft_values)
fft_filtered[cutoff:-cutoff] = 0  # Zero out high-frequency components

# Apply inverse FFT to reconstruct smoothed signal
reconstructed_signal = ifft(fft_filtered).real

# Ensure the reconstructed signal has the same length as the original
# if len(reconstructed_signal) == len(closing_prices):
min_length = min(len(reconstructed_signal), len(closing_prices))
reconstructed_signal = reconstructed_signal[:min_length]
closing_prices = closing_prices[:min_length]
dates = data.index[-min_length:]

# Plot original vs reconstructed stock prices with corrected alignment
plt.figure(figsize=(12, 6))
plt.plot(dates, closing_prices, label="Original Stock Prices", alpha=0.6)
plt.plot(dates, reconstructed_signal, label="FFT-Filtered Signal", linestyle='dashed', linewidth=2)
plt.xlabel("Date")
plt.ylabel("Stock Price (INR)")
plt.title("Stock Price - Original vs FFT Filtered (Fixed Alignment)")
plt.legend()
plt.grid()
plt.show()

# Plot FFT spectrum
plt.figure(figsize=(10, 5))
plt.plot(frequencies[:N//2], np.abs(fft_filtered)[:N//2])
plt.xlabel("Frequency")
plt.ylabel("Amplitude")
plt.title("FFT Spectrum of Stock Prices")
plt.grid()
plt.show()
