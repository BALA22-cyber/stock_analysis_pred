import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft, ifft
from time import sleep

# Fetch stock data
sleep(1)
ticker = "^NSEI"
sleep(1)
data = yf.download(ticker, period="1y", interval="1d")  # Daily data for 6 months
sleep(1)
print(data.head(),data.shape)

data.to_csv('NIFTY50_max_data.csv')
# Get closing prices
closing_prices = data['Close'].dropna().values #drop missing values
N = len(closing_prices)

print("Total iteration: ",N)

# Apply FFT
fft_values = fft(closing_prices)
frequencies = np.fft.fftfreq(N)  # Get frequency components
print(frequencies.shape)

# Filter out high-frequency noise (keep dominant frequencies)
cutoff = int(N * 0.02)  # Keep only 5% of the frequencies (adjustable)
fft_filtered = np.copy(fft_values)
fft_filtered[cutoff:-cutoff] = 0  # Zero out high-frequency components

# Apply inverse FFT to reconstruct smoothed signal
reconstructed_signal = ifft(fft_filtered)#.imaginary
# reconstructed_signal = ifft(fft_values).real
# # Plot original vs reconstructed stock prices

fig, axs = plt.subplots(1, 2, figsize=(20, 6))
# plt.figure(figsize=(12, 6))
# plt.plot(data.index, closing_prices, label="Original Stock Prices", alpha=0.6)
# plt.plot(data.index, reconstructed_signal, label="FFT-Filtered Signal", linestyle='dashed', linewidth=2)
# plt.xlabel("Date")
# plt.ylabel("Stock Price (INR)")
# plt.title("Tata Motors Stock Price - Original vs FFT Filtered")
# plt.legend()
# plt.grid()
# plt.show()

# Plot FFT spectrum
# plt.figure(figsize=(10,5))
# plt.plot(frequencies[:N//2], np.abs(fft_values)[:N//2])
# plt.xlabel("Frequency")
# plt.ylabel("Amplitude")
# plt.title("FFT Spectrum of Tata Motors Stock Prices")
# plt.grid()
# Plot FFT spectrum
axs[0].plot(data.index, closing_prices, label="Original Stock Prices", alpha=0.6)
axs[0].plot(data.index, reconstructed_signal, label="FFT-Filtered Signal", linestyle='dashed', linewidth=2)
axs[0].set_xlabel("Date")
axs[0].set_ylabel("Stock Price (INR)")
axs[0].set_title("Tata Motors Stock Price - Original vs FFT Filtered")
axs[0].legend()
axs[0].grid()
axs[1].plot(frequencies[:N//2], np.abs(fft_values)[:N//2])
axs[1].set_xlabel("Frequency")
axs[1].set_ylabel("Amplitude")
axs[1].set_title("FFT Spectrum of Tata Motors Stock Prices")
axs[1].grid()

plt.tight_layout()
plt.show()

