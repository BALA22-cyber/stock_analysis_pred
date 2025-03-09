import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Create synthetic stock price data
t = np.linspace(0, 1, 500, False)  # 1 second timeframe
sig = np.sin(3*np.pi*2*t) + 0.5*np.sin(80*2*np.pi*t)  # Signal with two frequencies
df = pd.DataFrame({'time':t, 'price':sig})
print(df.shape)

# Perform the Fourier Transform and compute the spectral density (power spectrum)
fft_vals = np.fft.fft(df['price'])
fft_abs = np.abs(fft_vals)
fft_norm = fft_abs / len(df['price'])  # normalization
freq = np.fft.fftfreq(len(df['price']), t[1] - t[0])  # frequencies present in the signal

# Identify the dominant frequency
dominant_freq = freq[np.argmax(fft_norm)]

# Plotting the data
fig, axs = plt.subplots(2)
fig.suptitle('Fourier Analysis on Stock Price Data')
axs[0].plot(df['time'], df['price'])
axs[0].set(xlabel='Time (s)', ylabel='Price')

axs[1].plot(freq, fft_norm)
axs[1].set(xlabel='Frequency (Hz)', ylabel='Normalized amplitude')

plt.show()

print(f"The dominant frequency in the stock price data is: {dominant_freq} Hz")        