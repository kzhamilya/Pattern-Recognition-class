import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import FastICA

# Parameters
t = np.linspace(0, 5, 500)  # Time vector
sine_freq = 2.0  #Frequency of sine signal in Hz
cosine_freq = 0.8  # Frequency of cosine signal in Hz

# Generate signals
sine_signal = np.sin(2 * np.pi * sine_freq * t)
cosine_signal = np.cos(2 * np.pi * cosine_freq * t)

# Add noise to signals
noise = 0.1 * np.random.normal(size=t.shape)
sine_signal += noise
cosine_signal += noise

# Mix signals
mixed_signals = np.c_[sine_signal, cosine_signal].T
A = np.array([[0.8, 0.3], [0.6, 1.2]])  # Different mixing matrix
mixed_signals = A @ mixed_signals  # Apply mixing

# Separate signals using ICA
ica = FastICA(n_components=2)
separated_signals = ica.fit_transform(mixed_signals.T).T  # Revert to original shape

# Plot mixed and separated signals
fig, axs = plt.subplots(2, 2, figsize=(12, 8))

# Original signals
axs[0, 0].plot(t, sine_signal, label='Sine Signal')
axs[0, 0].plot(t, cosine_signal, label='Cosine Signal')
axs[0, 0].set_title("Original Signals with Noise")
axs[0, 0].legend()

# Mixed signals
axs[0, 1].plot(t, mixed_signals[0, :], label='Mixed Signal 1')
axs[0, 1].plot(t, mixed_signals[1, :], label='Mixed Signal 2')
axs[0, 1].set_title("Mixed Signals")
axs[0, 1].legend()

# Separated signals
axs[1, 0].plot(t, separated_signals[0, :], label='Separated Signal 1')
axs[1, 0].plot(t, separated_signals[1, :], label='Separated Signal 2')
axs[1, 0].set_title("Separated Signals (ICA)")
axs[1, 0].legend()

# Display plots
for ax in axs.flat:
    ax.set_xlabel("Time")
    ax.set_ylabel("Amplitude")
    ax.grid()

plt.tight_layout()
plt.show()
