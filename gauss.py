import numpy as np
import scipy.io.wavfile as wav

# Parameters
fs = 44100  # Sampling frequency (Hz)
T = 2       # Duration in seconds (chosen to ensure function decay)
t = np.linspace(-T, T, 2 * T * fs)  # Time vector

# Generate the Gaussian function
V_in = np.exp(-t**2)

# Normalize to fit in the range [-1,1] for audio output
V_in = V_in / np.max(np.abs(V_in))

# Convert to 16-bit PCM format
V_in_pcm = np.int16(V_in * 32767)

# Save as a WAV file
wav_filename = "/mnt/data/V_in_gaussian.wav"
wav.write(wav_filename, fs, V_in_pcm)

wav_filename