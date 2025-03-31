import numpy as np
import scipy.io.wavfile as wav
import os
from datetime import datetime

# Basic parameters
fs = 44100  # Sampling frequency (Hz)
T = 5       # Duration in seconds
t = np.linspace(-T/2, T/2, T * fs)  # Time vector centered at 0

# RC circuit parameters
R = 10e3    # 10kΩ resistance
C = 100e-9  # 100nF capacitance
RC = R * C  # Time constant

# Create output folder
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), f"output_{timestamp}")
os.makedirs(output_dir, exist_ok=True)

# Alpha values to test
alpha_values = [0.1, 0.2, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 7.5, 10.0]

# List to store filenames and results
generated_files = []
theoretical_results = []

def generate_gaussian(alpha, normalize=False, modulate=True, carrier_freq=440):
    """Generate a Gaussian function with given alpha parameter."""
    # Create the Gaussian envelope
    gaussian = np.exp(-alpha * t**2)
    
    # Apply modulation to make it audible
    if modulate:
        output = gaussian * np.sin(2 * np.pi * carrier_freq * t)
    else:
        output = gaussian
    
    # Normalize if requested
    if normalize:
        output = output / np.max(np.abs(output))
    
    # Convert to 16-bit PCM format for WAV
    output_pcm = np.int16(output * 32767)
    return output, output_pcm

# Generate Gaussians for each alpha
for alpha in alpha_values:
    # Generate modulated Gaussian for better audibility
    _, wave_pcm = generate_gaussian(alpha, normalize=True, modulate=True)
    
    # Calculate theoretical values 
    V_infinity = (1/RC) * np.sqrt(np.pi/alpha)
    inverse_V_squared = (RC**2/np.pi) * alpha
    
    # Save as WAV file
    filename = f"gaussian_alpha_{alpha:.1f}.wav"
    filepath = os.path.join(output_dir, filename)
    wav.write(filepath, fs, wave_pcm)
    
    generated_files.append(filepath)
    theoretical_results.append((alpha, V_infinity, inverse_V_squared))
    
    print(f"Generated Gaussian with alpha={alpha:.1f}: {filepath}")

print(f"\nAll files saved to: {output_dir}")
print(f"Expected slope: {RC**2/np.pi:.6e}")

generated_files
