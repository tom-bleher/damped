import numpy as np
import scipy.io.wavfile as wav
import os
import matplotlib.pyplot as plt
from datetime import datetime

# Parameters
fs = 44100  # Sampling frequency (Hz)
T = 5      # Duration in seconds (increased to ensure complete function decay)
t = np.linspace(-T/2, T/2, T * fs)  # Time vector centered at 0

# Circuit parameters
R = 10e3    # Resistance in ohms (10kΩ)
C = 100e-9  # Capacitance in farads (100nF)
RC = R * C  # RC time constant

# Create output directory with timestamp
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), f"output_{timestamp}")
os.makedirs(output_dir, exist_ok=True)

# List of alpha values - using clear float values instead of logarithmic spacing
alpha_values = [0.1, 0.2, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 7.5, 10.0]  # Clear float values

# List to store all generated filenames and theoretical values
generated_files = []
theoretical_results = []

def generate_gaussian(alpha, normalize=False, modulate=True, carrier_freq=440, gain=1.0, sustained_tone=False):
    """
    Generate a Gaussian function with the given alpha parameter.
    
    Parameters:
        alpha: Controls the width of the Gaussian
        normalize: Whether to normalize the output to [-1,1]
        modulate: Whether to apply frequency modulation to make it audible
        carrier_freq: Frequency of the carrier wave in Hz
        gain: Additional gain factor to apply before normalization
        sustained_tone: If True, creates a longer sustained tone like online generators
    """
    if sustained_tone:
        # Create a sustained tone with fade-in and fade-out
        # Use a constant amplitude in the middle with Gaussian edges for smooth transition
        fade_duration = T / 10  # 10% of total duration for fade in/out
        center_duration = T - (2 * fade_duration)
        
        # Create envelope with fade in/out
        fade_in_samples = int(fade_duration * fs)
        fade_out_samples = int(fade_duration * fs)
        center_samples = len(t) - fade_in_samples - fade_out_samples
        
        # Create envelope parts
        fade_in_env = np.linspace(0, 1, fade_in_samples) ** 2  # Squared for smoother fade
        center_env = np.ones(center_samples)
        fade_out_env = np.linspace(1, 0, fade_out_samples) ** 2
        
        # Combine envelope parts
        envelope = np.concatenate([fade_in_env, center_env, fade_out_env])
        
        # Generate the tone with this envelope
        V_in = envelope * np.sin(2 * np.pi * carrier_freq * t)
    else:
        # Original Gaussian implementation
        V_in = np.exp(-alpha * t**2)
        
        # Apply frequency modulation if requested
        if modulate:
            # Modulate the Gaussian with a sine wave at carrier_freq
            V_in = V_in * np.sin(2 * np.pi * carrier_freq * t)
    
    # Apply additional gain
    V_in = V_in * gain
    
    if normalize:
        # Normalize to fit in the range [-1,1] for audio output
        V_in = V_in / np.max(np.abs(V_in))
    
    # Convert to 16-bit PCM format
    V_in_pcm = np.int16(V_in * 32767)
    return V_in, V_in_pcm

# Create a summary file
summary_file = os.path.join(output_dir, "experiment_summary.txt")
with open(summary_file, 'w') as f:
    f.write(f"Gaussian Experiment for RC Circuit (R={R}Ω, C={C}F, RC={RC}s)\n")
    f.write(f"Date: {timestamp}\n\n")
    f.write("alpha,theoretical_V_infinity,1/V_infinity^2,filename,modulated\n")

# Loop through different alpha values
for alpha in alpha_values:
    # Only generate modulated version for better audibility
    V_in_mod, V_in_pcm_mod = generate_gaussian(alpha, normalize=True, modulate=True, carrier_freq=440, gain=1.0)
    
    # Calculate theoretical values for the experiment
    V_infinity = (1/RC) * np.sqrt(np.pi/alpha)
    inverse_V_infinity_squared = (RC**2/np.pi) * alpha
    
    # Save only the modulated version as WAV file
    wav_filename_mod = f"V_in_gaussian_alpha_{alpha:.1f}_modulated.wav"
    full_path_mod = os.path.join(output_dir, wav_filename_mod)
    wav.write(full_path_mod, fs, V_in_pcm_mod)
    
    generated_files.append(full_path_mod)
    theoretical_results.append((alpha, V_infinity, inverse_V_infinity_squared))
    
    # Append to summary file
    with open(summary_file, 'a') as f:
        f.write(f"{alpha},{V_infinity},{inverse_V_infinity_squared},{wav_filename_mod},Yes\n")
    
    print(f"Generated modulated Gaussian wave with alpha={alpha:.1f}:")
    print(f"  - Modulated (audible): {full_path_mod}")

# Create plot of expected experimental results
plt.figure(figsize=(10, 6))
alphas = [result[0] for result in theoretical_results]
inv_v_infinity_squared = [result[2] for result in theoretical_results]

plt.plot(alphas, inv_v_infinity_squared, 'o-')
plt.xlabel('Alpha')
plt.ylabel('1/V_infinity^2')
plt.title(f'Expected Linear Relationship: 1/V_infinity^2 vs Alpha\nSlope should be R²C²/π = {RC**2/np.pi:.6e}')
plt.grid(True)
plt.xscale('log')
plt.savefig(os.path.join(output_dir, 'expected_results.png'))

print(f"\nExperiment setup complete. Files saved to {output_dir}")
print(f"Summary file created at {summary_file}")
print(f"Expected slope of 1/V_infinity^2 vs alpha: {RC**2/np.pi:.6e} (= R²C²/π)")

# Create additional audio tones that are easier to hear (like online tone generators)
print("\nGenerating additional sustained tones for better audibility...")
test_frequencies = [440, 1000, 2000, 4000]  # Common test frequencies including A4 (440Hz)

# Create a visualization of the waveforms
plt.figure(figsize=(12, 10))

for i, freq in enumerate(test_frequencies):
    # Generate sustained tone
    tone_name = f"sustained_tone_{freq}Hz.wav"
    V_in_sustained, V_in_pcm_sustained = generate_gaussian(
        alpha=1.0,  # Alpha doesn't matter for sustained tones
        normalize=True,
        modulate=False,  # Not needed since we generate the tone directly
        carrier_freq=freq,
        gain=1.0,
        sustained_tone=True
    )
    
    # Save the sustained tone
    tone_path = os.path.join(output_dir, tone_name)
    wav.write(tone_path, fs, V_in_pcm_sustained)
    generated_files.append(tone_path)
    
    # Add to visualization
    plt.subplot(len(test_frequencies), 1, i+1)
    # Plot just a small section to see the waveform clearly
    sample_window = 1000
    center_idx = len(t) // 2
    plt.plot(t[center_idx:center_idx+sample_window], 
             V_in_sustained[center_idx:center_idx+sample_window])
    plt.title(f"Sustained Tone at {freq} Hz")
    plt.grid(True)
    
    print(f"  - Created sustained tone: {tone_path}")

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'tone_waveforms.png'))
print(f"  - Saved waveform visualization to {os.path.join(output_dir, 'tone_waveforms.png')}")

# Generate a frequency sweep as well (like many online tone generators offer)
print("\nGenerating frequency sweep...")
sweep_duration = 5  # seconds
sweep_t = np.linspace(0, sweep_duration, int(sweep_duration * fs))
start_freq = 200
end_freq = 2000
# Linear frequency sweep
sweep_freq = start_freq + (end_freq - start_freq) * sweep_t / sweep_duration
# Generate the sweep signal
sweep_signal = np.sin(2 * np.pi * np.cumsum(sweep_freq) / fs)
# Apply fade in/out
fade_samples = int(0.1 * fs)  # 100ms fade
fade_in = np.linspace(0, 1, fade_samples) ** 2
fade_out = np.linspace(1, 0, fade_samples) ** 2
sweep_signal[:fade_samples] *= fade_in
sweep_signal[-fade_samples:] *= fade_out
# Normalize and convert to PCM
sweep_signal = sweep_signal / np.max(np.abs(sweep_signal))
sweep_signal_pcm = np.int16(sweep_signal * 32767)
# Save the sweep
sweep_path = os.path.join(output_dir, "frequency_sweep.wav")
wav.write(sweep_path, fs, sweep_signal_pcm)
generated_files.append(sweep_path)
print(f"  - Created frequency sweep: {sweep_path}")

print(f"\nExperiment setup complete. Files saved to {output_dir}")
print(f"Summary file created at {summary_file}")
print(f"Expected slope of 1/V_infinity^2 vs alpha: {RC**2/np.pi:.6e} (= R²C²/π)")

# Return the list of generated files
generated_files
