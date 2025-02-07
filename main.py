import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from audio_processor import get_frequency_data
import sounddevice as sd
from scipy.signal import find_peaks

# File path of the audio file
FILE_PATH = 'music/bee.mp3'  # Replace with your audio file path

# Get frequency data
stft_mag, time_axis, freq_axis, y, sr = get_frequency_data(FILE_PATH)

# Define different hop lengths to test
hop_length = 16


# Get frequency data
stft_mag, time_axis, freq_axis, y, sr = get_frequency_data(FILE_PATH, hop_length=hop_length)
pitch = np.log(freq_axis)
pitch[np.isneginf(pitch)] = 0 # get rid of log(0)s
avg_pitch_per_frame = np.sum(pitch[:, np.newaxis] * stft_mag, axis=0) / np.sum(stft_mag, axis=0) # mean pitch at each frame


# Print the shape of the STFT matrix and the hop length
print(f"Hop length: {hop_length}")
print(f"STFT shape: {stft_mag.shape}")
print(f"Time axis (first 10 values): {time_axis[:10]}")
print(f"Number of time values: {len(time_axis)}")
print("-" * 50)


## DRUMBEAT DETECTION
# Define the frequency range for drumbeats (e.g., 20 Hz to 200 Hz)
low_freq = 20
high_freq = 100
low_bin = int(low_freq * (stft_mag.shape[0] / sr))
high_bin = int(high_freq * (stft_mag.shape[0] / sr))
# Sum the magnitudes in the drumbeat frequency range
drum_energy = np.sum(stft_mag[low_bin:high_bin, :], axis=0)
# Normalize the energy
drum_energy = drum_energy / np.max(drum_energy)
# Detect peaks in the energy signal
peaks, _ = find_peaks(drum_energy, height=0.1)  # Adjust the height threshold as needed
# Convert peak indices to time
times = time_axis[peaks]
# Print the detected drumbeat times
print("Detected drumbeat times (s):", times)


# Set up the plot
fig, ax = plt.subplots(figsize=(12, 6))
line, = ax.plot(pitch, np.zeros_like(pitch))
ax.set_ylim(0, np.max(stft_mag))
ax.set_xlabel('Pitch log(Hz)')
ax.set_ylabel('Magnitude')
ax.set_title('Pitch Magnitude')


# Global variables for animation
start_time = None
audio_playing = False
current_frame = 0

def animate(frame):
    global start_time, audio_playing, current_frame
    current_time = frame / 1000  # Convert frame to seconds

    if not audio_playing:
        start_time = current_time
        sd.play(y, sr)
        audio_playing = True

    # Update plot
    current_frame = int(current_time * sr / hop_length)  # 512 is the hop_length used in STFT
    if current_frame < stft_mag.shape[1]:
        line.set_ydata(stft_mag[:, current_frame])
        ax.set_title(f'Pitch Magnitude at Time: {current_time:.2f} s')
        
        # Clear previous average pitch line
        # Remove previous average pitch line before adding a new one
        for l in ax.lines:
            if l.get_color() == 'r':  # Identify the red line
                l.remove()

        # Add red line at average pitch for the current frame
        ax.axvline(x=avg_pitch_per_frame[current_frame], color='r', linestyle='--', label="Avg Pitch")

        # Print drum beat-ness
        if current_frame in peaks:
            print(f"Drum beat detected at {current_time:.2f} s")
        else:
            print(0)

    else:
        # If we've reached the end of the audio, stop the animation
        anim.event_source.stop()
        sd.stop()

    return line,

# Set up the animation
anim = FuncAnimation(fig, animate, frames=np.arange(0, len(y)/sr*1000, 50),
                     interval=50, blit=False, repeat=False)

plt.tight_layout()
plt.show()

# Stop audio playback when the plot is closed
sd.stop()
