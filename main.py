
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from audio_processor import get_frequency_data
import sounddevice as sd

# File path of the audio file
FILE_PATH = 'music/y.mp3'  # Replace with your audio file path

# Get frequency data
stft_mag, time_axis, freq_axis, y, sr = get_frequency_data(FILE_PATH)

# Set up the plot
fig, ax = plt.subplots(figsize=(12, 6))
line, = ax.plot(freq_axis, np.zeros_like(freq_axis))
ax.set_ylim(0, np.max(stft_mag))
ax.set_xlabel('Frequency (Hz)')
ax.set_ylabel('Magnitude')
ax.set_title('Frequency Magnitude')

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
    current_frame = int(current_time * sr / 512)  # 512 is the hop_length used in STFT
    if current_frame < stft_mag.shape[1]:
        line.set_ydata(stft_mag[:, current_frame])
        ax.set_title(f'Frequency Magnitude at Time: {current_time:.2f} s')
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
