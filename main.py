import matplotlib
matplotlib.use('TkAgg')

import pyaudio
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from scipy.fftpack import fft
from scipy.signal import firwin, lfilter
import time
from tkinter import Tk, Label, Scale, HORIZONTAL, Button, Toplevel, simpledialog
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# ---------------- Audio Setup ----------------
count = [0]

CHUNK = 1024 * 2             # samples per frame
FORMAT = pyaudio.paInt16     # audio format
CHANNELS = 1                 # mono
RATE = 44100                 # samples per second
AMPLITUDE_LIMIT = 4096

# Filter default parameters
FILTER_CUTOFF = 1000        # Hz
FILTER_ORDER = 101          # Filter order

# PyAudio instance
p = pyaudio.PyAudio()

# Audio stream
stream = p.open(
    format=FORMAT,
    channels=CHANNELS,
    rate=RATE,
    input=True,
    output=True,
    frames_per_buffer=CHUNK
)

# ---------------- Plot Setup ----------------
fig, (ax1, ax2) = plt.subplots(2, figsize=(15, 7))
x = np.arange(0, 2 * CHUNK, 2)  # waveform samples
xf = np.linspace(0, RATE, CHUNK)  # frequency bins

line, = ax1.plot(x, np.random.rand(CHUNK), '-', lw=2)
line_fft, = ax2.semilogx(xf, np.random.rand(CHUNK), '-', lw=2)

ax1.set_title('AUDIO WAVEFORM')
ax1.set_xlabel('samples')
ax1.set_ylabel('volume')
ax1.set_ylim(-AMPLITUDE_LIMIT, AMPLITUDE_LIMIT)
ax1.set_xlim(0, 2 * CHUNK)
plt.setp(ax1, xticks=[0, CHUNK, 2 * CHUNK], yticks=[-AMPLITUDE_LIMIT, 0, AMPLITUDE_LIMIT])

ax2.set_xlim(20, RATE / 2)
print('stream started')

# ---------------- GUI Setup ----------------
root = Tk()
root.title("Real-time Audio Spectrum Analyzer")

# Pitch label
pitch_label = Label(root, text="Detected Pitch: -- Hz", font=("Arial", 14))
pitch_label.pack(pady=10)

# Filter cutoff slider
filter_slider = Scale(root, from_=20, to=10000, orient=HORIZONTAL, label="Filter Cutoff Frequency (Hz)")
filter_slider.set(FILTER_CUTOFF)
filter_slider.pack(pady=10)

# Distortion gain slider
distortion_slider = Scale(root, from_=1.0, to=10.0, orient=HORIZONTAL, resolution=0.1, label="Distortion Gain")
distortion_slider.set(2.0)
distortion_slider.pack(pady=10)

# ---------------- Audio Effects ----------------
def apply_fir_filter(data, cutoff):
    # Normalize cutoff frequency to Nyquist frequency
    nyquist = RATE / 2.0
    normalized_cutoff = cutoff / nyquist
    
    # Ensure cutoff is within valid range (0 < cutoff < 1)
    normalized_cutoff = max(0.01, min(0.99, normalized_cutoff))
    
    taps = firwin(FILTER_ORDER, normalized_cutoff)
    filtered_data = lfilter(taps, 1.0, data)
    return filtered_data

def lowPass(x, f):
    X = fft(x)
    n = len(X)

    # Convert cutoff frequency in Hz to index in FFT array
    cutoff_idx = int(f * n / RATE)
    
    # Make sure cutoff index is within valid range
    cutoff_idx = max(1, min(n//2, cutoff_idx))
    
    H = np.zeros(n)
    H[:cutoff_idx] = 1
    H[-cutoff_idx:] = 1
    
    Y = X * H
    y_filtered = np.fft.ifft(Y).real  # Get real part after inverse FFT
    
    # Clip to amplitude limits
    y_filtered = np.clip(y_filtered, -AMPLITUDE_LIMIT, AMPLITUDE_LIMIT)
    return y_filtered


def apply_distortion(data, gain):
    return np.clip(data * gain, -AMPLITUDE_LIMIT, AMPLITUDE_LIMIT)

def detect_pitch(data):
    data = data - np.mean(data)
    
    # Apply window to reduce edge effects
    windowed = data * np.hanning(len(data))
    
    # Autocorrelation
    corr = np.correlate(windowed, windowed, mode='full')
    corr = corr[len(corr)//2:]
    
    # Normalize
    if np.max(corr) > 0:
        corr = corr / np.max(corr)
    else:
        return 0
    
    # Find peaks in autocorrelation (skip the first peak at lag 0)
    # Look for minimum lag corresponding to reasonable pitch range (80-2000 Hz)
    min_period = int(RATE / 2000)  # ~22 samples for 2000 Hz
    max_period = int(RATE / 80)    # ~551 samples for 80 Hz
    
    if max_period >= len(corr):
        return 0
    
    # Find the highest peak in the valid range
    search_corr = corr[min_period:max_period]
    if len(search_corr) == 0:
        return 0
        
    peak_idx = np.argmax(search_corr) + min_period
    
    # Check if the peak is significant enough
    if corr[peak_idx] > 0.3:  # Threshold for pitch confidence
        pitch = RATE / peak_idx
        return pitch
    
    return 0

# ---------------- Sandbox Feature ----------------
def convInputSide(x, h):
    y = np.zeros(len(x) + len(h) - 1)
    for i in range(len(y)):
        for j in range(len(h)):
            if 0 <= i - j < len(x):
                y[i] += h[j] * x[i - j]
    return y

def capture_chunk():
    snapshot_data = data_np_distorted.copy()
    
    sandbox_window = Toplevel(root)
    sandbox_window.title("Audio Sandbox")
    
    fig_sandbox, ax_sandbox = plt.subplots(figsize=(10, 4))
    x_sandbox = np.arange(len(snapshot_data))
    line_sandbox, = ax_sandbox.plot(x_sandbox, snapshot_data, '-', lw=2)
    ax_sandbox.set_title("Captured Audio Chunk")
    ax_sandbox.set_ylim(-AMPLITUDE_LIMIT, AMPLITUDE_LIMIT)
    
    canvas_sandbox = FigureCanvasTkAgg(fig_sandbox, master=sandbox_window)
    canvas_sandbox.get_tk_widget().pack()

    # Filter cutoff slider in sandbox
    filter_slider_sandbox = Scale(sandbox_window, from_=20, to=10000, orient=HORIZONTAL, label="Filter Cutoff Hz")
    filter_slider_sandbox.set(FILTER_CUTOFF)
    filter_slider_sandbox.pack(pady=5)

    def apply_lowpass():
        cutoff = filter_slider_sandbox.get()
        filtered = apply_fir_filter(snapshot_data, cutoff)
        line_sandbox.set_ydata(filtered)
        canvas_sandbox.draw()

    lp_button = Button(sandbox_window, text="Apply Low-Pass Filter", command=apply_lowpass)
    lp_button.pack(pady=5)

    def apply_custom_kernel():
        kernel_str = simpledialog.askstring("Input Kernel", "Enter comma-separated values (e.g. 0.2,0.5,0.2):")
        if kernel_str:
            try:
                kernel = np.array([float(k.strip()) for k in kernel_str.split(',')])
                # Use numpy's convolution
                processed = np.convolve(snapshot_data, kernel, mode='same')
                
                # Ensure the processed signal is within amplitude limits
                processed = np.clip(processed, -AMPLITUDE_LIMIT, AMPLITUDE_LIMIT)
                
                line_sandbox.set_ydata(processed)
                canvas_sandbox.draw()
            except ValueError:
                print("Invalid kernel format. Please use comma-separated numbers.")



    kernel_button = Button(sandbox_window, text="Apply Custom Kernel", command=apply_custom_kernel)
    kernel_button.pack(pady=5)

    def reset_signal():
        line_sandbox.set_ydata(snapshot_data)
        canvas_sandbox.draw()

    reset_button = Button(sandbox_window, text="Reset Signal", command=reset_signal)
    reset_button.pack(pady=5)

capture_button = Button(root, text="Capture Chunk", command=capture_chunk)
capture_button.pack(pady=10)

# Global variable to store current processed data
data_np_distorted = np.zeros(CHUNK)

# ---------------- Animation ----------------
def animate(i):
    global data_np_distorted
    try:
        data = stream.read(CHUNK, exception_on_overflow=False)
        data_np = np.frombuffer(data, dtype='h')

        # Get slider values
        current_filter_cutoff = filter_slider.get()
        current_distortion_gain = distortion_slider.get()

        # Apply FIR filter and distortion
        data_np_filtered = apply_fir_filter(data_np, current_filter_cutoff)
        data_np_distorted = apply_distortion(data_np_filtered, current_distortion_gain)

        # Update waveform
        line.set_ydata(data_np_distorted)

        # FFT update
        yf = fft(data_np_distorted)
        line_fft.set_ydata(np.abs(yf[0:CHUNK]) / (512 * CHUNK))

        # Pitch detection
        pitch = detect_pitch(data_np_distorted)
        if pitch > 0:
            pitch_label.config(text=f"Detected Pitch: {pitch:.2f} Hz")
        else:
            pitch_label.config(text="Detected Pitch: -- Hz")

        count[0] += 1
        canvas.draw()
    except Exception as e:
        print(f"Error in animation: {e}")

# ---------------- Closing ----------------
def on_close():
    print("Closing")
    if count[0] > 0:
        frame_rate = count[0] / (time.time() - start_time)
        print('average frame rate = {:.0f} FPS'.format(frame_rate))
    
    stream.stop_stream()
    stream.close()
    p.terminate()
    print('stream stopped')
    root.quit()

# ---------------- Embed Figure ----------------
canvas = FigureCanvasTkAgg(fig, master=root)
canvas.get_tk_widget().pack(pady=10)

# ---------------- Main ----------------
start_time = time.time()
anim = animation.FuncAnimation(fig, animate, blit=False, interval=30)
root.protocol("WM_DELETE_WINDOW", on_close)
root.mainloop()