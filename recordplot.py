import numpy as np
import matplotlib.pyplot as plt
import pylab
import wave
import pyaudio 
import struct
import msvcrt


CHUNK = 2048
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 48000
RECORD_SECONDS = 5 # recording will be terminated after 5 sec

p = pyaudio.PyAudio()

stream = p.open(format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                frames_per_buffer=CHUNK)





# create matplotlib figure and axes
fig, ax = plt.subplots(1, figsize=(15, 7))

# variable for plotting
x = np.arange(0, 2 * CHUNK, 2)

# create a line object with random data
line = ax.plot(x, np.random.rand(CHUNK), '-', lw=2)[0]
# basic formatting for the axes
ax.set_title('AUDIO WAVEFORM')
ax.set_xlabel('samples')
ax.set_ylabel('volume')
ax.set_xlim(0, 2 * CHUNK)
plt.setp(ax, xticks=[0, CHUNK, 2 * CHUNK], yticks=[-1000, 1000])
# show the plot
plt.show(block=False)


print("* recording")
frames = []
for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):    
    data = stream.read(CHUNK)
    frames.append(data)

    result = np.frombuffer(data, dtype=np.int16)
    line.set_ydata(result)
    fig.canvas.draw()
    fig.canvas.flush_events()



filename = "output.wav"

# Save the recorded data as a WAV file
wf = wave.open(filename, 'wb')
wf.setnchannels(CHANNELS)
wf.setsampwidth(p.get_sample_size(FORMAT))
wf.setframerate(RATE)
wf.writeframes(b''.join(frames))
wf.close()