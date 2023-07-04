from scipy import signal
from scipy.fft import fftshift
import matplotlib.pyplot as plt
import numpy as np

fs = 1E-1
T = 10000
t = np.arange(0, T, 1 / fs)


x = np.sum(np.array([np.sin(5 * i * t) for i in range(50)]), axis = 0)

print(x.shape)

f, t, Sxx = signal.spectrogram(x, fs)

plt.pcolormesh(t, f, np.log(Sxx))

plt.ylabel('Frequency [Hz]')

plt.xlabel('Time [sec]')
plt.colorbar()

plt.show()
