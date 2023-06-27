import numpy as np
from gwpy.timeseries import TimeSeries
import matplotlib.pyplot as plt

dt = 0.01

x = np.arange(0, 100, dt)
y = np.sin(x) + x * np.sin(4 * x)

data = TimeSeries(y, dt = dt)
specgram = data.spectrogram(2, fftlength=1, overlap=.5) ** (1/2.)

plot = specgram.imshow(norm='log')
ax = plot.gca()
ax.set_yscale('log')
ax.colorbar(label=r'Gravitational-wave amplitude [strain/$\sqrt{\mathrm{Hz}}$]')
plot.show()