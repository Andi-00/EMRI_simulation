from scipy import signal
from scipy.fft import fftshift
from matplotlib import pyplot as plt
import numpy as np


from gwpy.timeseries import TimeSeries

plt.rcParams['pgf.rcfonts'] = False
plt.rcParams['font.serif'] = []
plt.rcParams['font.family'] = 'serif'
plt.rcParams['text.usetex'] = True
plt.rcParams['axes.formatter.useoffset'] = False
plt.rcParams['lines.linewidth'] = 2
plt.rcParams['errorbar.capsize'] = 2
plt.rcParams['grid.linewidth'] = 0.5
plt.rcParams['axes.labelsize'] = 20
plt.rcParams['axes.titlesize'] =20
plt.rcParams['xtick.labelsize'] = 14
plt.rcParams['ytick.labelsize'] = 14
plt.rcParams['legend.title_fontsize'] = 14
plt.rcParams['legend.fontsize'] = 14
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['savefig.bbox'] = 'tight'
plt.rcParams['savefig.pad_inches'] = 0.1

#plt.rcParams['savefig.transparent'] = True
plt.rcParams['figure.figsize'] = (8, 6)


fs = 1E4
T = 1
t = np.arange(0, T, 1 / fs)


x = np.sum(np.array([np.sin(5 * i * t + 5 * (i % 2) * t ** 2 ) for i in range(20)]), axis = 0)

data = TimeSeries(x, dt = 1 / fs)

fig, ax = plt.subplots()
ax.plot(data, color = "royalblue", zorder = 10)
ax.set_xlabel("time $t$ [s]")

plt.savefig("./Simulation/Timeseries.png")

hq = data.q_transform(qrange=(8,128), frange=(10,500), logf=True, whiten=False)
fig4 = hq.plot(figsize=[12, 10])
ax = fig4.gca()
fig4.colorbar(label="Normalised energy")
ax.grid(False)
ax.set_yscale('log')
ax.set_xlabel('Time [s]')

plt.show()

# This is the spectogram using scipy

# print(x.shape)

# f, t, Sxx = signal.spectrogram(x, fs)

# plt.pcolormesh(t, f, np.log(Sxx))

# plt.ylabel('Frequency [Hz]')

# plt.xlabel('Time [sec]')
# plt.colorbar()

# plt.show()
