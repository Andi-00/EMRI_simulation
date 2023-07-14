import numpy as np
from matplotlib import pyplot as plt
from gwpy.spectrogram import Spectrogram

data0 = np.swapaxes(np.genfromtxt("Simulation/data/data_000000.csv", delimiter=","), 0, 1)
data1 = np.swapaxes(np.genfromtxt("Simulation/data/data_000001.csv", delimiter=","), 0, 1)
data2 = np.swapaxes(np.genfromtxt("Simulation/data/data_000002.csv", delimiter=","), 0, 1)
data3 = np.swapaxes(np.genfromtxt("Simulation/data/data_000003.csv", delimiter=","), 0, 1)
data4 = np.swapaxes(np.genfromtxt("Simulation/data/data_000004.csv", delimiter=","), 0, 1)

print(data0.shape, data1.shape, data2.shape, data3.shape, data4.shape)

x = np.arange(61) - 0.5
y = np.arange(-1, 2001) * 5E-5 - 0.5 * 5E-5

z = data


fig, ax = plt.subplots()

ax.pcolormesh(x, y, z, norm = "log", vmin = 2E-5 * np.max(z))
ax.set_yscale("log")

ax.set_ylim(1E-4, 1E-1 + 1E-4)



plt.savefig("./test_grid.png")
