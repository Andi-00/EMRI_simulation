import numpy as np
from matplotlib import pyplot as plt
from gwpy.spectrogram import Spectrogram

data = np.genfromtxt("Simulation/params.csv", delimiter=",")

m = data[:, 0]

for mass in m:
    print("{:.2e}".format(mass))