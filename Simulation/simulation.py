import sys
import os

from gwpy.timeseries import TimeSeries

import matplotlib.pyplot as plt
import numpy as np

from few.trajectory.inspiral import EMRIInspiral
from few.amplitude.romannet import RomanAmplitude
from few.amplitude.interp2dcubicspline import Interp2DAmplitude
from few.waveform import FastSchwarzschildEccentricFlux, SlowSchwarzschildEccentricFlux, GenerateEMRIWaveform
from few.utils.utility import (get_overlap,
                               get_mismatch,
                               get_fundamental_frequencies,
                               get_separatrix,
                               get_mu_at_t,
                               get_p_at_t,
                               get_kerr_geo_constants_of_motion,
                               xI_to_Y,
                               Y_to_xI)

from few.utils.ylm import GetYlms
from few.utils.modeselector import ModeSelector
from few.summation.interpolatedmodesum import CubicSplineInterpolant
from few.waveform import SchwarzschildEccentricWaveformBase
from few.summation.interpolatedmodesum import InterpolatedModeSum
from few.summation.directmodesum import DirectModeSum
from few.utils.constants import *
from few.summation.aakwave import AAKSummation
from few.waveform import Pn5AAKWaveform, AAKWaveformBase

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

use_gpu = False

# keyword arguments for inspiral generator (RunSchwarzEccFluxInspiral)
inspiral_kwargs={
        "DENSE_STEPPING": 0,  # we want a sparsely sampled trajectory
        "max_init_len": int(1e3),  # all of the trajectories will be well under len = 1000
    }

# keyword arguments for inspiral generator (RomanAmplitude)
amplitude_kwargs = {
    "max_init_len": int(1e3),  # all of the trajectories will be well under len = 1000
    "use_gpu": use_gpu  # GPU is available in this class
}

# keyword arguments for Ylm generator (GetYlms)
Ylm_kwargs = {
    "assume_positive_m": False  # if we assume positive m, it will generate negative m for all m>0
}

# keyword arguments for summation generator (InterpolatedModeSum)
sum_kwargs = {
    "use_gpu": use_gpu,  # GPU is availabel for this type of summation
    "pad_output": False,
}

# THE FOLLOWING THREAD COMMANDS DO NOT WORK ON THE M1 CHIP, BUT CAN BE USED WITH OLDER MODELS
# EVENTUALLY WE WILL PROBABLY REMOVE OMP WHICH NOW PARALLELIZES WITHIN ONE WAVEFORM AND LEAVE IT TO
# THE USER TO PARALLELIZE FOR MANY WAVEFORMS ON THEIR OWN.

# set omp threads one of two ways
# num_threads = 4

# this is the general way to set it for all computations
# from few.utils.utility import omp_set_num_threads
# omp_set_num_threads(num_threads)

few = FastSchwarzschildEccentricFlux(
    inspiral_kwargs=inspiral_kwargs,
    amplitude_kwargs=amplitude_kwargs,
    Ylm_kwargs=Ylm_kwargs,
    sum_kwargs=sum_kwargs,
    use_gpu=use_gpu,
    # num_threads=num_threads,  # 2nd way for specific classes
)



gen_wave = GenerateEMRIWaveform("Pn5AAKWaveform")

# parameters
T = 0.2  # years
dt = 5  # seconds

M = 1e4  # solar mass
mu = 1  # solar mass

dist = 1.0  # distance in Gpc

p0 = 12.0
e0 = 0.2
x0 = 0.99  # will be ignored in Schwarzschild waveform

qS = 1E-6  # polar sky angle
phiS = 0.0  # azimuthal viewing angle


# spin related variables
a = 0.6  # will be ignored in Schwarzschild waveform
qK = 1E-6  # polar spin angle
phiK = 0.0  # azimuthal viewing angle


# Phases in r, theta and phi
Phi_phi0 = 0
Phi_theta0 = 0
Phi_r0 = 0

# Generate the random parameters for the EMRIs
# The parameters include M, mu / d, a, e0 and p0
def gen_parameters(N):
    n = np.random.uniform(4, 7, N)
    M = 10 ** n
    n = np.random.uniform(-2, 2, N)
    md = 10 ** n
    a = np.random.uniform(0, 1, N)
    e0 = np.random.uniform(0, 0.7, N)
    p0 = np.random.uniform(10, 16, N)

    parameters = np.array([np.array([M[i], md[i], a[i], e0[i], p0[i]]) for i in range(N)])

    return parameters

# Generate the strain h from the parameters par and returns it
def gen_strain(par):
    M = par[:, 0]
    mu = M * 9E-5
    d = mu / par[:, 1]
    a = par[:, 2]
    e0 = par[:, 3]
    p0 = par[:, 4]

    h = np.array([gen_wave(M[i], mu[i], a[i], p0[i], e0[i], x0, d[i], qS, phiS, qK, phiK, Phi_phi0, Phi_theta0, Phi_r0, T=T, dt=dt) for i in range(len(d))], dtype = object)
    
    return h

# Generate the spectrograms of the given strains
def gen_specs(hs):

    specs = []

    for h in hs:
        
        # Fill the strain array with zeros, so that all have the same length
        # independant of the signal duration
        hp = np.pad(h.real, (0, 1262326 - len(h)))
        hp = h.real

        ts = TimeSeries(hp, dt = dt)

        data = ts.spectrogram(1E4, nproc = 10) ** (1/2.)


        specs.append(data)

    return specs

# Save the spectrogram files as csv data
def save_files(specs):

    for i in range(len(specs)):
        np.savetxt("Simulation/data/data_{:06}.csv".format(i), np.array(specs[i]), delimiter = ",")




# gernerate 5 data sets
# data = gen_parameters(5)

# Save the parameters in a csv file
# np.savetxt("Simulation/params.csv", np.array(data), delimiter = ",")


data = np.genfromtxt("Simulation/params.csv", delimiter= ",")




h = gen_strain(data)
spec = gen_specs(h)
save_files(spec)




for s in spec:
    print(s.shape)

# h = gen_wave(M, mu, a, p0, e0, x0, dist, qS, phiS, qK, phiK, Phi_phi0, Phi_theta0, Phi_r0, T=T, dt=dt)


# data2 = TimeSeries(h[1].real, dt = dt)
# data3 = TimeSeries(h[2].real, dt = dt)

from gwpy.plot import Plot

# plot = Plot(data1, data2, data3)
# plot.show()

# Strain values below a 0.01 percent of the maximum value are not taken into account

for i in range(len(spec)):
    plot = spec[i].imshow(norm='log', vmin = 1E-2 * np.max(np.array(spec[i])))
    ax = plot.gca()
    ax.set_yscale('log')
    ax.set_ylim(1E-4, 1E-1)
    ax.grid(False)
    # ax.set_xlabel("Time $t$ [day]")
    ax.set_ylabel("Frequency $f$ [Hz]")
    ax.colorbar(
        label=r'Gravitational-wave amplitude [strain/$\sqrt{\mathrm{Hz}}$]')
    
    ax.set_title("Spectrogram of the wave", y = 1.02)
    plot.savefig("presentation/sensity_1E-2_{:02}.png".format(i))




