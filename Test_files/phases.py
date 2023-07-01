import sys
import os

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

plt.rcParams["axes.titlesize"] = 32
plt.rcParams["axes.labelsize"] = 30
plt.rcParams["axes.titleweight"] = "bold"
plt.rcParams["xtick.labelsize"] = 22
plt.rcParams["ytick.labelsize"] = 22
plt.rcParams["legend.fontsize"] = 22
plt.rcParams["axes.linewidth"] = 1.2
plt.rcParams["scatter.marker"] = "."



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
T = 0.001  # years
dt = 10.0  # seconds

T = dt * 10000 / (365 * 24 * 3600)

M = 1e7  # solar mass
mu = 1e2  # solar mass

dist = 1.0  # distance in Gpc

p0 = 12.0
e0 = 0.2
x0 = 1.0  # will be ignored in Schwarzschild waveform

qS = 0.0  # polar sky angle
phiS = 0.0  # azimuthal viewing angle


# spin related variables
a = 0.9  # will be ignored in Schwarzschild waveform
qK = np.pi / 4  # polar spin angle
phiK = 0.0  # azimuthal viewing angle


# Phases in r, theta and phi
Phi_phi0 = np.random.uniform(0, 2 * np.pi)
Phi_theta0 = np.random.uniform(0, 2 * np.pi)
Phi_r0 = np.random.uniform(0, 2 * np.pi)

h = gen_wave(
    M,
    mu,
    a,
    p0,
    e0,
    x0,
    dist,
    qS,
    phiS,
    qK,
    phiK,
    Phi_phi0,
    Phi_theta0,
    Phi_r0,
    T=T,
    dt=dt,
)

wave = few(M, mu, p0, e0, qS, phiS, dt=dt, T=T, dist = dist) 

fig, ax = plt.subplots(figsize = (20, 9))

ax.plot(h.real, color = "royalblue")
ax.plot(wave.real, color = "crimson")

ax.grid()

traj = EMRIInspiral(func="pn5", enforce_schwarz_sep=True)

t, p, e, Y, Phi_phi, Phi_r, Phi_theta = traj(M, mu, a, p0, e0, x0,
                                             Phi_phi0=Phi_phi0, Phi_theta0=Phi_theta0, Phi_r0=Phi_r0, T=T, dt = dt, upsample=True, fix_t=True)

from matplotlib import cm
cmap = cm.get_cmap('plasma')

r = p / (1 + e * np.cos(Phi_r))

x = r * np.cos(Phi_phi)
y = r * np.sin(Phi_phi)

fig, ax = plt.subplots(figsize = (20, 9), subplot_kw={"projection": "3d"})

m = 100

x = r * np.sin(Phi_theta) * np.cos(Phi_phi)
y = r * np.sin(Phi_theta) * np.sin(Phi_phi)
z = r * np.cos(Phi_theta)

for i in range(1, m):

    color = cmap(i / m)

    k = int(len(wave.real) / m)
    a = x[k * (i - 1) : k * i + 1]
    b = y[k * (i - 1) : k * i + 1]
    c = z[k * (i - 1) : k * i + 1]

    ax.plot(a, b, c, color = color)

fig.colorbar(cm.ScalarMappable(cmap=cmap), ax=ax,  label = "time")

# temp = np.arange(-20, 20, 0.2)

# ax.plot(np.sin(qK) * temp, np.zeros_like(temp), np.cos(qK) * temp, color = "black", lw = 5)
# ax.scatter(0, 0, 0, s = 8000, color = "black")

plt.savefig("trajectory_plot.png")

plt.show()




# wave_generator = Pn5AAKWaveform(inspiral_kwargs=inspiral_kwargs, sum_kwargs=sum_kwargs, use_gpu=False)

# # set initial parameters
# M = 1e6
# mu = 1e1
# a = 0.2
# p0 = 14.0
# e0 = 0.6
# iota0 = 0.1
# Y0 = np.cos(iota0)
# Phi_phi0 = 0.2
# Phi_theta0 = 1.2
# Phi_r0 = 0.8


# qS = 0.2
# phiS = 0.2
# qK = 0.8
# phiK = 0.8
# dist = 1.0
# mich = False
# dt = 30.0
# T = 0.01

# waveform = wave_generator(M, mu, a, p0, e0, Y0, dist, qS, phiS, qK, phiK,
#                           Phi_phi0=Phi_phi0, Phi_theta0=Phi_theta0, Phi_r0=Phi_r0, mich=True, dt=dt, T=T)

# t = np.arange(len(waveform)) * dt
# plt.plot(t, waveform.real, color = "royalblue")
# plt.plot(t, waveform.imag, color = "crimson")

# plt.show()