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
dt = 0.1  # seconds

# M = 1e7  # solar mass
mu = 10  # solar mass

dist = 1.0  # distance in Gpc

p0 = 12.0
e0 = 0.2
x0 = 1.0  # will be ignored in Schwarzschild waveform

qS = 0.0  # polar sky angle
phiS = 0.0  # azimuthal viewing angle


# spin related variables
a = 0.6  # will be ignored in Schwarzschild waveform
qK = 0  # polar spin angle
phiK = 0.0  # azimuthal viewing angle


# Phases in r, theta and phi
Phi_phi0 = 0
Phi_theta0 = 0
Phi_r0 = 0


from matplotlib import cm
cmap = cm.get_cmap('plasma')


fig, ax = plt.subplots(figsize = (20, 9))




for M in range(4, 8):
    h = gen_wave(10 ** M, mu, a, p0, e0, x0, dist, qS, phiS, qK, phiK, Phi_phi0, Phi_theta0, Phi_r0, T=T, dt=dt)

    col = cmap((M - 4)/ 3)

    t = np.arange(len(h.real)) * dt
    ax.plot(t, h.real * 1E22, color = col)

ax.set_xlabel("time $t /$a")
ax.set_ylabel("strain $h_+ / 10^{22}$")
ax.grid()

cbar = plt.colorbar(cm.ScalarMappable(cmap=cmap), ax=ax, label = "mass $M / M_\odot$", ticks = np.arange(0, 1.1, 1/3))
cbar.ax.set_yticklabels(["$10^4$", "$10^5$", "$10^6$", "$10^7$"])

ax.set_title("Mass comparison", y = 1.02)
plt.savefig("Massen_vgl.png")


# h0 = gen_wave(10 ** 4, mu, a, p0, e0, x0, dist, qS, phiS, qK, phiK, Phi_phi0, Phi_theta0, Phi_r0, T=T, dt=dt)
# h1 = gen_wave(10 ** 7, mu, a, p0, e0, x0, dist, qS, phiS, qK, phiK, Phi_phi0, Phi_theta0, Phi_r0, T=T, dt=dt)

# t0 = np.arange(len(h0.real)) * dt
# t1 = np.arange(len(h1.real)) * dt

# ax[0].plot(t0[: 300], h0.real[: 300] * 1E22, color = cmap(0), label = "$M = 10^4 \cdot M_\odot $")
# ax[1].plot(t1[: 300000], h1.real[: 300000] * 1E22, color = cmap(1 / 3), label = "$M = 10^7 \cdot M_\odot $")

# ax[0].set_ylabel("strain $h_+ / 10^{22}$")
# ax[1].set_ylabel("strain $h_+ / 10^{22}$")

# ax[0].set_title("Timescales for different $M$", y = 1.02)
# ax[1].set_xlabel("time $t$ / s")


# ax[0].legend()
# ax[1].legend()

# ax[0].grid()
# ax[1].grid()

# plt.savefig("Massen_timescales.png")



plt.show()