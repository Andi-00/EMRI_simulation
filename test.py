import sys
import os

from gwpy.timeseries import TimeSeries

from matplotlib import pyplot as plt
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
plt.rcParams["axes.grid"] = True


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


alpha = 1

# parameters
M = 1e7
mu = 1e4
p0 = 11.0
e0 = 0.7
theta = np.pi/3  # polar viewing angle
phi = np.pi/4  # azimuthal viewing angle

theta, phi = 0, 0

dt = 30.0
# dist = 1

wave = few(M, mu, p0, e0, theta, phi, dist = 1, dt=dt, T = 0.5)  #  assumes dt = 10.0 for max T = 1.0 year


print(wave)
print(wave.shape)
print(len(wave.real))

t = np.arange(len(wave.real)) * dt


gen_wave = GenerateEMRIWaveform("FastSchwarzschildEccentricFlux")

fig, ax = plt.subplots(figsize = (20, 9))

# ax.plot(t[-1000:], wave.real[-1000:], color = "royalblue")
# ax.plot(t[-1000:], wave.imag[-1000:], color = "crimson")

ax.plot(t, wave.real, color = "royalblue")

ax.set_xlabel("Time $t$ [s]")
ax.set_ylabel("Strain $h$")
ax.set_title("EMRI for a mass ratio of {:.1E}".format(mu / M), y = 1.02)

plt.savefig("EMRI_wave.png")


traj = EMRIInspiral(func="SchwarzEccFlux")



# run trajectory
# must include for generic inputs, will fix a = 0 and x = 1.0
t, p, e, x, Phi_phi, Phi_theta, Phi_r = traj(M, mu, 0.0, p0, e0, 1.0, new_t=t, upsample=True, fix_t=True)

r = p / (1 + e * np.cos(Phi_r))


print(min(r))

from matplotlib import cm
cmap = cm.get_cmap('plasma')

x = r * np.cos(Phi_phi)
y = r * np.sin(Phi_phi)

fig, ax = plt.subplots(figsize = (20, 9))

m = 100
x = r * np.cos(Phi_phi)
y = r * np.sin(Phi_phi) 

for i in range(1, m):

    color = cmap(i / m)

    k = int(len(wave.real) / m)
    a = x[k * (i - 1) : k * i + 1]
    b = y[k * (i - 1) : k * i + 1]

    ax.plot(a, b, color = color)

fig.colorbar(cm.ScalarMappable(cmap=cmap), ax=ax,  label = "time")

ax.set_xlabel("Dimensionless x")
ax.set_ylabel("Dimensionless y")
ax.set_title("Trajectory of the second mass in the $x$-$y$-plane", y = 1.02)

plt.savefig("2d_trajectory.png")

fig, ax = plt.subplots(figsize = (18, 9), subplot_kw={"projection": "3d"})


x = r * np.sin(Phi_theta) * np.cos(Phi_phi)
y = r * np.sin(Phi_theta) * np.sin(Phi_phi)
z = r * np.cos(Phi_theta)


m = 100

for i in range(1, m):

    color = cmap(i / m)

    k = int(len(wave.real) / m)
    a = x[k * (i - 1) : k * i + 1]
    b = y[k * (i - 1) : k * i + 1]
    c = z[k * (i - 1) : k * i + 1]

    ax.plot(a, b, c, color = color)

ax.scatter(0,1,0, s = 800, color = "black")

fig.colorbar(cm.ScalarMappable(cmap=cmap), ax=ax, label = "time")

ax.set_xlabel("Dimensionless x", fontsize = 22, labelpad = 10)
ax.set_ylabel("Dimensionless y", fontsize = 22, labelpad = 10)
ax.set_zlabel("Dimensionless z", fontsize = 22, labelpad = 10)
ax.set_title("Trajectory of the second mass", y = 1.02)

plt.savefig("3d_trajectory.png")

plt.show()