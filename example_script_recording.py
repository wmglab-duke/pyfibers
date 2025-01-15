"""Example use case of wmglab_neuron.

NOTE this is for development only
"""

from __future__ import annotations

import sys

import matplotlib.pyplot as plt
import numpy as np
from neuron import h

# sys.path.append(r'C:\nrn\lib\python') #noqa: E800
sys.path.append(r'/Applications/NEURON-7.8/lib/python')

from pyfibers import FiberModel, ScaledStim, build_fiber  # noqa: E402

nodecount = 133

model = FiberModel.SMALL_MRG_INTERPOLATION  # type of fiber model

# create fiber
fiber = build_fiber(diameter=5.7, fiber_model=model, temperature=37, n_sections=nodecount)

# create curve of potentials
fiber.potentials = fiber.point_source_potentials(0, 1000, fiber.length / 2, 1, 1)
plt.plot(fiber.potentials)

# Arbitrarily set point_source_potentials arguments to generate some sort of mock potentials for recording.


# create biphasic square wave to use as a stimulation waveform
waveform = np.concatenate((np.zeros(20), np.ones(20), np.zeros(1)))

# parameters
time_step = 0.005  # timestep
time_stop = 10  # duration of simulation

# Create instance of ScaledStim class
stimulation = ScaledStim(waveform=waveform, dt=time_step, tstop=time_stop)

# turn on saving gating parameters and Vm before running the simulations for thresholds
fiber.record_gating()
fiber.record_vm()
fiber.record_im(allsec=True)
fiber.record_vext()


# run threshold search
ap, time = stimulation.find_threshold(fiber)
# %%
plt.figure()
for key, value in fiber.gating.items():
    plt.plot(stimulation.time, value[fiber.loc_index(0.6)], label=key)
plt.legend()
ax2 = plt.gca().twinx()
plt.sca(ax2)
plt.plot(stimulation.time, fiber.vm[fiber.loc_index(0.6)], label='vm', color='k')
plt.ylabel('Vm')
plt.xlim(0, 4)

# %% Record single fiber action potenital
rec_potentials = fiber.point_source_potentials(0, 100, fiber.length * 3 / 4, 1, 1)
plt.figure()
plt.plot(rec_potentials)
plt.show()

downsample = 1
sfap, ds_time = fiber.record_sfap(rec_potentials, downsample=downsample)
plt.figure()
plt.plot(ds_time, sfap)
plt.xlim(0, 4)
plt.show()

# turn on saving gating parameters and Vm before running the simulations for thresholds
fiber.record_gating(recording_dt=1)
fiber.record_vm(recording_dt=1)

ds_time = h.Vector().record(h._ref_t, 1)

# run threshold search
ap, time = stimulation.find_threshold(fiber)

plt.figure()
for key, value in fiber.gating.items():
    plt.plot(ds_time, value[fiber.loc_index(0.6)], label=key)
plt.legend()
ax2 = plt.gca().twinx()
plt.sca(ax2)
plt.plot(ds_time, fiber.vm[fiber.loc_index(0.6)], label='vm', color='k')
plt.ylabel('Vm')
plt.xlim(0, 4)
plt.show()


# %% turn on saving gating parameters and Vm before running the simulations for thresholds
rec_vec = h.Vector([1, 2.5, 3])
fiber.record_gating(recording_tvec=rec_vec)
fiber.record_vm(recording_tvec=rec_vec)


# run threshold search
ap, time = stimulation.find_threshold(fiber)

plt.figure()
for key, value in fiber.gating.items():
    plt.plot(rec_vec, value[fiber.loc_index(0.6)], label=key)
plt.legend()
ax2 = plt.gca().twinx()
plt.sca(ax2)
plt.plot(ds_time, fiber.vm[fiber.loc_index(0.6)], label='vm', color='k')
plt.ylabel('Vm')
plt.xlim(0, 4)
plt.show()
