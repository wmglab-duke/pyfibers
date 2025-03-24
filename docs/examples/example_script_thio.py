"""Example use case of wmglab_neuron.

NOTE this is for development only
"""

from __future__ import annotations

import sys

import matplotlib.pyplot as plt
import numpy as np

from pyfibers import FiberModel, ScaledStim, build_fiber  # noqa: E402

sys.path.append(r'C:\nrn\lib\python')  # noqa: E800


length = 2500

diameter = 0.5

model = FiberModel.THIO_CUTANEOUS  # type of fiber model
# NOTE: Issue with autonomic is somehow APs are being detected at negative time. Need to investigate

# create fiber
fiber = build_fiber(diameter=diameter, fiber_model=model, temperature=37, length=length)

# create curve of potentials
fiber.potentials = fiber.point_source_potentials(0, 100, fiber.length / 5, 1, 1)
plt.plot(fiber.potentials)

# create biphasic square wave to use as a stimulation waveform
waveform = np.concatenate((np.zeros(100), np.ones(20), np.zeros(1)))

# parameters
time_step = 0.005  # timestep
time_stop = 10  # duration of simulation

# Create instance of ScaledStim class
stimulation = ScaledStim(waveform=waveform, dt=time_step, tstop=time_stop, dt_init_ss=5)

# turn on saving gating parameters and Vm before running the simulations for thresholds
fiber.set_save_vm()

# run threshold search
# amp, ap = stimulation.find_threshold(fiber, stimamp_top=-1)

# report the threshold amplitude with print statement to terminal
# print(f'Threshold for {diameter} micron {model}: {amp} (mA)')

# run a finite amp (i.e., one amplitude, not in a bisection search as was done above)
ap, time = stimulation.run_sim(-1, fiber)

plt.figure()
plt.plot(stimulation.time, fiber.vm[fiber.loc_index(0.9)], label='end')
plt.plot(stimulation.time, fiber.vm[fiber.loc_index(0.5)], label='center')
plt.plot(stimulation.time, fiber.vm[fiber.loc_index(0.7)], label='7')

plt.axvline(0.5, color='red')
plt.legend()
plt.ylim(-80, 80)
print(f'cv={fiber.measure_cv(start=0.5)}')
